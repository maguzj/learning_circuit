import numpy as np
from circuit_utils import Circuit
import jax
import jax.numpy as jnp
from update_rules import *

class SubspaceIndexer:
    def __init__(self, ne, var_indices, nklist, dklist, fixed_values=None, order='C'):
        self.ne = ne # number of edges
        self.var_indices = np.array(var_indices, dtype=int) # indices of variable edges
        self.var_ne = len(var_indices) # number of variable edges
        self.nklist = np.array(nklist, dtype=int) # number of states per variable edge
        self.dklist = np.array(dklist, dtype=float) # dk per variable edge
        self.order = order # ????
        self.fixed_values = (
            np.zeros(ne) if fixed_values is None else np.array(fixed_values, dtype=float)
        )
        self.shape = tuple(self.nklist)
        self.S = int(np.prod(self.nklist))

    def index_from_k(self, k):
        """Map a full k-vector to integer index (0-based)."""
        d = np.rint((k[self.var_indices] / self.dklist) - 1).astype(int)
        return np.ravel_multi_index(tuple(d), self.shape, order=self.order)

    def k_from_index(self, idx):
        """Inverse: index -> full k-vector (with fixed entries)."""
        d = np.array(np.unravel_index(idx, self.shape, order=self.order))
        k = self.fixed_values.copy()
        k[self.var_indices] = (d + 1) * self.dklist
        return k

from scipy.sparse import csr_matrix

def build_Q_subspace(indexer, rate_fn, inputs, targets, boundary='reflecting'):
    """
    Construct sparse rate matrix Q for a subspace indexer.
    indexer : SubspaceIndexer instance
    rate_fn : function rate_fn(k, var_dim, dir)
    boundary : 'periodic' | 'reflecting' | 'absorbing'
    """
    var_ne = indexer.var_ne
    var_indices = indexer.var_indices 
    nklist = indexer.nklist
    shape = indexer.shape
    S = indexer.S
    dklist = indexer.dklist
    ne = indexer.ne

    rows, cols, data = [], [], []

    for s in range(S):
        # recover the full k-vector (includes fixed entries)
        k = indexer.k_from_index(s)
        out_sum = 0.0

        # loop only over variable dimensions
        for dim in var_indices:
            for dir in (-1, +1):
                d = np.array(np.unravel_index(s, shape, order=indexer.order))
                d2 = d.copy()
                d2[dim] += dir

                if d2[dim] < 0 or d2[dim] >= nklist[dim]:
                    if boundary == 'periodic':
                        d2[dim] %= nklist[dim]
                    elif boundary in ('reflecting', 'absorbing'):
                        continue

                s2 = np.ravel_multi_index(tuple(d2), shape, order=indexer.order)
                r = float(rate_fn(k, dim, dir, inputs, targets))
                if r > 0:
                    rows.append(s)
                    cols.append(s2)
                    data.append(r)
                    out_sum += r

        rows.append(s)
        cols.append(s)
        data.append(-out_sum)

    return csr_matrix((data, (rows, cols)), shape=(S, S))


# ---------- Rate function factory (diffusion + rectified drift) ----------
def rate_fn_factory(diffusion, f_component):
    """
    diffusion: scalar or array-like length var_ne. Contributes D / dk^2 to both ± neighbors.
    f_component: callable f_component(k, var_dim,dir, datapoint) -> float
        Component of the 'force' along the variable dimension var_dim.

    Returns:
      make_rate_fn(indexer) -> rate_fn(k, var_dim, dir, datapoint)
    """
    def _as_array(x, var_ne):
        x = np.asarray(x, dtype=float) if np.ndim(x) else float(x)
        if np.ndim(x) == 0:
            return np.full(var_ne, float(x))
        return np.asarray(x, dtype=float)

    def make_rate_fn(indexer: SubspaceIndexer):
        D = _as_array(diffusion, indexer.var_ne)

        def rate_fn(k, var_dim, dir, inputs, targets):
            dk = indexer.dklist[var_dim]
            diff_part = D[var_dim] / (dk * dk)
            drive = float(f_component(k, var_dim, inputs, targets))
            drift_part = max(dir * drive, 0.0) / dk
            return diff_part + drift_part

        return rate_fn

    return make_rate_fn



def s_contrastive_single(conductances, incidence_matrix,Q_inputs,Q_outputs, Q_clamped, eta,inputs, true_output):
    V_free = Circuit.s_solve(conductances, incidence_matrix, Q_inputs, inputs)
    y_free = Q_outputs.T.dot(V_free)
    nudge = y_free+ eta * (true_output-y_free)
    clamped_inputs  = jnp.concatenate((inputs,nudge),axis = 0)
    V_clamped = Circuit.s_solve(conductances, incidence_matrix, Q_clamped, clamped_inputs)
    voltage_drop_free = incidence_matrix.T.dot(V_free)
    power_free =np.sum(conductances*(voltage_drop_free**2))
    voltage_drop_clamped = incidence_matrix.T.dot(V_clamped)
    power_clamped =np.sum(conductances*(voltage_drop_clamped**2))
    contrast = power_clamped - power_free
    return contrast

@jax.jit
def s_contrastive(conductances, incidence_matrix,Q_inputs,Q_outputs, Q_clamped, eta, inputs, true_output):
    batch_contrastive = jax.vmap(s_contrastive_single, in_axes=(None, None, None, None, None, None, 0, 0))
    contrastive_values = batch_contrastive(conductances, incidence_matrix,Q_inputs,Q_outputs, Q_clamped,eta,  inputs, true_output)
    return jnp.mean(contrastive_values)

@jax.jit
def s_grad_contrastive(conductances, incidence_matrix,Q_inputs,Q_outputs, Q_clamped, eta, inputs, true_output):
    ''' Compute the gradient of the contrastive function '''
    grad_func = jax.grad(s_contrastive, argnums=0)
    return grad_func(conductances, incidence_matrix,Q_inputs,Q_outputs, Q_clamped, eta, inputs, true_output)


def find_success_indices(indexer, incidence_matrix, Q_inputs, Q_outputs, inputs, targets, eps):
    """Indices i with s_mse(k_i, ...) < eps."""
    mse = np.array([
        s_mse(indexer.k_from_index(i), incidence_matrix, Q_inputs, Q_outputs, inputs, targets)
        for i in range(indexer.S)
    ], dtype=float)
    success = np.flatnonzero(mse < eps)
    return success, mse


# mfpt_periodic.py
# Sparse MFPT / Exit-time solvers for CTMCs, constant or m-phase periodic generators.
# Requires: numpy, scipy (sparse, splu, gmres, expm_multiply)

# from __future__ import annotations
import numpy as np
from typing import Iterable, List, Sequence, Tuple, Optional
from scipy.sparse import csr_matrix, csc_matrix, diags, issparse
from scipy.sparse.linalg import splu, spsolve, LinearOperator, gmres, lgmres, expm_multiply


# --------- Utilities ---------

def _as_csr(M):
    return M if (issparse(M) and M.format == "csr") else csr_matrix(M)

def _submatrix(Q_full: csr_matrix, inds: Sequence[int]) -> csr_matrix:
    """Return Q_full[inds, inds] as CSR."""
    return _as_csr(Q_full.tocsr()[inds, :][:, inds])

def _complement(n: int, inds: Sequence[int]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(inds, dtype=int)] = False
    return np.nonzero(mask)[0]

def _apply_A(Q: csr_matrix, v: np.ndarray, tau: float) -> np.ndarray:
    """Return (exp(-Q * tau) @ v) via Krylov; Q is sparse."""
    return expm_multiply((-Q) * tau, v)

from scipy.sparse.linalg import onenormest

def apply_A_chunked(Q: csr_matrix, v: np.ndarray, tau: float, smax = 5.0) -> np.ndarray:
    n1 = onenormest(Q)
    K  = max(1, int(np.ceil(n1 * tau / smax)))
    dt = tau / K
    y = v
    for _ in range(K):
        y = expm_multiply((-Q) * dt, y)
    return y


def _compute_B(Q: csr_matrix, tau: float, lu: Optional[splu] = None) -> Tuple[np.ndarray, splu]:
    """
    B = Q^{-1} (I - A) 1, where A = exp(-Q tau).
    Returns (B, LU), caching LU if provided.
    """
    n = Q.shape[0]
    ones = np.ones(n)
    A1 = _apply_A(Q, ones, tau)       # A @ 1
    rhs = ones - A1
    if lu is None:
        lu = splu(Q.tocsc())          # M-matrix: should be solvable on the relevant class
    return lu.solve(rhs), lu


def compute_B_chunked(Q: csr_matrix, tau: float, lu: Optional[splu] = None, smax = 5.0) -> Tuple[np.ndarray, splu]:
    n1 = onenormest(Q)
    K  = max(1, int(np.ceil(n1 * tau / smax)))
    dt = tau / K
    n  = Q.shape[0]
    ones = np.ones(n)
    if lu is None:
        lu = splu(Q.tocsc())

    # one safe small-step
    AΔ1 = expm_multiply((-Q) * dt, ones)
    BΔ  = lu.solve(ones - AΔ1)  # Q^{-1}(I - e^{-QΔ})1

    b = np.zeros(n)
    w = BΔ.copy()
    for _ in range(K):
        b += w                  # accumulate sum
        w = expm_multiply((-Q) * dt, w)  # w <- e^{-QΔ} w
    return b, lu



# --------- Constant generator (homogeneous) ---------

def mfpt_constant(Q_full: csr_matrix, B_inds: Sequence[int]) -> np.ndarray:
    """
    MFPT to B for time-homogeneous CTMC with generator Q_full.
    Returns T on all states (T=0 on B).
    Solves: -Q_CC T_C = 1, T_B = 0.
    """
    Q = _as_csr(Q_full)
    n = Q.shape[0]
    B = np.asarray(B_inds, dtype=int)
    C = _complement(n, B)

    QC = _submatrix(Q, C)
    ones = np.ones(len(C))
    # Solve without explicit inverse
    TC = spsolve(-QC, ones)
    T = np.zeros(n)
    T[C] = TC
    return T


def exit_time_constant(Q_full: csr_matrix, B_inds: Sequence[int]) -> np.ndarray:
    """
    Mean exit time from B for homogeneous CTMC.
    Solves: -Q_BB E = 1 on B.
    """
    Q = _as_csr(Q_full)
    n = Q.shape[0]
    B = np.asarray(B_inds, dtype=int)
    QBB = _submatrix(Q, B)
    ones = np.ones(len(B))
    EB = spsolve(-QBB, ones)
    E = np.zeros(n)
    E[B] = EB

    return E


# --------- Periodic (m phases, piecewise constant) ---------

def _linear_operator_I_minus_prod_A(Q_list: List[csr_matrix],
                                     tau_list: Sequence[float],
                                     n: int) -> LinearOperator:
    """
    Return LinearOperator M ~ (I - A_m ... A_1).
    matvec(x): y = x - A_m(...A_1 x).
    """
    taus = list(tau_list)
    def matvec(x):
        y = x
        for Q, tau in zip(Q_list, taus):      # A_1, then A_2, ..., A_m
            y = _apply_A(Q, y, tau)
        return x - y                          # (I - A_m...A_1) x
    return LinearOperator((n, n), matvec=matvec, dtype=float)

def make_M_op_chunked(Q_list, tau_list, n, max_scale=5.0):
    def mv(x):
        y = x
        for Q, tau in zip(Q_list, tau_list):
            y = apply_A_chunked(Q, y, tau, smax=max_scale)
        return x - y
    return LinearOperator((n, n), matvec=mv, dtype=float)


def _rhs_periodic(Q_list: List[csr_matrix],
                  tau_list: Sequence[float],
                  B_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Build RHS:  -(A_m...A_{r+1}) B_r  summed over r=1..m.
    Uses O(m^2) expm_multiply calls (fine for modest m).
    """
    m = len(Q_list)
    n = Q_list[0].shape[0]
    rhs = np.zeros(n)
    for r in range(m):
        v = B_vectors[r].copy()
        for k in range(r+1, m):               # apply A_{r+1},...,A_m to B_r
            v = _apply_A(Q_list[k], v, tau_list[k])
        rhs -= v
    return rhs

def rhs_periodic_chunked(Q_list, tau_list, B_list, max_scale=5.0):
    m = len(Q_list)
    rhs = np.zeros_like(B_list[0])
    for r in range(m):
        v = B_list[r].copy()
        for k in range(r+1, m):
            v = apply_A_chunked(Q_list[k], v, tau_list[k], smax=max_scale)
        rhs -= v
    return rhs


def _precond_sum_tauQ(Q_list: List[csr_matrix], tau_list: Sequence[float]) -> Optional[splu]:
    """
    Optional left-preconditioner for (I - A_m...A_1): M ≈ sum tau_i * Q_i (M-matrix).
    Returns LU factor or None if factorization fails.
    """
    try:
        n = Q_list[0].shape[0]
        M = csr_matrix((n, n))
        for Q, tau in zip(Q_list, tau_list):
            M = M + Q * tau
        return splu(M.tocsc())
    except Exception:
        return None


def _solve_periodic_block(Q_full_list: List[csr_matrix],
                          active_inds: Sequence[int],
                          tau_list: Sequence[float],
                          *,
                          gmres_tol: float = 1e-10,
                          gmres_maxiter: int = 500,
                          use_lgmres: bool = False) -> np.ndarray:
    """
    Core solver for periodic MFPT/exit-time at cycle start.
    - Restrict to active index set I (I = C for MFPT-to-B, or I = B for exit-from-B).
    - Build Q_i = Q_full_i[I, I] for all phases.
    - Compute B_i = Q_i^{-1}(I - A_i) 1 via one expm_multiply + one sparse solve per phase.
    - Solve (I - A_m...A_1) U0 = -(sum over r of A_{r+1}...A_m B_r) using GMRES/LGMRES.
    Returns U0 on the active set I.
    """
    m = len(Q_full_list)
    assert m == len(tau_list), "Q_full_list and tau_list must have same length."

    # Subgenerators on active set
    Qs = [_submatrix(Qf, active_inds) for Qf in Q_full_list]
    n = Qs[0].shape[0]

    # B_i and (optional) LU cache for Q_i
    B_vecs = []
    LU_cache = []
    for Qi, tau in zip(Qs, tau_list):
        # Bi, LU = _compute_B(Qi, tau)
        Bi, LU = compute_B_chunked(Qi, tau)
        B_vecs.append(Bi)
        LU_cache.append(LU)

    # Linear operator for (I - A_m...A_1)
    M = _linear_operator_I_minus_prod_A(Qs, tau_list, n)
    # M = make_M_op_chunked(Qs, tau_list, n)    

    # RHS vector
    b = _rhs_periodic(Qs, tau_list, B_vecs)
    # b = rhs_periodic_chunked(Qs, tau_list, B_vecs)
    

    # Optional preconditioner M_prec ≈ (Σ tau_i Q_i)^{-1}
    LU_prec = _precond_sum_tauQ(Qs, tau_list)
    if LU_prec is not None:
        def prec_mv(v):
            return LU_prec.solve(v)
        P = LinearOperator((n, n), matvec=prec_mv, dtype=float)
    else:
        P = None

    solver = lgmres if use_lgmres else gmres
    U0, info = solver(M, b, M=P, atol=gmres_tol, maxiter=gmres_maxiter)
    if info != 0:
        # Non-zero info means not fully converged (or break-down).
        # You can still return U0; caller may inspect residuals externally.
        print("Not converged after {} iterations. Norm(A.x-b) = {}\n".format(info,np.linalg.norm(M.dot(U0)-b)))
    else:
        print("Converged. Norm(A.x-b) = {}\n".format(np.linalg.norm(M.dot(U0)-b)))
    return U0


# --------- Public API (periodic; m phases) ---------

def mfpt_periodic(Q_full_list: List[csr_matrix],
                  B_inds: Sequence[int],
                  tau_list: Sequence[float],
                  *,
                  gmres_tol: float = 1e-10,
                  gmres_maxiter: int = 1000,
                  use_lgmres: bool = False) -> np.ndarray:
    """
    MFPT to B at cycle start (phase index 0) for an m-phase periodic CTMC.
    Inputs:
      - Q_full_list: [Q^(1), ..., Q^(m)] full generators on S (CSR/csc).
      - B_inds: indices of the target set B (0-based).
      - tau_list: durations [tau1, ..., taum] for each phase, sum = period T.
    Returns:
      - T0: MFPT vector on all states (0 on B), evaluated at the start of phase 1.
    """
    n = Q_full_list[0].shape[0]
    B = np.asarray(B_inds, dtype=int)
    C = _complement(n, B)

    # Solve on active set C
    T0_C = _solve_periodic_block(Q_full_list, C, tau_list,
                                 gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                 use_lgmres=use_lgmres)
    T0 = np.zeros(n)
    T0[C] = T0_C
    return T0


def exit_time_periodic(Q_full_list: List[csr_matrix],
                       B_inds: Sequence[int],
                       tau_list: Sequence[float],
                       *,
                       gmres_tol: float = 1e-10,
                       gmres_maxiter: int = 500,
                       use_lgmres: bool = False) -> np.ndarray:
    """
    Mean exit time from B at cycle start for an m-phase periodic CTMC.
    Returns E0 on B (length |B|). (Outside B, the quantity is not defined.)
    """
    n = Q_full_list[0].shape[0]
    B = np.asarray(B_inds, dtype=int)
    E0_B = _solve_periodic_block(Q_full_list, B, tau_list,
                                 gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                                 use_lgmres=use_lgmres)
    
    E0 = np.zeros(n)
    E0[B] = E0_B

    return E0


# --------- Optional: evaluate inside a period once T0 is known ---------

def propagate_inside_phase(T0_active: np.ndarray,
                           Q_full_list: List[csr_matrix],
                           active_inds: Sequence[int],
                           tau_list: Sequence[float],
                           s: float) -> np.ndarray:
    """
    Given the solution at phase start on the active set (C for MFPT, B for exit),
    evaluate U(s) inside the current cycle at time s ∈ [0, T).
    Uses piecewise formula: U(s) = exp(-Q_k * (s - t_{k-1})) U(t_{k-1}) - Q_k^{-1}(I - exp(-Q_k * (s - t_{k-1}))) 1
    where k is the phase that contains time s.

    Returns U(s) on the active set.
    """
    Qs = [_submatrix(Qf, active_inds) for Qf in Q_full_list]
    taus = list(tau_list)
    T = sum(taus)
    s_mod = s % T

    # walk through phases
    acc = 0.0
    U = T0_active.copy()
    for k, (Qk, tau) in enumerate(zip(Qs, taus)):
        if s_mod <= acc + tau:
            dt = s_mod - acc
            # U(dt) = e^{-Qk dt} U - Qk^{-1}(I - e^{-Qk dt}) 1
            ones = np.ones_like(U)
            U = _apply_A(Qk, U, dt) - splu(Qk.tocsc()).solve(ones - _apply_A(Qk, ones, dt))
            return U
        else:
            # advance full phase: U <- A_k U - B_k
            ones = np.ones_like(U)
            AkU = _apply_A(Qk, U, tau)
            Ak1 = _apply_A(Qk, ones, tau)
            Bk = splu(Qk.tocsc()).solve(ones - Ak1)
            U = AkU - Bk
            acc += tau

    # Should not reach here; numerical fallbacks:
    return U
