"""Experimental training variants built on top of the current LearningCircuit API.

The previous version of this module predated ``LearningCircuit.train`` and relied on
removed fields such as ``learning_step``, ``Q_free``, ``CL.MSE`` and the old
``learning`` base class. The code below keeps the public variant classes but rewires
them to the modern training/update flow used in ``learning.py`` and
``update_rules.py``.
"""

from typing import Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import jax.numpy as jnp
from scipy.sparse import hstack
from tqdm import tqdm

from circuit_utils import Circuit
from learning import LearningCircuit
from update_rules import (
    cl_update,
    cl_update_ferro_antiferro,
    gd_update,
    gd_update_ferro_antiferro,
)
from utils import circuit_input_batch


def _identity_modulation(conductances):
    if isinstance(conductances, jnp.ndarray):
        return jnp.ones_like(conductances)
    return np.ones_like(conductances)


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _stack_constraints(circuit: LearningCircuit, left, right):
    if circuit.jax:
        return jnp.concatenate((jnp.asarray(left), jnp.asarray(right)), axis=1)
    return hstack([left, right])


def _solve_batch(circuit: LearningCircuit, Q, inputs):
    if circuit.model_type == "ferro":
        if circuit.jax:
            return Circuit.s_solve_batch(
                circuit.conductances,
                circuit.incidence_matrix,
                jnp.asarray(Q),
                jnp.asarray(inputs),
            )
        return circuit.solve_batch(Q, inputs)

    if circuit.model_type == "antiferro":
        if circuit.jax:
            return Circuit.s_solve_batch_ferro_antiferro(
                circuit.conductances,
                circuit.incidence_matrix,
                jnp.asarray(Q),
                jnp.asarray(inputs),
            )
        return circuit.solve_batch_ferro_antiferro(Q, inputs)

    raise ValueError("model_type must be either 'ferro' or 'antiferro'")


def _clip_conductances(circuit: LearningCircuit):
    clip = jnp.clip if circuit.jax else np.clip
    sign = jnp.sign if circuit.jax else np.sign
    where = jnp.where if circuit.jax else np.where

    conductances = clip(circuit.conductances, circuit.min_k, circuit.max_k)
    epsilon = getattr(circuit, "epsilon", 1e-6)
    conductances = where(
        (conductances > -epsilon) & (conductances < epsilon),
        sign(conductances) * epsilon,
        conductances,
    )
    circuit.conductances = conductances


def _ensure_histories(circuit: LearningCircuit):
    if getattr(circuit, "loss_history", None) is None:
        circuit.loss_history = []
    if getattr(circuit, "power_history", None) is None:
        circuit.power_history = []
    if getattr(circuit, "energy_history", None) is None:
        circuit.energy_history = []
    if getattr(circuit, "checkpoint_epochs", None) is None:
        circuit.checkpoint_epochs = []
    if not hasattr(circuit, "current_power"):
        circuit.current_power = 0.0
    if not hasattr(circuit, "current_energy"):
        circuit.current_energy = 0.0


def _record_checkpoint(circuit: LearningCircuit, loss_epoch: float):
    _ensure_histories(circuit)
    circuit.loss_history.append(loss_epoch)
    circuit.power_history.append(float(circuit.current_power))
    circuit.energy_history.append(float(circuit.current_energy))
    circuit.checkpoint_epochs.append(int(circuit.epoch))


def _save_local_state(circuit: LearningCircuit, save_path: str):
    circuit.save_local(save_path + "_conductances.csv")


def _base_gd_update(circuit: LearningCircuit):
    if circuit.model_type == "antiferro":
        return gd_update_ferro_antiferro
    return gd_update


def _base_cl_update(circuit: LearningCircuit):
    if circuit.model_type == "antiferro":
        return cl_update_ferro_antiferro
    return cl_update


def thresholded_gd_update(
    circuit: LearningCircuit,
    batch: Tuple[np.ndarray, np.ndarray],
    modulating_f,
    lr: float,
    threshold: float,
):
    delta_k, loss = _base_gd_update(circuit)(circuit, batch, modulating_f, lr)
    abs_fn = jnp.abs if circuit.jax else np.abs
    mask = abs_fn(delta_k) > (lr * threshold)
    return delta_k * mask, loss


def thresholded_cl_update(
    circuit: LearningCircuit,
    batch: Tuple[np.ndarray, np.ndarray],
    modulating_f,
    lr: float,
    eta: float,
    threshold: float,
):
    delta_k, loss = _base_cl_update(circuit)(circuit, batch, modulating_f, lr, eta)
    abs_fn = jnp.abs if circuit.jax else np.abs
    mask = abs_fn(delta_k) > (lr * threshold)
    return delta_k * mask, loss


def directed_aging_update(
    circuit: LearningCircuit,
    batch: Tuple[np.ndarray, np.ndarray],
    modulating_f,
    lr: float,
):
    inputs = circuit_input_batch(
        circuit.jax,
        batch[0],
        circuit.indices_inputs,
        circuit.current_bool,
        circuit.n,
    )
    targets = batch[1]

    states = _solve_batch(circuit, circuit.Q_inputs, inputs)
    outputs = np.dot(np.asarray(states), _to_dense(circuit.Q_outputs))
    voltage_drops = np.dot(np.asarray(states), _to_dense(circuit.incidence_matrix))

    circuit.current_power = float(np.mean(np.sum(circuit.conductances * (voltage_drops ** 2), axis=1)))
    circuit.current_energy += float(np.sum(circuit.conductances * (voltage_drops ** 2)))

    delta_k = lr * (voltage_drops ** 2).mean(axis=0) * np.asarray(modulating_f(circuit.conductances))
    loss = float(np.mean(0.5 * (outputs - targets) ** 2))
    return delta_k, loss


def generalized_cl_update(
    circuit: LearningCircuit,
    batch: Tuple[np.ndarray, np.ndarray],
    modulating_f,
    lr: float,
    eta: float,
    exp_conductances: float,
    q_outputs=None,
):
    inputs = circuit_input_batch(
        circuit.jax,
        batch[0],
        circuit.indices_inputs,
        circuit.current_bool,
        circuit.n,
    )
    targets = batch[1]
    q_outputs = circuit.Q_outputs if q_outputs is None else q_outputs
    q_clamped = _stack_constraints(circuit, circuit.Q_inputs, q_outputs)

    states_free = _solve_batch(circuit, circuit.Q_inputs, inputs)
    y_free = np.dot(np.asarray(states_free), _to_dense(q_outputs))
    nudge = y_free + eta * (targets - y_free)
    clamped_inputs = np.concatenate((np.asarray(inputs), nudge), axis=1)
    states_clamped = _solve_batch(circuit, q_clamped, clamped_inputs)

    vd_free = np.dot(np.asarray(states_free), _to_dense(circuit.incidence_matrix))
    vd_clamped = np.dot(np.asarray(states_clamped), _to_dense(circuit.incidence_matrix))

    modulation = np.asarray(modulating_f(circuit.conductances))
    conductance_factor = np.asarray(circuit.conductances) ** exp_conductances
    delta_k = lr * (1.0 / eta) * ((vd_clamped ** 2 - vd_free ** 2).mean(axis=0))
    delta_k = delta_k * conductance_factor * modulation

    circuit.current_power = float(np.mean(np.sum(circuit.conductances * (vd_free ** 2), axis=1)))
    circuit.current_energy += float(np.sum(circuit.conductances * (vd_free ** 2)))

    loss = float(np.mean(0.5 * (y_free - targets) ** 2))
    return delta_k, loss


class restricted_CL(LearningCircuit):
    """Gradient-descent variant with small updates masked out."""

    def train_GD(
        self,
        threshold: float,
        train_data,
        learning_rate: float,
        modulating_f: Optional[Callable] = None,
        max_epochs: Optional[int] = None,
        verbose: bool = True,
        model_state_path: str = "model_states/",
        print_every: int = 100,
    ) -> Dict[str, list]:
        modulating_f = _identity_modulation if modulating_f is None else modulating_f
        if max_epochs is not None:
            self.max_epochs = max_epochs
        return self.train(
            train_data,
            thresholded_gd_update,
            learning_rate,
            modulating_f,
            threshold,
            verbose=verbose,
            model_state_path=model_state_path,
            print_every=print_every,
        )

    def train_CL(
        self,
        threshold: float,
        eta: float,
        train_data,
        learning_rate: float,
        modulating_f: Optional[Callable] = None,
        max_epochs: Optional[int] = None,
        verbose: bool = True,
        model_state_path: str = "model_states/",
        print_every: int = 100,
    ) -> Dict[str, list]:
        modulating_f = _identity_modulation if modulating_f is None else modulating_f
        if max_epochs is not None:
            self.max_epochs = max_epochs
        return self.train(
            train_data,
            thresholded_cl_update,
            learning_rate,
            modulating_f,
            eta,
            threshold,
            verbose=verbose,
            model_state_path=model_state_path,
            print_every=print_every,
        )


class robust_CL(LearningCircuit):
    """Variants that combine the base task with robustness-oriented updates."""

    def set_outputs2(self, indices_outputs2):
        self.indices_outputs2 = indices_outputs2
        self.Q_outputs2 = self.constraint_matrix(indices_outputs2)
        return self.Q_outputs2

    def train_DA(
        self,
        learning_rate,
        train_data,
        n_epochs,
        save_global=False,
        save_state=False,
        save_path="trained_circuit",
        save_every=1,
        verbose=True,
        modulating_f: Optional[Callable] = None,
    ):
        _ensure_histories(self)
        modulating_f = _identity_modulation if modulating_f is None else modulating_f

        if self.epoch == 0:
            loss_epoch = _run_epoch(self, train_data, directed_aging_update, learning_rate, modulating_f, False)
            _record_checkpoint(self, loss_epoch)
            if save_state:
                _save_local_state(self, save_path)

        epochs = tqdm(range(1, n_epochs + 1)) if verbose else range(1, n_epochs + 1)
        for epoch in epochs:
            loss_epoch = _run_epoch(self, train_data, directed_aging_update, learning_rate, modulating_f, True)
            self.epoch += 1
            if epoch % save_every == 0:
                _record_checkpoint(self, loss_epoch)
                if save_state:
                    _save_local_state(self, save_path)
            if verbose:
                epochs.set_description(f"Epoch: {epoch}/{n_epochs} | Loss: {loss_epoch:.2e}")

        if save_global:
            self.save_global(save_path + "_global.json")
            self.save_graph(save_path + "_graph.json")
        return _history_dict(self)

    def train_CL_tandem(
        self,
        lr_1,
        eta_1,
        lr_2,
        eta_2,
        train_data_1,
        train_data_2,
        n_epochs,
        save_global=False,
        save_state=False,
        save_path="trained_circuit",
        save_every=1,
        verbose=True,
        modulating_f: Optional[Callable] = None,
    ):
        _ensure_histories(self)
        modulating_f = _identity_modulation if modulating_f is None else modulating_f

        if self.epoch == 0:
            loss_epoch = _run_epoch(self, train_data_1, _base_cl_update(self), lr_1, modulating_f, False, eta_1)
            _record_checkpoint(self, loss_epoch)
            if save_state:
                _save_local_state(self, save_path)

        epochs = tqdm(range(1, n_epochs + 1)) if verbose else range(1, n_epochs + 1)
        for epoch in epochs:
            _run_epoch(self, train_data_2, _base_cl_update(self), lr_2, modulating_f, True, eta_2)
            loss_epoch = _run_epoch(self, train_data_1, _base_cl_update(self), lr_1, modulating_f, True, eta_1)
            self.epoch += 1
            if epoch % save_every == 0:
                _record_checkpoint(self, loss_epoch)
                if save_state:
                    _save_local_state(self, save_path)
            if verbose:
                epochs.set_description(f"Epoch: {epoch}/{n_epochs} | Loss: {loss_epoch:.2e}")

        if save_global:
            self.save_global(save_path + "_global.json")
            self.save_graph(save_path + "_graph.json")
        return _history_dict(self)

    def train_CL_tandem_random(
        self,
        lr_1,
        eta_1,
        lr_2,
        eta_2,
        train_data_1,
        train_data_2,
        n_epochs,
        n_outputs_r,
        save_global=False,
        save_state=False,
        save_path="trained_circuit",
        save_every=1,
        verbose=True,
        n_attempts=20,
        modulating_f: Optional[Callable] = None,
    ):
        _ensure_histories(self)
        modulating_f = _identity_modulation if modulating_f is None else modulating_f

        edges = list(self.graph.edges)
        valid_indices = [i for i, (u, v) in enumerate(edges) if u not in self.indices_inputs and v not in self.indices_inputs]
        edge_array = np.asarray(edges)

        if self.epoch == 0:
            loss_epoch = _run_epoch(self, train_data_1, _base_cl_update(self), lr_1, modulating_f, False, eta_1)
            _record_checkpoint(self, loss_epoch)
            if save_state:
                _save_local_state(self, save_path)

        epochs = tqdm(range(1, n_epochs + 1)) if verbose else range(1, n_epochs + 1)
        for epoch in epochs:
            for i in np.random.permutation(len(train_data_2)):
                robust_outputs = _sample_acyclic_edge_outputs(self.graph, edges, edge_array, valid_indices, n_outputs_r, n_attempts)
                self.set_outputs2(robust_outputs)
                _run_single_batch(
                    self,
                    train_data_2[i],
                    lambda circuit, batch, mod_f, lr, eta: generalized_cl_update(
                        circuit,
                        batch,
                        mod_f,
                        lr,
                        eta,
                        0.0,
                        q_outputs=circuit.Q_outputs2,
                    ),
                    lr_2,
                    modulating_f,
                    True,
                    eta_2,
                )

            loss_epoch = _run_epoch(self, train_data_1, _base_cl_update(self), lr_1, modulating_f, True, eta_1)
            self.epoch += 1
            if epoch % save_every == 0:
                _record_checkpoint(self, loss_epoch)
                if save_state:
                    _save_local_state(self, save_path)
            if verbose:
                epochs.set_description(f"Epoch: {epoch}/{n_epochs} | Loss: {loss_epoch:.2e}")

        if save_global:
            self.save_global(save_path + "_global.json")
            self.save_graph(save_path + "_graph.json")
        return _history_dict(self)

    def train_CL_DA_tandem(
        self,
        lr_1,
        eta_1,
        lr_2,
        train_data_1,
        n_epochs,
        save_global=False,
        save_state=False,
        save_path="trained_circuit",
        save_every=1,
        verbose=True,
        modulating_f: Optional[Callable] = None,
    ):
        _ensure_histories(self)
        modulating_f = _identity_modulation if modulating_f is None else modulating_f

        if self.epoch == 0:
            loss_epoch = _run_epoch(self, train_data_1, _base_cl_update(self), lr_1, modulating_f, False, eta_1)
            _record_checkpoint(self, loss_epoch)
            if save_state:
                _save_local_state(self, save_path)

        epochs = tqdm(range(1, n_epochs + 1)) if verbose else range(1, n_epochs + 1)
        for epoch in epochs:
            _run_epoch(self, train_data_1, directed_aging_update, lr_2, modulating_f, True)
            loss_epoch = _run_epoch(self, train_data_1, _base_cl_update(self), lr_1, modulating_f, True, eta_1)
            self.epoch += 1
            if epoch % save_every == 0:
                _record_checkpoint(self, loss_epoch)
                if save_state:
                    _save_local_state(self, save_path)
            if verbose:
                epochs.set_description(f"Epoch: {epoch}/{n_epochs} | Loss: {loss_epoch:.2e}")

        if save_global:
            self.save_global(save_path + "_global.json")
            self.save_graph(save_path + "_graph.json")
        return _history_dict(self)


class generalized_CL(LearningCircuit):
    """Coupled learning with conductance-dependent modulation."""

    def train_CL_gen(
        self,
        exp_conductances,
        learning_rate,
        eta,
        train_data,
        n_epochs,
        save_global=False,
        save_state=False,
        save_path="trained_circuit",
        save_every=1,
        verbose=True,
        modulating_f: Optional[Callable] = None,
    ):
        _ensure_histories(self)
        modulating_f = _identity_modulation if modulating_f is None else modulating_f

        update_fn = lambda circuit, batch, mod_f, lr, current_eta, current_exp: generalized_cl_update(
            circuit,
            batch,
            mod_f,
            lr,
            current_eta,
            current_exp,
        )

        if self.epoch == 0:
            loss_epoch = _run_epoch(self, train_data, update_fn, learning_rate, modulating_f, False, eta, exp_conductances)
            _record_checkpoint(self, loss_epoch)
            if save_state:
                _save_local_state(self, save_path)

        epochs = tqdm(range(1, n_epochs + 1)) if verbose else range(1, n_epochs + 1)
        for epoch in epochs:
            loss_epoch = _run_epoch(self, train_data, update_fn, learning_rate, modulating_f, True, eta, exp_conductances)
            self.epoch += 1
            if epoch % save_every == 0:
                _record_checkpoint(self, loss_epoch)
                if save_state:
                    _save_local_state(self, save_path)
            if verbose:
                epochs.set_description(f"Epoch: {epoch}/{n_epochs} | Loss: {loss_epoch:.2e}")

        if save_global:
            self.save_global(save_path + "_global.json")
            self.save_graph(save_path + "_graph.json")
        return _history_dict(self)


class negative_coupling(LearningCircuit):
    """Deprecated placeholder for a removed solver path."""

    def train_GD(self, *args, **kwargs):
        raise NotImplementedError(
            "negative_coupling is not available in the current codebase because "
            "Circuit no longer provides the negative-coupling solvers used by the "
            "original implementation."
        )


def _run_single_batch(
    circuit: LearningCircuit,
    batch,
    update_fn,
    lr,
    modulating_f,
    apply_update,
    *update_params,
):
    delta_k, loss = update_fn(circuit, batch, modulating_f, lr, *update_params)
    if loss < circuit.best_error:
        circuit.best_error = loss
        circuit.best_conductances = np.array(circuit.conductances, copy=True)
    if apply_update:
        circuit.conductances = circuit.conductances - delta_k
        _clip_conductances(circuit)
    return float(loss)


def _run_epoch(
    circuit: LearningCircuit,
    train_data,
    update_fn,
    lr,
    modulating_f,
    apply_update,
    *update_params,
):
    loss_acc = 0.0
    for i in np.random.permutation(len(train_data)):
        loss_acc += _run_single_batch(
            circuit,
            train_data[i],
            update_fn,
            lr,
            modulating_f,
            apply_update,
            *update_params,
        )
    return loss_acc / len(train_data)


def _sample_acyclic_edge_outputs(graph, edges, edge_array, valid_indices, n_outputs_r, n_attempts):
    for _ in range(n_attempts):
        sampled_indices = np.random.choice(valid_indices, n_outputs_r, replace=False)
        sampled_edges = [edges[index] for index in sampled_indices]
        subgraph = graph.edge_subgraph(sampled_edges)
        try:
            nx.find_cycle(subgraph)
        except nx.NetworkXNoCycle:
            return edge_array[sampled_indices]
    raise RuntimeError(f"Could not find an acyclic edge subset after {n_attempts} attempts")


def _history_dict(circuit: LearningCircuit):
    _ensure_histories(circuit)
    return {
        "checkpoint_epochs": circuit.checkpoint_epochs,
        "loss": circuit.loss_history,
        "power": circuit.power_history,
        "energy": circuit.energy_history,
    }
