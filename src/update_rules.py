import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import hstack
from typing import List, Tuple, Union, Optional
from utils import *


from learning import LearningCircuit
from circuit_utils import Circuit

def s_mse_single(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output):
    ''' Compute the mean square error. '''
    # input_vector =  self.circuit_input(inputs, self.indices_inputs, self.current_bool)
    V = Circuit.s_solve(conductances, incidence_matrix, Q_inputs, inputs)
    predicted_output = Q_outputs.T.dot(V)
    return 0.5*jnp.mean((predicted_output - true_output)**2)

@jax.jit
def s_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_outputs):
    ''' Compute the mean square error for batches of inputs and outputs. '''
    batch_mse = jax.vmap(s_mse_single, in_axes=(None, None, None, None, 0, 0))
    mse_values = batch_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_outputs)
    return jnp.mean(mse_values) 


@jax.jit
def s_grad_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output):
    ''' Compute the gradient of the mean square error. '''
    grad_func = jax.grad(s_mse, argnums=0)
    return grad_func(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output)


def gd_update(
    circuit: LearningCircuit,
    batch: Tuple[np.ndarray, np.ndarray],
    modulating_f,
    lr: float) -> Tuple[np.ndarray, float, float]:

    circuit_batch = (circuit_input_batch(circuit.jax, batch[0], circuit.indices_inputs, circuit.current_bool, circuit.n), batch[1])
    inputs, targets = circuit_batch
    grad = s_grad_mse(
        circuit.conductances,
        circuit.incidence_matrix,
        circuit.Q_inputs,
        circuit.Q_outputs,
        inputs,
        targets)
    
    delta_k = lr * grad * modulating_f(circuit.conductances)
    # logging
    states = Circuit.s_solve_batch(
        circuit.conductances,
        circuit.incidence_matrix,
        circuit.Q_inputs,
        inputs)
    preds = states.dot(circuit.Q_outputs)
    loss = float(np.mean(0.5 * (preds - targets)**2))
    power = float(((states.dot(circuit.incidence_matrix))**2 * circuit.conductances).mean())
    circuit.current_power = power
    return delta_k, loss, power


def cl_update(
    circuit: LearningCircuit,
    batch: Tuple[np.ndarray, np.ndarray],
    modulating_f,
    lr: float,
    eta:float) -> Tuple[np.ndarray, float, float]:
    """
    Coupled Learning update rule.
    """

    circuit_batch = (circuit_input_batch(circuit.jax, batch[0], circuit.indices_inputs, circuit.current_bool, circuit.n), batch[1])
    inputs, targets = circuit_batch

    # free phase
    states_free = circuit.solve_batch(circuit.Q_inputs,inputs)
    y_free = np.dot(states_free, circuit.Q_outputs.toarray())
    # clamped phase
    nudge = y_free + eta * (targets - y_free)
    clamped_inputs = np.concatenate((inputs, nudge), axis=1)

    if not hasattr(circuit, 'Q_clamped'):
        circuit.Q_clamped = hstack([circuit.Q_inputs, circuit.Q_outputs])

    states_clamped = circuit.solve_batch(circuit.Q_clamped, clamped_inputs)

    # voltage drops
    vd_free = np.dot(states_free,circuit.incidence_matrix.toarray())
    vd_clamped = np.dot(states_clamped,circuit.incidence_matrix.toarray())
    # delta conductances
    delta_k = (1/eta) * ((vd_clamped**2 - vd_free**2).mean(axis=0))
    # logging
    loss = float(np.mean(0.5 * (y_free - targets)**2))
    power = float(np.mean(circuit.conductances * (vd_free**2)))
    circuit.current_power = power
    circuit.current_energy += circuit.current_power
    return lr * delta_k, loss, circuit.current_power