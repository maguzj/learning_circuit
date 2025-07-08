# utils.py
import csv
import json
import numpy as np
import jax.numpy as jnp
import jax
from scipy.sparse import bmat

'''
*****************************************************************************************************
*****************************************************************************************************

										DATA PROCESSING

*****************************************************************************************************
*****************************************************************************************************
'''


def split_data_into_batches( input_data, output_data, batch_size):
        '''
        Convert the data into batches.

        Parameters
        ----------
        input_data : np.array
            Input data with dimensions (n_data, len(indices_inputs))
        output_data : np.array
            Output data with dimensions (n_data, len(indices_outputs))
        batch_size : int
            The size of each batch.

        Returns
        -------
        batches : list of tuples
            List of batches, each containing a tuple of (input_batch, output_batch).
        '''
        n_data = len(input_data)
        if n_data != len(output_data):
            raise ValueError("Input and output data must have the same length.")
        if n_data < batch_size:
            raise ValueError("Batch size must be smaller than the length of the data.")

        n_batches = n_data // batch_size
        batches = [
            (input_data[i * batch_size: (i + 1) * batch_size],
             output_data[i * batch_size: (i + 1) * batch_size])
            for i in range(n_batches)
        ]

        # Check if there are leftover data points after full batches
        if n_data % batch_size != 0:
            batches.append(
                (input_data[n_batches * batch_size:], output_data[n_batches * batch_size:])
            )

        return batches


def np_circuit_input(input_nodes, indices_nodes, current_bool, n_nodes):
    ''' Compute the input vector f for the circuit. 
    
    Parameters
    ----------
    input_nodes : np.array
        Array with the current or voltages at the nodes specified by indices_nodes.
    indices_nodes : np.array
        Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].
    current_bool : np.array
        Boolean array specifying if the input_nodes are currents or voltages. If an entry is True, the corresponding input_node is a current. If False, it is a voltage.

    Returns
    -------
    f : np.array
        Source vector f. f has size n + len(indices_nodes).
    '''
    n_voltage_input = len(current_bool) -(current_bool).sum()
    f = np.zeros(n_nodes + n_voltage_input)
    f[indices_nodes[current_bool]] = input_nodes[current_bool]
    f[n_nodes:] = input_nodes[~current_bool]
    return f

def np_circuit_input_batch(input_nodes, indices_nodes, current_bool, n_nodes):
    fs = [np_circuit_input(iv, indices_nodes, current_bool, n_nodes) for iv in input_nodes]
    return np.stack(fs, axis=0)

def jax_circuit_input(input_nodes, indices_nodes, current_bool, n_nodes):
    n_voltage_input = len(current_bool) -(current_bool).sum()
    f = jnp.zeros(n_nodes + n_voltage_input)
    f = f.at[indices_nodes[current_bool]].set(input_nodes[current_bool])
    f = f.at[n_nodes:].set(input_nodes[~current_bool])
    return f


jax_circuit_input_batch = jax.vmap(jax_circuit_input, in_axes=(0, None, None, None))

# @jax.jit
# def jax_circuit_input_batch(input_nodes, indices_nodes, current_bool, n_nodes):
#     batch_circuit_input = jax.vmap(jax_circuit_input, in_axes=(0, None, None, None))
#     return batch_circuit_input(input_nodes, indices_nodes, current_bool, n_nodes)


def circuit_input_batch(is_jax, input_nodes, indices_nodes, current_bool, n_nodes):
    if is_jax:
        return jax_circuit_input_batch(input_nodes, indices_nodes, current_bool, n_nodes)
    else:
        return np_circuit_input_batch(input_nodes, indices_nodes, current_bool, n_nodes)


'''
*****************************************************************************************************
*****************************************************************************************************

										Hessians

*****************************************************************************************************
*****************************************************************************************************
'''


def hessian(jax, conductances, incidence_matrix):
    if jax:
        return 2*jnp.dot(incidence_matrix*conductances,jnp.transpose(incidence_matrix))
    else:
        return 2*(incidence_matrix*conductances).dot(incidence_matrix.T)
    
def extended_hessian(jax, hessian, Q):
    if jax:
        ext_hess = jnp.block([[hessian, Q],[jnp.transpose(Q), jnp.zeros(shape=(jnp.shape(Q)[1],jnp.shape(Q)[1]))]])
        return ext_hess
    else:
        ext_hess = bmat([[hessian, Q], [Q.T, None]], format='csr', dtype=float)
        return ext_hess
    
def hessian_ferro_antiferro(jax, conductances, incidence_matrix):
    if jax:
        abs_im = jnp.abs(incidence_matrix)
        abs_k = jnp.abs(conductances)
        return jnp.dot(abs_im*(abs_k-conductances),jnp.transpose(abs_im))+jnp.dot(incidence_matrix*(abs_k+conductances),jnp.transpose(incidence_matrix))
    else:
        abs_im = np.abs(incidence_matrix)
        abs_k = np.abs(conductances)
        return (abs_im*(abs_k-conductances)).dot(abs_im.T)+(incidence_matrix*(abs_k+conductances)).dot(incidence_matrix.T)





'''
*****************************************************************************************************
*****************************************************************************************************

										IMPORT / EXPORT

*****************************************************************************************************
*****************************************************************************************************
'''




def save_to_csv(filename, data, mode='a'):
    """
    Save data to a CSV file.
    
    Parameters:
    - filename: Name of the file to save to.
    - data: The data to save (should be a list or array).
    - mode: File mode ('w' for write, 'a' for append). Default is 'a'.
    """
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def load_from_csv(filename):
    """
    Load data from a CSV file.
    
    Parameters:
    - filename: Name of the file to load from.
    
    Returns:
    - A list of lists containing the data.
    """
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(item) for item in row])
    return data


def to_json_compatible(item):
    ''' Convert JAX arrays to JSON compatible formats, assuming item is a JAX array. '''
    if isinstance(item, (jnp.ndarray, np.ndarray, list)):
        return np.array(item).tolist()  # np.array() ensures compatibility and handles JAX arrays too
    return item


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_array(data, key, default, use_jax):
    array_data = data.get(key, default)
    if array_data is not None:
        return jnp.array(array_data) if use_jax else np.array(array_data)
    return None
