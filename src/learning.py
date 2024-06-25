import numpy as np
from circuit_utils import Circuit
from network_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix, csc_array
import jax.numpy as jnp
import jax
from jax import jit, vmap
import json
import csv
from scipy.sparse import hstack

class learning(Circuit):
    '''
    Class for linear learning circuits.

    Attributes
    ----------
    graph : networkx.Graph
        Graph of the circuit.
    n : int
        Number of nodes in the circuit.
    ne : int
        Number of edges in the circuit.
    pts : np.array
        Positions of the nodes.
    '''
    def __init__(self,graph, conductances, name = 'circuit', min_k = 1.e-6, max_k = 1.e6, jax = False, loss_history = None, checkpoint_iterations = None, power_history = None, energy_history = None, best_conductances = None, best_error = None):
        ''' Initialize the coupled learning circuit.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of the circuit.
        conductances : np.array
            Conductances of the edges of the circuit.
        '''
        super().__init__(graph, jax=jax)
        self.jax = jax
        self.set_conductances(conductances)
        self.name = name
        
        self.min_k = min_k
        self.max_k = max_k
        if loss_history is None:
            self.loss_history = []
        else:
            self.loss_history = loss_history
        if checkpoint_iterations is None or checkpoint_iterations == []:
            self.checkpoint_iterations = []
            self.learning_step = 0
        else:
            self.learning_step = checkpoint_iterations[-1]
            self.checkpoint_iterations = checkpoint_iterations
        if power_history is None or power_history == []:
            self.power_history = []
            self.current_power = 0
        else:
            self.power_history = power_history
            self.current_power = power_history[-1]
        if energy_history is None or energy_history == []:
            self.energy_history = []
            self.current_energy = 0
        else:
            self.energy_history = energy_history
            self.current_energy = energy_history[-1]
        if best_conductances is None:
            self.best_conductances = self.conductances
        else:
            self.best_conductances = best_conductances
        if best_error is None:
            self.best_error = np.inf
        else:
            self.best_error = best_error

    def _clip_conductances(self):
        ''' Clip the conductances to be between min_k and max_k.
        '''
        self.conductances = np.clip(self.conductances, self.min_k, self.max_k)

    def _jax_clip_conductances(self):
        ''' Clip the conductances to be between min_k and max_k.
        '''
        self.conductances = jnp.clip(self.conductances, self.min_k, self.max_k)

    '''
	*****************************************************************************************************
	*****************************************************************************************************

										TASK

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def set_inputs(self, indices_inputs, current_bool):
        ''' Set the inputs of the circuit.

        Parameters
        ----------
        indices_inputs : np.array
            Indices of the nodes that are inputs.
        current_bool : np.array
            Boolean array indicating if the input is a current source or a voltage source.

        Returns
        -------
        Q_inputs : scipy.sparse.csr_matrix or jnp.array
            Constraint matrix Q_inputs: a sparse constraint rectangular matrix of size n x len(indices_inputs[current_bool]). Its entries are only 1 or 0.
        '''
        self.indices_inputs = indices_inputs
        self.current_bool = current_bool
        Q_inputs = self.constraint_matrix(indices_inputs[~current_bool])
        self.Q_inputs = Q_inputs
        return Q_inputs

    def set_outputs(self, indices_outputs):
        ''' Set the outputs of the circuit.

        Parameters
        ----------
        indices_outputs : np.array
            Indices of the nodes that are outputs.

        Returns
        -------
        Q_outputs : scipy.sparse.csr_matrix or jnp.array
            Constraint matrix Q_outputs: a sparse constraint rectangular matrix of size n x len(indices_outputs). Its entries are only 1 or 0.
        '''
        self.indices_outputs = indices_outputs
        Q_outputs = self.constraint_matrix(indices_outputs)
        self.Q_outputs = Q_outputs
        return Q_outputs
    
    def square_error(self, predicted_output, true_output):
        ''' Compute the square error. '''
        return 0.5*(predicted_output - true_output)**2

    def mse(self, predicted_output, true_output):
        ''' Compute the mean square error. '''
        return np.mean(self.square_error(predicted_output, true_output))

    @staticmethod
    def s_mse_single(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output):
        ''' Compute the mean square error. '''
        # input_vector =  self.circuit_input(inputs, self.indices_inputs, self.current_bool)
        V = Circuit.s_solve(conductances, incidence_matrix, Q_inputs, inputs)
        predicted_output = Q_outputs.T.dot(V)
        return 0.5*jnp.mean((predicted_output - true_output)**2)

    @staticmethod
    @jit
    def s_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_outputs):
        ''' Compute the mean square error for batches of inputs and outputs. '''
        batch_mse = vmap(learning.s_mse_single, in_axes=(None, None, None, None, 0, 0))
        mse_values = batch_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_outputs)
        return jnp.mean(mse_values) 

    @staticmethod
    @jit
    def s_grad_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output):
        ''' Compute the gradient of the mean square error. '''
        grad_func = jax.grad(learning.s_mse, argnums=0)
        return grad_func(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output)

    @staticmethod
    @jit
    def s_hessian_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output):
        ''' Compute the hessian of the mean square error. '''
        hessian_func = jax.hessian(learning.s_mse, argnums=0)
        return hessian_func(conductances, incidence_matrix, Q_inputs, Q_outputs, inputs, true_output)
        
    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

								TRAINING: COUPLED LEARNING AND GRADIENT DESCENT

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def _dk_CL(self, eta, batch, Q_clamped, Q_outputs):
        ''' Compute the change in conductances, dk, according to the input,output data in the batch using Coupled Learning.
        The new conductances are computed as: k_new = k_old - learning_rate*dk. 
        
        Parameters
        ----------
        learning_rate : float
            Learning rate.
        eta : float
            Nudge rate.
        batch : tuple of np.array
            Tuple of input_data and true_output data.
        Q_clamped : scipy.sparse.csr_matrix
            Constraint matrix Q_clamped: a sparse constraint rectangular matrix of size n x (len(voltage_indices_inputs) + len(indices_outputs)).

        Returns
        -------
        delta_conductances: np.array
            Change in conductances.
        mse : float
            Mean square error.
        '''
        delta_conductances = np.zeros(self.ne)
        batch_size = len(batch[0])
        mse = 0
        power = 0

        for input_data, true_output in zip(*batch):
            input_vector = self.circuit_input(input_data, self.indices_inputs, self.current_bool)
            free_state = self.solve(self.Q_inputs, input_vector)
            output_data = Q_outputs.T.dot(free_state)
            mse += self.mse(output_data, true_output)
            nudge = output_data + eta * (true_output - output_data)
            clamped_input_vector = np.concatenate((input_vector, nudge))
            clamped_state = self.solve(Q_clamped, clamped_input_vector)
            voltage_drop_free = self.incidence_matrix.T.dot(free_state)
            voltage_drop_clamped = self.incidence_matrix.T.dot(clamped_state)
            power = power + np.sum(self.conductances*(voltage_drop_free**2))
            self.current_energy += power
            delta_conductances = delta_conductances + 1.0/eta * (voltage_drop_clamped**2 - voltage_drop_free**2)

        delta_conductances = delta_conductances/batch_size
        mse = mse/batch_size
        self.current_power = power/batch_size

        return delta_conductances, mse

    def train_CL(self, learning_rate, eta, train_data, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1,verbose=True):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)
        self.learning_rate = learning_rate
        self.eta = eta

        # training
        self.Q_clamped = hstack([self.Q_inputs, self.Q_outputs])
        n_batches = len(train_data)
        for epoch in epochs:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                delta_conductances, loss = self._dk_CL(eta, batch, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
                # update
                self.conductances = self.conductances - learning_rate*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            # save
            if (epoch + 1) % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step - 1)
                self.power_history.append(power_per_epoch)
                self.energy_history.append(self.current_energy)
                if save_state:
                    self.save_local(save_path+'_conductances.csv')
            if verbose:
                epochs.set_description('Epoch: {}/{} | Loss: {:.2e}'.format(epoch,n_epochs, loss_per_epoch))
        # end of training
        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')

        return self.checkpoint_iterations, self.loss_history, self.power_history, self.energy_history
    
    @staticmethod
    @jit
    def _s_dk_GD(circuit_batch, conductances, incidence_matrix, Q_inputs, Q_outputs, indices_inputs, current_bool, n):
        ''' Compute the change in conductances, dk, according to the input,output data in the batch using Gradient Descent.
        
        Parameters
        ----------
        circuit_batch : tuple of np.array
            Tuple of input_data and true_output data.

        Returns
        -------
        delta_conductances: np.array
            Change in conductances.
        '''
        delta_conductances = learning.s_grad_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, circuit_batch[0], circuit_batch[1])
        return delta_conductances

    def train_GD(self, learning_rate, train_data, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)
        self.learning_rate = learning_rate

        # training
        n_batches = len(train_data) 
        for epoch in epochs:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                circuit_batch = (Circuit.s_circuit_input_batch(batch[0], self.indices_inputs, self.current_bool, self.n), batch[1])
                free_states = Circuit.s_solve_batch(self.conductances, self.incidence_matrix, self.Q_inputs, circuit_batch[0])
                power_array = (free_states.dot(self.incidence_matrix)**2).dot(self.conductances)
                self.current_power = np.mean(power_array)
                self.current_energy += np.sum(power_array)
                loss = self.mse(free_states.dot(self.Q_outputs),circuit_batch[1])
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
                # update
                delta_conductances = self._s_dk_GD(circuit_batch, self.conductances, self.incidence_matrix, self.Q_inputs, self.Q_outputs, self.indices_inputs, self.current_bool, self.n)
                self.conductances = self.conductances - learning_rate*delta_conductances
                self._jax_clip_conductances()
                self.learning_step = self.learning_step + 1
                
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            # save
            if (epoch + 1) % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step - 1)
                self.power_history.append(power_per_epoch)
                self.energy_history.append(self.current_energy)
                if save_state:
                    self.save_local(save_path+'_conductances.csv')
            if verbose:
                epochs.set_description('Epoch: {}/{} | Loss: {:.2e}'.format(epoch,n_epochs, loss_per_epoch))

        # end of training
        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')

        return self.checkpoint_iterations, self.loss_history, self.power_history, self.energy_history

    def reset_training(self):
        ''' Reset the training. '''
        self.learning_step = 0
        self.checkpoint_iterations = []
        self.loss_history = []
        self.power_history = []
        self.energy_history = []
        self.current_power = 0
        self.current_energy = 0


    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PRUNE AND REWIRE

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def prune_edge(self, edge):
        ''' Prune an edge of the circuit. '''
        self._remove_edge(edge)
        # reset the incidence matrix
        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)
        if jax:
            self.incidence_matrix = jnp.array(self.incidence_matrix.todense())

        # reset the task
        if jax:
            self.jax_set_task(self.indices_source, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)
        else:
            self.set_task(self.indices_source, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)

    def prune_edge_bunch(self, edge):
        ''' Prune an edge of the circuit. '''
        pass
    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										SAVE, LOAD, AND EXPORT

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def save_graph(self, path):
        ''' Save the graph of the circuit in JSON format '''
        with open(path, 'w') as f:
            json.dump({
                "nodes":list(self.graph.nodes),
                "pts":self.pts.tolist(),
                "edges":list(self.graph.edges)},f)

    def save_global(self, path):
        ''' 
        Save the attributes of the circuit in JSON format to the specified path.

        Parameters:
            path (str): The file path where the JSON data should be written.

        Raises:
            ValueError: If any attribute is not JSON-serializable.
            IOError: If there is an error writing to the file.
        '''
        try:
            # Dictionary comprehension to clean and convert data
            dic = {
                "name": self.name,
                "n": self.n,
                "ne": self.ne,
                "learning_rate": self.learning_rate,
                "learning_step": self.learning_step,
                "jax": self.jax,
                "min_k": self.min_k,
                "max_k": self.max_k,
                "indices_inputs": to_json_compatible(self.indices_inputs),
                "indices_outputs": to_json_compatible(self.indices_outputs),
                "current_bool": to_json_compatible(self.current_bool),
                "loss_history": to_json_compatible(self.loss_history),
                "energy_history": to_json_compatible(self.energy_history),
                "power_history": to_json_compatible(self.power_history),
                "checkpoint_iterations": to_json_compatible(self.checkpoint_iterations),
                "best_conductances": to_json_compatible(self.best_conductances),
                "best_error": to_json_compatible(self.best_error),
            }

            # Writing to file
            with open(path, 'w') as file:
                json.dump(dic, file, indent=4)
            print("Data successfully saved to JSON.")
        except TypeError as e:
            print(f"Error converting to JSON: {e}")
        except IOError as e:
            print(f"Error writing to file: {e}")
        

    def save_local(self, path):
        ''' Save the current conductances in CSV format. '''
        # if the file already exists, append the conductances to the file
        if self.learning_step == 0:
            save_to_csv(path, self.conductances.tolist(), mode='w')
        else:
            save_to_csv(path, self.conductances.tolist())

        



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_circuit(self, title=None, lw = 0.5, point_size = 100, highlight_nodes = False, figsize = (4,4), highlighted_point_size = 200, filename = None):
        ''' Plot the circuit.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        pos_edges = np.array([np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]).T for edge in self.graph.edges()])
        fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
        for i in range(len(pos_edges)):
            axs.plot(pos_edges[i,0], pos_edges[i,1], c = 'black', lw = lw, zorder = 1)
        axs.scatter(posX, posY, s = point_size, c = 'black', zorder = 2)
        if highlight_nodes:
            # sources in red
            axs.scatter(posX[self.indices_source], posY[self.indices_source], s = highlighted_point_size, c = 'red', zorder = 10)
            # targets in blue. Check the type of target
            if self.target_type == 'node':
                axs.scatter(posX[self.indices_target], posY[self.indices_target], s = highlighted_point_size, c = 'blue', zorder = 10)
            elif self.target_type == 'edge':
                axs.scatter(posX[self.indices_target[:,0]], posY[self.indices_target[:,0]], s = highlighted_point_size, c = 'blue', zorder = 10)
                axs.scatter(posX[self.indices_target[:,1]], posY[self.indices_target[:,1]], s = 0.5*highlighted_point_size, c = 'blue', zorder = 10)
            # try:
            #     axs.scatter(posX[self.indices_target[:,1:]], posY[self.indices_target[:,1:]], s = highlighted_point_size, c = 'blue', zorder = 10)
            # except:
            #     axs.scatter(posX[self.indices_target], posY[self.indices_target], s = highlighted_point_size, c = 'blue', zorder = 10)
        axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)
        if filename:
            fig.savefig(filename, dpi = 300)


    '''
	*****************************************************************************************************
	*****************************************************************************************************

										JAX TO SPARSE AND VICEVERSA

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def jaxify(self):
        ''' Jaxify the circuit. '''
        
        converted = False
        if not self.jax:
            self.jax = True
            converted = True
        
        if not isinstance(self.Q_inputs, jnp.ndarray):
            self.Q_inputs = jnp.array(self.Q_inputs.todense())
            converted = True
        
        if not isinstance(self.Q_outputs, jnp.ndarray):
            self.Q_outputs = jnp.array(self.Q_outputs.todense())
            converted = True

        # if not isinstance(self.Q_clamped, jnp.ndarray):
        #     self.Q_clamped = jnp.array(self.Q_clamped.todense())
        #     converted = True

        if not isinstance(self.incidence_matrix, jnp.ndarray):
            self.incidence_matrix = jnp.array(self.incidence_matrix.todense())
            converted = True

        if not isinstance(self.conductances, jnp.ndarray):
            self.conductances = jnp.array(self.conductances)
            converted = True

        if converted:
            print('Converted to jax')
        else:
            print('Already jaxified')

    def sparsify(self):
        ''' Sparsify the circuit. '''
        converted = False
        if self.jax:
            self.jax = False
            converted = True
        
        if isinstance(self.Q_inputs, jnp.ndarray):
            self.Q_inputs = csr_matrix(self.Q_inputs, dtype = np.float64)
            converted = True

        if hasattr(self, "Q_clamped"):
            if isinstance(self.Q_clamped, jnp.ndarray):
                self.Q_clamped = csr_matrix(self.Q_clamped, dtype = np.float64)
                converted = True

        if isinstance(self.incidence_matrix, jnp.ndarray):
            self.incidence_matrix = csc_array(self.incidence_matrix, dtype = np.float64)
            converted = True

        if isinstance(self.conductances, jnp.ndarray):
            self.conductances = np.array(self.conductances, dtype = np.float64)
            converted = True

        if converted:
            print('Converted to sparse')
        else:
            print('Already sparse')
        


'''
*****************************************************************************************************
*****************************************************************************************************

									UTILS

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


import json
import numpy as np
import jax.numpy as jnp

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_array(data, key, default, use_jax):
    array_data = data.get(key, default)
    if array_data is not None:
        return jnp.array(array_data) if use_jax else np.array(array_data)
    return None

def create_cl_from_json(jsonfile_global, jsonfile_graph, csv_local=None, new_train=False):
    # Load data from JSON files
    data_global = load_json(jsonfile_global)

    # Graph and Conductances
    graph = network_from_json(jsonfile_graph)
    if csv_local:
        conductances = load_from_csv(csv_local)[-1] 
    else:
        conductances = np.ones(len(graph.edges))

    # Use JAX or NumPy based on the 'jax' attribute from JSON
    use_jax = data_global['jax']
    array_type = jnp if use_jax else np
    conductances = array_type.array(conductances)

    # Manage training-related data
    training_defaults = {'loss_history': None, 'energy_history': None, 'power_history': None, 'checkpoint_iterations': None, 'learning_step': 0}
    if not new_train:
        training_data = {key: data_global[key] for key in ['loss_history', 'energy_history', 'power_history', 'checkpoint_iterations', 'learning_step']}
    else:
        training_data = training_defaults

    # Task-specific settings
    indices_inputs = get_array(data_global, 'indices_inputs', None, use_jax)
    indices_outputs = get_array(data_global, 'indices_outputs', None, use_jax)

    # Create the learning object
    circuit = learning(
        graph,
        conductances,
        name = data_global['name'],
        min_k = data_global['min_k'],
        max_k = data_global['max_k'],
        jax = use_jax,
        loss_history = training_data['loss_history'],
        checkpoint_iterations = training_data['checkpoint_iterations'],
        power_history = training_data['power_history'],
        energy_history = training_data['energy_history'],
        best_conductances = get_array(data_global, 'best_conductances', None, use_jax),
        best_error = data_global.get('best_error')
        )
    
    # Set some attributes
    circuit.learning_rate = data_global['learning_rate']


    # Set task
    current_bool = get_array(data_global, 'current_bool', None, use_jax)
    _ = circuit.set_inputs(indices_inputs, current_bool)
    _ = circuit.set_outputs(indices_outputs)

    return circuit

