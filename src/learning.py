import numpy as np
from circuit_utils import Circuit
from network_utils import *
from tqdm import tqdm
from scipy.sparse import csr_matrix, csc_array
import jax.numpy as jnp
import json
from typing import List, Tuple
from utils import *




class LearningCircuit(Circuit):
    '''
    Class for linear LearningCircuit circuits.

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
    VERSION        = "1.0"
    DEFAULT_PARAMS = {
        "min_k": 1e-6,
        "max_k": 1e6,
        "jax": False,
        "l2_regularization": 0,
        "l2_center": 0,
        "loss_history": None,
        "checkpoint_iterations": None,
        "power_history": None,
        "energy_history": None,
        "best_conductances": None,
        "best_error": None,
        "name": "network",
        "indices_outputs": None,
        "indices_inputs": None,
        "current_bool": None
        # and any others
    }
    def __init__(self,graph, conductances, **params):
        ''' Initialize the coupled LearningCircuit circuit.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of the circuit.
        conductances : np.array
            Conductances of the edges of the circuit.
        '''
        super().__init__(graph, jax=params.get("jax", self.DEFAULT_PARAMS["jax"]))
        self.set_conductances(conductances)
        # self.name = name

        # merge defaults + passed-in options
        merged = {**self.DEFAULT_PARAMS, **params}

        # always coerce into the right array type
        def _asarray(x, dtype=None):
            if x is None:
                return None
            if self.jax:
                return jnp.asarray(x, dtype=dtype)
            else:
                return np.asarray(x, dtype=dtype)
            
         
        self.min_k  = merged["min_k"]
        self.max_k  = merged["max_k"]
        self.l2_regularization = merged["l2_regularization"]
        self.l2_center = merged["l2_center"]
        self.name = merged["name"]
        self.jax = merged["jax"]
        self.indices_inputs  = _asarray(merged["indices_inputs"],  dtype=int)
        self.indices_outputs = _asarray(merged["indices_outputs"], dtype=int)
        self.current_bool    = _asarray(merged["current_bool"],    dtype=bool)

        # self.indices_inputs = merged["indices_inputs"]
        # self.indices_outputs = merged["indices_outputs"]
        # self.current_bool = merged["current_bool"]

        if merged["loss_history"] is None:
            self.loss_history = []
        else:
            self.loss_history = merged["loss_history"]
        if merged["checkpoint_iterations"] is None or merged["checkpoint_iterations"] == []:
            self.checkpoint_iterations = []
            self.learning_step = 0
        else:
            self.learning_step = merged["checkpoint_iterations"][-1]
            self.checkpoint_iterations = merged["checkpoint_iterations"]
        if merged["power_history"] is None or merged["power_history"] == []:
            self.power_history = []
            self.current_power = 0
        else:
            self.power_history = merged["power_history"]
            self.current_power = merged["power_history"][-1]
        if merged["energy_history"] is None or merged["energy_history"] == []:
            self.energy_history = []
            self.current_energy = 0
        else:
            self.energy_history = merged["energy_history"]
            self.current_energy = merged["energy_history"][-1]
        if merged["best_conductances"] is None:
            self.best_conductances = self.conductances
        else:
            self.best_conductances = merged["best_conductances"]
        if merged["best_error"] is None:
            self.best_error = np.inf
        else:
            self.best_error = merged["best_error"]

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
    

    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

								TRAINING: COUPLED LEARNING AND GRADIENT DESCENT

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def train(
        self,
        train_data: List[Tuple[np.ndarray, np.ndarray]],
        n_epochs: int,
        update_fn,
        lr: float,
        modulating_f,
        *update_params, 
        verbose: bool = True,
        save_state: bool = False,
        save_global: bool = False,
        save_path: str = 'trained_circuit') -> dict:

        it = tqdm(range(1, n_epochs + 1)) if verbose else range(1, n_epochs + 1)
        n_batches = len(train_data)

        # Initial state stats of one peoch
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches)
            loss_acc, power_acc = 0.0, 0.0
            for i in indices:
                batch = train_data[i]
                delta_k, loss, power = update_fn(self, batch, modulating_f, lr, *update_params)
                # stats
                loss_acc += loss
                power_acc += power
                self.current_energy += power
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances.copy()
                if save_state:
                    self.save_local(save_path+'_conductances.csv')

            loss_epoch = loss_acc / n_batches
            power_epoch = power_acc / n_batches
            self.current_power = power_epoch
            self.loss_history.append(loss_epoch)
            self.power_history.append(self.current_power)
            self.energy_history.append(self.current_energy)
            self.checkpoint_iterations.append(self.learning_step)
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        # training
        for epoch in it:
            indices = np.random.permutation(n_batches)
            loss_acc, power_acc = 0.0, 0.0

            for i in indices:
                batch = train_data[i]
                delta_k, loss, power = update_fn(self, batch, modulating_f, lr, *update_params)
                # apply & clip
                self.conductances -= delta_k
                clip = jnp.clip if self.jax else np.clip
                self.conductances = clip(
                    self.conductances, self.min_k, self.max_k
                )

                # stats
                loss_acc += loss
                power_acc += power
                self.current_energy += power
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances.copy()
                self.learning_step += 1

            # end epoch
            loss_epoch = loss_acc / n_batches
            power_epoch = power_acc / n_batches
            self.current_power = power_epoch
            self.loss_history.append(loss_epoch)
            self.power_history.append(self.current_power)
            self.energy_history.append(self.current_energy)
            self.checkpoint_iterations.append(self.learning_step)
            if verbose:
                it.set_description(f"Epoch {epoch}/{n_epochs} Loss={loss_epoch:.3e}")
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        # end of training
        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')
        return {
            'iterations': self.checkpoint_iterations,
            'loss': self.loss_history,
            'power': self.power_history,
            'energy': self.energy_history,
        }




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
                "version": self.VERSION,
                # dump all the DEFAULT_PARAMS (in case they changed from default)
                **{k: getattr(self, k) for k in self.DEFAULT_PARAMS},
                "n": self.n,
                "ne": self.ne,
                # "learning_rate": self.learning_rate,
                "learning_step": self.learning_step,
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

        
    @classmethod
    def from_json(cls, path, graph, conductances):
        data = json.load(open(path))
        params = {k: data.get(k, default)
                  for k, default in cls.DEFAULT_PARAMS.items()}
        obj = cls(graph,
                  conductances,
                  **params)
        # restore histories if present…
        obj.loss_history          = data.get("loss_history", [])
        obj.power_history         = data.get("power_history", [])
        obj.energy_history        = data.get("energy_history", [])
        obj.checkpoint_iterations = data.get("checkpoint_iterations", [])
        obj.best_conductances     = np.array(data.get("best_conductances", conductances))
        obj.best_error            = data.get("best_error", np.inf)

        _ = obj.set_inputs(obj.indices_inputs, obj.current_bool)
        _ = obj.set_outputs(obj.indices_outputs)
        return obj


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
            self.Q_inputs = jnp.array(self.Q_inputs.toarray())
            converted = True
        
        if not isinstance(self.Q_outputs, jnp.ndarray):
            self.Q_outputs = jnp.array(self.Q_outputs.toarray())
            converted = True

        # if not isinstance(self.Q_clamped, jnp.ndarray):
        #     self.Q_clamped = jnp.array(self.Q_clamped.toarray())
        #     converted = True

        if not isinstance(self.incidence_matrix, jnp.ndarray):
            self.incidence_matrix = jnp.array(self.incidence_matrix.toarray())
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

def lc_from_json(jsonfile_global, jsonfile_graph, csv_local=None, new_train=False):
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

    # Create the LearningCircuit object
    circuit = LearningCircuit(
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
    # circuit.learning_rate = data_global['learning_rate']


    # Set task
    current_bool = get_array(data_global, 'current_bool', None, use_jax)
    _ = circuit.set_inputs(indices_inputs, current_bool)
    _ = circuit.set_outputs(indices_outputs)

    return circuit


