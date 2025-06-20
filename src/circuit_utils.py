import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import copy
import jax
import jax.numpy as jnp
from scipy.linalg import solve as scipy_solve
import itertools
from  matplotlib.collections import LineCollection
from matplotlib.collections import EllipseCollection
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from jax import jit, vmap
from voronoi_utils import get_voronoi_polygons
import cmocean

class Circuit(object):
    ''' Class to simulate a circuit with trainable conductances 
    
    Parameters
    ----------
    graph   :   str or networkx.Graph
                If str, it is the path to the file containing the graph. If networkx.Graph, it is the graph itself.
    
    Attributes
    ----------
    graph   :   networkx.Graph
                Graph specifying the nodes and edges in the network. A conductance parameter is associated with each edge. A trainable edge will
		        be updated during training.
    n : int
        Number of nodes in the graph.
    ne : int
        Number of edges in the graph.
    pts: numpy.ndarray
        Positions of the nodes in the graph.
    jax : bool
        If True, the class is using jax. If False, the class is using scipy.sparse.
    indicence_matrix : scipy.sparse.csr_matrix or jnp.array
        Incidence matrix of the graph. The incidence matrix is oriented.
    '''

    def __init__(self, graph, jax=False):
        if type(graph) == str:
            self.graph = nx.read_gpickle(graph)
        else:
            self.graph = copy.deepcopy(graph)

        self.n = len(self.graph.nodes)
        self.ne = len(self.graph.edges)
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in self.graph.nodes])
        self.jax = jax

        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)
        if jax:
            self.incidence_matrix = jnp.array(self.incidence_matrix.todense())
            

    def set_conductances(self, conductances):
        # conductances is a list of floats
        assert len(conductances) == self.ne, 'conductances must have the same length as the number of edges'
        # if list, convert to numpy array
        if self.jax:
            conductances = jnp.array(conductances)
        else:
            conductances = np.array(conductances)
        self.conductances = conductances

    def _hessian(self):
        ''' Compute the Hessian of the network with respect to the conductances. '''
        return 2*(self.incidence_matrix*self.conductances).dot(self.incidence_matrix.T)
    
    @staticmethod
    def _s_hessian(conductances,incidence_matrix):
        ''' Compute the Hessian of the network with respect to the conductances. '''
        return 2*jnp.dot(incidence_matrix*conductances,jnp.transpose(incidence_matrix))
    
    @staticmethod
    def _s_hessian_with_negative_couplings(conductances,deltap,deltam):
        ''' Compute the Hessian of the network with respect to the generalized conductances. The incidence matrix is the difference of deltap and deltam '''
        return 2*(jnp.dot(deltap*jnp.abs(conductances),jnp.transpose(deltap))+jnp.dot(deltam*jnp.abs(conductances),jnp.transpose(deltam))-jnp.dot(deltap*conductances,jnp.transpose(deltam))-jnp.dot(deltam*conductances,jnp.transpose(deltap)))


    def constraint_matrix(self, indices_nodes):
        ''' Compute the constraint matrix Q given by indices_nodes.
        Q corresponds to the projector onto the space of indices_nodes. 
        
        Parameters
        ----------
        indices_nodes : np.array
            Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].
            if the entries are arrays of two indices, the constraints are  flow constraints from the first to the second node.
            else, the constraints are a node constraints.

        Returns
        -------
        Q : scipy.sparse.csr_matrix (jax=false) or jnp.array (jax=true)
            Constraint matrix Q: a rectangular matrix of size n x len(indices_nodes).
        '''
        shape = indices_nodes.shape
        # Check indicesNodes is a non-empty array
        if shape[0] == 0:
            raise ValueError('indicesNodes must be a non-empty array.')

        if self.jax:
            if len(shape) == 1:
                Q = jnp.zeros(shape=(self.n, shape[0]))
                Q = Q.at[indices_nodes, jnp.arange(shape[0])].set(1)
            elif len(shape) == 2 and shape[1] == 2:
                Q = jnp.zeros(shape=(self.n, shape[0]))
                Q = Q.at[indices_nodes[:,0], jnp.arange(shape[0])].set(1)
                Q = Q.at[indices_nodes[:,1], jnp.arange(shape[0])].set(-1)
        else:
            if len(shape) == 1:
                Q = csr_matrix((np.ones(shape[0]), (indices_nodes, np.arange(shape[0]))), shape=(self.n, shape[0]))
            elif len(shape) == 2 and shape[1] == 2:
                Q = csr_matrix((np.ones(shape[0]), (indices_nodes[:,0], np.arange(shape[0]))), shape=(self.n, shape[0])) + csr_matrix((-np.ones(shape[0]), (indices_nodes[:,1], np.arange(shape[0]))), shape=(self.n, shape[0]))
            else:
                raise ValueError('indicesNodes must be a 1D array or a 2D array with shape (integer, 2).')
                
        return Q
    
    def _extended_hessian(self, Q):
        ''' Extend the hessian of the network with the constraint matrix Q. 

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q

        Returns
        -------
        H : scipy.sparse.csr_matrix
            Extended Hessian. H is a sparse matrix of size (n + len(indices_nodes)) x (n + len(indices_nodes)).
        
        '''
        ext_hess = bmat([[self._hessian(), Q], [Q.T, None]], format='csr', dtype=float)
        return ext_hess

    @staticmethod
    def _s_extended_hessian(hessian,Q):
        ''' Extend the hessian of the network with the constraint matrix Q. 

        Parameters
        ----------
        hessian : jnp.array
            Hessian of the network
        Q : jnp.array
            Constraint matrix Q

        Returns
        -------
        H : jnp.array
            Extended Hessian. H is a dense matrix of size (n + len(indices_nodes)) x (n + len(indices_nodes)).
        
        '''
        ext_hess = jnp.block([[hessian, Q],[jnp.transpose(Q), jnp.zeros(shape=(jnp.shape(Q)[1],jnp.shape(Q)[1]))]])
        return ext_hess

    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										NUMERICAL INTEGRATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def circuit_input(self, input_nodes, indices_nodes, current_bool):
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
        f = np.zeros(self.n + n_voltage_input)
        f[indices_nodes[current_bool]] = input_nodes[current_bool]
        f[self.n:] = input_nodes[~current_bool]
        
        return f

    @staticmethod
    def s_circuit_input(input_nodes, indices_nodes, current_bool, n):
        ''' Compute the input vector f for the circuit. 
        
        Parameters
        ----------
        input_nodes : np.array
            Array with the current or voltages at the nodes specified by indices_nodes.
        indices_nodes : np.array
            Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].
        current_bool : np.array
            Boolean array specifying if the input_nodes are currents or voltages. If an entry is True, the corresponding input_node is a current. If False, it is a voltage.
        n : int
            Number of nodes in the graph.

        Returns
        -------
        f : np.array
            Source vector f. f has size n + len(indices_nodes).
        '''
        n_voltage_input = len(current_bool) -(current_bool).sum()
        f = jnp.zeros(n + n_voltage_input)
        f = f.at[indices_nodes[current_bool]].set(input_nodes[current_bool])
        f = f.at[n:].set(input_nodes[~current_bool])
        
        return f

    @staticmethod
    def s_circuit_input_batch(input_nodes, indices_nodes, current_bool, n):
        batch_circuit_input = vmap(Circuit.s_circuit_input, in_axes=(0, None, None, None))
        return batch_circuit_input(input_nodes, indices_nodes, current_bool, n)
        
    

    def solve(self, Q, input_vector):
        ''' Solve the circuit with the constraint matrix Q and the input_vector.

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q
        input_vector : np.array
            Size self.n + len(indices_nodes).
            First self.n entries correspond to imposed currents, the rest to imposed voltages.

        Returns
        -------
        V : np.array
            Solution vector V. V has size n.
        '''
        assert len(input_vector) == self.n + Q.shape[1], "Source vector f has the wrong size."

        H = self._extended_hessian(Q)
        # f_extended = np.hstack([np.zeros(self.n), f])
        # solve the system
        V = spsolve(H, input_vector)[:self.n]
        return V

    @staticmethod
    @jit
    def s_solve(conductances, incidence_matrix, Q, input_vector):
        ''' Solve the circuit with the constraint matrix Q and the source vector input_vector.

        Parameters
        ----------
        Q : jnp.array
            Constraint matrix Q
        input_vector : np.array
            Source vector input_vector. input_vector has size self.n + len(indices_nodes).

        Returns
        -------
        V : np.array
            Solution vector V. V has size n.
        '''
        H = Circuit._s_extended_hessian(Circuit._s_hessian(conductances,incidence_matrix),Q)
        # f_extended = jnp.hstack([jnp.zeros(jnp.shape(incidence_matrix)[0]), f])
        # solve the system
        V = jax.scipy.linalg.solve(H, input_vector)[:jnp.shape(incidence_matrix)[0]]
        return V

    @staticmethod
    @jit
    def s_solve_batch(conductances, incidence_matrix, Q, input_vectors):
        ''' Solve the circuit with the constraint matrix Q and the source vector input_vector.

        Parameters
        ----------
        Q : jnp.array
            Constraint matrix Q
        input_vector : np.array
            Source vector input_vector. input_vector has size self.n + len(indices_nodes).

        Returns
        -------
        V : np.array
            Solution vector V. V has size n.
        '''
        batch_solve = vmap(Circuit.s_solve, in_axes=(None, None, None, 0))
        return batch_solve(conductances, incidence_matrix, Q, input_vectors)
    


    @staticmethod
    @jit
    def s_solve_neg(conductances, deltap, deltam, Q, input_vector):
        ''' Solve the circuit with the constraint matrix Q and the source vector input_vector.

        Parameters
        ----------
        Q : jnp.array
            Constraint matrix Q
        input_vector : np.array
            Source vector input_vector. input_vector has size self.n + len(indices_nodes).

        Returns
        -------
        V : np.array
            Solution vector V. V has size n.
        '''
        H = Circuit._s_extended_hessian(Circuit._s_hessian_with_negative_couplings(conductances,deltap, deltam),Q)
        # f_extended = jnp.hstack([jnp.zeros(jnp.shape(incidence_matrix)[0]), f])
        # solve the system
        V = jax.scipy.linalg.solve(H, input_vector)[:jnp.shape(deltap)[0]]
        return V


    @staticmethod
    @jit
    def s_solve_batch_neg(conductances, deltap, deltam, Q, input_vectors):
        ''' Solve the circuit with the constraint matrix Q and the source vector input_vector.

        Parameters
        ----------
        Q : jnp.array
            Constraint matrix Q
        input_vector : np.array
            Source vector input_vector. input_vector has size self.n + len(indices_nodes).

        Returns
        -------
        V : np.array
            Solution vector V. V has size n.
        '''
        batch_solve = vmap(Circuit.s_solve_neg, in_axes=(None, None, None, None, 0))
        return batch_solve(conductances, deltap, deltam, Q, input_vectors)
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										CIRCUIT REDUCTION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def _star_to_mesh_onenode(self, node):
        ''' Reduces the circuit by using the star-mesh transformation on node.

        Parameters
        ----------
        node : int
            Index of the node to be transformed.

        '''
        # determine the neighbors of node and the sum of the conductances of the edges between node and its neighbors
        neighbors = list(self.graph.neighbors(node))
        sum_conductances = sum([self.graph[node][neighbor]['weight'] for neighbor in neighbors])

        # add the edges between the neighbors of node with new weights (conductances) corresponding to the product of the conductances of the edges between node and its neighbors divided by the sum of the conductances of the edges between node and its neighbors
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                eff_conductance = (self.graph[neighbors[i]][node]['weight'])*(self.graph[neighbors[j]][node]['weight'])/sum_conductances
                # add the edge if it does not exist
                if not self.graph.has_edge(neighbors[i], neighbors[j]):
                    self.graph.add_edge(neighbors[i], neighbors[j], weight = eff_conductance)
                # otherwise, update the weight (parallel conductances)
                else:
                    self.graph[neighbors[i]][neighbors[j]]['weight'] += eff_conductance

        # remove the node from the graph
        self.graph.remove_node(node)

    def star_to_mesh(self, nodes, all_but_nodes=False):
        ''' Reduces the circuit by using the star-mesh transformation on all nodes in nodes.
        
        Parameters
        ----------
        nodes : np.array
            Array with the indices of the nodes to be transformed.
        all_but_nodes : bool, optional
            If True, the star-mesh transformation is applied to all nodes except those in nodes. The default is False.
        '''
        # check that nodes is a non-empty array
        if len(nodes) == 0:
            raise ValueError('nodes must be a non-empty array.')
        # check that the nodes exist
        if not all([node in self.graph.nodes for node in nodes]):
            raise ValueError('Some of the nodes do not exist.')
        # check that the nodes are not repeated
        if len(nodes) != len(set(nodes)):
            raise ValueError('Some of the nodes are repeated.')
        # check that the graph will have at least 2 nodes
        if (len(nodes) >= self.n - 1 and not all_but_nodes) or (len(nodes) < 2 and all_but_nodes):
            raise ValueError('The graph will have less than 2 nodes.')

        if all_but_nodes:
            nodes = [node for node in self.graph.nodes() if node not in nodes]

        # apply the star-mesh transformation to all nodes in nodes
        for node in nodes:
            self._star_to_mesh_onenode(node)

        self.n = self.graph.number_of_nodes()
        self.ne = self.graph.number_of_edges()
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in self.graph.nodes])
        self.weight_to_conductance()
            
    def _remove_edge(self, edge):
        ''' Remove the edge from the graph. '''
        # determine the index of the edge in the list of edges
        index_edge = list(self.graph.edges).index(tuple(edge))
        self.graph.remove_edge(*edge)
        self.n = self.graph.number_of_nodes()
        self.ne = self.graph.number_of_edges()
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in self.graph.nodes])
        # remove the corresponding conductance
        self.conductances = np.delete(self.conductances, index_edge)


        
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										EFFECTIVE CONDUCTANCES

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def power_dissipated(self, voltages):
        ''' Compute the power dissipated in the circuit for the given voltages. '''
        # check that the conductances have been set
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        # check that the voltages have the right size
        if len(voltages) != self.n:
            raise ValueError('Voltages have the wrong size.')
        # compute the power dissipated
        voltage_drop = self.incidence_matrix.T.dot(voltages)
        return np.sum(self.conductances*(voltage_drop**2))

    def effective_conductance(self, array_pair_indices_nodes, seed = 0):
        ''' 
        Compute the effective conductance between the pair of nodes represented contained in array_pair_indices_nodes.

        Parameters
        ----------
        array_pair_indices_nodes : np.array
            Array with the pairs of indices of the nodes for which we want the effective conductance. The order of the elements does not matter.
        seed : int, optional
            Seed for the random number generator. The default is 0.

        Returns
        -------
        voltage_matrix : np.array
            Matrix of size n_unique_nodes x n_unique_nodes. Each row corresponds to the (DV)^2/2 between all the different pairs of unique nodes in array_pair_indices_nodes.
        power_vector : np.array
            Vector of size n_unique_nodes. Each entry corresponds to the power dissipated in the circuit for the corresponding row in voltage_matrix.
        effective_conductance : np.array
            Vector of size len(array_pair_indices_nodes). Each entry corresponds to the effective conductance between the pair of nodes in array_pair_indices_nodes.
        '''
        # check that the conductances have been set
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        # check that the indices_nodes are valid
        if array_pair_indices_nodes.shape[1] != 2:
            raise ValueError('array_pair_indices_nodes must be a 2D array with shape (integer, 2).')


        # extract the unique indices of the nodes
        indices_nodes = np.unique(array_pair_indices_nodes)
        # check that the nodes exist
        if not all([node in self.graph.nodes for node in indices_nodes]):
            raise ValueError('Some of the nodes do not exist.')

        # generate array of all possible permutations of the indices_nodes
        all_possible_pairs = np.array(list(itertools.combinations(indices_nodes, 2)))

        # sort the arrays for better performance
        sorted_array_pair_indices_nodes = np.sort(array_pair_indices_nodes, axis=1)
        sorted_all_possible_pairs = np.sort(all_possible_pairs, axis=1)

        matching_indices = []

        for pair in sorted_array_pair_indices_nodes:
            index = np.where(np.all(sorted_all_possible_pairs == pair, axis=1))[0]
            if index.size != 0:
                matching_indices.append(index[0])


        # generate the linear system
        voltage_matrix = []
        power_vector = []

        np.random.seed(seed)

        for i in range(len(all_possible_pairs)):
            f = np.random.rand(len(indices_nodes))
            # generate the constraint matrix
            Q = self.constraint_matrix(indices_nodes, restrictionType = 'node')
            # solve the system
            V_free = self.solve(Q, f)
            # compute the voltage drops square based on the free state voltages and the array_pair_indices_nodes
            voltage_drops = np.array([(V_free[all_possible_pairs[i,0]] - V_free[all_possible_pairs[i,1]])**2 for i in range(len(all_possible_pairs))])
            # compute the power dissipated
            power_dissipated = self.power_dissipated(V_free)
            # add the voltage drops to the voltage matrix. Same for power
            voltage_matrix.append(voltage_drops/2)
            power_vector.append(power_dissipated)

        # compute the effective conductance
        voltage_matrix = np.array(voltage_matrix)
        power_vector = np.array(power_vector)
        # solve the linear system
        effective_conductance = scipy_solve(voltage_matrix, power_vector)

        # find the effective conductance corresponding to the pair of nodes in array_pair_indices_nodes
        effective_conductance = effective_conductance[matching_indices]

        return voltage_matrix, power_vector, effective_conductance



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										EIGENVALUES AND EIGENVECTORS

	*****************************************************************************************************
	*****************************************************************************************************
	'''



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_node_state(self, node_state, title = None, lw = 0.5, cmap = 'RdYlBu_r', size_factor = 100, prop = True, figsize = (4,4), filename = None, ax = None):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        node_state : np.array
            State of the nodes in the graph. node_state has size n.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        norm = mplcolors.Normalize(vmin=np.min(node_state), vmax=np.max(node_state))
        if prop:
            size = size_factor*np.abs(node_state[:])
        else:   
            size = size_factor
        if ax is not None:
            ax.scatter(posX, posY, s = size, c = node_state[:],edgecolors = 'black',linewidth = lw,  cmap = cmap, norm = norm)
            ax.set( aspect='equal')
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # show the colorbar
            # ax.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5)
            # set the title of each subplot to be the corresponding eigenvalue in scientific notation
            ax.set_title(title)
        else:
            fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
            axs.scatter(posX, posY, s = size, c = node_state[:],edgecolors = 'black',linewidth = lw,  cmap = cmap, norm = norm)
            axs.set( aspect='equal')
            # remove ticks
            axs.set_xticks([])
            axs.set_yticks([])
            # show the colorbar
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.5)
            # set the title of each subplot to be the corresponding eigenvalue in scientific notation
            axs.set_title(title)
            if filename is not None:
                fig.savefig(filename, dpi = 300, bbox_inches='tight')

        print('Warning: this function is going to be deprecated. Use node_state_to_ax instead.')

    def plot_edge_state(self, edge_state, title = None,lw = 0.5, cmap = 'YlOrBr', figsize = (4,4), minmax = None, filename = None, background_color = '0.75'):
        ''' Plot the state of the edges in the graph.

        Parameters
        ----------
        edge_state : np.array
            State of the edges in the graph. edge_state has size ne.
        '''
        _cmap = plt.cm.get_cmap(cmap)
        pos_edges = np.array([np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]).T for edge in self.graph.edges()])
        if minmax:
            norm = plt.Normalize(vmin=minmax[0], vmax=minmax[1])
        else:
            norm = plt.Normalize(vmin=np.min(edge_state), vmax=np.max(edge_state))
        fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
        for i in range(len(pos_edges)):
            axs.plot(pos_edges[i,0], pos_edges[i,1], color = _cmap(norm(edge_state[i])), linewidth = lw)
        axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # show the colorbar
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.5)
        axs.set_facecolor(background_color)
        fig.set_facecolor(background_color)
        # remove frame
        for spine in axs.spines.values():
            spine.set_visible(False)
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)
        if filename:
            fig.savefig(filename, dpi = 300)
        
        print('Warning: this function is going to be deprecated. Use edge_state_to_ax instead.')

    def edge_state_to_ax(self, ax, edge_state, vmin = None, vmax = None, cmap = cmocean.cm.matter, plot_mode = 'lines', lw = 1, zorder = 2, autoscale = True, annotate = False, alpha = 1, truncate = False, truncate_value = 0.1,shrink_factor = 0.3, color_scale = 'linear', mask = None, mask_value = 0):
        '''
        Plot the state of the edges in the graph.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object where the plot will be drawn.
        edge_state : np.array
            State of the edges in the graph. edge_state has size ne.
        vmin : float or None, optional
            Minimum value of the colormap. If None, vmin is set to the minimum value of edge_state. The default is None.
        vmax : float or None, optional
            Maximum value of the colormap. If None, vmax is set to the maximum value of edge_state. The default is None.
        cmap : str, optional
            Colormap. The default is 'YlOrBr'.
        plot_mode : str, optional
            Plot mode. Either 'lines' or 'arrows'. The default is 'lines'.
        lw : float, optional
            Line width. The default is 1.
        zorder : int, optional
            zorder of the edges. The default is 2.
        autoscale : bool, optional
            If True, the axes are autoscaled. The default is True.
        annotate : bool, optional
            If True, the edges are annotated with their index. The default is False.
        alpha : float, optional
            Alpha of the edges. The default is 1.
        truncate : bool, optional
            If True, only the edges with norm(edge_state) larger than truncate_value are plotted. The default is False.
        truncate_value : float, optional
            Value to truncate the edges. The default is 0.1.
        shrink_factor : float, optional
            Factor to shrink the arrows in plot_mode='arrows'. The default is 0.3.
        color_scale : str, optional
            Scale of the colormap. Either 'linear' or 'log'. The default is 'linear'.
        mask : np.array, optional
            Mask to apply to the edges. If None, no mask is applied. The default is None.
        
        Returns
        -------
        plt.cm.ScalarMappable
            ScalarMappable object that can be used to add a colorbar to the plot.
        '''
        _cmap = plt.cm.get_cmap(cmap)
        # create the line collection object
        pos_edges = [np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]) for edge in self.graph.edges()]
        _edge_state = edge_state
        _abs_edge_state = np.abs(_edge_state)
        if truncate:
            # consider only the edges with norm(edge_state) larger than truncate_value
            pos_edges = [pos_edges[i] for i in range(len(pos_edges)) if _abs_edge_state[i] > truncate_value]
            _edge_state = _edge_state[_abs_edge_state > truncate_value]
            _abs_edge_state = np.abs(_edge_state)

        if mask is not None:
            pos_edges = [pos_edges[i] for i in range(len(pos_edges)) if mask[i]]
            _edge_state = _edge_state[mask]
            _abs_edge_state = np.abs(_edge_state)

        if plot_mode == 'lines':
            if vmin is None:
                vmin = np.min(edge_state)
            if vmax is None:
                vmax = np.max(edge_state)
            if color_scale == 'linear':
                norm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
            elif color_scale == 'log':
                norm = mplcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                raise ValueError('color_scale must be either "linear" or "log".')
            color_array = _cmap(norm(_edge_state)) #beware: norm is not abs value, it is the custom function
            lc = LineCollection(pos_edges, color = color_array, linewidths = lw, path_effects=[path_effects.Stroke(capstyle="round")],zorder=zorder, alpha = alpha)
            ax.add_collection(lc)
        elif plot_mode == 'arrows':
            if vmin is None:
                vmin = np.min(np.abs(edge_state))
            if vmax is None:
                vmax = np.max(np.abs(edge_state))
            if color_scale == 'linear':
                norm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
            elif color_scale == 'log':
                norm = mplcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                raise ValueError('color_scale must be either "linear" or "log".')
            color_array = _cmap(norm(_abs_edge_state))
            arrows = []
            for i in range(len(pos_edges)):
                start_pos = pos_edges[i][0]
                end_pos = pos_edges[i][1]
                mid_point = (start_pos + end_pos)/2
                start_pos_shrunk = start_pos + shrink_factor*(mid_point - start_pos)
                end_pos_shrunk = end_pos - shrink_factor*(end_pos - mid_point)
                start_pos, end_pos = start_pos_shrunk, end_pos_shrunk

                if _edge_state[i] < 0:
                    start_pos, end_pos = end_pos, start_pos

                arrow = FancyArrowPatch(start_pos, end_pos, 
                                    arrowstyle='simple,head_length=0.5, head_width=0.7, tail_width=0.2', 
                                    color=color_array[i], 
                                    linewidth=lw,
                                    shrinkA=0,
                                    shrinkB=0,
                                    mutation_scale= 0.8,
                                    path_effects=[path_effects.Stroke(capstyle="round")])
                arrows.append(arrow)
            pc = PatchCollection(arrows, match_original=True, zorder=zorder, alpha = alpha)
            ax.add_collection(pc)
        else:
            raise ValueError('plot_mode must be either "lines" or "arrows".')

        if annotate:
            for i in range(self.ne):
                ax.annotate(str(i), (np.mean(pos_edges[i][:,0]), np.mean(pos_edges[i][:,1])), fontsize = 2*lw, color = 'black', ha = 'center', va = 'center', zorder = 3, path_effects=[path_effects.withStroke(linewidth=2*lw,
                                                        foreground="w")])

        if autoscale:
            ax.autoscale()

        return plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    def node_state_to_ax(self, ax, node_state, vmin = None, vmax = None, cmap = 'coolwarm', plot_mode = 'ellipses', radius = 0.1, zorder = 2, autoscale = True,annotate = False, color_scale = 'linear'):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object where the plot will be drawn.
        node_state : np.array
            State of the nodes in the graph. node_state has size n.
        vmin : float or None, optional
            Minimum value of the colormap. If None, vmin is set to the minimum value of node_state. The default is None.
        vmax : float or None, optional
            Maximum value of the colormap. If None, vmax is set to the maximum value of node_state. The default is None.
        cmap : str, optional
            Colormap. The default is 'viridis'.
        plot_mode : str, optional
            Plot mode. Either 'ellipses' or 'voronoi'. The default is 'ellipses'.
        radius : float, optional
            Radius of the ellipses or the circles. The default is 0.1.
        zorder : int, optional
            zorder of the ellipses or the circles. The default is 2.
        autoscale : bool, optional
            If True, the axes are autoscaled. The default is True.
        annotate : bool, optional
            If True, the nodes are annotated with their index. The default is False.
        color_scale : str, optional
            Scale of the colormap. Either 'linear' or 'log'. The default is 'linear'.

        Returns
        -------
        plt.cm.ScalarMappable
            ScalarMappable object that can be used to add a colorbar to the plot.
        '''
        if vmin is None:
            vmin = np.min(node_state)
        if vmax is None:
            vmax = np.max(node_state)
        if color_scale == 'linear':
            norm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
        elif color_scale == 'log':
            norm = mplcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise ValueError('color_scale must be either "linear" or "log".')
        
        color_array = plt.cm.get_cmap(cmap)(norm(node_state))

        if plot_mode == 'ellipses':
            # create a collection of ellipses
            r = np.ones(self.n)*radius
            ec = EllipseCollection(r, r, np.zeros(self.n), color = color_array, linewidths = 1,offsets=self.pts,
                       offset_transform=ax.transData, zorder=zorder)
            ax.add_collection(ec)

            if autoscale:
                ax.autoscale()
        elif plot_mode == 'voronoi':
            # find the extreme points of the graph
            minX = np.min(self.pts[:,0])
            maxX = np.max(self.pts[:,0])
            minY = np.min(self.pts[:,1])
            maxY = np.max(self.pts[:,1])
            # plot them in a scatter plot
            ax.scatter(self.pts[:,0], self.pts[:,1], s = 0)
            # Autoscale before the polygons are plotted
            if autoscale:
                ax.autoscale()

            # create a collection of polygons
            polygons = get_voronoi_polygons(self.pts)
            pc = PolyCollection(polygons, color = color_array, linewidths = 1, zorder=zorder)
            ax.add_collection(pc)
        else:
            raise ValueError('plot_mode must be either "ellipses" or "voronoi".')

        if annotate:
            posX = self.pts[:,0]
            posY = self.pts[:,1]
            for i in range(self.n):
                ax.annotate(str(i), (posX[i], posY[i]), fontsize = 0.8*radius, color = 'black', ha = 'center', va = 'center', zorder = 3)
            
        

        # return the colorbar
        return plt.cm.ScalarMappable(norm=norm, cmap=cmap)
