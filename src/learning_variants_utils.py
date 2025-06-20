from learning import *
import jax
import json
import csv
from jax import jit

class restricted_CL(learning):
    @staticmethod
    @jit
    def _threshold_sstep_GD_NA(threshold,conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, learning_rate, min_k, max_k):
        ''' Perform a step of gradient descent over the MSE for Node Allostery '''
        grad_vec = jax.grad(CL.MSE_NA)(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target)
        mask = jnp.abs(grad_vec) > threshold
        grad_vec = grad_vec*mask
        new_conductances = conductances - learning_rate*grad_vec
        new_conductances = jnp.clip(new_conductances, min_k, max_k)
        return new_conductances

    @staticmethod
    @jit
    def _threshold_sstep_GD_EA(threshold,conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, learning_rate, min_k, max_k):
        ''' Perform a step of gradient descent over the MSE for Edge Allostery '''
        grad_vec = jax.grad(CL.MSE_EA)(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target)
        mask = jnp.abs(grad_vec) > threshold
        grad_vec = grad_vec*mask
        new_conductances = conductances - learning_rate*grad_vec
        new_conductances = jnp.clip(new_conductances, min_k, max_k)
        return new_conductances
    def _siterate_GD(self, threshold, n_steps, task):
        ''' Iterate gradient descent for n_steps.
        task is a string: "node" (node allostery) or "edge" (edge allostery)
        '''
        if task == "node":
            for i in range(n_steps):
                free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source)
                voltage_drop_free = free_state.dot(self.incidence_matrix)
                self.current_power = np.sum(self.conductances*(voltage_drop_free**2))
                self.current_energy += self.current_power
                self.learning_step += 1
                self.conductances = restricted_CL._threshold_sstep_GD_NA(threshold,self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.learning_rate, self.min_k, self.max_k)
        elif task == "edge":
            for i in range(n_steps):
                free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source)
                voltage_drop_free = free_state.dot(self.incidence_matrix)
                self.current_power = np.sum(self.conductances*(voltage_drop_free**2))
                self.current_energy += self.current_power
                self.learning_step += 1
                self.conductances = restricted_CL._threshold_sstep_GD_EA(threshold,self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.learning_rate, self.min_k, self.max_k)
        else:
            raise Exception('task must be "node" or "edge"')
        return self.conductances

    def train_GD(self, threshold, n_epochs, n_steps_per_epoch, verbose=True, pbar=False, log_spaced=False, save_global=False, save_state=False, save_path='trained_circuit'):
        ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of gradient descent.
        If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
        '''
        # exit message: not implemented yet
        raise Exception('Not implemented, not tested')

        if pbar:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)

        if log_spaced:
            n_steps = n_epochs * n_steps_per_epoch
            n_steps_per_epoch = log_partition(n_steps, n_epochs)
        else:
            actual_steps_per_epoch = n_steps_per_epoch

        # initial state
        if self.learning_step == 0:
            self.end_epoch.append(self.learning_step)
            loss = CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)
            if loss < self.best_error:
                self.best_error = loss
                self.best_conductances = self.conductances
            self.losses.append(loss)
            if save_state:
                self.save_local(save_path+'.csv')
        else: #to avoid double counting the initial state
            # remove the last element of power and energy
            self.power.pop()
            self.energy.pop()
            # set the current power and energy to the last element
            self.current_power = self.power[-1]
            self.current_energy = self.energy[-1]

        #training
        for epoch in epochs:
            if log_spaced:
                actual_steps_per_epoch = n_steps_per_epoch[epoch]
            conductances = self._siterate_GD(threshold,actual_steps_per_epoch, self.target_type)
            loss = CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)
            if loss < self.best_error:
                self.best_error = loss
                self.best_conductances = self.conductances
            self.losses.append(loss)
            self.power.append(self.current_power)
            self.energy.append(self.current_energy)
            if verbose:
                print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
            self.epoch += 1
            self.end_epoch.append(self.learning_step)
            if save_state:
                # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                self.save_local(save_path+'.csv')
        # at the end of training, compute the current power and current energy, and save global and save graph
        free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source)
        voltage_drop_free = free_state.dot(self.incidence_matrix)
        self.current_power = np.sum(self.conductances*(voltage_drop_free**2))
        self.current_energy += self.current_power
        self.power.append(self.current_power)
        self.energy.append(self.current_energy)

        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')
        return self.losses, conductances





class robust_CL(learning):
    def _dk_DA(self, batch):
        ''' Compute the change in conductances, dk, according to the input,output data in the batch using Directed Aging.
        The new conductances are computed as: k_new = k_old - learning_rate*dk. 
        
        Parameters
        ----------
        learning_rate : float
            Learning rate.
        batch : tuple of np.array
            Tuple of input_data and true_output data.

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
            output_data = self.Q_outputs.T.dot(free_state)
            mse += self.mse(output_data, true_output)
            # nudge = output_data + eta * (true_output - output_data)
            # clamped_input_vector = np.concatenate((input_vector, nudge))
            # clamped_state = self.solve(Q_clamped, clamped_input_vector)
            voltage_drop_free = self.incidence_matrix.T.dot(free_state)
            # voltage_drop_clamped = self.incidence_matrix.T.dot(clamped_state)
            power = power + np.sum(self.conductances*(voltage_drop_free**2))
            self.current_energy += power
            delta_conductances = delta_conductances + voltage_drop_free**2

        delta_conductances = delta_conductances/batch_size
        mse = mse/batch_size
        self.current_power = power/batch_size

        return delta_conductances, mse

    def train_DA(self, learning_rate, train_data, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1,verbose=True):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)
        self.learning_rate = learning_rate

        # set up
        n_batches = len(train_data)

        # initial error, power and energy. We run the training step without updating the conductances
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches)
            for i in indices:
                batch = train_data[i]
                delta_conductances, loss = self._dk_DA(batch)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
            
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            self.loss_history.append(loss_per_epoch)
            self.checkpoint_iterations.append(self.learning_step)
            self.power_history.append(power_per_epoch)
            self.energy_history.append(self.current_energy)
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        # training
        for epoch in epochs:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                delta_conductances, loss = self._dk_DA(batch)
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
            if epoch % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step)
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

    # here: define training for a second task with the same inputs but different outputs
    def set_outputs2(self, indices_outputs2):
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
        self.indices_outputs2 = indices_outputs2
        Q_outputs2 = self.constraint_matrix(indices_outputs2)
        self.Q_outputs2 = Q_outputs2
        return Q_outputs2

    def train_CL_tandem(self, lr_1, eta_1, lr_2, eta_2, train_data_1, train_data_2, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1,verbose=True):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)
        self.learning_rate = lr_1
        self.eta = eta_1

        # set up
        self.Q_clamped = hstack([self.Q_inputs, self.Q_outputs])
        self.Q_clamped_2 = hstack([self.Q_inputs, self.Q_outputs2])
        n_batches_1 = len(train_data_1)
        n_batches_2 = len(train_data_2)

        # initial error, power and energy. We run the training step without updating the conductances
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches_1)
            for i in indices:
                batch = train_data_1[i]
                delta_conductances, loss = self._dk_CL(eta_1, batch, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
            
            loss_per_epoch = loss_per_epoch/n_batches_1
            power_per_epoch = power_per_epoch/n_batches_1
            self.loss_history.append(loss_per_epoch)
            self.checkpoint_iterations.append(self.learning_step)
            self.power_history.append(power_per_epoch)
            self.energy_history.append(self.current_energy)
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        # training
        for epoch in epochs:
            indices_1 = np.random.permutation(n_batches_1)
            indices_2 = np.random.permutation(n_batches_2)
            loss_per_epoch = 0
            power_per_epoch = 0

            # robust training
            for i in indices_2:
                batch_2 = train_data_2[i]
                delta_conductances, loss = self._dk_CL(eta_2, batch_2, self.Q_clamped_2, self.Q_outputs2)
                # update
                self.conductances = self.conductances - lr_2*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            # original training
            for i in indices_1:
                batch_1 = train_data_1[i]
                delta_conductances, loss = self._dk_CL(eta_1, batch_1, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
                # update
                self.conductances = self.conductances - lr_1*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            
            loss_per_epoch = loss_per_epoch/n_batches_1
            power_per_epoch = power_per_epoch/n_batches_1
            # save
            if epoch % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step)
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

    def train_CL_tandem_random(self, lr_1, eta_1, lr_2, eta_2, train_data_1, train_data_2, n_epochs,n_outputs_r, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1,verbose=True, n_attempts=20):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)
        self.learning_rate = lr_1
        self.eta = eta_1

        edges = list(self.graph.edges)
        valid_indices = [index for index, (u, v) in enumerate(edges) if u not in self.indices_inputs and v not in self.indices_inputs]
        npedges = np.array(edges)
        # filtered_edges = np.array([edge for edge in edges if not np.any(np.isin(edge, self.indices_inputs))])

        # set up
        self.Q_clamped = hstack([self.Q_inputs, self.Q_outputs])
        n_batches_1 = len(train_data_1)
        n_batches_2 = len(train_data_2)

        # initial error, power and energy. We run the training step without updating the conductances
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches_1)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data_1[i]
                delta_conductances, loss = self._dk_CL(eta_1, batch, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
            
            loss_per_epoch = loss_per_epoch/n_batches_1
            power_per_epoch = power_per_epoch/n_batches_1
            self.loss_history.append(loss_per_epoch)
            self.checkpoint_iterations.append(self.learning_step)
            self.power_history.append(power_per_epoch)
            self.energy_history.append(self.current_energy)
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        for epoch in epochs:
            indices_1 = np.random.permutation(n_batches_1)
            indices_2 = np.random.permutation(n_batches_2)
            loss_per_epoch = 0
            power_per_epoch = 0

            # robust training
            for i in indices_2:
                batch_2 = train_data_2[i]
                # Loop to find a subgraph without cycles (10 attempts)
                for _ in range(n_attempts):
                    rnd_indices = np.random.choice(valid_indices, n_outputs_r, replace=False)
                    sub_g = self.graph.edge_subgraph([edges[index] for index in rnd_indices])
                    try:
                        nx.find_cycle(sub_g)
                        cycle_bool = True
                    except nx.exception.NetworkXNoCycle:
                        cycle_bool = False
                    if not cycle_bool:
                        break

                # if after 10 attempts we still have a cycle, we quit
                if cycle_bool:
                    raise Exception('Could not find a subgraph without cycles after {} attempts'.format(n_attempts))

                indices_outputs_robustness = npedges[rnd_indices]
                self.set_outputs2(indices_outputs_robustness)
                self.Q_clamped_2 = hstack([self.Q_inputs, self.Q_outputs2])
                delta_conductances, loss = self._dk_CL(eta_2, batch_2, self.Q_clamped_2, self.Q_outputs2)
                # update
                self.conductances = self.conductances - lr_2*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            # original training
            for i in indices_1:
                batch_1 = train_data_1[i]
                delta_conductances, loss = self._dk_CL(eta_1, batch_1, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
                # update
                self.conductances = self.conductances - lr_1*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            
            loss_per_epoch = loss_per_epoch/n_batches_1
            power_per_epoch = power_per_epoch/n_batches_1
            # save
            if epoch % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step)
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


    def train_CL_DA_tandem(self, lr_1, eta_1, lr_2, train_data_1, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1,verbose=True):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)
        self.learning_rate = lr_1
        self.eta = eta_1

        # set up
        self.Q_clamped = hstack([self.Q_inputs, self.Q_outputs])
        n_batches_1 = len(train_data_1)

        # initial error, power and energy. We run the training step without updating the conductances
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches_1)
            for i in indices:
                batch = train_data_1[i]
                delta_conductances, loss = self._dk_CL(eta_1, batch, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
            
            loss_per_epoch = loss_per_epoch/n_batches_1
            power_per_epoch = power_per_epoch/n_batches_1
            self.loss_history.append(loss_per_epoch)
            self.checkpoint_iterations.append(self.learning_step)
            self.power_history.append(power_per_epoch)
            self.energy_history.append(self.current_energy)
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        # training
        for epoch in epochs:
            indices_1 = np.random.permutation(n_batches_1)
            indices_2 = np.random.permutation(n_batches_1)
            loss_per_epoch = 0
            power_per_epoch = 0

            # robust training
            for i in indices_2:
                batch_2 = train_data_1[i]
                delta_conductances, loss = self._dk_DA(batch_2)
                # update
                self.conductances = self.conductances - lr_2*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            # original training
            for i in indices_1:
                batch_1 = train_data_1[i]
                delta_conductances, loss = self._dk_CL(eta_1, batch_1, self.Q_clamped, self.Q_outputs)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
                # update
                self.conductances = self.conductances - lr_1*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            
            loss_per_epoch = loss_per_epoch/n_batches_1
            power_per_epoch = power_per_epoch/n_batches_1
            # save
            if epoch % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step)
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
    





##### Generalized local learning rules
class generalized_CL(learning):
    def _dk_CL_gen(self, eta, batch, Q_clamped, Q_outputs, exp_conductances=0):
        ''' Compute the change in conductances, dk, according to the input,output data in the batch using Coupled Learning.
        
        Parameters
        ----------
        eta : float
            Nudge rate.
        batch : tuple of np.array
            Tuple of input_data and true_output data.
        Q_clamped : scipy.sparse.csr_matrix
            Constraint matrix Q_clamped: a sparse constraint rectangular matrix of size n x (len(voltage_indices_inputs) + len(indices_outputs)).
        Q_outputs : scipy.sparse.csr_matrix
            Constraint matrix Q_outputs: a sparse constraint rectangular matrix of size n x len(indices_outputs). Its entries are only 1 or 0.
        exp_conductances : float
            Exponential conductances. If 0, the conductances are not exponentiated.

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

        delta_conductances = (delta_conductances/batch_size)*(self.conductances**exp_conductances)
        mse = mse/batch_size
        self.current_power = power/batch_size

        return delta_conductances, mse
    
    def train_CL_gen(self, exp_conductances, learning_rate, eta, train_data, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1,verbose=True):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)
        self.learning_rate = learning_rate
        self.eta = eta

        # set up
        self.Q_clamped = hstack([self.Q_inputs, self.Q_outputs])
        n_batches = len(train_data)

        # initial error, power and energy. We run the training step without updating the conductances
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                delta_conductances, loss = self._dk_CL_gen(eta, batch, self.Q_clamped, self.Q_outputs, exp_conductances)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            self.loss_history.append(loss_per_epoch)
            self.checkpoint_iterations.append(self.learning_step)
            self.power_history.append(power_per_epoch)
            self.energy_history.append(self.current_energy)
            if save_state:
                self.save_local(save_path+'_conductances.csv')
            
        # training
        for epoch in epochs:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                delta_conductances, loss = self._dk_CL_gen(eta, batch, self.Q_clamped, self.Q_outputs, exp_conductances)
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
                # update
                if self.l2:
                    self.conductances = self.conductances - learning_rate*(delta_conductances+self.l2*(self.conductances-self.center))
                else:
                    self.conductances = self.conductances - learning_rate*delta_conductances
                self._clip_conductances()
                self.learning_step = self.learning_step + 1
            
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            # save
            if epoch % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step)
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
    

class negative_coupling(learning):

    @staticmethod
    def s_mse_single_neg(conductances, deltap, deltam, Q_inputs, Q_outputs, inputs, true_output):
        ''' Compute the mean square error. '''
        # input_vector =  self.circuit_input(inputs, self.indices_inputs, self.current_bool)
        V = Circuit.s_solve_neg(conductances, deltap, deltam, Q_inputs, inputs)
        predicted_output = Q_outputs.T.dot(V)
        return 0.5*jnp.mean((predicted_output - true_output)**2)

    @staticmethod
    @jit
    def s_mse_neg(conductances, deltap, deltam, Q_inputs, Q_outputs, inputs, true_outputs):
        ''' Compute the mean square error for batches of inputs and outputs. '''
        batch_mse = vmap(negative_coupling.s_mse_single_neg, in_axes=(None, None, None, None, None, 0, 0))
        mse_values = batch_mse(conductances, deltap, deltam, Q_inputs, Q_outputs, inputs, true_outputs)
        return jnp.mean(mse_values) 

    @staticmethod
    @jit
    def s_grad_mse_neg(conductances, deltap,deltam, Q_inputs, Q_outputs, inputs, true_output):
        ''' Compute the gradient of the mean square error. '''
        grad_func = jax.grad(negative_coupling.s_mse_neg, argnums=0)
        return grad_func(conductances, deltap,deltam, Q_inputs, Q_outputs, inputs, true_output)
    

    @staticmethod
    @jit
    def _s_dk_GD_neg(circuit_batch, conductances, deltap, deltam, Q_inputs, Q_outputs, indices_inputs, current_bool, n):
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
        delta_conductances = negative_coupling.s_grad_mse_neg(conductances, deltap, deltam, Q_inputs, Q_outputs, circuit_batch[0], circuit_batch[1])
        return delta_conductances

    def train_GD(self, learning_rate, train_data, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1, verbose=True):
        ''' Train the circuit for n_epochs. Each epoch consists of one passage of all the train_data.
        Have n_save_points save points, where the state of the circuit is saved if save_state is True.
        '''

        if verbose:
            epochs = tqdm(range(1,n_epochs+1))
        else:
            epochs = range(1,n_epochs+1)
        self.learning_rate = learning_rate

        # set up
        n_batches = len(train_data) 

        # deltap and deltam
        deltap = (jnp.abs(self.incidence_matrix)+self.incidence_matrix)/2
        deltam = (jnp.abs(self.incidence_matrix)-self.incidence_matrix)/2

        # initial error, power and energy. We run the training step without updating the conductances
        if self.learning_step == 0:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                circuit_batch = (Circuit.s_circuit_input_batch(batch[0], self.indices_inputs, self.current_bool, self.n), batch[1])
                free_states = Circuit.s_solve_batch_neg(self.conductances, deltap,deltam, self.Q_inputs, circuit_batch[0])
                power_array = (free_states.dot(self.incidence_matrix)**2).dot(self.conductances)
                self.current_power = np.mean(power_array)
                self.current_energy += np.sum(power_array)
                loss = self.mse(free_states.dot(self.Q_outputs),batch[1])
                loss_per_epoch += loss
                power_per_epoch += self.current_power
                # the loss is prior to the update of the conductances
                if loss < self.best_error:
                    self.best_error = loss
                    self.best_conductances = self.conductances
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            self.loss_history.append(loss_per_epoch)
            self.checkpoint_iterations.append(self.learning_step)
            self.power_history.append(power_per_epoch)
            self.energy_history.append(self.current_energy)
            if save_state:
                self.save_local(save_path+'_conductances.csv')

        # training
        for epoch in epochs:
            indices = np.random.permutation(n_batches)
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                circuit_batch = (Circuit.s_circuit_input_batch(batch[0], self.indices_inputs, self.current_bool, self.n), batch[1])
                free_states = Circuit.s_solve_batch_neg(self.conductances, deltap, deltam, self.Q_inputs, circuit_batch[0])
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
                delta_conductances = self._s_dk_GD_neg(circuit_batch, self.conductances, deltap, deltam, self.Q_inputs, self.Q_outputs, self.indices_inputs, self.current_bool, self.n)
                self.conductances = self.conductances - learning_rate*delta_conductances
                self._jax_clip_conductances()
                self.learning_step = self.learning_step + 1
                
            loss_per_epoch = loss_per_epoch/n_batches
            power_per_epoch = power_per_epoch/n_batches
            # save
            if epoch % save_every == 0:
                self.loss_history.append(loss_per_epoch)
                self.checkpoint_iterations.append(self.learning_step)
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