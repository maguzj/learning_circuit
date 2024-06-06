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
        if pbar:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)

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