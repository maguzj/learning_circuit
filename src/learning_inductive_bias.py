from learning import *
import jax
import json
import csv
from jax import jit

class inductive_bias(learning):
    @staticmethod
    @jit
    def _s_dk_GD_inductive_bias(exponent, circuit_batch, conductances, incidence_matrix, Q_inputs, Q_outputs, indices_inputs, current_bool, n):
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
        delta_conductances = learning.s_grad_mse(conductances, incidence_matrix, Q_inputs, Q_outputs, circuit_batch[0], circuit_batch[1])*((exponent**2)*jnp.power(conductances,2*(exponent-1)/exponent))
        return delta_conductances

    def train_GD_inductive_bias(self,exponent, learning_rate, train_data, n_epochs, save_global = False, save_state = False, save_path = 'trained_circuit', save_every = 1, verbose=True):
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
            loss_per_epoch = 0
            power_per_epoch = 0
            for i in indices:
                batch = train_data[i]
                circuit_batch = (Circuit.s_circuit_input_batch(batch[0], self.indices_inputs, self.current_bool, self.n), batch[1])
                free_states = Circuit.s_solve_batch(self.conductances, self.incidence_matrix, self.Q_inputs, circuit_batch[0])
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
                delta_conductances = self._s_dk_GD_inductive_bias(exponent, circuit_batch, self.conductances, self.incidence_matrix, self.Q_inputs, self.Q_outputs, self.indices_inputs, self.current_bool, self.n)
                # update
                if self.l2:
                    self.conductances = self.conductances - learning_rate*(delta_conductances+self.l2*(self.conductances-self.center))
                else:
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
