from smithers.ml.hosvd import hosvd
import torch
import numpy as np

class AHOSVD():
    def __init__(self, tensor, mode_number_list, batch_len):
        """
        Class that handles the construction of the functions needed
        to perform the dimensionality reduction of a given tensor,
        which has exactly one dimension (the first one) that is too large, 
        thus preventing standard HOSVD to run on current architectures.
        """
        self.tensor = tensor
        self.mode_number_list = mode_number_list
        self.batch_len = batch_len
        self.u_matrices = []
        self.proj_matrices = []
    
    def _incremental_average(self, current_list, new_list, index): #funziona correttamente
        matrices_list = []
        if index == 0:
            return new_list
        elif index > 0:
            for i in range(len(current_list)):
                matrices_list.append((index / (index + 1)) * current_list[i] + (1/(index + 1)) * new_list[i])
            return matrices_list
    
    def _partial_HOSVD(self, batch_from_tensor):
        HOSVD = hosvd(batch_from_tensor.shape)
        #HOSVD.fit(batch_from_tensor)
        #return HOSVD.modes_matrices[1:]
        HOSVD.fit_AHOSVD(batch_from_tensor)
        return HOSVD.modes_matrices


    def compute_u_matrices(self):
        for index in range(int(np.floor(self.tensor.shape[0]/self.batch_len))):
            self.u_matrices = self._incremental_average(self.u_matrices, self._partial_HOSVD(self.tensor[index * self.batch_len : (index+1) * self.batch_len]), index)

    def compute_proj_matrices(self):
        for i in range(len(self.u_matrices)):
            self.proj_matrices.append(self.u_matrices[i][ : , : self.mode_number_list[i+1]].t().conj())

    def tensor_reverse(self, tensor):
        incr_list = [i for i in range(len(tensor.shape))]
        incr_list.reverse()
        return torch.permute(tensor, tuple(incr_list))
    
    def project_single_observation(self, observation_tensor):
        for i, _ in enumerate(observation_tensor.shape):
            observation_tensor = torch.tensordot(self.proj_matrices[i], observation_tensor, ([1],[i]))
        return self.tensor_reverse(observation_tensor)
    
    def project_multiple_observations(self, observations_tensor):
        for i in range(len(self.proj_matrices)):
            observations_tensor = torch.tensordot(self.proj_matrices[i], observations_tensor, ([1],[i+1]))
        return self.tensor_reverse(observations_tensor)

# example
if __name__ == '__main__':
    import time
    tensor = torch.randn(50000, 4, 4, 256).to('cuda')
    start = time.time()
    ahosvd = AHOSVD(tensor, [20,3,3,50], 20)
    ahosvd.compute_u_matrices()
    print(f"le dimensioni delle matrici U sono {[ahosvd.u_matrices[i].shape for i in range(len(ahosvd.u_matrices))]}")
    ahosvd.compute_proj_matrices()
    end = time.time()
    test_observation = torch.randn(4, 4, 256).to('cuda')
    projected_single_obs = ahosvd.project_single_observation(test_observation)
    test_mult_obs = torch.randn(40, 4, 4, 256).to('cuda')
    test_mult_obs_output = ahosvd.project_multiple_observations(test_mult_obs)
    print(f'time needed: {end-start} seconds')
    print(projected_single_obs.shape)
    print(test_mult_obs_output.shape)