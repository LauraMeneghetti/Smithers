import torch
import numpy as np

class hosvd():
    def __init__(self, mode_number_list):
        """
        Class that handles Higher Order SVD.
        Use the tensor of interest shape as parameter, if all the hosvd modes are required in the reduction.

        :param list[int] mode_number_list: list containing the number of modes considered for each dimension

        :Example:
            >>> HOSVD = hosvd()
            >>> random_tensor = torch.randn(101,102,103)
            >>> HOSVD.fit(random_tensor)
            >>> transformed = HOSVD.transform(random_tensor)
            >>> inverted = HOSVD.inverse_transform(transformed)
            >>> random_test_tensor = torch.randn(101,102,103)
            >>> relative_error = torch.linalg.norm(HOSVD.inverse_transform(HOSVD.transform(random_test_tensor))-random_test_tensor)/torch.linalg.norm(random_test_tensor)
            >>> print('''The input tensor's shape is {}
                        The projected tensor's shape {}
                        The output tensor's shape is {}
                        The relative error on the test tensor is {:.4}'''.format(random_tensor.shape, transformed.shape, inverted.shape, relative_error))
        """
        self.modes_matrices = None
        self.singular_values = None
        self.modal_singular_values = []
        self.mode_number_list = list(mode_number_list)
    
    def unfolding(self, A, n):
        """
        Method that handles the unfolding of a tensor as a matrix

        :param torch.Tensor A: the input tensor
        :param int n: the dimension along which the unfolding is done
        """
        shape = A.shape
        tensor_dimensions = len(shape)
        size = np.prod(shape)
        size_list = list(range(tensor_dimensions))
        size_list[n] = 0
        size_list[0] = n
        n_rows = int(shape[n])
        n_columns = int(size / n_rows)
        return A.permute(size_list).reshape(n_rows, n_columns)

    def modalsvd(self, A, n):
        """
        Method that performs the standard SVD of an unfolded matrix defined from an input tensor
        along a given dimension

        :param torch.Tensor A: the input tensor
        :param int n: the index of the unfolding matrix being decomposed
        """
        return torch.linalg.svd(self.unfolding(A, n), full_matrices = True)

    def higherorderSVD_noS(self, A):
        """
        Mathod that performs the Higher Order SVD on a given tensor.
        This method DOES NOT return the singular value tensor S

        :param torch.Tensor A: the input tensor
        """
        U_matrices = []
        for i in range(len(A.shape)):
            u,sigma,_ = self.modalsvd(A,i)
            self.modal_singular_values.append(sigma/sigma[0])
            U_matrices.append(u)
        return U_matrices
    
    def higherorderSVD_AHOSVD(self, A):
        """
        Mathod that performs a partial Higher Order SVD on a given tensor.
        This is to be used in AHOSVD.
        This method DOES NOT return the singular value tensor S

        :param torch.Tensor A: the input tensor
        """
        U_matrices = []
        for i in range(1,len(A.shape)):
            u,sigma,_ = self.modalsvd(A,i)
            self.modal_singular_values.append(sigma/sigma[0])
            U_matrices.append(u)
        return U_matrices

    def higherorderSVD_withS(self, A):
        """
        Mathod that performs the Higher Order SVD on a given tensor.
        This method DOES return the singular value tensor S

        :param torch.Tensor A: the input tensor
        """
        U_matrices = []
        S = A.clone()
        for i in range(len(A.shape)):
            u,sigma,_ = self.modalsvd(A,i)
            self.modal_singular_values.append(sigma/sigma[0])
            U_matrices.append(u)
            S = torch.tensordot(S, u, dims=([0],[0]))
        return U_matrices, S

    def fit(self, A, return_S_tensor = False):
        """
        Create the reduced space for the given snapshots A using HOSVD

        :param torch.Tensor A: the input tensor
        :param bool return_S_tensor: state whether or not you are interested in the singular value tensor (requires more computations)
        """
        if not return_S_tensor:
            self.modes_matrices = self.higherorderSVD_noS(A)
        else: 
            self.modes_matrices, self.singular_values = self.higherorderSVD_withS(A)

    def fit_AHOSVD(self, A):
        self.modes_matrices = self.higherorderSVD_AHOSVD(A)

    def tensor_reverse(self, A):
        incr_list = [i for i in range(len(A.shape))]
        incr_list.reverse()
        return torch.permute(A, tuple(incr_list))

    def transform(self, A):
        """
        Reduces the given snapshots tensor

        :param torch.Tensor A: the input tensor
        """
        for i, _ in enumerate(A.shape):
            A = torch.tensordot(self.modes_matrices[i][:,:self.mode_number_list[i]].t().conj(), A, ([1],[i]))
        return self.tensor_reverse(A)

    def inverse_transform(self, UTA):
        """
        Reconstruct the full order solution from the projected one

        :param torch.Tensor UTA: the input tensor
        """
        for i, _ in enumerate(UTA.shape):
            UTA = torch.tensordot(self.modes_matrices[i][:,:self.mode_number_list[i]], UTA, ([1],[i]))
        return self.tensor_reverse(UTA)

    def reduce(self, A):
        """
        Reduces the given snapshots tensor

        :param torch.Tensor A: the input tensor

        .. note::
            Same as `transform`. Kept for backward compatibility.
        """
        return self.transform(A)

    def expand(self, UTA):
        """
        Reconstruct the full order solution from the projected one

        :param torch.Tensor UTA: the input tensor

        .. note::
            Same as `inverse_transform`. Kept for backward compatibility.
        """
        return self.inverse_transform(UTA)


def test_accuracy(tensor, rank):
    HOSVD = hosvd(rank)
    HOSVD.fit(tensor)
    red_tensor = HOSVD.transform(tensor)
    reconstruct_tensor = HOSVD.inverse_transform(red_tensor)
    error = torch.linalg.norm(reconstruct_tensor - tensor)
    relative_error = error / np.linalg.norm(tensor)
    return relative_error, HOSVD.singular_values
# example
if __name__ == '__main__':
    # for _ in range(5):
    #     random_tensor = torch.randn(101,102,103)
    #     err, ss = test_accuracy(random_tensor)
    #     print(f'relative error: {err}')
    #     print(ss)
    tensor = torch.zeros(100, 100, 100)
    for i in range(100):
        tensor[i, :, :] += i
    err, ss = test_accuracy(tensor, [1, 1, 1])
    print(f'relative error: {err}')
    print(ss)
