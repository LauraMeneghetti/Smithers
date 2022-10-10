'''
Module focused on the reduction of the ANN and implementaion of the
training and testing phases.
'''

import torch
import numpy as np

from smithers.ml.rednet import RedNet
from smithers.ml.fnn import FNN, training_fnn
from smithers.ml.tensor_product_layer import tensor_product_layer
from smithers.ml.utils import PossibleCutIdx, spatial_gradients, forward_dataset, projection
from smithers.ml.utils import randomized_svd
from smithers.ml.hosvd import hosvd
from smithers.ml.AHOSVD import AHOSVD


#from ATHENA.athena.active import ActiveSubspaces
from smithers.ml.pcemodel import PCEModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class NetAdapter():
    '''
    Class that handles the reduction of a pretrained ANN and implementation
    of the training and testing phases.
    '''
    def __init__(self, cutoff_idx, red_dim, red_method, inout_method):
        '''
        :param int cutoff_idx: value that identifies the cut-off layer
        :param int red_dim: dimension of the reduced space onto which we
            project the high-dimensional vectors
        :param str red_method: string that identifies the reduced method to
            use, e.g. 'AS', 'POD'
        :param str inout_method: string the represents the technique to use for
            the identification of the input-output map, e.g. 'PCE', 'ANN'

        :Example:

            >>> from smithers.ml.netadapter import NetAdapter
            >>> netadapter = NetAdapter(6, 50, 'POD', 'FNN')
            >>> original_network = import_net() # user defined method to load/build the original model
            >>> train_data = construct_dataset(path_to_dataset)
            >>> train_loader = load_dataset(train_data)
            >>> train_labels = train_data.targets
            >>> n_class = 10
            >>> red_model = netadapter.reduce_net(original_network, train_data, train_labels, train_loader, n_class)
        '''

        self.cutoff_idx = cutoff_idx
        self.red_dim = red_dim
        self.red_method = red_method
        self.inout_method = inout_method

    def _reduce_AS(self, pre_model, post_model, train_dataset):
        '''
        Function that performs the reduction using Active Subspaces (AS)
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param Dataset train_dataset: dataset containing the training
            images.
        :returns: tensor proj_mat representing the projection matrix
            for AS (n_feat x red_dim)
        :rtype: torch.Tensor
        '''
        input_type = train_dataset.__getitem__(0)[0].dtype
        grad = spatial_gradients(train_dataset, pre_model, post_model)
        asub = ActiveSubspaces(dim=self.red_dim, method='exact').to(device)
        asub.fit(gradients=grad)
        proj_mat = torch.tensor(asub.evects, dtype=input_type)

        return proj_mat

    def _reduce_POD(self, matrix_features):
        '''
        Function that performs the reduction using the Proper Orthogonal
        Decomposition (POD).
        :param torch.Tensor matrix_features: (n_images x n_feat) matrix
            containing the output of the pre-model that needs to be reduced.
        :returns: tensor proj_mat representing the projection matrix
            for POD (n_feat x red_dim).
        :rtype: torch.Tensor
        '''
        u = torch.svd(torch.transpose(matrix_features, 0, 1))[0]
        proj_mat = u[:, :self.red_dim]

        return proj_mat

    def _reduce_RandSVD(self, matrix_features): #MODIF
        '''
        Function that performs the reduction using the Randomized SVD (RandSVD).
        :param torch.Tensor matrix_features: (n_images x n_feat) matrix
            containing the output of the pre-model that needs to be reduced.
        :returns: tensor proj_mat representing the projection matrix
            obtained via RandSVD (n_feat x red_dim).
        :rtype: torch.Tensor
        '''
        matrix_features = matrix_features.to('cpu')
        u, s, v = randomized_svd(torch.transpose(matrix_features, 0, 1), self.red_dim)
        return u

    def _reduce_HOSVD(self, matrix_features): #MODIF
        '''
        Function that performs the reduction using the Higher order SVD (RandSVD).
        :param torch.Tensor matrix_features: (n_images x n_feat) matrix
            containing the output of the pre-model that needs to be reduced.
        :returns: tensor proj_mat representing the projection matrix
            obtained via RandSVD (n_feat x red_dim).
        :rtype: torch.Tensor
        '''
        #matrix_features = matrix_features.to('cpu')
        HOSVD = hosvd(None)
        u = None
        u, s, v = randomized_svd(torch.transpose(matrix_features, 0, 1), self.red_dim)
        return u

    def _reduce(self, pre_model, post_model, train_dataset, train_loader, device = device):
        '''
        Function that performs the reduction of the high dimensional
        output of the pre-model
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param Dataset train_dataset: dataset containing the training
            images.
        :param iterable train_loader: iterable object for loading the dataset.
            It iterates over the given dataset, obtained combining a
            dataset(images and labels) and a sampler.
        :returns: tensors matrix_red and proj_mat containing the reduced output
            of the pre-model (n_images x red_dim) and the projection matrix
            (n_feat x red_dim) respectively.
        :rtype: torch.tensor
    	'''
        #matrix_features = matrixize(pre_model, train_dataset, train_labels)
        matrix_features = forward_dataset(pre_model, train_loader).to(device)

        if self.red_method == 'AS':
            #code for AS
            proj_mat = self._reduce_AS(pre_model, post_model, train_dataset)

        elif self.red_method == 'POD':
            #code for POD
            proj_mat = self._reduce_POD(matrix_features)

        elif self.red_method == 'RandSVD': #MODIF
            #code for RandSVD
            proj_mat = self._reduce_RandSVD(matrix_features)

        elif self.red_method == 'HOSVD': #MODIF
            #code for RandSVD
            proj_mat = self._reduce_RandSVD(matrix_features)

        else:
            raise ValueError

        matrix_red = projection(proj_mat, train_loader, matrix_features)

        return matrix_red, proj_mat

    def _inout_mapping_FNN(self, matrix_red, train_labels, n_class):
        '''
        Function responsible for the creation of the input-output map using
        a Feedfoprward Neural Network (FNN).

        :param torch.tensor matrix_red: matrix containing the reduced output
       	    of the pre-model.
        :param torch.tensor train_labels: tensor representing the labels
            associated to each image in the train dataset.
        :param int n _class: number of classes that composes the dataset
        :return: trained model of the FNN
        :rtype: nn.Module
        '''
        n_neurons = 20
        targets = list(train_labels)
        fnn = FNN(self.red_dim, n_class, n_neurons).to(device)
        epochs = 500
        training_fnn(fnn, epochs, matrix_red.to(device), targets)

        return fnn

    def _inout_mapping_PCE(self, matrix_red, out_postmodel, train_loader,
                           train_labels):
        '''
        Function responsible for the creation of the input-output map using
        the Polynomial Chaos Expansion method (PCE).

        :param torch.tensor matrix_red: matrix containing the reduced output
            of the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset (images and labels) and a sampler.
        :param torch.tensor train_labels: tensor representing the labels
            associated to each image in the train dataset.
        :return: trained model of PCE layer and PCE coeff
        :rtype: list
        '''
        mean = torch.mean(matrix_red, 0).to(device)
        var = torch.std(matrix_red, 0).to(device)

        PCE_model = PCEModel(mean, var)
        coeff = PCE_model.Training(matrix_red, out_postmodel,
                                   train_labels[:matrix_red.shape[0]])[0]
        PCE_coeff = torch.FloatTensor(coeff).to(device)

        return [PCE_model, PCE_coeff]

    def _inout_mapping(self, matrix_red, n_class, out_model, train_labels,
                       train_loader):
        '''
        Function responsible for the creation of the input-output map.
        :param tensor matrix_red: matrix containing the reduced output
            of the pre-model.
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset
        :param int n _class: number of classes that composes the dataset
        :param nn.Sequential pre_model: sequential model representing
            the pre-model.
        :param nn.Sequential post_model: sequential model representing
            the pre-model.
        :param iterable train_loader: iterable object, it load the dataset for
            training. It iterates over the given dataset, obtained combining a
            dataset (images and labels) and a sampler.
        :param tensor train_labels: tensor representing the labels associated
            to each image in the train dataset.
        :return: trained model of FNN or list with the trained model of PCE and
            the corresponding PCE coefficients
        :rtype: nn.Module/list
        '''
        if self.inout_method == 'FNN':
            #code for FNN
            inout_map = self._inout_mapping_FNN(matrix_red, train_labels,
                                                n_class)

        elif self.inout_method == 'PCE':
            #code for PCE
            inout_map = self._inout_mapping_PCE(matrix_red, out_model,
                                                train_loader, train_labels)

        else:
            raise ValueError

        return inout_map

    def reduce_net(self, input_network, train_dataset, train_labels,
                   train_loader, n_class, device = device):
        '''
        Function that performs the reduction of the network
        :param nn.Sequential input_network: sequential model representing
            the input network. If the sequential model is not provided, but
            instead you have a nn.Module obj, see the function get_seq_model
            in utils.py.
        :param Dataset train_dataset: dataset containing the training
            images
        :param torch.Tensor train_labels: tensor representing the labels
            associated to each image in the train dataset
        :param iterable train_loader: iterable object for loading the dataset.
            It iterates over the given dataset, obtained combining a
            dataset(images and labels) and a sampler.
        :param int n _class: number of classes that composes the dataset
        :return: reduced net
        :rtype: nn.Module
        '''
        print('Initializing reduction. Chosen reduction method is: '+self.red_method, flush=True)
        input_type = train_dataset.__getitem__(0)[0].dtype
        possible_cut_idx = PossibleCutIdx(input_network)
        cut_idxlayer = possible_cut_idx[self.cutoff_idx]
        pre_model = input_network[:cut_idxlayer].to(device, dtype=input_type)
        post_model = input_network[cut_idxlayer:].to(device, dtype=input_type)
        out_model = forward_dataset(input_network, train_loader)
        matrix_red, proj_mat = self._reduce(pre_model, post_model, train_dataset, train_loader, device)
        inout_map = self._inout_mapping(matrix_red, n_class, out_model, train_labels, train_loader)
        reduced_net = RedNet(n_class, pre_model, proj_mat, inout_map)
        return reduced_net.to(device)

    def reduce_net_AHOSVD(self, input_network, train_dataset, train_labels, train_loader, n_class, mode_list_batch = [25, 35, 3, 3], device = device):
        print('Initializing reduction. Chosen reduction method is: AHOSVD', flush=True)
        input_type = train_dataset.__getitem__(0)[0].dtype
        possible_cut_idx = PossibleCutIdx(input_network)
        cut_idxlayer = possible_cut_idx[self.cutoff_idx]
        pre_model = input_network[:cut_idxlayer].to(device, dtype=input_type)
        post_model = input_network[cut_idxlayer:].to(device, dtype=input_type)

        #forward_dataset
        print('Inizio forwarding dataset', flush = True)
        out_model = torch.zeros(0).to(device)
        num_batch = len(train_loader)
        for idx_, (batch, target) in enumerate(train_loader):
            if idx_ >= num_batch:
                break
            batch = batch.to(device) #MODIF

            with torch.no_grad():
                outputs = pre_model(batch).to(device) #MODIF
                #outputs = torch.squeeze(outputs.flatten(1)).detach() da rimuovere per HOSVD
            out_model = torch.cat([out_model.to(device), outputs.to(device)]).to(device)
        print(f"La shape di out_model è {out_model.shape}", flush = True)
        print('Fine forwarding dataset', flush = True)
        # fine forward_dataset
        #spostiamo sulla cpu per liberare spazio in gpu
        pre_model = pre_model.to('cpu')
        post_model = post_model.to('cpu')
        #####
        ahosvd = AHOSVD(out_model, mode_list_batch, mode_list_batch[0])
        ahosvd.compute_u_matrices()
        ahosvd.compute_proj_matrices()
        proj_matrices = ahosvd.proj_matrices
        tensor_red = ahosvd.project_multiple_observations(out_model)
        print(f"La shape di tensor_red è {tensor_red.shape}", flush = True)
        # flattening dell'out_model ridotto
        flattened_red_out_model = torch.squeeze(tensor_red.flatten(1)).detach()
        # self.red_dim = flattened_red_out_model.shape[1]
        print(f'La shape di flattened_red_out_model è {flattened_red_out_model.shape}', flush = True)
        inout_map = self._inout_mapping(flattened_red_out_model, n_class, out_model, train_labels, train_loader)
        proj_matrices_layer = tensor_product_layer(proj_matrices)
        reduced_net = RedNet(n_class, pre_model, proj_matrices_layer, inout_map)
        return reduced_net.to(device)