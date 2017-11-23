from itertools import chain, combinations
import numpy as np
import theano.tensor as T
from lasagne.layers import DenseLayer, get_all_layers, get_all_param_values, get_all_params, get_output, InputLayer, \
    LSTMLayer, set_all_param_values
from nn.layers import LSTMLayer0Mask
from lasagne.nonlinearities import linear
from nn.nonlinearities import elu_plus_one
from .utilities import last_d_softmax

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams(1234)


class RecModel(object):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z):

        self.z_dim = z_dim

        self.max_length_0 = max_length_0
        self.max_length_1 = max_length_1
        self.max_length = max(self.max_length_0, self.max_length_1)

        self.embedding_dim_0 = embedding_dim_0
        self.embedding_dim_1 = embedding_dim_1

        self.dist_z = dist_z()

        self.mean_nn, self.cov_nn = self.nn_fn()

    def nn_fn(self):

        raise NotImplementedError()

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        mask_0 = T.switch(T.lt(x_0, 0), 0, 1)  # N * max(L)
        x_0_embedded *= T.shape_padright(mask_0)  # N * max(L) * E

        mask_1 = T.switch(T.lt(x_1, 0), 0, 1)  # N * max(L)
        x_1_embedded *= T.shape_padright(mask_1)  # N * max(L) * E

        means = get_output(self.mean_nn, [x_0_embedded, x_1_embedded])  # N * dim(z)
        covs = get_output(self.cov_nn, [x_0_embedded, x_1_embedded])  # N * dim(z)

        return means, covs

    def get_samples(self, x_0, x_0_embedded, x_1, x_1_embedded, num_samples, means_only=False):
        """
        :param x_0: N * max(L) matrix
        :param x_0_embedded: N * max(L) * E tensor
        :param x_1: N * max(L) matrix
        :param x_1_embedded: N * max(L) * E tensor
        :param num_samples: int
        :param means_only: bool

        :return samples: (S*N) * dim(z) matrix
        """

        means, covs = self.get_means_and_covs(x_0, x_0_embedded, x_1, x_1_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        return samples

    def kl_std_gaussian(self, x_0, x_0_embedded, x_1, x_1_embedded):
        """
        :param x_0: N * max(L) matrix
        :param x_0_embedded: N * max(L) * E tensor
        :param x_1: N * max(L) matrix
        :param x_1_embedded: N * max(L) * E tensor

        :return kl: N length vector
        """

        means, covs = self.get_means_and_covs(x_0, x_0_embedded, x_1, x_1_embedded)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return kl

    def get_samples_and_kl_std_gaussian(self, x_0, x_0_embedded, x_1, x_1_embedded, num_samples, means_only=False):

        means, covs = self.get_means_and_covs(x_0, x_0_embedded, x_1, x_1_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_params(self):

        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNN(RecModel):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim_0 + self.embedding_dim_1))

        l_mask = InputLayer((None, self.max_length))

        l_prev = l_in

        all_layers = []

        for h in range(self.nn_rnn_depth):

            l_prev = LSTMLayer(l_prev, num_units=self.nn_rnn_hid_units, mask_input=l_mask)

            all_layers.append(l_prev)

        return all_layers

    def nn_fn(self):

        l_in = InputLayer((None, self.nn_rnn_depth * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid(self, x_0, x_0_embedded, x_1, x_1_embedded):

        mask_0 = T.switch(T.lt(x_0, 0), 0, 1)  # N * max(L)
        mask_1 = T.switch(T.lt(x_1, 0), 0, 1)  # N * max(L)

        mask = T.maximum(mask_0, mask_1)  # N * max(L)

        h_prev = T.concatenate([x_0_embedded, x_1_embedded], axis=-1)  # N * max(L) * 2E

        all_h = []

        for h in range(len(self.rnn)):

            h_prev = self.rnn[h].get_output_for([h_prev, mask])  # N * max(L) * dim(hid)

            all_h.append(h_prev[:, -1])

        hid = T.concatenate(all_h, axis=-1)

        return hid

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        hid = self.get_hid(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (depth*dim(hid))

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNNSplitForwardFinal(RecModel):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        return [l_forward_0, l_forward_1]

    def nn_fn(self):

        l_in = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid(self, x_0, x_0_embedded, x_1, x_1_embedded):

        mask_0 = T.ge(x_0, 0)  # N * max(L)
        mask_1 = T.ge(x_1, 0)  # N * max(L)

        h_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_1 = self.rnn[1].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)

        return T.concatenate([h_0, h_1], axis=-1)  # N * (2*dim(hid))

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        hid = self.get_hid(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (2*dim(hid))

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))
        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNNSplitForwardBackwardFinal(RecModel):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_backward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True, backwards=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_backward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True, backwards=True)

        return [l_forward_0, l_backward_0, l_forward_1, l_backward_1]

    def nn_fn(self):

        l_in = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid(self, x_0, x_0_embedded, x_1, x_1_embedded):

        mask_0 = T.ge(x_0, 0)  # N * max(L)
        mask_1 = T.ge(x_1, 0)  # N * max(L)

        h_forward_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_backward_0 = self.rnn[1].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_0 = T.concatenate([h_forward_0, h_backward_0], axis=-1)  # N * (2*dim(hid))

        h_forward_1 = self.rnn[2].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)
        h_backward_1 = self.rnn[3].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)
        h_1 = T.concatenate([h_forward_1, h_backward_1], axis=-1)  # N * (2*dim(hid))

        return T.concatenate([h_0, h_1], axis=-1)  # N * (4*dim(hid))

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        hid = self.get_hid(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (4*dim(hid))

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))
        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNNSplitForwardBackwardFinalMultipleLatents(RecRNNSplitForwardBackwardFinal):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.num_z = nn_kwargs['num_z']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs)

        self.rnn = self.rnn_fn()

    def nn_fn(self):

        l_in = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.num_z*self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.num_z*self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        N = x_0.shape[0]

        hid = self.get_hid(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (4*dim(hid))

        means = get_output(self.mean_nn, hid).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)
        covs = get_output(self.cov_nn, hid).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))
        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNNSplitForwardBackwardAttention(RecModel):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.rnn = self.rnn_fn()

        self.attention_weights_nn = self.attention_weights_nn_fn()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        return [l_forward_0, l_backward_0, l_forward_1, l_backward_1]

    def nn_fn(self):

        l_in = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def attention_weights_nn_fn(self):

        l_in_0 = InputLayer((None, self.embedding_dim_0))

        l_out_0 = DenseLayer(l_in_0, self.max_length, nonlinearity=linear)

        l_in_1 = InputLayer((None, self.embedding_dim_1))

        l_out_1 = DenseLayer(l_in_1, self.max_length, nonlinearity=linear)

        return [l_out_0, l_out_1]

    def get_hid(self, x_0, x_0_embedded, x_1, x_1_embedded):

        mask_0 = T.ge(x_0, 0)  # N * max(L)
        mask_1 = T.ge(x_1, 0)  # N * max(L)

        mask_0_float = T.cast(mask_0, 'float32')
        mask_1_float = T.cast(mask_1, 'float32')

        h_forward_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * max(L) * dim(hid)
        h_backward_0 = self.rnn[1].get_output_for([x_0_embedded, mask_0])  # N * max(L) * dim(hid)
        hids_0 = T.concatenate([h_forward_0, h_backward_0], axis=-1)  # N * max(L) * (2*dim(hid))

        h_forward_1 = self.rnn[2].get_output_for([x_1_embedded, mask_1])  # N * max(L) * dim(hid)
        h_backward_1 = self.rnn[3].get_output_for([x_1_embedded, mask_1])  # N * max(L) * dim(hid)
        hids_1 = T.concatenate([h_forward_1, h_backward_1], axis=-1)  # N * max(L) * (2*dim(hid))

        x_0_embedded_avg = T.sum(x_0_embedded, axis=1) / T.shape_padright(T.sum(mask_0_float, axis=1))  # N * E
        attention_weights_0 = get_output(self.attention_weights_nn[0], x_0_embedded_avg)  # N * max(L)
        W_0 = T.switch(mask_0, attention_weights_0, -np.inf)  # N * max(L)
        W_0 = last_d_softmax(W_0)  # N * max(L)
        h_0 = T.sum(T.shape_padright(W_0) * hids_0, axis=1)  # N * (2*dim(hid))

        x_1_embedded_avg = T.sum(x_1_embedded, axis=1) / T.shape_padright(T.sum(mask_1_float, axis=1))  # N * E
        attention_weights_1 = get_output(self.attention_weights_nn[1], x_1_embedded_avg)  # N * max(L)
        W_1 = T.switch(mask_1, attention_weights_1, -np.inf)  # N * max(L)
        W_1 = last_d_softmax(W_1)  # N * max(L)
        h_1 = T.sum(T.shape_padright(W_1) * hids_1, axis=1)  # N * (2*dim(hid))

        return T.concatenate([h_0, h_1], axis=-1)  # N * (4*dim(hid))

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        hid = self.get_hid(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (4*dim(hid))

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)
        attention_weights_nn_params = get_all_params(get_all_layers(self.attention_weights_nn), trainable=True)

        return rnn_params + nn_params + attention_weights_nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))
        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        attention_weights_nn_params_vals = get_all_param_values(get_all_layers(self.attention_weights_nn))

        return [rnn_params_vals, nn_params_vals, attention_weights_nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals, attention_weights_nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)
        set_all_param_values(get_all_layers(self.attention_weights_nn), attention_weights_nn_params_vals)


class RecRNNSplitForwardBackwardAttentionMultipleLatents(RecModel):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.num_z = nn_kwargs['num_z']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.rnn = self.rnn_fn()

        self.attention_weights_nn = self.attention_weights_nn_fn()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer0Mask(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                     nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward_0 = LSTMLayer0Mask(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                      nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer0Mask(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                     nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward_1 = LSTMLayer0Mask(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                      nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        return [l_forward_0, l_backward_0, l_forward_1, l_backward_1]

    def nn_fn(self):

        l_in = InputLayer((None, 4*self.nn_rnn_hid_units))

        mean_nn = DenseLayer(l_in, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_in, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def attention_weights_nn_fn(self):

        l_in_0 = InputLayer((None, self.embedding_dim_0))

        l_out_0 = DenseLayer(l_in_0, self.num_z*self.max_length, nonlinearity=linear)

        l_in_1 = InputLayer((None, self.embedding_dim_1))

        l_out_1 = DenseLayer(l_in_1, self.num_z*self.max_length, nonlinearity=linear)

        return [l_out_0, l_out_1]

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded):

        N = x_0.shape[0]

        mask_0 = T.switch(T.lt(x_0, 0), 0, 1)  # N * max(L)
        mask_1 = T.switch(T.lt(x_1, 0), 0, 1)  # N * max(L)

        mask_0_float = T.cast(mask_0, 'float32')
        mask_1_float = T.cast(mask_1, 'float32')

        hids_forward_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * max(L) * dim(hid)
        hids_backward_0 = self.rnn[1].get_output_for([x_0_embedded, mask_0])  # N * max(L) * dim(hid)
        hids_0 = T.concatenate([hids_forward_0, hids_backward_0], axis=-1)  # N * max(L) * (2*dim(hid))

        hids_forward_1 = self.rnn[2].get_output_for([x_1_embedded, mask_1])  # N * max(L) * dim(hid)
        hids_backward_1 = self.rnn[3].get_output_for([x_1_embedded, mask_1])  # N * max(L) * dim(hid)
        hids_1 = T.concatenate([hids_forward_1, hids_backward_1], axis=-1)  # N * max(L) * (2*dim(hid))

        x_0_embedded_avg = T.sum(x_0_embedded, axis=1) / T.shape_padright(T.sum(mask_0_float, axis=1))  # N * E
        attention_weights_0 = get_output(self.attention_weights_nn[0], x_0_embedded_avg)  # N * (Z*max(L))
        attention_weights_0 = attention_weights_0.reshape((N, self.num_z, self.max_length))  # N * Z * max(L)
        W_0 = T.switch(T.tile(T.shape_padaxis(mask_0, 1), (1, self.num_z, 1)), attention_weights_0, -np.inf)  # N * Z *
        # max(L)
        W_0 = last_d_softmax(W_0)  # N * Z * max(L)
        h_0 = T.sum(T.shape_padright(W_0) * T.shape_padaxis(hids_0, 1), axis=2)  # N * Z * (2*dim(hid))

        x_1_embedded_avg = T.sum(x_1_embedded, axis=1) / T.shape_padright(T.sum(mask_1_float, axis=1))  # N * E
        attention_weights_1 = get_output(self.attention_weights_nn[1], x_1_embedded_avg)  # N * (Z*max(L))
        attention_weights_1 = attention_weights_1.reshape((N, self.num_z, self.max_length))  # N * Z * max(L)
        W_1 = T.switch(T.tile(T.shape_padaxis(mask_1, 1), (1, self.num_z, 1)), attention_weights_1, -np.inf)  # N * Z *
        # max(L)
        W_1 = last_d_softmax(W_1)  # N * Z * max(L)
        h_1 = T.sum(T.shape_padright(W_1) * T.shape_padaxis(hids_1, 1), axis=2)  # N * Z * (2*dim(hid))

        h = T.concatenate([h_0, h_1], axis=-1).reshape((N*self.num_z, 4*self.nn_rnn_hid_units))  # (N*Z) *
        # (4*dim(hid))

        means = get_output(self.mean_nn, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)
        covs = get_output(self.cov_nn, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)
        attention_weights_nn_params = get_all_params(get_all_layers(self.attention_weights_nn), trainable=True)

        return rnn_params + nn_params + attention_weights_nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))
        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        attention_weights_nn_params_vals = get_all_param_values(get_all_layers(self.attention_weights_nn))

        return [rnn_params_vals, nn_params_vals, attention_weights_nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals, attention_weights_nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)
        set_all_param_values(get_all_layers(self.attention_weights_nn), attention_weights_nn_params_vals)


class RecRNNSplitForwardBackwardFinalMulti(object):

    def __init__(self, z_dim, max_length, embedding_dim, dist_z, num_langs, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.dist_z = dist_z()

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        self.rnns = {i: self.rnn_fn() for i in range(num_langs)}

        self.nns = {i: self.nn_fn() for i in combinations(range(num_langs), 2)}

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim))

        l_mask = InputLayer((None, self.max_length))

        l_out = LSTMLayer(l_in, num_units=self.nn_rnn_hid_units, mask_input=l_mask,
                          nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        return l_out

    def nn_fn(self):

        l_in = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid(self, x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1):

        mask_0 = T.ge(x_0, 0)  # N * max(L)
        h_0 = self.rnns[lang_0].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)

        mask_1 = T.ge(x_1, 0)  # N * max(L)
        h_1 = self.rnns[lang_1].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)

        h = T.concatenate([h_0, h_1], axis=-1)  # N * (2*dim(hid))

        return h

    def get_means_and_covs(self, x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1):

        h = self.get_hid(x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1)  # N * (2*dim(hid))

        means = get_output(self.nns[(lang_0, lang_1)][0], h)  # N * dim(z)
        covs = get_output(self.nns[(lang_0, lang_1)][1], h)  # N * dim(z)

        return means, covs

    def get_samples_and_kl_std_gaussian(self, x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1, num_samples,
                                        means_only=False):

        if lang_0 < lang_1:
            min_lang = lang_0
            max_lang = lang_1
            x_min_lang = x_0
            x_min_lang_embedded = x_0_embedded
            x_max_lang = x_1
            x_max_lang_embedded = x_1_embedded
        else:
            min_lang = lang_1
            max_lang = lang_0
            x_min_lang = x_1
            x_min_lang_embedded = x_1_embedded
            x_max_lang = x_0
            x_max_lang_embedded = x_0_embedded

        means, covs = self.get_means_and_covs(x_min_lang, x_min_lang_embedded, x_max_lang, x_max_lang_embedded,
                                              min_lang, max_lang)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(list(self.rnns.values())), trainable=True)
        nn_params = get_all_params(get_all_layers(list(chain.from_iterable(self.nns.values()))), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        rnn_params_vals = {l: get_all_param_values(get_all_layers(self.rnns[l])) for l in self.rnns.keys()}
        nn_params_vals = {l: get_all_param_values(get_all_layers(self.nns[l])) for l in self.nns.keys()}

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        for l in self.rnns.keys():
            set_all_param_values(get_all_layers(self.rnns[l]), rnn_params_vals[l])

        for l in self.nns.keys():
            set_all_param_values(get_all_layers(self.nns[l]), nn_params_vals[l])


class RecModelSSL(object):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z):

        self.z_dim = z_dim

        self.max_length_0 = max_length_0
        self.max_length_1 = max_length_1
        self.max_length = max(self.max_length_0, self.max_length_1)

        self.embedding_dim_0 = embedding_dim_0
        self.embedding_dim_1 = embedding_dim_1

        self.dist_z = dist_z()

    def get_means_and_covs_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        raise NotImplementedError()

    def get_means_and_covs_0_only(self, x_0, x_0_embedded):

        raise NotImplementedError()

    def get_means_and_covs_1_only(self, x_1, x_1_embedded):

        raise NotImplementedError()

    def get_samples_and_kl_std_gaussian_both(self, x_0, x_0_embedded, x_1, x_1_embedded, num_samples, means_only=False):

        means, covs = self.get_means_and_covs_both(x_0, x_0_embedded, x_1, x_1_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_samples_and_kl_std_gaussian_0_only(self, x_0, x_0_embedded, num_samples, means_only=False):

        means, covs = self.get_means_and_covs_0_only(x_0, x_0_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_samples_and_kl_std_gaussian_1_only(self, x_1, x_1_embedded, num_samples, means_only=False):

        means, covs = self.get_means_and_covs_1_only(x_1, x_1_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_params(self):

        raise NotImplementedError()

    def get_param_values(self):

        raise NotImplementedError()

    def set_param_values(self, param_values):

        raise NotImplementedError()


class RecRNNSplitForwardBackwardFinalSSL(RecModelSSL):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        self.rnn = self.rnn_fn()

        self.mean_nn_both, self.cov_nn_both = self.nn_fn_both()
        self.mean_nn_0_only, self.cov_nn_0_only = self.nn_fn_single_only()
        self.mean_nn_1_only, self.cov_nn_1_only = self.nn_fn_single_only()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_backward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True, backwards=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_backward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True, backwards=True)

        return [l_forward_0, l_backward_0, l_forward_1, l_backward_1]

    def nn_fn_both(self):

        l_in = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def nn_fn_single_only(self):

        l_in = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid_0(self, x_0, x_0_embedded):

        mask_0 = T.ge(x_0, 0)  # N * max(L)

        h_forward_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_backward_0 = self.rnn[1].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_0 = T.concatenate([h_forward_0, h_backward_0], axis=-1)  # N * (2*dim(hid))

        return h_0

    def get_hid_1(self, x_1, x_1_embedded):

        mask_1 = T.ge(x_1, 0)  # N * max(L)

        h_forward_1 = self.rnn[2].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)
        h_backward_1 = self.rnn[3].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)
        h_1 = T.concatenate([h_forward_1, h_backward_1], axis=-1)  # N * (2*dim(hid))

        return h_1

    def get_hid_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        h_0 = self.get_hid_0(x_0, x_0_embedded)  # N * (2*dim(hid))
        h_1 = self.get_hid_1(x_1, x_1_embedded)  # N * (2*dim(hid))

        return T.concatenate([h_0, h_1], axis=-1)  # N * (4*dim(hid))

    def get_means_and_covs_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        hid = self.get_hid_both(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (4*dim(hid))

        means = get_output(self.mean_nn_both, hid)  # N * dim(z)
        covs = get_output(self.cov_nn_both, hid)  # N * dim(z)

        return means, covs

    def get_means_and_covs_0_only(self, x_0, x_0_embedded):

        hid = self.get_hid_0(x_0, x_0_embedded)  # N * (2*dim(hid))

        means = get_output(self.mean_nn_0_only, hid)  # N * dim(z)
        covs = get_output(self.cov_nn_0_only, hid)  # N * dim(z)

        return means, covs

    def get_means_and_covs_1_only(self, x_1, x_1_embedded):

        hid = self.get_hid_1(x_1, x_1_embedded)  # N * (2*dim(hid))

        means = get_output(self.mean_nn_1_only, hid)  # N * dim(z)
        covs = get_output(self.cov_nn_1_only, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn_both, self.cov_nn_both]), trainable=True)
        nn_0_only_params = get_all_params(get_all_layers([self.mean_nn_0_only, self.cov_nn_0_only]), trainable=True)
        nn_1_only_params = get_all_params(get_all_layers([self.mean_nn_1_only, self.cov_nn_1_only]), trainable=True)

        return rnn_params + nn_params + nn_0_only_params + nn_1_only_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn_both, self.cov_nn_both]))
        nn_0_only_params_vals = get_all_param_values(get_all_layers([self.mean_nn_0_only, self.cov_nn_0_only]))
        nn_1_only_params_vals = get_all_param_values(get_all_layers([self.mean_nn_1_only, self.cov_nn_1_only]))

        return [rnn_params_vals, nn_params_vals, nn_0_only_params_vals, nn_1_only_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals, nn_0_only_params_vals, nn_1_only_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_both, self.cov_nn_both]), nn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_0_only, self.cov_nn_0_only]), nn_0_only_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_1_only, self.cov_nn_1_only]), nn_1_only_params_vals)


class RecRNNSplitForwardBackwardFinalMultipleLatentsSSL(RecRNNSplitForwardBackwardFinalSSL):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.num_z = nn_kwargs['num_z']

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs)

    def nn_fn_both(self):

        l_in = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.num_z*self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.num_z*self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def nn_fn_single_only(self):

        l_in = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.num_z*self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.num_z*self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_means_and_covs_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        N = x_0.shape[0]

        hid = self.get_hid_both(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (4*dim(hid))

        means = get_output(self.mean_nn_both, hid).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)
        covs = get_output(self.cov_nn_both, hid).reshape((N, self.num_z, self.z_dim))  # N * dim(z)

        return means, covs

    def get_means_and_covs_0_only(self, x_0, x_0_embedded):

        N = x_0.shape[0]

        hid = self.get_hid_0(x_0, x_0_embedded)  # N * (2*dim(hid))

        means = get_output(self.mean_nn_0_only, hid).reshape((N, self.num_z, self.z_dim))  # N * dim(z)
        covs = get_output(self.cov_nn_0_only, hid).reshape((N, self.num_z, self.z_dim))  # N * dim(z)

        return means, covs

    def get_means_and_covs_1_only(self, x_1, x_1_embedded):

        N = x_1.shape[0]

        hid = self.get_hid_1(x_1, x_1_embedded)  # N * (2*dim(hid))

        means = get_output(self.mean_nn_1_only, hid).reshape((N, self.num_z, self.z_dim))  # N * dim(z)
        covs = get_output(self.cov_nn_1_only, hid).reshape((N, self.num_z, self.z_dim))  # N * dim(z)

        return means, covs


class RecRNNSplitForwardBackwardAttentionMultipleLatentsSSL(RecModelSSL):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        super().__init__(z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z)

        self.num_z = nn_kwargs['num_z']

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        self.rnn = self.rnn_fn()

        self.attention_weights_nn = self.attention_weights_nn_fn()

        self.mean_nn_both, self.cov_nn_both = self.nn_fn_both()
        self.mean_nn_0_only, self.cov_nn_0_only = self.nn_fn_single_only()
        self.mean_nn_1_only, self.cov_nn_1_only = self.nn_fn_single_only()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer0Mask(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                     nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward_0 = LSTMLayer0Mask(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                      nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer0Mask(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                     nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward_1 = LSTMLayer0Mask(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                      nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        return [l_forward_0, l_backward_0, l_forward_1, l_backward_1]

    def attention_weights_nn_fn(self):

        l_in_0 = InputLayer((None, self.embedding_dim_0))

        l_out_0 = DenseLayer(l_in_0, self.num_z*self.max_length, nonlinearity=linear)

        l_in_1 = InputLayer((None, self.embedding_dim_1))

        l_out_1 = DenseLayer(l_in_1, self.num_z*self.max_length, nonlinearity=linear)

        return [l_out_0, l_out_1]

    def nn_fn_both(self):

        l_in = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def nn_fn_single_only(self):

        l_in = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid_0(self, x_0, x_0_embedded):

        N = x_0.shape[0]

        mask_0 = T.switch(T.lt(x_0, 0), 0, 1)  # N * max(L)
        mask_0_float = T.cast(mask_0, 'float32')

        hids_forward_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * max(L) * dim(hid)
        hids_backward_0 = self.rnn[1].get_output_for([x_0_embedded, mask_0])  # N * max(L) * dim(hid)
        hids_0 = T.concatenate([hids_forward_0, hids_backward_0], axis=-1)  # N * max(L) * (2*dim(hid))

        x_0_embedded_avg = T.sum(x_0_embedded, axis=1) / T.shape_padright(T.sum(mask_0_float, axis=1))  # N * E
        attention_weights_0 = get_output(self.attention_weights_nn[0], x_0_embedded_avg)  # N * (Z*max(L))
        attention_weights_0 = attention_weights_0.reshape((N, self.num_z, self.max_length))  # N * Z * max(L)
        W_0 = T.switch(T.tile(T.shape_padaxis(mask_0, 1), (1, self.num_z, 1)), attention_weights_0, -np.inf)  # N * Z *
        # max(L)
        W_0 = last_d_softmax(W_0)  # N * Z * max(L)
        h_0 = T.sum(T.shape_padright(W_0) * T.shape_padaxis(hids_0, 1), axis=2)  # N * Z * (2*dim(hid))

        return h_0

    def get_hid_1(self, x_1, x_1_embedded):

        N = x_1.shape[0]

        mask_1 = T.switch(T.lt(x_1, 0), 0, 1)  # N * max(L)
        mask_1_float = T.cast(mask_1, 'float32')

        hids_forward_1 = self.rnn[2].get_output_for([x_1_embedded, mask_1])  # N * max(L) * dim(hid)
        hids_backward_1 = self.rnn[3].get_output_for([x_1_embedded, mask_1])  # N * max(L) * dim(hid)
        hids_1 = T.concatenate([hids_forward_1, hids_backward_1], axis=-1)  # N * max(L) * (2*dim(hid))

        x_1_embedded_avg = T.sum(x_1_embedded, axis=1) / T.shape_padright(T.sum(mask_1_float, axis=1))  # N * E
        attention_weights_1 = get_output(self.attention_weights_nn[1], x_1_embedded_avg)  # N * (Z*max(L))
        attention_weights_1 = attention_weights_1.reshape((N, self.num_z, self.max_length))  # N * Z * max(L)
        W_1 = T.switch(T.tile(T.shape_padaxis(mask_1, 1), (1, self.num_z, 1)), attention_weights_1, -np.inf)  # N * Z *
        # max(L)
        W_1 = last_d_softmax(W_1)  # N * Z * max(L)
        h_1 = T.sum(T.shape_padright(W_1) * T.shape_padaxis(hids_1, 1), axis=2)  # N * Z * (2*dim(hid))

        return h_1

    def get_hid_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        h_0 = self.get_hid_0(x_0, x_0_embedded)  # N * Z * (2*dim(hid))
        h_1 = self.get_hid_1(x_1, x_1_embedded)  # N * Z * (2*dim(hid))

        return T.concatenate([h_0, h_1], axis=-1)  # N * Z * (4*dim(hid))

    def get_means_and_covs_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        N = x_0.shape[0]

        h = self.get_hid_both(x_0, x_0_embedded, x_1, x_1_embedded)  # N * Z * (4*dim(hid))

        h = h.reshape((N*self.num_z, 4*self.nn_rnn_hid_units))  # (N*Z) * (4*dim(hid))

        means = get_output(self.mean_nn_both, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)
        covs = get_output(self.cov_nn_both, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)

        return means, covs

    def get_means_and_covs_0_only(self, x_0, x_0_embedded):

        N = x_0.shape[0]

        h = self.get_hid_0(x_0, x_0_embedded)  # N * Z * (2*dim(hid))

        h = h.reshape((N*self.num_z, 2*self.nn_rnn_hid_units))  # (N*Z) * (2*dim(hid))

        means = get_output(self.mean_nn_0_only, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)
        covs = get_output(self.cov_nn_0_only, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)

        return means, covs

    def get_means_and_covs_1_only(self, x_1, x_1_embedded):

        N = x_1.shape[0]

        h = self.get_hid_1(x_1, x_1_embedded)  # N * Z * (2*dim(hid))

        h = h.reshape((N*self.num_z, 2*self.nn_rnn_hid_units))  # (N*Z) * (2*dim(hid))

        means = get_output(self.mean_nn_1_only, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)
        covs = get_output(self.cov_nn_1_only, h).reshape((N, self.num_z, self.z_dim))  # N * Z * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        attention_weights_nn_params = get_all_params(get_all_layers(self.attention_weights_nn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn_both, self.cov_nn_both]), trainable=True)
        nn_0_only_params = get_all_params(get_all_layers([self.mean_nn_0_only, self.cov_nn_0_only]), trainable=True)
        nn_1_only_params = get_all_params(get_all_layers([self.mean_nn_1_only, self.cov_nn_1_only]), trainable=True)

        return rnn_params + attention_weights_nn_params + nn_params + nn_0_only_params + nn_1_only_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        attention_weights_nn_params_vals = get_all_param_values(get_all_layers(self.attention_weights_nn))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn_both, self.cov_nn_both]))
        nn_0_only_params_vals = get_all_param_values(get_all_layers([self.mean_nn_0_only, self.cov_nn_0_only]))
        nn_1_only_params_vals = get_all_param_values(get_all_layers([self.mean_nn_1_only, self.cov_nn_1_only]))

        return [rnn_params_vals, attention_weights_nn_params_vals, nn_params_vals, nn_0_only_params_vals,
                nn_1_only_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, attention_weights_nn_params_vals, nn_params_vals, nn_0_only_params_vals,
         nn_1_only_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers(self.attention_weights_nn), attention_weights_nn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_both, self.cov_nn_both]), nn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_0_only, self.cov_nn_0_only]), nn_0_only_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_1_only, self.cov_nn_1_only]), nn_1_only_params_vals)


class RecRNNSplitForwardBackwardFinalTwoLatentsDependentSSL(object):

    def __init__(self, z_dim, max_length_0, max_length_1, embedding_dim_0, embedding_dim_1, dist_z, nn_kwargs):

        self.z_dim = z_dim

        self.max_length_0 = max_length_0
        self.max_length_1 = max_length_1
        self.max_length = max(self.max_length_0, self.max_length_1)

        self.embedding_dim_0 = embedding_dim_0
        self.embedding_dim_1 = embedding_dim_1

        self.dist_z = dist_z()

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        self.rnn = self.rnn_fn()

        self.mean_nn_both_z1, self.cov_nn_both_z1, self.mean_nn_both_z0, self.cov_nn_both_z0 = self.nn_fn_both()
        self.mean_nn_0_only_z1, self.cov_nn_0_only_z1, self.mean_nn_0_only_z0, self.cov_nn_0_only_z0 = \
            self.nn_fn_single_only()
        self.mean_nn_1_only_z1, self.cov_nn_1_only_z1, self.mean_nn_1_only_z0, self.cov_nn_1_only_z0 = \
            self.nn_fn_single_only()

    def rnn_fn(self):

        l_in_0 = InputLayer((None, self.max_length, self.embedding_dim_0))

        l_mask_0 = InputLayer((None, self.max_length))

        l_forward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_backward_0 = LSTMLayer(l_in_0, num_units=self.nn_rnn_hid_units, mask_input=l_mask_0,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True, backwards=True)

        l_in_1 = InputLayer((None, self.max_length, self.embedding_dim_1))

        l_mask_1 = InputLayer((None, self.max_length))

        l_forward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        l_backward_1 = LSTMLayer(l_in_1, num_units=self.nn_rnn_hid_units, mask_input=l_mask_1,
                                 nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True, backwards=True)

        return [l_forward_0, l_backward_0, l_forward_1, l_backward_1]

    def nn_fn_both(self):

        l_in_hid = InputLayer((None, 4 * self.nn_rnn_hid_units))

        l_prev = l_in_hid

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn_z1 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn_z1 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        l_in_hid_and_z1 = InputLayer((None, (4 * self.nn_rnn_hid_units) + self.z_dim))

        l_prev = l_in_hid_and_z1

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn_z0 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn_z0 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn_z1, cov_nn_z1, mean_nn_z0, cov_nn_z0

    def nn_fn_single_only(self):

        l_in_hid = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in_hid

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn_z1 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn_z1 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        l_in_hid_and_z1 = InputLayer((None, (2 * self.nn_rnn_hid_units) + self.z_dim))

        l_prev = l_in_hid_and_z1

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn_z0 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn_z0 = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn_z1, cov_nn_z1, mean_nn_z0, cov_nn_z0

    def get_hid_0(self, x_0, x_0_embedded):

        mask_0 = T.ge(x_0, 0)  # N * max(L)

        h_forward_0 = self.rnn[0].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_backward_0 = self.rnn[1].get_output_for([x_0_embedded, mask_0])  # N * dim(hid)
        h_0 = T.concatenate([h_forward_0, h_backward_0], axis=-1)  # N * (2*dim(hid))

        return h_0

    def get_hid_1(self, x_1, x_1_embedded):

        mask_1 = T.ge(x_1, 0)  # N * max(L)

        h_forward_1 = self.rnn[2].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)
        h_backward_1 = self.rnn[3].get_output_for([x_1_embedded, mask_1])  # N * dim(hid)
        h_1 = T.concatenate([h_forward_1, h_backward_1], axis=-1)  # N * (2*dim(hid))

        return h_1

    def get_hid_both(self, x_0, x_0_embedded, x_1, x_1_embedded):

        h_0 = self.get_hid_0(x_0, x_0_embedded)  # N * (2*dim(hid))
        h_1 = self.get_hid_1(x_1, x_1_embedded)  # N * (2*dim(hid))

        return T.concatenate([h_0, h_1], axis=-1)  # N * (4*dim(hid))

    def get_samples_and_kl_std_gaussian_both(self, x_0, x_0_embedded, x_1, x_1_embedded, num_samples, means_only=False):

        hid = self.get_hid_both(x_0, x_0_embedded, x_1, x_1_embedded)  # N * (4*dim(hid))
        hid_rep = T.tile(hid, (num_samples, 1))  # (S*N) * (4*dim(hid))

        means_z1 = T.tile(get_output(self.mean_nn_both_z1, hid), (num_samples, 1))  # (S*N) * dim(z)
        covs_z1 = T.tile(get_output(self.cov_nn_both_z1, hid), (num_samples, 1))  # (S*N) * dim(z)

        if means_only:

            z_1 = means_z1  # (S*N) * dim(z)

            means_z0 = get_output(self.mean_nn_both_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)
            covs_z0 = get_output(self.cov_nn_both_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)

            z_0 = means_z0

        else:

            z_1 = self.dist_z.get_samples(1, [means_z1, covs_z1])  # (S*N) * dim(z)

            means_z0 = get_output(self.mean_nn_both_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)
            covs_z0 = get_output(self.cov_nn_both_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)

            z_0 = self.dist_z.get_samples(1, [means_z0, covs_z0])  # (S*N) * dim(z)

        z = T.stack([z_0, z_1], axis=1)  # (S*N) * 2 * dim(z)

        means = T.stack([means_z0, means_z1], axis=1)
        covs = T.stack([covs_z0, covs_z1], axis=1)

        kl = (1./num_samples) * -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2),
                                             axis=range(1, means.ndim))

        return z, kl

    def get_samples_and_kl_std_gaussian_0_only(self, x_0, x_0_embedded, num_samples, means_only=False):

        hid = self.get_hid_0(x_0, x_0_embedded)  # N * (2*dim(hid))
        hid_rep = T.tile(hid, (num_samples, 1))  # (S*N) * (2*dim(hid))

        means_z1 = get_output(self.mean_nn_0_only_z1, hid)  # N * dim(z)
        covs_z1 = get_output(self.cov_nn_0_only_z1, hid)  # N * dim(z)

        if means_only:

            z_1 = T.tile(means_z1, [num_samples] + [1]*(means_z1.ndim - 1))  # (S*N) * dim(z)

            means_z0 = get_output(self.mean_nn_0_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)
            covs_z0 = get_output(self.cov_nn_0_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)

            z_0 = means_z0

        else:

            z_1 = self.dist_z.get_samples(num_samples, [means_z1, covs_z1])  # (S*N) * dim(z)

            means_z0 = get_output(self.mean_nn_0_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)
            covs_z0 = get_output(self.cov_nn_0_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)

            z_0 = self.dist_z.get_samples(1, [means_z0, covs_z0])  # (S*N) * dim(z)

        z = T.stack([z_0, z_1], axis=1)  # (S*N) * 2 * dim(z)

        means = T.stack([means_z0, means_z1], axis=1)
        covs = T.stack([covs_z0, covs_z1], axis=1)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return z, kl

    def get_samples_and_kl_std_gaussian_1_only(self, x_1, x_1_embedded, num_samples, means_only=False):

        hid = self.get_hid_0(x_1, x_1_embedded)  # N * (2*dim(hid))
        hid_rep = T.tile(hid, (num_samples, 1))  # (S*N) * (2*dim(hid))

        means_z1 = get_output(self.mean_nn_1_only_z1, hid)  # N * dim(z)
        covs_z1 = get_output(self.cov_nn_1_only_z1, hid)  # N * dim(z)

        if means_only:

            z_1 = T.tile(means_z1, [num_samples] + [1]*(means_z1.ndim - 1))  # (S*N) * dim(z)

            means_z0 = get_output(self.mean_nn_1_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)
            covs_z0 = get_output(self.cov_nn_1_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)

            z_0 = means_z0

        else:

            z_1 = self.dist_z.get_samples(num_samples, [means_z1, covs_z1])  # (S*N) * dim(z)

            means_z0 = get_output(self.mean_nn_1_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)
            covs_z0 = get_output(self.cov_nn_1_only_z0, T.concatenate([hid_rep, z_1], axis=-1))  # (S*N) * dim(z)

            z_0 = self.dist_z.get_samples(1, [means_z0, covs_z0])  # (S*N) * dim(z)

        z = T.stack([z_0, z_1], axis=1)  # (S*N) * 2 * dim(z)

        means = T.stack([means_z0, means_z1], axis=1)
        covs = T.stack([covs_z0, covs_z1], axis=1)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return z, kl

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn_both_z1, self.cov_nn_both_z1, self.mean_nn_both_z0,
                                                   self.cov_nn_both_z0]), trainable=True)
        nn_0_only_params = get_all_params(get_all_layers([self.mean_nn_0_only_z1, self.cov_nn_0_only_z1,
                                                          self.mean_nn_0_only_z0, self.cov_nn_0_only_z0]),
                                          trainable=True)
        nn_1_only_params = get_all_params(get_all_layers([self.mean_nn_1_only_z1, self.cov_nn_1_only_z1,
                                                          self.mean_nn_1_only_z0, self.cov_nn_1_only_z0]),
                                          trainable=True)

        return rnn_params + nn_params + nn_0_only_params + nn_1_only_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn_both_z1, self.cov_nn_both_z1,
                                                              self.mean_nn_both_z0, self.cov_nn_both_z0]))
        nn_0_only_params_vals = get_all_param_values(get_all_layers([self.mean_nn_0_only_z1, self.cov_nn_0_only_z1,
                                                                     self.mean_nn_0_only_z0, self.cov_nn_0_only_z0]))
        nn_1_only_params_vals = get_all_param_values(get_all_layers([self.mean_nn_1_only_z1, self.cov_nn_1_only_z1,
                                                                     self.mean_nn_1_only_z0, self.cov_nn_1_only_z0]))

        return [rnn_params_vals, nn_params_vals, nn_0_only_params_vals, nn_1_only_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals, nn_0_only_params_vals, nn_1_only_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_both_z1, self.cov_nn_both_z1, self.mean_nn_both_z0,
                                             self.cov_nn_both_z0]), nn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_0_only_z1, self.cov_nn_0_only_z1, self.mean_nn_0_only_z0,
                                             self.cov_nn_0_only_z0]), nn_0_only_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn_1_only_z1, self.cov_nn_1_only_z1, self.mean_nn_1_only_z0,
                                             self.cov_nn_1_only_z0]), nn_1_only_params_vals)


class RecRNNSplitForwardBackwardFinalSSLMulti(object):

    def __init__(self, z_dim, max_length, embedding_dim, dist_z, num_langs, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.dist_z = dist_z()

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        self.rnns = {i: self.rnn_fn() for i in range(num_langs)}

        self.only_nns = {i: self.nn_fn_single_only() for i in range(num_langs)}

        self.both_nns = {i: self.nn_fn_both() for i in combinations(range(num_langs), 2)}

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim))

        l_mask = InputLayer((None, self.max_length))

        l_out = LSTMLayer(l_in, num_units=self.nn_rnn_hid_units, mask_input=l_mask,
                          nonlinearity=self.nn_rnn_hid_nonlinearity, only_return_final=True)

        return l_out

    def nn_fn_single_only(self):

        l_in = InputLayer((None, self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def nn_fn_both(self):

        l_in = InputLayer((None, 2 * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid_only(self, x, x_embedded, lang):

        mask = T.ge(x, 0)  # N * max(L)

        h = self.rnns[lang].get_output_for([x_embedded, mask])  # N * dim(hid)

        return h

    def get_hid_both(self, x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1):

        h_0 = self.get_hid_only(x_0, x_0_embedded, lang_0)  # N * dim(hid)
        h_1 = self.get_hid_only(x_1, x_1_embedded, lang_1)  # N * dim(hid)

        h = T.concatenate([h_0, h_1], axis=-1)  # N * (2*dim(hid))

        return h

    def get_means_and_covs_only(self, x, x_embedded, lang):

        h = self.get_hid_only(x, x_embedded, lang)  # N * dim(hid)

        means = get_output(self.only_nns[lang][0], h)  # N * dim(z)
        covs = get_output(self.only_nns[lang][1], h)  # N * dim(z)

        return means, covs

    def get_means_and_covs_both(self, x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1):

        h = self.get_hid_both(x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1)  # N * (2*dim(hid))

        means = get_output(self.both_nns[(lang_0, lang_1)][0], h)  # N * dim(z)
        covs = get_output(self.both_nns[(lang_0, lang_1)][1], h)  # N * dim(z)

        return means, covs

    def get_samples_and_kl_std_gaussian_only(self, x, x_embedded, lang, num_samples, means_only=False):

        means, covs = self.get_means_and_covs_only(x, x_embedded, lang)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_samples_and_kl_std_gaussian_both(self, x_0, x_0_embedded, x_1, x_1_embedded, lang_0, lang_1, num_samples,
                                             means_only=False):

        if lang_0 < lang_1:
            min_lang = lang_0
            max_lang = lang_1
            x_min_lang = x_0
            x_min_lang_embedded = x_0_embedded
            x_max_lang = x_1
            x_max_lang_embedded = x_1_embedded
        else:
            min_lang = lang_1
            max_lang = lang_0
            x_min_lang = x_1
            x_min_lang_embedded = x_1_embedded
            x_max_lang = x_0
            x_max_lang_embedded = x_0_embedded

        means, covs = self.get_means_and_covs_both(x_min_lang, x_min_lang_embedded, x_max_lang, x_max_lang_embedded,
                                                   min_lang, max_lang)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(list(self.rnns.values())), trainable=True)
        only_nn_params = get_all_params(get_all_layers(list(chain.from_iterable(self.only_nns.values()))),
                                        trainable=True)
        both_nn_params = get_all_params(get_all_layers(list(chain.from_iterable(self.both_nns.values()))),
                                        trainable=True)

        return rnn_params + only_nn_params + both_nn_params

    def get_param_values(self):

        rnn_params_vals = {l: get_all_param_values(get_all_layers(self.rnns[l])) for l in self.rnns.keys()}
        only_nn_params_vals = {l: get_all_param_values(get_all_layers(self.only_nns[l])) for l in self.only_nns.keys()}
        both_nn_params_vals = {l: get_all_param_values(get_all_layers(self.both_nns[l])) for l in self.both_nns.keys()}

        return [rnn_params_vals, only_nn_params_vals, both_nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, only_nn_params_vals, both_nn_params_vals] = param_values

        for l in self.rnns.keys():
            set_all_param_values(get_all_layers(self.rnns[l]), rnn_params_vals[l])

        for l in self.only_nns.keys():
            set_all_param_values(get_all_layers(self.only_nns[l]), only_nn_params_vals[l])

        for l in self.both_nns.keys():
            set_all_param_values(get_all_layers(self.both_nns[l]), both_nn_params_vals[l])
