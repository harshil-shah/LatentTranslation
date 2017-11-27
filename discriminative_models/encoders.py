import theano.tensor as T
from lasagne.layers import get_all_layers, get_all_params, get_all_param_values, InputLayer, set_all_param_values
from nn.layers import LSTMLayer0Mask


class Encoder(object):

    def __init__(self, z_dim, max_length, embedding_dim):

        self.z_dim = z_dim

        self.max_length = max_length

        self.embedding_dim = embedding_dim

        self.nn = self.nn_fn()

    def nn_fn(self):

        raise NotImplementedError()

    def get_hid(self, x, x_embedded):

        raise NotImplementedError()


class RNNSearchEncoder(Encoder):

    def __init__(self, z_dim, max_length, embedding_dim, nn_kwargs):

        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        super().__init__(z_dim, max_length, embedding_dim)

    def nn_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim))

        l_mask = InputLayer((None, self.max_length))

        l_forward = LSTMLayer0Mask(l_in, num_units=int(self.z_dim / 2), mask_input=l_mask,
                                   nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_backward = LSTMLayer0Mask(l_in, num_units=int(self.z_dim / 2), mask_input=l_mask,
                                    nonlinearity=self.nn_rnn_hid_nonlinearity, backwards=True)

        return [l_forward, l_backward]

    def get_z(self, x, x_embedded):

        mask = T.ge(x, 0)  # N * max(L)

        z_forward = self.nn[0].get_output_for([x_embedded, mask])  # N * dim(hid)
        z_backward = self.nn[1].get_output_for([x_embedded, mask])  # N * dim(hid)
        z = T.concatenate([z_forward, z_backward], axis=-1)  # N * (2*dim(hid))

        return z

    def get_params(self):

        nn_params = get_all_params(get_all_layers(self.nn), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers(self.nn))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.nn), nn_params_vals)
