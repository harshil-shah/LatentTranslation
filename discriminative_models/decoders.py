import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer, get_all_layers, get_all_params, get_all_param_values, get_output, InputLayer, \
    RecurrentLayer, set_all_param_values
from lasagne.nonlinearities import linear, tanh
from nn.layers import CanvasRNNLayer, RNNSearchLayer
from model.utilities import last_d_softmax


class Decoder(object):

    def __init__(self, z_dim, max_length, embedding_dim, embedder):

        self.z_dim = z_dim
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.embedder = embedder

    def get_target_embeddings(self, x_pre_padded, z):

        raise NotImplementedError

    def get_probs(self, x_embedded, x_embedded_dropped, z, all_embeddings, mode='all'):

        N = x_embedded.shape[0]

        x_pre_padded = T.concatenate([T.zeros((N, 1, self.embedding_dim)), x_embedded_dropped], axis=1)[:, :-1]  # N *
        # max(L) * E

        target_embeddings = self.get_target_embeddings(x_pre_padded, z)  # N * max(L) * E

        probs_numerators = T.sum(x_embedded * target_embeddings, axis=-1)  # N * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # N * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # N * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_x(self, x, x_embedded, x_embedded_dropped, z, all_embeddings):

        x_padding_mask = T.ge(x, 0)  # N * max(L)

        probs = self.get_probs(x_embedded, x_embedded_dropped, z, all_embeddings, mode='true')  # N * max(L)
        probs += T.cast(1.e-5, 'float32')  # N * max(L)

        log_p_x = T.sum(x_padding_mask * T.log(probs), axis=-1)  # N

        return log_p_x

    def beam_search(self, z, all_embeddings, beam_size):

        N = z.shape[0]

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            x_pre_padded = T.concatenate([T.zeros((N*beam_size, 1, self.embedding_dim)), active_paths_embedded],
                                         axis=1)[:, :-1]  # (N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings(x_pre_padded, T.repeat(z, beam_size, 0))
            target_embeddings = target_embeddings[:, l].reshape((N, beam_size, self.embedding_dim))

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return best_scores_l, active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[all_embeddings]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')


class RNNSearchDecoder(Decoder):

    def __init__(self, z_dim, max_length, embedding_dim, embedder, nn_kwargs):

        super().__init__(z_dim, max_length, embedding_dim, embedder)

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_in, self.nn = self.nn_fn()

    def nn_fn(self):

        l_in_x = InputLayer((None, self.max_length, self.embedding_dim))

        l_in_z = InputLayer((None, self.max_length, self.z_dim))

        l_rnn = RNNSearchLayer(l_in_x, l_in_z, self.nn_rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x, l_in_z), l_out

    def get_target_embeddings(self, x_pre_padded, z):

        target_embeddings = get_output(self.nn, {self.nn_in[0]: x_pre_padded,
                                                 self.nn_in[1]: z})  # N * max(L) * E

        return target_embeddings

    def get_params(self):

        nn_params = get_all_params(get_all_layers(self.nn), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers(self.nn))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.nn), nn_params_vals)


class CanvasRNNDecoder(Decoder):

    def __init__(self, z_dim, max_length, embedding_dim, embedder, nn_kwargs):

        super().__init__(z_dim, max_length, embedding_dim, embedder)

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn = self.nn_fn()
        self.read_attention_nn = self.read_attention_nn_fn()

        self.W = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim + 3*self.embedding_dim,
                                                                     self.embedding_dim))), name='W')

    def nn_fn(self):

        l_in_z = InputLayer((None, self.max_length, self.z_dim))

        l_out = CanvasRNNLayer(l_in_z, self.nn_rnn_hid_units, max_length=self.max_length,
                               embedding_size=self.embedding_dim, nonlinearity=self.nn_rnn_hid_nonlinearity,
                               only_return_final=True)

        return l_out

    def read_attention_nn_fn(self):

        l_in = InputLayer((None, self.max_length, self.z_dim))

        l_prev = l_in

        for h in range(2):

            l_prev = DenseLayer(l_prev, num_units=500, nonlinearity=tanh)

        l_out = DenseLayer(l_prev, num_units=self.max_length**2, nonlinearity=None)

        return l_out

    def read_attention(self, z):

        N = z.shape[0]

        read_attention = get_output(self.read_attention_nn, z).reshape((N, self.max_length, self.max_length))  # N *
        # max(L) * max(L)

        read_attention_mask = T.switch(T.eq(T.tril(T.ones((self.max_length, self.max_length)), -2), 0), -np.inf, 1.)
        # max(L) * max(L)

        read_attention_pre_softmax = read_attention * T.shape_padleft(read_attention_mask)  # N * max(L) * max(L)

        read_attention_pre_softmax = T.switch(T.isinf(read_attention_pre_softmax), -np.inf, read_attention_pre_softmax)
        # N * max(L) * max(L)

        read_attention_softmax = last_d_softmax(read_attention_pre_softmax)  # N * max(L) * max(L)

        read_attention_softmax = T.switch(T.isnan(read_attention_softmax), 0,
                                          read_attention_softmax)  # N * max(L) * max(L)

        return read_attention_softmax

    def get_target_embeddings(self, x_pre_padded, z):

        read_attention = self.read_attention(z)  # N * max(L) * max(L)

        total_written = T.batched_dot(read_attention, x_pre_padded)  # N * max(L) * E

        canvases = get_output(self.nn, z)  # N * max(L) * E

        target_embeddings = T.dot(T.concatenate((z, total_written, x_pre_padded, canvases), axis=-1),
                                  self.W)  # N * max(L) * E

        return target_embeddings

    def get_params(self):

        nn_params = get_all_params(get_all_layers(self.nn), trainable=True)
        read_attention_nn_params = get_all_params(get_all_layers(self.read_attention_nn), trainable=True)

        return nn_params + read_attention_nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers(self.nn))
        read_attention_nn_params_vals = get_all_param_values(get_all_layers(self.read_attention_nn))

        return [nn_params_vals, read_attention_nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals, read_attention_nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.nn), nn_params_vals)
        set_all_param_values(get_all_layers(self.read_attention_nn), read_attention_nn_params_vals)
