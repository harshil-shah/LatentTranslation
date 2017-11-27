import theano
import theano.tensor as T
from lasagne.layers import get_all_layers, get_all_params, get_all_param_values, get_output, InputLayer, \
    RecurrentLayer, set_all_param_values
from lasagne.nonlinearities import linear
from nn.layers import RNNSearchLayer
from model.utilities import last_d_softmax


class RNNSearchDecoder(object):

    def __init__(self, enc_h_dim, max_length, embedding_dim, embedder, nn_kwargs):

        self.enc_h_dim = enc_h_dim
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.alignment_nn = self.alignment_nn_fn()

        self.nn_in, self.nn = self.nn_fn()

    def alignment_nn_fn(self):

        l_in_h = InputLayer((None, self.max_length, self.nn_rnn_hid_units + self.enc_h_dim))

        l_out = RecurrentLayer(l_in_h, 1, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return l_out

    def nn_fn(self):

        l_in_x = InputLayer((None, self.max_length, self.embedding_dim))

        l_in_enc_h = InputLayer((None, self.max_length, self.enc_h_dim))

        l_rnn = RNNSearchLayer(l_in_x, l_in_enc_h, self.nn_rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x, l_in_enc_h), l_out

    def get_probs(self, x_embedded, x_embedded_dropped, enc_h, all_embeddings, mode='all'):

        N = x_embedded.shape[0]

        x_pre_padded = T.concatenate([T.zeros((N, 1, self.embedding_dim)), x_embedded_dropped], axis=1)[:, :-1]  # N *
        # max(L) * E

        target_embeddings = get_output(self.nn, {self.nn_in[0]: x_pre_padded,
                                                 self.nn_in[1]: enc_h})  # N * max(L) * E

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

    def log_p_x(self, x, x_embedded, x_embedded_dropped, enc_h, all_embeddings):

        x_padding_mask = T.ge(x, 0)  # N * max(L)

        probs = self.get_probs(x_embedded, x_embedded_dropped, enc_h, all_embeddings, mode='true')  # N * max(L)
        probs += T.cast(1.e-5, 'float32')  # N * max(L)

        log_p_x = T.sum(x_padding_mask * T.log(probs), axis=-1)  # N

        return log_p_x

    def beam_search(self, enc_h, all_embeddings, beam_size):

        N = enc_h.shape[0]

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            x_pre_padded = T.concatenate([T.zeros((N*beam_size, 1, self.embedding_dim)), active_paths_embedded],
                                         axis=1)[:, :-1]  # (N*B) * max(L) * E

            nn_input = {self.nn_in[0]: x_pre_padded,
                        self.nn_in[1]: T.repeat(enc_h, beam_size, 0)}

            target_embeddings = get_output(self.nn, nn_input)[:, l].reshape((N, beam_size, self.embedding_dim))
            # N * B * E

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

    def get_params(self):

        nn_params = get_all_params(get_all_layers(self.nn), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers(self.nn))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.nn), nn_params_vals)
