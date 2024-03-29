import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import ConcatLayer, DenseLayer, get_all_layers, get_all_param_values, get_all_params, get_output, \
    InputLayer, LSTMLayer, RecurrentLayer, ReshapeLayer, set_all_param_values
from lasagne.nonlinearities import linear
from nn.layers import CanvasRNNLayer, LSTMLayer0Mask, RNNSearchLayer
from nn.nonlinearities import elu_plus_one

from .utilities import last_d_softmax

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()


class GenAUTRWords(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_canvas_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_canvas_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']
        self.nn_canvas_rnn_time_steps = nn_kwargs['rnn_time_steps']

        try:
            self.nn_canvas_rnn_separate_updates = nn_kwargs['rnn_separate_updates']
        except KeyError:
            self.nn_canvas_rnn_separate_updates = False

        self.nn_read_attention_nn_depth = nn_kwargs['read_attention_nn_depth']
        self.nn_read_attention_nn_hid_units = nn_kwargs['read_attention_nn_hid_units']
        self.nn_read_attention_nn_hid_nonlinearity = nn_kwargs['read_attention_nn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.canvas_rnn = self.canvas_rnn_fn()

        self.canvas_update_params = self.init_canvas_update_params()

        self.read_attention_nn = self.read_attention_nn_fn()

    def init_canvas_update_params(self):

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim + 3*self.embedding_dim,
                                                                       self.embedding_dim))), name='W_x_to_x')

        canvas_update_params = [W_x_to_x]

        return canvas_update_params

    def read_attention_nn_fn(self):

        l_in = InputLayer((None, self.z_dim))

        l_prev = l_in

        for h in range(self.nn_read_attention_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_read_attention_nn_hid_units,
                                nonlinearity=self.nn_read_attention_nn_hid_nonlinearity)

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

    def canvas_rnn_fn(self):

        l_in = InputLayer((None, None, self.z_dim))

        l_1 = CanvasRNNLayer(l_in, num_units=self.nn_canvas_rnn_hid_units, max_length=self.max_length,
                             embedding_size=self.embedding_dim, separate_updates=self.nn_canvas_rnn_separate_updates,
                             only_return_final=True)

        return l_1

    def log_p_z(self, z):

        return self.dist_z.log_density(z)

    def get_canvases(self, z, num_time_steps=None):
        """
        :param z: N * dim(z) matrix
        :param num_time_steps: int, number of RNN time steps to use

        :return canvases: N * max(L) * E tensor
        :return canvas_gate_sums: N * max(L) matrix
        """

        if num_time_steps is None:
            num_time_steps = self.nn_canvas_rnn_time_steps

        z_rep = T.tile(z.reshape((z.shape[0], 1, z.shape[1])), (1, num_time_steps, 1))  # N * T * dim(z)

        canvases = get_output(self.canvas_rnn, T.cast(z_rep, 'float32'))  # N * T * dim(hid)

        return canvases

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        read_attention = self.read_attention(z)  # (S*N) * max(L) * max(L)

        total_written = T.batched_dot(read_attention, x_dropped)  # (S*N) * max(L) * E

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))

        target_embeddings = T.dot(T.concatenate((z_rep, total_written, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_x(self, x, x_embedded, x_embedded_dropped, z, all_embeddings):
        """
        :param x: N * max(L) tensor
        :param x_embedded: N * max(L) * E tensor
        :param x_embedded_dropped: N * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)
        x_rep_padding_mask = T.switch(T.lt(x_rep, 0), 0, 1)  # (S*N) * max(L)

        x_embedded_rep = T.tile(x_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_embedded_dropped_rep = T.tile(x_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        canvases = self.get_canvases(z)

        probs = self.get_probs(x_embedded_rep, x_embedded_dropped_rep, z, canvases, all_embeddings, mode='true')
        # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        # probs = theano_print_min_max(probs, 'probs')

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

        N = z.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z, num_time_steps)  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z)  # N * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, canvases, all_embeddings, W_x_to_x, z_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((N*beam_size, self.max_length,
                                                                                 self.embedding_dim))
                                          ).reshape((N, beam_size, self.max_length, self.embedding_dim))[:, :, l]  # N *
            # B * E

            target_embeddings = T.dot(T.concatenate((z_rep, total_written, active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

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

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[canvases, all_embeddings,
                                                                      self.canvas_update_params[-1],
                                                                      T.cast(z_rep, 'float32')]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def beam_search_samples(self, z, all_embeddings, beam_size, num_samples, num_time_steps=None):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        log_p_z = self.log_p_z(z)  # (S*N)

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, beam_size, 1))  # (S*N) * B * dim(z)

        canvases = self.get_canvases(z, num_time_steps)  # (S*N) * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z)  # (S*N) * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, log_p_z, canvases, all_embeddings, W_x_to_x, z_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            active_paths_current_embedded = T.tile(active_paths_current_embedded, (num_samples, 1, 1, 1))  # (S*N) * B *
            # max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((num_samples*N*beam_size,
                                                                                 self.max_length, self.embedding_dim)))
            # (S*N*B) * max(L) * E

            total_written = total_written.reshape((num_samples*N, beam_size, self.max_length,
                                                   self.embedding_dim))[:, :, l]  # (S*N) * B * E

            target_embeddings = T.dot(T.concatenate((z_rep, total_written, active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # (S*N) * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[log_p_z, canvases, all_embeddings,
                                                                      self.canvas_update_params[-1],
                                                                      T.cast(z_rep, 'float32')]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def generate_text(self, z, all_embeddings, num_time_steps=None):

        N = z.shape[0]

        canvases = self.get_canvases(z, num_time_steps)

        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):

            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, x_prev_sampled_embedded, z, canvases,
                                           all_embeddings, mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            #

            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            probs_argmax = self.get_probs(x_prev_argmax_embedded, x_prev_argmax_embedded, z, canvases, all_embeddings,
                                          mode='all')  # N * max(L) * D

            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        return x_sampled[-1], x_argmax[-1], updates

    def generate_z_prior(self, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        return z

    def generate_output_prior(self, all_embeddings, num_samples, beam_size, num_time_steps=None):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        # _, attention = self.get_canvases(z, num_time_steps)
        attention = T.zeros((z.shape[0], self.max_length))

        outputs = [z, x_gen_sampled, x_gen_argmax, x_gen_beam, attention]

        return outputs, updates

    def generate_output_posterior_fn(self, x, z, all_embeddings, beam_size, num_time_steps=None):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        generate_output_posterior = theano.function(inputs=[x],
                                                    outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam],
                                                    updates=updates,
                                                    allow_input_downcast=True
                                                    )

        return generate_output_posterior

    def get_params(self):

        canvas_rnn_params = get_all_params(self.canvas_rnn, trainable=True)

        read_attention_nn_params = get_all_params(self.read_attention_nn, trainable=True)

        return canvas_rnn_params + read_attention_nn_params + self.canvas_update_params

    def get_param_values(self):

        canvas_rnn_params_vals = get_all_param_values(self.canvas_rnn)

        read_attention_nn_params_vals = get_all_param_values(self.read_attention_nn)

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [canvas_rnn_params_vals, read_attention_nn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):

        [canvas_rnn_params_vals, read_attention_nn_params_vals, canvas_update_params_vals] = param_values

        set_all_param_values(self.canvas_rnn, canvas_rnn_params_vals)

        set_all_param_values(self.read_attention_nn, read_attention_nn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])


class GenAUTRWordsTwoLatents(GenAUTRWords):

    def init_canvas_update_params(self):

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (2*self.z_dim + 3*self.embedding_dim,
                                                                       self.embedding_dim))), name='W_x_to_x')

        canvas_update_params = [W_x_to_x]

        return canvas_update_params

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * 2 * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        read_attention = self.read_attention(z[:, 1])  # (S*N) * max(L) * max(L)

        total_written = T.batched_dot(read_attention, x_dropped)  # (S*N) * max(L) * E

        z_rep = T.tile(T.shape_padaxis(z.reshape((SN, 2*self.z_dim)), 1), (1, self.max_length, 1))

        target_embeddings = T.dot(T.concatenate((z_rep, total_written, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_x(self, x, x_embedded, x_embedded_dropped, z, all_embeddings):
        """
        :param x: N * max(L) tensor
        :param x_embedded: N * max(L) * E tensor
        :param x_embedded_dropped: N * max(L) * E tensor
        :param z: (S*N) * 2 * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)
        x_rep_padding_mask = T.switch(T.lt(x_rep, 0), 0, 1)  # (S*N) * max(L)

        x_embedded_rep = T.tile(x_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_embedded_dropped_rep = T.tile(x_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        canvases = self.get_canvases(z[:, 0])

        probs = self.get_probs(x_embedded_rep, x_embedded_dropped_rep, z, canvases, all_embeddings, mode='true')
        # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

        N = z.shape[0]

        z_0_rep = T.tile(T.shape_padaxis(z[:, 0], 1), (1, beam_size, 1))  # N * B * dim(z)
        z_1_rep = T.tile(T.shape_padaxis(z[:, 1], 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z[:, 0], num_time_steps)  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z[:, 1])  # N * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, canvases, all_embeddings, W_x_to_x, z_0_rep,
                         z_1_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((N*beam_size, self.max_length,
                                                                                 self.embedding_dim))
                                          ).reshape((N, beam_size, self.max_length, self.embedding_dim))[:, :, l]  # N *
            # B * E

            target_embeddings = T.dot(T.concatenate((z_0_rep, z_1_rep, total_written,
                                                     active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

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

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[canvases, all_embeddings,
                                                                      self.canvas_update_params[-1],
                                                                      T.cast(z_0_rep, 'float32'),
                                                                      T.cast(z_1_rep, 'float32')]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def beam_search_samples(self, z, all_embeddings, beam_size, num_samples, num_time_steps=None):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        log_p_z = self.log_p_z(z)  # (S*N)

        z_0_rep = T.tile(T.shape_padaxis(z[:, 0], 1), (1, beam_size, 1))  # (S*N) * B * dim(z)
        z_1_rep = T.tile(T.shape_padaxis(z[:, 1], 1), (1, beam_size, 1))  # (S*N) * B * dim(z)

        canvases = self.get_canvases(z[:, 0], num_time_steps)  # (S*N) * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z[:, 1])  # (S*N) * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, log_p_z, canvases, all_embeddings, W_x_to_x,
                         z_0_rep, z_1_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            active_paths_current_embedded = T.tile(active_paths_current_embedded, (num_samples, 1, 1, 1))  # (S*N) * B *
            # max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((num_samples*N*beam_size,
                                                                                 self.max_length, self.embedding_dim)))
            # (S*N*B) * max(L) * E

            total_written = total_written.reshape((num_samples*N, beam_size, self.max_length,
                                                   self.embedding_dim))[:, :, l]  # (S*N) * B * E

            target_embeddings = T.dot(T.concatenate((z_0_rep, z_1_rep, total_written,
                                                     active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # (S*N) * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[log_p_z, canvases, all_embeddings,
                                                                      self.canvas_update_params[-1],
                                                                      T.cast(z_0_rep, 'float32'),
                                                                      T.cast(z_1_rep, 'float32')]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def generate_text(self, z, all_embeddings, num_time_steps=None):

        N = z.shape[0]

        canvases = self.get_canvases(z[:, 0], num_time_steps)

        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):

            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, x_prev_sampled_embedded, z, canvases,
                                           all_embeddings, mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            #

            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            probs_argmax = self.get_probs(x_prev_argmax_embedded, x_prev_argmax_embedded, z, canvases, all_embeddings,
                                          mode='all')  # N * max(L) * D

            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        return x_sampled[-1], x_argmax[-1], updates

    def generate_z_prior(self, num_samples):

        z = self.dist_z.get_samples(dims=[1, 2, self.z_dim], num_samples=num_samples)  # S * 2 * dim(z)

        return z

    def generate_output_prior(self, all_embeddings, num_samples, beam_size, num_time_steps=None):

        z = self.dist_z.get_samples(dims=[1, 2, self.z_dim], num_samples=num_samples)  # S * 2 * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        # _, attention = self.get_canvases(z, num_time_steps)
        attention = T.zeros((z.shape[0], self.max_length))

        outputs = [z, x_gen_sampled, x_gen_argmax, x_gen_beam, attention]

        return outputs, updates


class GenAUTRWordsTwoLatentsTwoReadAttentions(GenAUTRWordsTwoLatents):

    def init_canvas_update_params(self):

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (2*self.z_dim + 4*self.embedding_dim,
                                                                       self.embedding_dim))), name='W_x_to_x')

        canvas_update_params = [W_x_to_x]

        return canvas_update_params

    def read_attention_nn_fn(self):

        l_in = InputLayer((None, self.z_dim))

        l_prev = l_in

        for h in range(self.nn_read_attention_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_read_attention_nn_hid_units,
                                nonlinearity=self.nn_read_attention_nn_hid_nonlinearity)

        l_out = DenseLayer(l_prev, num_units=2 * (self.max_length**2), nonlinearity=None)

        return l_out

    def read_attention(self, z):

        N = z.shape[0]

        read_attention = get_output(self.read_attention_nn, z).reshape((2*N, self.max_length, self.max_length))  # (2*N)
        # * max(L) * max(L)

        read_attention_mask = T.switch(T.eq(T.tril(T.ones((self.max_length, self.max_length)), -2), 0), -np.inf, 1.)
        # max(L) * max(L)

        read_attention_pre_softmax = read_attention * T.shape_padleft(read_attention_mask)  # (2*N) * max(L) * max(L)

        read_attention_pre_softmax = T.switch(T.isinf(read_attention_pre_softmax), -np.inf, read_attention_pre_softmax)
        # (2*N) * max(L) * max(L)

        read_attention_softmax = last_d_softmax(read_attention_pre_softmax)  # (2*N) * max(L) * max(L)

        read_attention_softmax = T.switch(T.isnan(read_attention_softmax), 0,
                                          read_attention_softmax)  # (2*N) * max(L) * max(L)

        return read_attention_softmax

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * 2 * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        read_attention = self.read_attention(z[:, 1])  # (2*S*N) * max(L) * max(L)

        total_written = T.batched_dot(read_attention, T.tile(x_dropped, (2, 1, 1)))  # (2*S*N) * max(L) * E
        total_written = total_written.reshape((SN, self.max_length, 2*self.embedding_dim))  # (S*N) * max(L) * (2*E)

        z_rep = T.tile(T.shape_padaxis(z.reshape((SN, 2*self.z_dim)), 1), (1, self.max_length, 1))

        target_embeddings = T.dot(T.concatenate((z_rep, total_written, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

        N = z.shape[0]

        z_0_rep = T.tile(T.shape_padaxis(z[:, 0], 1), (1, beam_size, 1))  # N * B * dim(z)
        z_1_rep = T.tile(T.shape_padaxis(z[:, 1], 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z[:, 0], num_time_steps)  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z[:, 1])  # (2*N) * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, canvases, all_embeddings, W_x_to_x, z_0_rep,
                         z_1_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          T.tile(active_paths_current_embedded.reshape((N*beam_size, self.max_length,
                                                                                        self.embedding_dim)),
                                                 (2, 1, 1))
                                          ).reshape((N, beam_size, self.max_length, 2*self.embedding_dim))[:, :, l]  # N
            # * B * (2*E)

            target_embeddings = T.dot(T.concatenate((z_0_rep, z_1_rep, total_written,
                                                     active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

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

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[canvases, all_embeddings,
                                                                      self.canvas_update_params[-1],
                                                                      T.cast(z_0_rep, 'float32'),
                                                                      T.cast(z_1_rep, 'float32')]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def beam_search_samples(self, z, all_embeddings, beam_size, num_samples, num_time_steps=None):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        log_p_z = self.log_p_z(z)  # (S*N)

        z_0_rep = T.tile(T.shape_padaxis(z[:, 0], 1), (1, beam_size, 1))  # (S*N) * B * dim(z)
        z_1_rep = T.tile(T.shape_padaxis(z[:, 1], 1), (1, beam_size, 1))  # (S*N) * B * dim(z)

        canvases = self.get_canvases(z[:, 0], num_time_steps)  # (S*N) * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z[:, 1])  # (2*S*N) * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, log_p_z, canvases, all_embeddings, W_x_to_x,
                         z_0_rep, z_1_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            active_paths_current_embedded = T.tile(active_paths_current_embedded, (num_samples, 1, 1, 1))  # (S*N) * B *
            # max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          T.tile(active_paths_current_embedded.reshape((num_samples*N*beam_size,
                                                                                        self.max_length,
                                                                                        self.embedding_dim)),
                                                 (2, 1, 1)))
            # (2*S*N*B) * max(L) * E

            total_written = total_written.reshape((num_samples*N, beam_size, self.max_length,
                                                   2*self.embedding_dim))[:, :, l]  # (S*N) * B * (2*E)

            target_embeddings = T.dot(T.concatenate((z_0_rep, z_1_rep, total_written,
                                                     active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # (S*N) * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[log_p_z, canvases, all_embeddings,
                                                                      self.canvas_update_params[-1],
                                                                      T.cast(z_0_rep, 'float32'),
                                                                      T.cast(z_1_rep, 'float32')]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')


class GenAUTRWordsMultipleLatents(GenAUTRWords):

    def init_canvas_update_params(self):

        W_z_contrib = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length,
                                                                          self.nn_canvas_rnn_time_steps))))

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim + 3*self.embedding_dim,
                                                                       self.embedding_dim))), name='W_x_to_x')

        canvas_update_params = [W_z_contrib, W_x_to_x]

        return canvas_update_params

    def read_attention_nn_fn(self):

        l_in = InputLayer((None, self.nn_canvas_rnn_time_steps, self.z_dim))

        l_prev = l_in

        for h in range(self.nn_read_attention_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_read_attention_nn_hid_units,
                                nonlinearity=self.nn_read_attention_nn_hid_nonlinearity)

        l_out = DenseLayer(l_prev, num_units=self.max_length**2, nonlinearity=None)

        return l_out

    def get_canvases(self, z, num_time_steps=None):
        """
        :param z: N * dim(z) matrix
        :param num_time_steps: int, number of RNN time steps to use

        :return canvases: N * max(L) * E tensor
        :return canvas_gate_sums: N * max(L) matrix
        """

        if num_time_steps is None:
            num_time_steps = self.nn_canvas_rnn_time_steps

        canvases = get_output(self.canvas_rnn, T.cast(z, 'float32'))  # N * T * dim(hid)

        return canvases

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * T * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        read_attention = self.read_attention(z)  # (S*N) * max(L) * max(L)

        total_written = T.batched_dot(read_attention, x_dropped)  # (S*N) * max(L) * E

        W_z_contrib = T.shape_padright(T.shape_padleft(last_d_softmax(self.canvas_update_params[0])))  # 1 * max(L) * T
        # * 1

        z_contrib = T.sum(W_z_contrib * T.shape_padaxis(z, 1), axis=2)  # (S*N) * max(L) * dim(z)

        target_embeddings = T.dot(T.concatenate((z_contrib, total_written, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

        N = z.shape[0]

        canvases = self.get_canvases(z, num_time_steps)  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z)  # N * max(L) * max(L)

        W_z_contrib = T.shape_padright(T.shape_padleft(last_d_softmax(self.canvas_update_params[0])))  # 1 * max(L) * T
        # * 1

        z_contrib = T.sum(W_z_contrib * T.shape_padaxis(z, 1), axis=2)  # (S*N) * max(L) * dim(z)

        def step_forward(l, z_contrib_l, best_scores_lm1, active_paths_current, canvases, all_embeddings, W_x_to_x):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((N*beam_size, self.max_length,
                                                                                 self.embedding_dim))
                                          ).reshape((N, beam_size, self.max_length, self.embedding_dim))[:, :, l]  # N *
            # B * E

            z_contrib_l = T.tile(T.shape_padaxis(z_contrib_l, 1), (1, beam_size, 1))

            target_embeddings = T.dot(T.concatenate((z_contrib_l, total_written, active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

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

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=[T.arange(self.max_length),
                                                                  T.cast(z_contrib.dimshuffle((1, 0, 2)), 'float32')],
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[canvases, all_embeddings,
                                                                      self.canvas_update_params[-1]]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def beam_search_samples(self, z, all_embeddings, beam_size, num_samples, num_time_steps=None):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        log_p_z = self.log_p_z(z)  # (S*N)

        canvases = self.get_canvases(z, num_time_steps)  # (S*N) * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z)  # (S*N) * max(L) * max(L)

        W_z_contrib = T.shape_padright(T.shape_padleft(last_d_softmax(self.canvas_update_params[0])))  # 1 * max(L) * T
        # * 1

        z_contrib = T.sum(W_z_contrib * T.shape_padaxis(z, 1), axis=2)  # (S*N) * max(L) * dim(z)

        def step_forward(l, z_contrib_l, best_scores_lm1, active_paths_current, log_p_z, canvases, all_embeddings,
                         W_x_to_x):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            active_paths_current_embedded = T.tile(active_paths_current_embedded, (num_samples, 1, 1, 1))  # (S*N) * B *
            # max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((num_samples*N*beam_size,
                                                                                 self.max_length, self.embedding_dim)))
            # (S*N*B) * max(L) * E

            total_written = total_written.reshape((num_samples*N, beam_size, self.max_length,
                                                   self.embedding_dim))[:, :, l]  # (S*N) * B * E

            z_contrib_l = T.tile(T.shape_padaxis(z_contrib_l, 1), (1, beam_size, 1))  # (S*N) * B * dim(z)

            target_embeddings = T.dot(T.concatenate((z_contrib_l, total_written, active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # (S*N) * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=[T.arange(self.max_length),
                                                                  T.cast(z_contrib.dimshuffle((1, 0, 2)), 'float32')],
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[log_p_z, canvases, all_embeddings,
                                                                      self.canvas_update_params[-1]]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def generate_z_prior(self, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.nn_canvas_rnn_time_steps, self.z_dim], num_samples=num_samples)  # S *
        # T * dim(z)

        return z

    def generate_output_prior(self, all_embeddings, num_samples, beam_size, num_time_steps=None):

        z = self.dist_z.get_samples(dims=[1, self.nn_canvas_rnn_time_steps, self.z_dim], num_samples=num_samples)  # S *
        # T * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        # _, attention = self.get_canvases(z, num_time_steps)
        attention = T.zeros((z.shape[0], self.max_length))

        outputs = [z, x_gen_sampled, x_gen_argmax, x_gen_beam, attention]

        return outputs, updates


class GenRNNMultipleLatents(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.rnn_in, self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in_z = InputLayer((None, self.max_length, self.z_dim))

        l_in_x = InputLayer((None, self.max_length, self.embedding_dim))

        l_in = ConcatLayer([l_in_z, l_in_x], axis=-1)

        l_1 = LSTMLayer(l_in, num_units=self.nn_rnn_hid_units, nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_concat = ConcatLayer([l_in_z, l_1], axis=-1)

        l_out = RecurrentLayer(l_concat, num_units=self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=None)

        return (l_in_z, l_in_x), l_out

    def log_p_z(self, z):

        return self.dist_z.log_density(z)

    def get_probs(self, x, x_dropped, z, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * max(L) * dim(z) tensor
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        target_embeddings = get_output(self.rnn, {self.rnn_in[0]: z,
                                                  self.rnn_in[1]: x_pre_padded})  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_x(self, x, x_embedded, x_embedded_dropped, z, all_embeddings):
        """
        :param x: N * max(L) tensor
        :param x_embedded: N * max(L) * E tensor
        :param x_embedded_dropped: N * max(L) * E tensor
        :param z: (S*N) * max(L) * dim(z) tensor
        :param all_embeddings: D * E matrix

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)
        x_rep_padding_mask = T.switch(T.lt(x_rep, 0), 0, 1)  # (S*N) * max(L)

        x_embedded_rep = T.tile(x_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_embedded_dropped_rep = T.tile(x_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        probs = self.get_probs(x_embedded_rep, x_embedded_dropped_rep, z, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

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

            # rnn_input = T.concatenate((T.repeat(z, beam_size, 0), x_pre_padded), axis=-1)  # (N*B) * max(L) * (dim(z)+E)

            rnn_input = {self.rnn_in[0]: T.repeat(z, beam_size, 0),
                         self.rnn_in[1]: x_pre_padded}

            target_embeddings = get_output(self.rnn, rnn_input)[:, l].reshape((N, beam_size, self.embedding_dim))
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

    def beam_search_samples(self, z, all_embeddings, beam_size, num_samples, num_time_steps=None):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        log_p_z = self.log_p_z(z)  # (S*N)

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            active_paths_embedded = T.tile(active_paths_embedded, (num_samples, 1, 1))  # (S*N*B) *  max(L) * E

            x_pre_padded = T.concatenate([T.zeros((num_samples*N*beam_size, 1, self.embedding_dim)),
                                          active_paths_embedded], axis=1)[:, :-1]  # (S*N*B) * max(L) * E

            # rnn_input = T.concatenate((T.repeat(z, beam_size, 0), x_pre_padded), axis=-1)  # (N*B) * max(L) * (dim(z)+E)

            rnn_input = {self.rnn_in[0]: T.repeat(z, beam_size, 0),
                         self.rnn_in[1]: x_pre_padded}

            target_embeddings = get_output(self.rnn, rnn_input)[:, l].reshape((num_samples*N, beam_size,
                                                                               self.embedding_dim))  # (S*N) * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

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

    def generate_text(self, z, all_embeddings, num_time_steps=None):

        N = z.shape[0]

        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):

            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, x_prev_sampled_embedded, z,
                                           all_embeddings, mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            #

            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            probs_argmax = self.get_probs(x_prev_argmax_embedded, x_prev_argmax_embedded, z, all_embeddings,
                                          mode='all')  # N * max(L) * D

            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        return x_sampled[-1], x_argmax[-1], updates

    def generate_z_prior(self, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.max_length, self.z_dim], num_samples=num_samples)  # S * max(L) *
        # dim(z)

        return z

    def generate_output_prior(self, all_embeddings, num_samples, beam_size, num_time_steps=None):

        z = self.dist_z.get_samples(dims=[1, self.max_length, self.z_dim], num_samples=num_samples)  # S * max(L) *
        # dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        # _, attention = self.get_canvases(z, num_time_steps)
        attention = T.zeros((z.shape[0], self.max_length))

        outputs = [z, x_gen_sampled, x_gen_argmax, x_gen_beam, attention]

        return outputs, updates

    def generate_output_posterior_fn(self, x, z, all_embeddings, beam_size, num_time_steps=None):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        generate_output_posterior = theano.function(inputs=[x],
                                                    outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam],
                                                    updates=updates,
                                                    allow_input_downcast=True
                                                    )

        return generate_output_posterior

    def get_params(self):

        rnn_params = get_all_params(self.rnn, trainable=True)

        return rnn_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(self.rnn)

        return [rnn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals] = param_values

        set_all_param_values(self.rnn, rnn_params_vals)


class GenRNNSearch(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.nn_in, self.nn = self.nn_fn()

    def nn_fn(self):

        l_in_x = InputLayer((None, self.max_length, self.embedding_dim))

        l_in_z = InputLayer((None, self.max_length, self.z_dim))

        l_rnn = RNNSearchLayer(l_in_x, l_in_z, self.nn_rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x, l_in_z), l_out

    def get_probs(self, x_embedded, x_embedded_dropped, z, all_embeddings, mode='all'):

        N = x_embedded.shape[0]

        x_pre_padded = T.concatenate([T.zeros((N, 1, self.embedding_dim)), x_embedded_dropped], axis=1)[:, :-1]  # N *
        # max(L) * E

        target_embeddings = get_output(self.nn, {self.nn_in[0]: x_pre_padded,
                                                 self.nn_in[1]: z})  # N * max(L) * E

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

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)
        x_embedded_rep = T.tile(x_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_embedded_dropped_rep = T.tile(x_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        x_rep_padding_mask = T.ge(x_rep, 0)  # (S*N) * max(L)

        probs = self.get_probs(x_embedded_rep, x_embedded_dropped_rep, z, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

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

            nn_input = {self.nn_in[0]: x_pre_padded,
                        self.nn_in[1]: T.repeat(z, beam_size, 0)}

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

    def beam_search_samples(self, z, log_p_z, all_embeddings, beam_size, num_samples):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            active_paths_embedded = T.tile(active_paths_embedded, (num_samples, 1, 1))  # (S*N*B) *  max(L) * E

            x_pre_padded = T.concatenate([T.zeros((num_samples*N*beam_size, 1, self.embedding_dim)),
                                          active_paths_embedded], axis=1)[:, :-1]  # (S*N*B) * max(L) * E

            nn_input = {self.nn_in[0]: x_pre_padded,
                        self.nn_in[1]: T.repeat(z, beam_size, 0)}

            target_embeddings = get_output(self.nn, nn_input)[:, l].reshape((num_samples*N, beam_size,
                                                                             self.embedding_dim))  # (S*N) * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

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


class GenVNMT(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.rnn_x_0 = self.rnn_x_0_fn()
        self.nn_z = self.nn_z_fn()
        self.rnn_x_1_in, self.rnn_x_1 = self.rnn_x_1_fn()

    def rnn_x_0_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim))

        l_mask = InputLayer((None, self.max_length))

        l_forward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                   mask_input=l_mask)
        l_backward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                    mask_input=l_mask, backwards=True)

        return l_forward, l_backward

    def nn_z_fn(self):

        l_in = InputLayer((None, self.rnn_hid_units))

        l_out = DenseLayer(l_in, self.z_dim * 2, nonlinearity=linear)

        return l_out

    def rnn_x_1_fn(self):

        l_in_x_1 = InputLayer((None, self.max_length, self.embedding_dim + self.z_dim))

        l_in_h = InputLayer((None, self.max_length, self.rnn_hid_units))

        l_rnn = RNNSearchLayer(l_in_x_1, l_in_h, self.rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x_1, l_in_h), l_out

    def get_h(self, x_0, x_0_embedded):

        mask = T.ge(x_0, 0)  # N * max(L)

        h_forward = self.rnn_x_0[0].get_output_for([x_0_embedded, mask])  # N * max(L) * (dim(hid)/2)
        h_backward = self.rnn_x_0[1].get_output_for([x_0_embedded, mask])  # N * max(L) * (dim(hid)/2)

        h = T.concatenate([h_forward, h_backward], axis=-1)  # N * max(L) * dim(hid)

        return h

    def get_z_means_and_covs(self, h, x_0):

        mask = T.ge(x_0, 0)  # N * max(L)

        h_avg = T.sum(h, axis=1) / T.shape_padright(T.sum(mask, axis=1))  # N * dim(hid)

        means_and_covs = get_output(self.nn_z, h_avg)  # N * (2*dim(z))

        means = means_and_covs[:, :self.z_dim]  # N * dim(z)
        covs = elu_plus_one(means_and_covs[:, -self.z_dim:])  # N * dim(z)

        return means, covs

    def get_samples_z(self, h, x_0, num_samples):

        means, covs = self.get_z_means_and_covs(h, x_0)

        samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        return samples

    def log_p_z(self, h, z, x_0):

        S = T.cast(z.shape[0] / x_0.shape[0], 'int32')

        means, covs = self.get_z_means_and_covs(h, x_0)

        means = T.tile(means, (S, 1))
        covs = T.tile(covs, (S, 1))

        log_p_z = self.dist_z.log_density(z, [means, covs])  # (S*N)

        return log_p_z

    def kl(self, h, x_0, means_rec, covs_rec):

        means_gen, covs_gen = self.get_z_means_and_covs(h, x_0)

        kl = 0.5 * T.sum((covs_rec/covs_gen) + (((means_gen - means_rec)**2)/covs_gen) - T.ones_like(means_gen) +
                         T.log(covs_gen/covs_rec), axis=range(1, means_rec.ndim))

        return kl

    def get_target_embeddings(self, h, z, x_1_pre_padded):

        target_embeddings = get_output(self.rnn_x_1, {
            self.rnn_x_1_in[0]: T.cast(T.concatenate([x_1_pre_padded, T.tile(T.shape_padaxis(z, 1),
                                                                             (1, self.max_length, 1))],
                                                     axis=-1), 'float32'),
            self.rnn_x_1_in[1]: T.cast(h, 'float32'),
        })  # N * max(L) * E

        return target_embeddings

    def get_probs(self, h, x_1_embedded, x_1_embedded_dropped, z, all_embeddings, mode='all'):

        N = x_1_embedded.shape[0]

        x_1_pre_padded = T.concatenate([T.zeros((N, 1, self.embedding_dim)), x_1_embedded_dropped], axis=1)[:, :-1]
        # N * max(L) * E

        target_embeddings = self.get_target_embeddings(h, z, x_1_pre_padded)  # N * max(L) * E

        probs_numerators = T.sum(x_1_embedded * target_embeddings, axis=-1)  # N * max(L)

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

    def log_p_x_1(self, h, z, x_1, x_1_embedded, x_1_embedded_dropped, all_embeddings):

        N = x_1.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        h_rep = T.tile(h, (S, 1, 1))

        x_1_rep = T.tile(x_1, (S, 1))  # (S*N) * max(L)
        x_1_embedded_rep = T.tile(x_1_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_1_embedded_dropped_rep = T.tile(x_1_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        probs = self.get_probs(h_rep, x_1_embedded_rep, x_1_embedded_dropped_rep, z, all_embeddings, mode='true')
        # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        x_1_rep_padding_mask = T.ge(x_1_rep, 0)  # (S*N) * max(L)

        log_p_x_1 = T.sum(x_1_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x_1

    def beam_search(self, h, z, all_embeddings, beam_size):

        N = z.shape[0]

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        h_rep = T.repeat(h, beam_size, 0)  # (N*B) * max(L) * dim(hid)

        z_rep = T.repeat(z, beam_size, 0)  # (N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((N*beam_size, 1, self.embedding_dim)), active_paths_embedded],
                                           axis=1)[:, :-1]  # (N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings(h_rep, z_rep, x_1_pre_padded)  # (N*B) * max(L) * E
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

    def beam_search_samples(self, h, z, log_p_z, all_embeddings, beam_size, num_samples):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        h_rep = T.tile(T.repeat(h, beam_size, 0), (num_samples, 1, 1))  # (S*N*B) * max(L) * dim(hid)

        z_rep = T.repeat(z, beam_size, 0)  # (S*N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            active_paths_embedded = T.tile(active_paths_embedded, (num_samples, 1, 1))  # (S*N*B) *  max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((num_samples*N*beam_size, 1, self.embedding_dim)),
                                            active_paths_embedded], axis=1)[:, :-1]  # (S*N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings(h_rep, z_rep, x_1_pre_padded)  # (S*N*B) * max(L) * E
            target_embeddings = target_embeddings[:, l].reshape((num_samples*N, beam_size, self.embedding_dim))  # (S*N)
            # * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

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

        rnn_x_0_params = get_all_params(get_all_layers(self.rnn_x_0), trainable=True)
        nn_z_params = get_all_params(get_all_layers(self.nn_z), trainable=True)
        rnn_x_1_params = get_all_params(get_all_layers(self.rnn_x_1), trainable=True)

        return rnn_x_0_params + nn_z_params + rnn_x_1_params

    def get_param_values(self):

        rnn_x_0_params_vals = get_all_param_values(get_all_layers(self.rnn_x_0))
        nn_z_params_vals = get_all_param_values(get_all_layers(self.nn_z))
        rnn_x_1_params_vals = get_all_param_values(get_all_layers(self.rnn_x_1))

        return [rnn_x_0_params_vals, nn_z_params_vals, rnn_x_1_params_vals]

    def set_param_values(self, param_values):

        [rnn_x_0_params_vals, nn_z_params_vals, rnn_x_1_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn_x_0), rnn_x_0_params_vals)
        set_all_param_values(get_all_layers(self.nn_z), nn_z_params_vals)
        set_all_param_values(get_all_layers(self.rnn_x_1), rnn_x_1_params_vals)


# class GenVNMTNoSkip(GenVNMT):
#
#     def rnn_x_1_fn(self):
#
#         l_in_x_1 = InputLayer((None, self.max_length, self.embedding_dim))
#
#         l_in_h = InputLayer((None, self.max_length, self.z_dim))
#
#         l_rnn = RNNSearchLayer(l_in_x_1, l_in_h, self.rnn_hid_units, max_length=self.max_length,
#                                nonlinearity=self.rnn_hid_nonlinearity)
#
#         l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)
#
#         return (l_in_x_1, l_in_h), l_out
#
#     def get_z_means_and_covs(self, x_0, x_0_embedded):
#
#         N = x_0.shape[0]
#
#         h = self.get_h(x_0, x_0_embedded)  # N * max(L) * dim(hid)
#
#         means_and_covs = get_output(self.nn_z, h.reshape((N*self.max_length, self.rnn_hid_units)))  # (N*max(L)) *
#         # (2*dim(z))
#
#         means = means_and_covs[:, :self.z_dim].reshape((N, self.max_length, self.z_dim))  # N * max(L) * dim(z)
#         covs = elu_plus_one(means_and_covs[:, -self.z_dim:].reshape((N, self.max_length, self.z_dim)))  # N * max(L) *
#         # dim(z)
#
#         return means, covs
#
#     def get_target_embeddings(self, x_0, x_0_embedded, z, x_1_pre_padded):
#
#         target_embeddings = get_output(self.rnn_x_1, {
#             self.rnn_x_1_in[0]: T.cast(x_1_pre_padded, 'float32'),
#             self.rnn_x_1_in[1]: T.cast(z, 'float32'),
#         })  # N * max(L) * E
#
#         return target_embeddings


class GenVNMTMultiIndicator(object):

    def __init__(self, num_langs, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.num_langs = num_langs
        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.rnn_x_0 = self.rnn_x_0_fn()
        self.nn_z = self.nn_z_fn()
        self.rnn_x_1_in, self.rnn_x_1 = self.rnn_x_1_fn()

    def rnn_x_0_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim + self.num_langs))

        l_mask = InputLayer((None, self.max_length))

        l_forward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                   mask_input=l_mask)
        l_backward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                    mask_input=l_mask, backwards=True)

        return l_forward, l_backward

    def nn_z_fn(self):

        l_in = InputLayer((None, self.rnn_hid_units))

        l_out = DenseLayer(l_in, self.z_dim * 2, nonlinearity=linear)

        return l_out

    def rnn_x_1_fn(self):

        l_in_x_1 = InputLayer((None, self.max_length, self.embedding_dim + self.z_dim + self.num_langs))

        l_in_h = InputLayer((None, self.max_length, self.rnn_hid_units))

        l_rnn = RNNSearchLayer(l_in_x_1, l_in_h, self.rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x_1, l_in_h), l_out

    def get_h(self, l_0, x_0, x_0_embedded):

        l_0_one_hot = T.extra_ops.to_one_hot(l_0, self.num_langs)  # N * num_l
        l_0_rep = T.tile(T.shape_padaxis(l_0_one_hot, 1), (1, self.max_length, 1))  # N * max(L) * num_l

        mask = T.ge(x_0, 0)  # N * max(L)

        rnn_input = T.concatenate((x_0_embedded, l_0_rep), axis=-1)  # N * max(L) * (E + num_l)

        h_forward = self.rnn_x_0[0].get_output_for([rnn_input, mask])  # N * max(L) * (dim(hid)/2)
        h_backward = self.rnn_x_0[1].get_output_for([rnn_input, mask])  # N * max(L) * (dim(hid)/2)

        h = T.concatenate([h_forward, h_backward], axis=-1)  # N * max(L) * dim(hid)

        return h

    def get_z_means_and_covs(self, h, x_0):

        mask = T.ge(x_0, 0)  # N * max(L)

        h_avg = T.sum(h, axis=1) / T.shape_padright(T.sum(mask, axis=1))  # N * dim(hid)

        means_and_covs = get_output(self.nn_z, h_avg)  # N * (2*dim(z))

        means = means_and_covs[:, :self.z_dim]  # N * dim(z)
        covs = elu_plus_one(means_and_covs[:, -self.z_dim:])  # N * dim(z)

        return means, covs

    def get_samples_z(self, h, x_0, num_samples):

        means, covs = self.get_z_means_and_covs(h, x_0)

        samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        return samples

    def log_p_z(self, z, h, x_0):

        S = T.cast(z.shape[0] / x_0.shape[0], 'int32')

        means, covs = self.get_z_means_and_covs(h, x_0)

        means = T.tile(means, (S, 1))
        covs = T.tile(covs, (S, 1))

        log_p_z = self.dist_z.log_density(z, [means, covs])  # (S*N)

        return log_p_z

    def kl(self, h, x_0, means_rec, covs_rec):

        means_gen, covs_gen = self.get_z_means_and_covs(h, x_0)

        kl = 0.5 * T.sum((covs_rec/covs_gen) + (((means_gen - means_rec)**2)/covs_gen) - T.ones_like(means_gen) +
                         T.log(covs_gen/covs_rec), axis=range(1, means_rec.ndim))

        return kl

    def get_target_embeddings(self, l_1, h, z, x_1_pre_padded):

        N = x_1_pre_padded.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))  # N * max(L) * dim(z)

        l_1_one_hot = T.extra_ops.to_one_hot(l_1.reshape((1,)), self.num_langs)  # 1 * num_l
        l_1_rep = T.tile(T.shape_padleft(l_1_one_hot, 1), (N, self.max_length, 1))  # N * max(L) * num_l

        rnn_x_1_input = T.concatenate([x_1_pre_padded, z_rep, l_1_rep], axis=-1)  # N * max(L) * (E + dim(z) + num_l)

        target_embeddings = get_output(self.rnn_x_1, {
            self.rnn_x_1_in[0]: T.cast(rnn_x_1_input, 'float32'),
            self.rnn_x_1_in[1]: T.cast(h, 'float32'),
        })  # N * max(L) * E

        return target_embeddings

    def get_probs(self, l_1, h, x_1_embedded, x_1_embedded_dropped, z, all_embeddings, mode='all'):

        N = x_1_embedded.shape[0]

        x_1_pre_padded = T.concatenate([T.zeros((N, 1, self.embedding_dim)), x_1_embedded_dropped], axis=1)[:, :-1]
        # N * max(L) * E

        target_embeddings = self.get_target_embeddings(l_1, h, z, x_1_pre_padded)  # N * max(L) * E

        probs_numerators = T.sum(x_1_embedded * target_embeddings, axis=-1)  # N * max(L)

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

    def log_p_x_1(self, l_1, h, z, x_1, x_1_embedded, x_1_embedded_dropped, all_embeddings):

        N = x_1.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        h_rep = T.tile(h, (S, 1, 1))  # (S*N) * max(L) * dim(hid)

        x_1_rep = T.tile(x_1, (S, 1))  # (S*N) * max(L)
        x_1_embedded_rep = T.tile(x_1_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_1_embedded_dropped_rep = T.tile(x_1_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        probs = self.get_probs(l_1, h_rep, x_1_embedded_rep, x_1_embedded_dropped_rep, z, all_embeddings,
                               mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        x_1_rep_padding_mask = T.ge(x_1_rep, 0)  # (S*N) * max(L)

        log_p_x_1 = T.sum(x_1_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x_1

    def beam_search(self, l_0, l_1, x_0, x_0_embedded, z, all_embeddings, beam_size):

        N = z.shape[0]

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        h = self.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)
        h_rep = T.repeat(h, beam_size, 0)  # (N*B) * max(L) * dim(hid)

        z_rep = T.repeat(z, beam_size, 0)  # (N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((N*beam_size, 1, self.embedding_dim)), active_paths_embedded],
                                           axis=1)[:, :-1]  # (N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings(l_1, h_rep, z_rep, x_1_pre_padded)  # (N*B) * max(L)
            # * E
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

    def beam_search_samples(self, l_0, l_1, x_0, x_0_embedded, z, log_p_z, all_embeddings, beam_size, num_samples):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        h = self.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)
        h_rep = T.tile(T.repeat(h, beam_size, 0), (num_samples, 1, 1))  # (S*N*B) * max(L) * dim(hid)

        z_rep = T.tile(T.repeat(z, beam_size, 0), (num_samples, 1))  # (S*N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            active_paths_embedded = T.tile(active_paths_embedded, (num_samples, 1, 1))  # (S*N*B) *  max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((num_samples*N*beam_size, 1, self.embedding_dim)),
                                            active_paths_embedded], axis=1)[:, :-1]  # (S*N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings(l_1, h_rep, z_rep, x_1_pre_padded)  # (S*N*B) *
            # max(L) * E
            target_embeddings = target_embeddings[:, l].reshape((num_samples*N, beam_size, self.embedding_dim))  # (S*N)
            # * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

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

        rnn_x_0_params = get_all_params(get_all_layers(self.rnn_x_0), trainable=True)
        nn_z_params = get_all_params(get_all_layers(self.nn_z), trainable=True)
        rnn_x_1_params = get_all_params(get_all_layers(self.rnn_x_1), trainable=True)

        return rnn_x_0_params + nn_z_params + rnn_x_1_params

    def get_param_values(self):

        rnn_x_0_params_vals = get_all_param_values(get_all_layers(self.rnn_x_0))
        nn_z_params_vals = get_all_param_values(get_all_layers(self.nn_z))
        rnn_x_1_params_vals = get_all_param_values(get_all_layers(self.rnn_x_1))

        return [rnn_x_0_params_vals, nn_z_params_vals, rnn_x_1_params_vals]

    def set_param_values(self, param_values):

        [rnn_x_0_params_vals, nn_z_params_vals, rnn_x_1_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn_x_0), rnn_x_0_params_vals)
        set_all_param_values(get_all_layers(self.nn_z), nn_z_params_vals)
        set_all_param_values(get_all_layers(self.rnn_x_1), rnn_x_1_params_vals)


# class GenVNMTMultiIndicatorNoSkip(GenVNMTMultiIndicator):
#
#     def rnn_x_1_fn(self):
#
#         l_in_x_1 = InputLayer((None, self.max_length, self.embedding_dim + self.num_langs))
#
#         l_in_h = InputLayer((None, self.max_length, self.z_dim))
#
#         l_rnn = RNNSearchLayer(l_in_x_1, l_in_h, self.rnn_hid_units, max_length=self.max_length,
#                                nonlinearity=self.rnn_hid_nonlinearity)
#
#         l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)
#
#         return (l_in_x_1, l_in_h), l_out
#
#     def get_z_means_and_covs(self, h, x_0):
#
#         N = h.shape[0]
#
#         means_and_covs = get_output(self.nn_z, h.reshape((N*self.max_length, self.rnn_hid_units)))  # (N*max(L)) *
#         # (2*dim(z))
#
#         means = means_and_covs[:, :self.z_dim].reshape((N, self.max_length, self.z_dim))  # N * max(L) * dim(z)
#         covs = elu_plus_one(means_and_covs[:, -self.z_dim:].reshape((N, self.max_length, self.z_dim)))  # N * max(L) *
#         # dim(z)
#
#         return means, covs
#
#     def get_target_embeddings(self, l_1, h, z, x_1_pre_padded):
#
#         N = x_1_pre_padded.shape[0]
#
#         l_1_one_hot = T.extra_ops.to_one_hot(l_1.reshape((1,)), self.num_langs)  # 1 * num_l
#         l_1_rep = T.tile(T.shape_padleft(l_1_one_hot, 1), (N, self.max_length, 1))  # N * max(L) * num_l
#
#         rnn_x_1_input = T.concatenate([x_1_pre_padded, l_1_rep], axis=-1)  # N * max(L) * (E + dim(z) + num_l)
#
#         target_embeddings = get_output(self.rnn_x_1, {
#             self.rnn_x_1_in[0]: T.cast(rnn_x_1_input, 'float32'),
#             self.rnn_x_1_in[1]: T.cast(z, 'float32'),
#         })  # N * max(L) * E
#
#         return target_embeddings


class GenVNMTJoint(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.rnn_x_0 = self.rnn_x_0_fn()
        self.rnn_hid_x_1 = self.rnn_hid_x_1_fn()
        self.rnn_x_1_in, self.rnn_x_1 = self.rnn_x_1_fn()

    def rnn_x_0_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim + self.z_dim))

        l_rnn = LSTMLayer(l_in, self.rnn_hid_units, nonlinearity=self.rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, nonlinearity=linear, W_hid_to_hid=T.zeros)

        return l_out

    def rnn_hid_x_1_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim))

        l_mask = InputLayer((None, self.max_length))

        l_forward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                   mask_input=l_mask)
        l_backward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                    mask_input=l_mask, backwards=True)

        return l_forward, l_backward

    def rnn_x_1_fn(self):

        l_in_x_1 = InputLayer((None, self.max_length, self.embedding_dim + self.z_dim))

        l_in_h = InputLayer((None, self.max_length, self.rnn_hid_units))

        l_rnn = RNNSearchLayer(l_in_x_1, l_in_h, self.rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x_1, l_in_h), l_out

    def get_h(self, x_0, x_0_embedded):

        mask = T.ge(x_0, 0)  # N * max(L)

        h_forward = self.rnn_hid_x_1[0].get_output_for([x_0_embedded, mask])  # N * max(L) * (dim(hid)/2)
        h_backward = self.rnn_hid_x_1[1].get_output_for([x_0_embedded, mask])  # N * max(L) * (dim(hid)/2)

        h = T.concatenate([h_forward, h_backward], axis=-1)  # N * max(L) * dim(hid)

        return h

    def get_samples_z(self, num_samples):

        samples = self.dist_z.get_samples(num_samples, dims=(1, self.z_dim))  # S * dim(z)

        return samples

    def log_p_z(self, z):

        log_p_z = self.dist_z.log_density(z)  # S

        return log_p_z

    def kl(self, means_rec, covs_rec):

        kl = -0.5 * T.sum(-covs_rec - means_rec**2 + T.ones_like(means_rec) + T.log(covs_rec),
                          axis=range(1, means_rec.ndim))

        return kl

    def get_probs(self, x_embedded, target_embeddings, all_embeddings, mode='all'):

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

    def get_target_embeddings_x_0(self, z, x_0_pre_padded):

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))  # N * max(L) * dim(z)

        rnn_input = T.cast(T.concatenate((x_0_pre_padded, z_rep), axis=-1), 'float32')

        target_embeddings = get_output(self.rnn_x_0, rnn_input)  # N * max(L) * E

        return target_embeddings

    def log_p_x_0(self, z, x_0, x_0_embedded, x_0_embedded_dropped, all_embeddings):

        N = x_0.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        x_0_rep = T.tile(x_0, (S, 1))  # (S*N) * max(L)
        x_0_embedded_rep = T.tile(x_0_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_0_embedded_dropped_rep = T.tile(x_0_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        x_0_pre_padded_rep = T.concatenate([T.zeros((S*N, 1, self.embedding_dim)), x_0_embedded_dropped_rep],
                                           axis=1)[:, :-1]  # (S*N) * max(L) * E

        target_embeddings = self.get_target_embeddings_x_0(z, x_0_pre_padded_rep)  # (S*N) * max(L) * E

        probs = self.get_probs(x_0_embedded_rep, target_embeddings, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        x_0_rep_padding_mask = T.ge(x_0_rep, 0)  # (S*N) * max(L)

        log_p_x_0 = T.sum(x_0_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x_0

    def get_target_embeddings_x_1(self, x_0, x_0_embedded, z, x_1_pre_padded):

        h = self.get_h(x_0, x_0_embedded)  # N * max(L) * dim(hid)

        target_embeddings = get_output(self.rnn_x_1, {
            self.rnn_x_1_in[0]: T.cast(T.concatenate([x_1_pre_padded, T.tile(T.shape_padaxis(z, 1),
                                                                             (1, self.max_length, 1))],
                                                     axis=-1), 'float32'),
            self.rnn_x_1_in[1]: T.cast(h, 'float32'),
        })  # N * max(L) * E

        return target_embeddings

    def log_p_x_1(self, x_0, x_0_embedded, z, x_1, x_1_embedded, x_1_embedded_dropped, all_embeddings):

        N = x_1.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        x_0_rep = T.tile(x_0, (S, 1))  # (S*N) * max(L)
        x_0_embedded_rep = T.tile(x_0_embedded, (S, 1, 1))  # (S*N) * max(L) * E

        x_1_rep = T.tile(x_1, (S, 1))  # (S*N) * max(L)
        x_1_embedded_rep = T.tile(x_1_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_1_embedded_dropped_rep = T.tile(x_1_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        x_1_pre_padded_rep = T.concatenate([T.zeros((S*N, 1, self.embedding_dim)), x_1_embedded_dropped_rep],
                                           axis=1)[:, :-1]  # (S*N) * max(L) * E

        target_embeddings = self.get_target_embeddings_x_1(x_0_rep, x_0_embedded_rep, z, x_1_pre_padded_rep)  # (S*N) *
        # max(L) * E

        probs = self.get_probs(x_1_embedded_rep, target_embeddings, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        x_1_rep_padding_mask = T.ge(x_1_rep, 0)  # (S*N) * max(L)

        log_p_x_1 = T.sum(x_1_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x_1

    def beam_search(self, x_0, x_0_embedded, z, all_embeddings, beam_size):

        N = z.shape[0]

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        x_0_rep = T.repeat(x_0, beam_size, 0)  # (N*B) * max(L)
        x_0_embedded_rep = T.repeat(x_0_embedded, beam_size, 0)  # (N*B) * max(L) * E
        z_rep = T.repeat(z, beam_size, 0)  # (N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((N*beam_size, 1, self.embedding_dim)), active_paths_embedded],
                                           axis=1)[:, :-1]  # (N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings_x_1(x_0_rep, x_0_embedded_rep, z_rep, x_1_pre_padded)
            # (N*B) * max(L) * E
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

    def beam_search_samples(self, x_0, x_0_embedded, z, log_p_z, all_embeddings, beam_size, num_samples):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        x_0_rep = T.tile(T.repeat(x_0, beam_size, 0), (num_samples, 1))  # (S*N*B) * max(L)
        x_0_embedded_rep = T.tile(T.repeat(x_0_embedded, beam_size, 0), (num_samples, 1, 1))  # (S*N*B) * max(L) * E
        z_rep = T.repeat(z, beam_size, 0)  # (S*N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            active_paths_embedded = T.tile(active_paths_embedded, (num_samples, 1, 1))  # (S*N*B) *  max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((num_samples*N*beam_size, 1, self.embedding_dim)),
                                            active_paths_embedded], axis=1)[:, :-1]  # (S*N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings_x_1(x_0_rep, x_0_embedded_rep, z_rep, x_1_pre_padded)
            # (S*N*B) * max(L) * E
            target_embeddings = target_embeddings[:, l].reshape((num_samples*N, beam_size, self.embedding_dim))  # (S*N)
            # * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

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

        rnn_x_0_params = get_all_params(get_all_layers(self.rnn_x_0), trainable=True)
        rnn_hid_x_1_params = get_all_params(get_all_layers(self.rnn_hid_x_1), trainable=True)
        rnn_x_1_params = get_all_params(get_all_layers(self.rnn_x_1), trainable=True)

        return rnn_x_0_params + rnn_hid_x_1_params + rnn_x_1_params

    def get_param_values(self):

        rnn_x_0_params_vals = get_all_param_values(get_all_layers(self.rnn_x_0))
        rnn_hid_x_1_params_vals = get_all_param_values(get_all_layers(self.rnn_hid_x_1))
        rnn_x_1_params_vals = get_all_param_values(get_all_layers(self.rnn_x_1))

        return [rnn_x_0_params_vals, rnn_hid_x_1_params_vals, rnn_x_1_params_vals]

    def set_param_values(self, param_values):

        [rnn_x_0_params_vals, rnn_hid_x_1_params_vals, rnn_x_1_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn_x_0), rnn_x_0_params_vals)
        set_all_param_values(get_all_layers(self.rnn_hid_x_1), rnn_hid_x_1_params_vals)
        set_all_param_values(get_all_layers(self.rnn_x_1), rnn_x_1_params_vals)


class GenVNMTJointMultiIndicator(object):

    def __init__(self, num_langs, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.num_langs = num_langs
        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.rnn_x_0 = self.rnn_x_0_fn()
        self.rnn_hid_x_1 = self.rnn_hid_x_1_fn()
        self.rnn_x_1_in, self.rnn_x_1 = self.rnn_x_1_fn()

    def rnn_x_0_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim + self.z_dim + self.num_langs))

        l_rnn = LSTMLayer(l_in, self.rnn_hid_units, nonlinearity=self.rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, nonlinearity=linear, W_hid_to_hid=T.zeros)

        return l_out

    def rnn_hid_x_1_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim + self.num_langs))

        l_mask = InputLayer((None, self.max_length))

        l_forward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                   mask_input=l_mask)
        l_backward = LSTMLayer0Mask(l_in, int(self.rnn_hid_units/2), nonlinearity=self.rnn_hid_nonlinearity,
                                    mask_input=l_mask, backwards=True)

        return l_forward, l_backward

    def rnn_x_1_fn(self):

        l_in_x_1 = InputLayer((None, self.max_length, self.embedding_dim + self.z_dim + self.num_langs))

        l_in_h = InputLayer((None, self.max_length, self.rnn_hid_units))

        l_rnn = RNNSearchLayer(l_in_x_1, l_in_h, self.rnn_hid_units, max_length=self.max_length,
                               nonlinearity=self.rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_rnn, self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=linear)

        return (l_in_x_1, l_in_h), l_out

    def get_h(self, l_0, x_0, x_0_embedded):

        N = x_0.shape[0]

        l_0_one_hot = T.extra_ops.to_one_hot(l_0.reshape((1,)), self.num_langs)  # 1 * num_l
        l_0_rep = T.tile(T.shape_padleft(l_0_one_hot, 1), (N, self.max_length, 1))  # N * max(L) * num_l

        mask = T.ge(x_0, 0)  # N * max(L)

        rnn_input = T.concatenate((x_0_embedded, l_0_rep), axis=-1)  # N * max(L) * (E + num_l)

        h_forward = self.rnn_hid_x_1[0].get_output_for([rnn_input, mask])  # N * max(L) * (dim(hid)/2)
        h_backward = self.rnn_hid_x_1[1].get_output_for([rnn_input, mask])  # N * max(L) * (dim(hid)/2)

        h = T.concatenate([h_forward, h_backward], axis=-1)  # N * max(L) * dim(hid)

        return h

    def get_samples_z(self, num_samples):

        samples = self.dist_z.get_samples(num_samples, dims=(1, self.z_dim))  # S * dim(z)

        return samples

    def log_p_z(self, z):

        log_p_z = self.dist_z.log_density(z)  # S

        return log_p_z

    def kl(self, means_rec, covs_rec):

        kl = -0.5 * T.sum(-covs_rec - means_rec**2 + T.ones_like(means_rec) + T.log(covs_rec),
                          axis=range(1, means_rec.ndim))

        return kl

    def get_probs(self, x_embedded, target_embeddings, all_embeddings, mode='all'):

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

    def get_target_embeddings_x_0(self, z, l_0, x_0_pre_padded):

        N = z.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))  # N * max(L) * dim(z)

        l_0_one_hot = T.extra_ops.to_one_hot(l_0.reshape((1,)), self.num_langs)  # 1 * num_l
        l_0_rep = T.tile(T.shape_padleft(l_0_one_hot, 1), (N, self.max_length, 1))  # N * max(L) * num_l

        rnn_input = T.cast(T.concatenate((x_0_pre_padded, z_rep, l_0_rep), axis=-1), 'float32')

        target_embeddings = get_output(self.rnn_x_0, rnn_input)  # N * max(L) * E

        return target_embeddings

    def log_p_x_0(self, z, l_0, x_0, x_0_embedded, x_0_embedded_dropped, all_embeddings):

        N = x_0.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        x_0_rep = T.tile(x_0, (S, 1))  # (S*N) * max(L)
        x_0_embedded_rep = T.tile(x_0_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_0_embedded_dropped_rep = T.tile(x_0_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        x_0_pre_padded_rep = T.concatenate([T.zeros((S*N, 1, self.embedding_dim)), x_0_embedded_dropped_rep],
                                           axis=1)[:, :-1]  # (S*N) * max(L) * E

        target_embeddings = self.get_target_embeddings_x_0(z, l_0, x_0_pre_padded_rep)  # (S*N) * max(L) * E

        probs = self.get_probs(x_0_embedded_rep, target_embeddings, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        x_0_rep_padding_mask = T.ge(x_0_rep, 0)  # (S*N) * max(L)

        log_p_x_0 = T.sum(x_0_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x_0

    def get_target_embeddings_x_1(self, l_1, h, z, x_1_pre_padded):

        N = x_1_pre_padded.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))  # N * max(L) * dim(z)

        l_1_one_hot = T.extra_ops.to_one_hot(l_1.reshape((1,)), self.num_langs)  # 1 * num_l
        l_1_rep = T.tile(T.shape_padleft(l_1_one_hot, 1), (N, self.max_length, 1))  # N * max(L) * num_l

        rnn_x_1_input = T.concatenate([x_1_pre_padded, z_rep, l_1_rep], axis=-1)  # N * max(L) * (E + dim(z) + num_l)

        target_embeddings = get_output(self.rnn_x_1, {
            self.rnn_x_1_in[0]: T.cast(rnn_x_1_input, 'float32'),
            self.rnn_x_1_in[1]: T.cast(h, 'float32'),
        })  # N * max(L) * E

        return target_embeddings

    def log_p_x_1(self, l_1, h, z, x_1, x_1_embedded, x_1_embedded_dropped, all_embeddings):

        N = x_1.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        h_rep = T.tile(h, (S, 1, 1))

        x_1_rep = T.tile(x_1, (S, 1))  # (S*N) * max(L)
        x_1_embedded_rep = T.tile(x_1_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_1_embedded_dropped_rep = T.tile(x_1_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        x_1_pre_padded_rep = T.concatenate([T.zeros((S*N, 1, self.embedding_dim)), x_1_embedded_dropped_rep],
                                           axis=1)[:, :-1]  # (S*N) * max(L) * E

        target_embeddings = self.get_target_embeddings_x_1(l_1, h_rep, z, x_1_pre_padded_rep)  # (S*N) *
        # max(L) * E

        probs = self.get_probs(x_1_embedded_rep, target_embeddings, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-5, 'float32')  # (S*N) * max(L)

        x_1_rep_padding_mask = T.ge(x_1_rep, 0)  # (S*N) * max(L)

        log_p_x_1 = T.sum(x_1_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x_1

    def beam_search(self, l_0, l_1, x_0, x_0_embedded, z, all_embeddings, beam_size):

        N = z.shape[0]

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        h = self.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)
        h_rep = T.repeat(h, beam_size, 0)  # (N*B) * max(L) * dim(hid)

        z_rep = T.repeat(z, beam_size, 0)  # (N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((N*beam_size, 1, self.embedding_dim)), active_paths_embedded],
                                           axis=1)[:, :-1]  # (N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings_x_1(l_1, h_rep, z_rep, x_1_pre_padded)  # (N*B) * max(L)
            # * E
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

    def beam_search_samples(self, l_0, l_1, x_0, x_0_embedded, z, log_p_z, all_embeddings, beam_size, num_samples):

        N = T.cast(z.shape[0] / num_samples, 'int32')

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        h = self.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)
        h_rep = T.tile(T.repeat(h, beam_size, 0), (num_samples, 1, 1))  # (S*N*B) * max(L) * dim(hid)

        z_rep = T.tile(T.repeat(z, beam_size, 0), (num_samples, 1))  # (S*N*B) * dim(z)

        def step_forward(l, best_scores_lm1, active_paths_current, all_embeddings):

            active_paths_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_paths_embedded = active_paths_embedded.reshape((N * beam_size, self.max_length, self.embedding_dim))
            # (N*B) * max(L) * E

            active_paths_embedded = T.tile(active_paths_embedded, (num_samples, 1, 1))  # (S*N*B) *  max(L) * E

            x_1_pre_padded = T.concatenate([T.zeros((num_samples*N*beam_size, 1, self.embedding_dim)),
                                            active_paths_embedded], axis=1)[:, :-1]  # (S*N*B) * max(L) * E

            target_embeddings = self.get_target_embeddings_x_1(l_1, h_rep, z_rep, x_1_pre_padded)  # (S*N*B) *
            # max(L) * E
            target_embeddings = target_embeddings[:, l].reshape((num_samples*N, beam_size, self.embedding_dim))  # (S*N)
            # * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * B * D

            probs = last_d_softmax(probs_denominators)  # (S*N) * B * D

            scores = T.shape_padright(T.tile(best_scores_lm1, (num_samples, 1))) + T.log(probs) + \
                     ((1./self.max_length) * T.shape_padright(log_p_z, 2))  # (S*N) * B * D
            scores = T.mean(scores.reshape((num_samples, N, beam_size, self.vocab_size)), axis=0)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return T.cast(best_scores_l, 'float32'), active_paths_new

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

        rnn_x_0_params = get_all_params(get_all_layers(self.rnn_x_0), trainable=True)
        rnn_hid_x_1_params = get_all_params(get_all_layers(self.rnn_hid_x_1), trainable=True)
        rnn_x_1_params = get_all_params(get_all_layers(self.rnn_x_1), trainable=True)

        return rnn_x_0_params + rnn_hid_x_1_params + rnn_x_1_params

    def get_param_values(self):

        rnn_x_0_params_vals = get_all_param_values(get_all_layers(self.rnn_x_0))
        rnn_hid_x_1_params_vals = get_all_param_values(get_all_layers(self.rnn_hid_x_1))
        rnn_x_1_params_vals = get_all_param_values(get_all_layers(self.rnn_x_1))

        return [rnn_x_0_params_vals, rnn_hid_x_1_params_vals, rnn_x_1_params_vals]

    def set_param_values(self, param_values):

        [rnn_x_0_params_vals, rnn_hid_x_1_params_vals, rnn_x_1_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn_x_0), rnn_x_0_params_vals)
        set_all_param_values(get_all_layers(self.rnn_hid_x_1), rnn_hid_x_1_params_vals)
        set_all_param_values(get_all_layers(self.rnn_x_1), rnn_x_1_params_vals)
