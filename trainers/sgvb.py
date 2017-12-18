from itertools import combinations
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_all_params, get_output, InputLayer, DenseLayer
from lasagne.nonlinearities import linear
from nn.nonlinearities import elu_plus_one
from lasagne.updates import norm_constraint


def embedder(x, all_embeddings):

    all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

    return all_embeddings[x]


def cut_off(x, eos_ind):

    def step(x_l, x_lm1):

        x_l = T.switch(T.eq(x_lm1, eos_ind), -1, x_l)
        x_l = T.switch(T.eq(x_lm1, -1), -1, x_l)

        return T.cast(x_l, 'int32')

    x_cut_off, _ = theano.scan(step,
                               sequences=x.T,
                               outputs_info=T.zeros((x.shape[0],), 'int32'),
                               )

    return x_cut_off.T


class SGVBWords(object):

    def __init__(self, generative_model_0, generative_model_1, recognition_model, z_dim, max_length_0, max_length_1,
                 vocab_size_0, vocab_size_1, embedding_dim_0, embedding_dim_1, dist_z_gen, dist_x_0_gen, dist_x_1_gen,
                 dist_z_rec, gen_nn_0_kwargs, gen_nn_1_kwargs, rec_nn_kwargs, eos_ind_0, eos_ind_1):

        self.z_dim = z_dim
        self.max_length_0 = max_length_0
        self.max_length_1 = max_length_1
        self.vocab_size_0 = vocab_size_0
        self.vocab_size_1 = vocab_size_1
        self.embedding_dim_0 = embedding_dim_0
        self.embedding_dim_1 = embedding_dim_1

        self.all_embeddings_0 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size_0, embedding_dim_0))))
        self.all_embeddings_1 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size_1, embedding_dim_1))))

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_0_gen = dist_x_0_gen
        self.dist_x_1_gen = dist_x_1_gen

        self.gen_nn_0_kwargs = gen_nn_0_kwargs
        self.gen_nn_1_kwargs = gen_nn_1_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model_0, self.generative_model_1 = self.init_generative_models(generative_model_0,
                                                                                       generative_model_1)

        self.recognition_model = self.init_recognition_models(recognition_model)

        self.eos_ind_0 = eos_ind_0
        self.eos_ind_1 = eos_ind_1

    def init_generative_models(self, generative_model_0, generative_model_1):

        generative_model_0 = generative_model_0(self.z_dim, self.max_length_0, self.vocab_size_0, self.embedding_dim_0,
                                                embedder, self.dist_z_gen, self.dist_x_0_gen, self.gen_nn_0_kwargs)

        generative_model_1 = generative_model_1(self.z_dim, self.max_length_1, self.vocab_size_1, self.embedding_dim_1,
                                                embedder, self.dist_z_gen, self.dist_x_1_gen, self.gen_nn_1_kwargs)

        return generative_model_0, generative_model_1

    def init_recognition_models(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length_0, self.max_length_1, self.embedding_dim_0,
                                              self.embedding_dim_1, self.dist_z_rec, self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo(self, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian(x_0, x_0_embedded, x_1, x_1_embedded,
                                                                       num_samples)  # (S*N) * dim(z) and N

        if drop_mask_0 is None:
            x_0_embedded_dropped = x_0_embedded
        else:
            x_0_embedded_dropped = x_0_embedded * T.shape_padright(drop_mask_0)

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_0 = self.generative_model_0.log_p_x(x_0, x_0_embedded, x_0_embedded_dropped, z, self.all_embeddings_0)
        # (S*N)

        log_p_x_1 = self.generative_model_1.log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z, self.all_embeddings_1)
        # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(beta * kl)

        return elbo, T.sum(kl)

    def elbo_fn(self, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        beta = T.scalar('beta')

        drop_mask_0 = T.matrix('drop_mask_0')  # N * max(L)
        drop_mask_1 = T.matrix('drop_mask_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x_0, x_1, num_samples, beta, drop_mask_0, drop_mask_1)

        params = self.generative_model_0.get_params() + self.generative_model_1.get_params() + \
                 self.recognition_model.get_params() + [self.all_embeddings_0, self.all_embeddings_1]
        grads = T.grad(-elbo, params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x_0, x_1, beta, drop_mask_0, drop_mask_1],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples, beam_size, num_time_steps=None):

        z = self.generative_model_0.generate_z_prior(num_samples)  # S * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        return theano.function(inputs=[],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn(self, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)
        x_1_embedded = embedder(x_1, self.all_embeddings_1)

        z = self.recognition_model.get_samples(x_0, x_0_embedded, x_1, x_1_embedded, 1, means_only=True)  # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        return theano.function(inputs=[x_0, x_1],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def translate_fn(self, from_index, beam_size, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        if from_index == 0:
            all_embeddings_from = self.all_embeddings_0
            all_embeddings_to = self.all_embeddings_1
            eos_ind_to = self.eos_ind_1
        else:
            all_embeddings_from = self.all_embeddings_1
            all_embeddings_to = self.all_embeddings_0
            eos_ind_to = self.eos_ind_0

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        if from_index == 0:

            z = self.recognition_model.get_samples(x_from, x_from_embedded, x_to_best_guess, x_to_best_guess_embedded,
                                                   1, means_only=True)  # N * dim(z)

            x_to = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)

        else:

            z = self.recognition_model.get_samples(x_to_best_guess, x_to_best_guess_embedded, x_from, x_from_embedded,
                                                   1, means_only=True)  # N * dim(z)

            x_to = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)

        x_to = cut_off(x_to, eos_ind_to)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, from_index, beam_size, num_samples, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        if from_index == 0:
            all_embeddings_from = self.all_embeddings_0
            all_embeddings_to = self.all_embeddings_1
            eos_ind_to = self.eos_ind_1
        else:
            all_embeddings_from = self.all_embeddings_1
            all_embeddings_to = self.all_embeddings_0
            eos_ind_to = self.eos_ind_0

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        if from_index == 0:

            z = self.recognition_model.get_samples(x_from, x_from_embedded, x_to_best_guess, x_to_best_guess_embedded,
                                                   num_samples)  # (S*N) * dim(z)

            x_to = self.generative_model_1.beam_search_samples(z, self.all_embeddings_1, beam_size, num_samples,
                                                               num_time_steps)

        else:

            z = self.recognition_model.get_samples(x_to_best_guess, x_to_best_guess_embedded, x_from, x_from_embedded,
                                                   num_samples)  # (S*N) * dim(z)

            x_to = self.generative_model_0.beam_search_samples(z, self.all_embeddings_0, beam_size, num_samples,
                                                               num_time_steps)

        x_to = cut_off(x_to, eos_ind_to)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )


class SGVBWordsSSL(object):

    def __init__(self, generative_model_0, generative_model_1, recognition_model, z_dim, max_length_0, max_length_1,
                 vocab_size_0, vocab_size_1, embedding_dim_0, embedding_dim_1, dist_z_gen, dist_x_0_gen, dist_x_1_gen,
                 dist_z_rec, gen_nn_0_kwargs, gen_nn_1_kwargs, rec_nn_kwargs, eos_ind_0, eos_ind_1):

        self.z_dim = z_dim
        self.max_length_0 = max_length_0
        self.max_length_1 = max_length_1
        self.vocab_size_0 = vocab_size_0
        self.vocab_size_1 = vocab_size_1
        self.embedding_dim_0 = embedding_dim_0
        self.embedding_dim_1 = embedding_dim_1

        self.all_embeddings_0 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size_0, embedding_dim_0))))
        self.all_embeddings_1 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size_1, embedding_dim_1))))

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_0_gen = dist_x_0_gen
        self.dist_x_1_gen = dist_x_1_gen

        self.gen_nn_0_kwargs = gen_nn_0_kwargs
        self.gen_nn_1_kwargs = gen_nn_1_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model_0, self.generative_model_1 = self.init_generative_models(generative_model_0,
                                                                                       generative_model_1)

        self.recognition_model = self.init_recognition_models(recognition_model)

        self.eos_ind_0 = eos_ind_0
        self.eos_ind_1 = eos_ind_1

        self.params = self.generative_model_0.get_params() + self.generative_model_1.get_params() + \
            self.recognition_model.get_params() + [self.all_embeddings_0, self.all_embeddings_1]

    def init_generative_models(self, generative_model_0, generative_model_1):

        generative_model_0 = generative_model_0(self.z_dim, self.max_length_0, self.vocab_size_0, self.embedding_dim_0,
                                                embedder, self.dist_z_gen, self.dist_x_0_gen, self.gen_nn_0_kwargs)

        generative_model_1 = generative_model_1(self.z_dim, self.max_length_1, self.vocab_size_1, self.embedding_dim_1,
                                                embedder, self.dist_z_gen, self.dist_x_1_gen, self.gen_nn_1_kwargs)

        return generative_model_0, generative_model_1

    def init_recognition_models(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length_0, self.max_length_1, self.embedding_dim_0,
                                              self.embedding_dim_1, self.dist_z_rec, self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo_both(self, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_0, x_0_embedded, x_1, x_1_embedded,
                                                                            num_samples)  # (S*N) * dim(z) and N

        if drop_mask_0 is None:
            x_0_embedded_dropped = x_0_embedded
        else:
            x_0_embedded_dropped = x_0_embedded * T.shape_padright(drop_mask_0)

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_0 = self.generative_model_0.log_p_x(x_0, x_0_embedded, x_0_embedded_dropped, z, self.all_embeddings_0)
        # (S*N)

        log_p_x_1 = self.generative_model_1.log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z, self.all_embeddings_1)
        # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(beta * kl)

        return (1. / x_0.shape[0]) * elbo, T.mean(kl)

    def symbolic_elbo_0_only(self, x_0, num_samples, beta=None, drop_mask_0=None):

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian_0_only(x_0, x_0_embedded, num_samples)  # (S*N) *
        # dim(z) and N

        if drop_mask_0 is None:
            x_0_embedded_dropped = x_0_embedded
        else:
            x_0_embedded_dropped = x_0_embedded * T.shape_padright(drop_mask_0)

        log_p_x_0 = self.generative_model_0.log_p_x(x_0, x_0_embedded, x_0_embedded_dropped, z, self.all_embeddings_0)
        # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * log_p_x_0) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * log_p_x_0) - T.sum(beta * kl)

        return (1. / x_0.shape[0]) * elbo, T.mean(kl)

    def symbolic_elbo_1_only(self, x_1, num_samples, beta=None, drop_mask_1=None):

        x_1_embedded = embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian_1_only(x_1, x_1_embedded, num_samples)  # (S*N) *
        # dim(z) and N

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_1 = self.generative_model_1.log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z, self.all_embeddings_1)
        # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * log_p_x_1) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * log_p_x_1) - T.sum(beta * kl)

        return (1. / x_1.shape[0]) * elbo, T.mean(kl)

    def symbolic_elbo_all(self, x_0_only, x_1_only, x_0_both, x_1_both, num_samples, beta=None, drop_mask_0_only=None,
                          drop_mask_1_only=None, drop_mask_0_both=None, drop_mask_1_both=None):

        elbo_0_only, _ = self.symbolic_elbo_0_only(x_0_only, num_samples, beta, drop_mask_0_only)
        elbo_1_only, _ = self.symbolic_elbo_1_only(x_1_only, num_samples, beta, drop_mask_1_only)
        elbo_both, kl_both = self.symbolic_elbo_both(x_0_both, x_1_both, num_samples, beta, drop_mask_0_both,
                                                     drop_mask_1_both)

        return elbo_0_only + elbo_1_only + elbo_both, kl_both

    def elbo_fn_both(self, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo_both(x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def elbo_fn_0_only(self, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)

        elbo, kl = self.symbolic_elbo_0_only(x_0, num_samples)

        elbo_fn = theano.function(inputs=[x_0],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def elbo_fn_1_only(self, num_samples):

        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo_1_only(x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x_0_only = T.imatrix('x_0_only')  # N * max(L)
        x_1_only = T.imatrix('x_1_only')  # N * max(L)
        x_0_both = T.imatrix('x_0_both')  # N * max(L)
        x_1_both = T.imatrix('x_1_both')  # N * max(L)

        beta = T.scalar('beta')

        drop_mask_0_only = T.matrix('drop_mask_0_only')  # N * max(L)
        drop_mask_1_only = T.matrix('drop_mask_1_only')  # N * max(L)
        drop_mask_0_both = T.matrix('drop_mask_0_both')  # N * max(L)
        drop_mask_1_both = T.matrix('drop_mask_1_both')  # N * max(L)

        elbo, kl_both = self.symbolic_elbo_all(x_0_only, x_1_only, x_0_both, x_1_both, num_samples, beta,
                                               drop_mask_0_only, drop_mask_1_only, drop_mask_0_both, drop_mask_1_both)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x_0_only, x_1_only, x_0_both, x_1_both, beta, drop_mask_0_only,
                                            drop_mask_1_only, drop_mask_0_both, drop_mask_1_both],
                                    outputs=[elbo, kl_both],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples, beam_size, num_time_steps=None):

        z = self.generative_model_0.generate_z_prior(num_samples)  # S * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        return theano.function(inputs=[],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_both(self, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)
        x_1_embedded = embedder(x_1, self.all_embeddings_1)

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_0, x_0_embedded, x_1, x_1_embedded, 1,
                                                                           means_only=True)  # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        x_beam_0 = cut_off(x_beam_0, self.eos_ind_0)
        x_beam_1 = cut_off(x_beam_1, self.eos_ind_1)

        return theano.function(inputs=[x_0, x_1],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_0_only(self, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_0_only(x_0, x_0_embedded, 1, means_only=True)
        # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        x_beam_0 = cut_off(x_beam_0, self.eos_ind_0)
        x_beam_1 = cut_off(x_beam_1, self.eos_ind_1)

        return theano.function(inputs=[x_0],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_1_only(self, beam_size, num_time_steps=None):

        x_1 = T.imatrix('x_1')  # N * max(L)

        x_1_embedded = embedder(x_1, self.all_embeddings_1)

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_1_only(x_1, x_1_embedded, 1, means_only=True)
        # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        x_beam_0 = cut_off(x_beam_0, self.eos_ind_0)
        x_beam_1 = cut_off(x_beam_1, self.eos_ind_1)

        return theano.function(inputs=[x_1],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def translate_fn(self, from_index, beam_size, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        if from_index == 0:
            all_embeddings_from = self.all_embeddings_0
            all_embeddings_to = self.all_embeddings_1
            eos_ind_to = self.eos_ind_1
        else:
            all_embeddings_from = self.all_embeddings_1
            all_embeddings_to = self.all_embeddings_0
            eos_ind_to = self.eos_ind_0

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        if from_index == 0:

            z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_from, x_from_embedded, x_to_best_guess,
                                                                               x_to_best_guess_embedded, 1,
                                                                               means_only=True)  # N * dim(z)

            x_to = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)

        else:

            z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_to_best_guess,
                                                                               x_to_best_guess_embedded, x_from,
                                                                               x_from_embedded, 1, means_only=True)
            # N * dim(z)

            x_to = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)

        x_to = cut_off(x_to, eos_ind_to)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, from_index, beam_size, num_samples, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        if from_index == 0:
            all_embeddings_from = self.all_embeddings_0
            all_embeddings_to = self.all_embeddings_1
            eos_ind_to = self.eos_ind_1
        else:
            all_embeddings_from = self.all_embeddings_1
            all_embeddings_to = self.all_embeddings_0
            eos_ind_to = self.eos_ind_0

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        if from_index == 0:

            z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_from, x_from_embedded, x_to_best_guess,
                                                                               x_to_best_guess_embedded, num_samples)
            # (S*N) * dim(z)

            x_to = self.generative_model_1.beam_search_samples(z, self.all_embeddings_1, beam_size, num_samples,
                                                               num_time_steps)

        else:

            z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_to_best_guess, x_to_best_guess_embedded,
                                                                               x_from, x_from_embedded, num_samples)
            # (S*N) * dim(z)

            x_to = self.generative_model_0.beam_search_samples(z, self.all_embeddings_0, beam_size, num_samples,
                                                               num_time_steps)

        x_to = cut_off(x_to, eos_ind_to)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )


class SGVBWordsMulti(object):

    def __init__(self, num_langs, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim,
                 dist_z_gen, dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.num_langs = num_langs

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings = [theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
                               for l in range(num_langs)]

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_gen = dist_x_gen

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_models = self.init_generative_models(generative_model)
        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

        generative_model_params = [g.get_params() for g in self.generative_models]
        generative_model_params_flat = [p for params in generative_model_params for p in params]

        self.params = generative_model_params_flat + self.recognition_model.get_params() + self.all_embeddings

    def init_generative_models(self, generative_model):

        generative_models = []

        for l in range(self.num_langs):

            generative_models.append(generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                                      embedder, self.dist_z_gen, self.dist_x_gen,
                                                      self.gen_nn_kwargs))

        return generative_models

    def init_recognition_model(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec,
                                              self.num_langs, self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo_both(self, l_0, l_1, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings[l_0])  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian(x_0, x_0_embedded, x_1, x_1_embedded, l_0, l_1,
                                                                       num_samples)  # (S*N) * dim(z) and N

        if drop_mask_0 is None:
            x_0_embedded_dropped = x_0_embedded
        else:
            x_0_embedded_dropped = x_0_embedded * T.shape_padright(drop_mask_0)

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_0 = self.generative_models[l_0].log_p_x(x_0, x_0_embedded, x_0_embedded_dropped, z,
                                                        self.all_embeddings[l_0])  # (S*N)

        log_p_x_1 = self.generative_models[l_1].log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z,
                                                        self.all_embeddings[l_1])  # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(beta * kl)

        return (1. / x_0.shape[0]) * elbo, T.mean(kl)

    def symbolic_elbo_all(self, xs_both, num_samples, beta=None, drop_masks_both=None):

        if drop_masks_both is None:
            drop_masks_both = {l: [None, None] for l in combinations(range(self.num_langs), 2)}

        elbos_both = []
        kls_both = []

        for pair in combinations(range(self.num_langs), 2):
            l_0 = pair[0]
            l_1 = pair[1]

            elbo_both, kl_both = self.symbolic_elbo_both(l_0, l_1, xs_both[(l_0, l_1)][0], xs_both[(l_0, l_1)][1],
                                                         num_samples, beta, drop_masks_both[(l_0, l_1)][0],
                                                         drop_masks_both[(l_0, l_1)][1])

            elbos_both.append(elbo_both)
            kls_both.append(kl_both)

        return T.sum(elbos_both), T.sum(kls_both)

    def elbo_fn(self, l_0, l_1, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo_both(l_0, l_1, x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        xs_both = {l: [T.imatrix('x_' + str(l[0]) + '_both'), T.imatrix('x_' + str(l[1]) + '_both')]
                   for l in combinations(range(self.num_langs), 2)}

        beta = T.scalar('beta')

        drop_masks_both = {l: [T.matrix('drop_mask_' + str(l[0]) + '_both'), T.matrix('drop_mask_' + str(l[1]) +
                                                                                      '_both')]
                           for l in combinations(range(self.num_langs), 2)}

        elbo, kls_both = self.symbolic_elbo_all(xs_both, num_samples, beta, drop_masks_both)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        xs_both_flat = [item for sublist in xs_both.values() for item in sublist]
        drop_masks_both_flat = [item for sublist in drop_masks_both.values() for item in sublist]

        inputs = xs_both_flat + [beta] + drop_masks_both_flat

        optimiser = theano.function(inputs=inputs,
                                    outputs=[elbo, kls_both],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_outputs_z(self, z, beam_size, num_time_steps):

        outputs = []

        for l in range(self.num_langs):

            x_beam_l = self.generative_models[l].beam_search(z, self.all_embeddings[l], beam_size, num_time_steps)  # S
            # * max(L)

            x_beam_l = cut_off(x_beam_l, self.eos_ind)

            outputs.append(x_beam_l)

        return outputs

    def generate_output_prior_fn(self, num_samples, beam_size, num_time_steps=None):

        z = self.generative_models[0].generate_z_prior(num_samples)  # S * dim(z)

        outputs = self.generate_outputs_z(z, beam_size, num_time_steps)

        return theano.function(inputs=[],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_both(self, l_0, l_1, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings[l_0])
        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian(x_0, x_0_embedded, x_1, x_1_embedded, l_0, l_1, 1,
                                                                      means_only=True)  # N * dim(z)

        outputs = self.generate_outputs_z(z, beam_size, num_time_steps)

        return theano.function(inputs=[x_0, x_1],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def translate_fn(self, l_from, l_to, beam_size, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        all_embeddings_from = self.all_embeddings[l_from]
        all_embeddings_to = self.all_embeddings[l_to]

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian(x_from, x_from_embedded, x_to_best_guess,
                                                                      x_to_best_guess_embedded, l_from, l_to, 1,
                                                                      means_only=True)  # N * dim(z)

        x_to = self.generative_models[l_to].beam_search(z, self.all_embeddings[l_to], beam_size, num_time_steps)

        x_to = cut_off(x_to, self.eos_ind)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, l_from, l_to, beam_size, num_samples, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        all_embeddings_from = self.all_embeddings[l_from]
        all_embeddings_to = self.all_embeddings[l_to]

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian(x_from, x_from_embedded, x_to_best_guess,
                                                                      x_to_best_guess_embedded, l_from, l_to,
                                                                      num_samples)  # (S*N) * dim(z)

        x_to = self.generative_models[l_to].beam_search_samples(z, self.all_embeddings[l_to], beam_size, num_samples,
                                                                num_time_steps)

        x_to = cut_off(x_to, self.eos_ind)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def set_embedding_vals(self, embedding_values):

        for l in range(self.num_langs):

            self.all_embeddings[l].set_value(embedding_values[l])

    def set_generative_models_params_vals(self, generative_models_params_vals):

        for l in range(self.num_langs):

            self.generative_models[l].set_param_values(generative_models_params_vals[l])


class SGVBWordsSSLMulti(object):

    def __init__(self, num_langs, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim,
                 dist_z_gen, dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.num_langs = num_langs

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings = [theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
                               for l in range(num_langs)]

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_gen = dist_x_gen

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_models = self.init_generative_models(generative_model)
        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

        generative_model_params = [g.get_params() for g in self.generative_models]
        generative_model_params_flat = [p for params in generative_model_params for p in params]

        self.params = generative_model_params_flat + self.recognition_model.get_params() + self.all_embeddings

    def init_generative_models(self, generative_model):

        generative_models = []

        for l in range(self.num_langs):

            generative_models.append(generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                                      embedder, self.dist_z_gen, self.dist_x_gen,
                                                      self.gen_nn_kwargs))

        return generative_models

    def init_recognition_model(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec,
                                              self.num_langs, self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo_only(self, l, x, num_samples, beta=None, drop_mask=None):

        x_embedded = embedder(x, self.all_embeddings[l])  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian_only(x, x_embedded, l, num_samples)  # (S*N) *
        # dim(z) and N

        if drop_mask is None:
            x_embedded_dropped = x_embedded
        else:
            x_embedded_dropped = x_embedded * T.shape_padright(drop_mask)

        log_p_x = self.generative_models[l].log_p_x(x, x_embedded, x_embedded_dropped, z, self.all_embeddings[l])
        # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * log_p_x) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * log_p_x) - T.sum(beta * kl)

        return (1. / x.shape[0]) * elbo, T.mean(kl)

    def symbolic_elbo_both(self, l_0, l_1, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings[l_0])  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_0, x_0_embedded, x_1, x_1_embedded, l_0,
                                                                            l_1, num_samples)  # (S*N) * dim(z) and N

        if drop_mask_0 is None:
            x_0_embedded_dropped = x_0_embedded
        else:
            x_0_embedded_dropped = x_0_embedded * T.shape_padright(drop_mask_0)

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_0 = self.generative_models[l_0].log_p_x(x_0, x_0_embedded, x_0_embedded_dropped, z,
                                                        self.all_embeddings[l_0])  # (S*N)

        log_p_x_1 = self.generative_models[l_1].log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z,
                                                        self.all_embeddings[l_1])  # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * (log_p_x_0 + log_p_x_1)) - T.sum(beta * kl)

        return (1. / x_0.shape[0]) * elbo, T.mean(kl)

    def symbolic_elbo_all(self, xs_only, xs_both, num_samples, beta=None, drop_masks_only=None, drop_masks_both=None):

        if drop_masks_only is None:
            drop_masks_only = {l: None for l in range(self.num_langs)}

        if drop_masks_both is None:
            drop_masks_both = {l: [None, None] for l in combinations(range(self.num_langs), 2)}

        elbos_only = []

        for l in range(self.num_langs):
            elbo_l_only, _ = self.symbolic_elbo_only(l, xs_only[l], num_samples, beta, drop_masks_only[l])
            elbos_only.append(elbo_l_only)

        elbos_both = []
        kls_both = []

        for pair in combinations(range(self.num_langs), 2):
            l_0 = pair[0]
            l_1 = pair[1]

            elbo_both, kl_both = self.symbolic_elbo_both(l_0, l_1, xs_both[(l_0, l_1)][0], xs_both[(l_0, l_1)][1],
                                                         num_samples, beta, drop_masks_both[(l_0, l_1)][0],
                                                         drop_masks_both[(l_0, l_1)][1])

            elbos_both.append(elbo_both)
            kls_both.append(kl_both)

        return T.sum(elbos_only) + T.sum(elbos_both), T.sum(kls_both)

    def elbo_fn(self, l_0, l_1, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo_both(l_0, l_1, x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        xs_only = {l: T.imatrix('x_' + str(l) + '_only') for l in range(self.num_langs)}
        xs_both = {l: [T.imatrix('x_' + str(l[0]) + '_both'), T.imatrix('x_' + str(l[1]) + '_both')]
                   for l in combinations(range(self.num_langs), 2)}

        beta = T.scalar('beta')

        drop_masks_only = {l: T.matrix('drop_mask_' + str(l) + '_only') for l in range(self.num_langs)}
        drop_masks_both = {l: [T.matrix('drop_mask_' + str(l[0]) + '_both'), T.matrix('drop_mask_' + str(l[1]) +
                                                                                      '_both')]
                           for l in combinations(range(self.num_langs), 2)}

        elbo, kls_both = self.symbolic_elbo_all(xs_only, xs_both, num_samples, beta, drop_masks_only, drop_masks_both)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        xs_both_flat = [item for sublist in xs_both.values() for item in sublist]
        drop_masks_both_flat = [item for sublist in drop_masks_both.values() for item in sublist]

        inputs = list(xs_only.values()) + xs_both_flat + [beta] + list(drop_masks_only.values()) + drop_masks_both_flat

        optimiser = theano.function(inputs=inputs,
                                    outputs=[elbo, kls_both],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_outputs_z(self, z, beam_size, num_time_steps):

        outputs = []

        for l in range(self.num_langs):

            x_beam_l = self.generative_models[l].beam_search(z, self.all_embeddings[l], beam_size, num_time_steps)  # S
            # * max(L)

            x_beam_l = cut_off(x_beam_l, self.eos_ind)

            outputs.append(x_beam_l)

        return outputs

    def generate_output_prior_fn(self, num_samples, beam_size, num_time_steps=None):

        z = self.generative_models[0].generate_z_prior(num_samples)  # S * dim(z)

        outputs = self.generate_outputs_z(z, beam_size, num_time_steps)

        return theano.function(inputs=[],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_only(self, l, beam_size, num_time_steps=None):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = embedder(x, self.all_embeddings[l])

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_only(x, x_embedded, l, 1, means_only=True)
        # N * dim(z)

        outputs = self.generate_outputs_z(z, beam_size, num_time_steps)

        return theano.function(inputs=[x],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_both(self, l_0, l_1, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings[l_0])
        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_0, x_0_embedded, x_1, x_1_embedded, l_0,
                                                                           l_1, 1, means_only=True)  # N * dim(z)

        outputs = self.generate_outputs_z(z, beam_size, num_time_steps)

        return theano.function(inputs=[x_0, x_1],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def translate_fn(self, l_from, l_to, beam_size, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        all_embeddings_from = self.all_embeddings[l_from]
        all_embeddings_to = self.all_embeddings[l_to]

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_from, x_from_embedded, x_to_best_guess,
                                                                           x_to_best_guess_embedded, l_from, l_to, 1,
                                                                           means_only=True)  # N * dim(z)

        x_to = self.generative_models[l_to].beam_search(z, self.all_embeddings[l_to], beam_size, num_time_steps)

        x_to = cut_off(x_to, self.eos_ind)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, l_from, l_to, beam_size, num_samples, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        all_embeddings_from = self.all_embeddings[l_from]
        all_embeddings_to = self.all_embeddings[l_to]

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_from, x_from_embedded, x_to_best_guess,
                                                                           x_to_best_guess_embedded, l_from, l_to,
                                                                           num_samples)  # (S*N) * dim(z)

        x_to = self.generative_models[l_to].beam_search_samples(z, self.all_embeddings[l_to], beam_size, num_samples,
                                                                num_time_steps)

        x_to = cut_off(x_to, self.eos_ind)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def set_embedding_vals(self, embedding_values):

        for l in range(self.num_langs):

            self.all_embeddings[l].set_value(embedding_values[l])

    def set_generative_models_params_vals(self, generative_models_params_vals):

        for l in range(self.num_langs):

            self.generative_models[l].set_param_values(generative_models_params_vals[l])


class SGVBWordsSSLMultiRecurrentZ(object):

    def __init__(self, num_langs, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim,
                 dist_z_gen, dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.num_langs = num_langs

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings = [theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
                               for l in range(num_langs)]

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_gen = dist_x_gen

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_models = self.init_generative_models(generative_model)
        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

        self.z_nn_forward = self.z_nn_fn()
        self.z_nn_backward = self.z_nn_fn()

        z_nn_forward_params = get_all_params(self.z_nn_forward, trainable=True)
        z_nn_backward_params = get_all_params(self.z_nn_backward, trainable=True)

        generative_model_params = [g.get_params() for g in self.generative_models]
        generative_model_params_flat = [p for params in generative_model_params for p in params]

        self.params = z_nn_forward_params + z_nn_backward_params + generative_model_params_flat + \
            self.recognition_model.get_params() + self.all_embeddings

    def init_generative_models(self, generative_model):

        generative_models = []

        for l in range(self.num_langs):

            generative_models.append(generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                                      embedder, self.dist_z_gen, self.dist_x_gen,
                                                      self.gen_nn_kwargs))

        return generative_models

    def init_recognition_model(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec,
                                              self.num_langs, self.rec_nn_kwargs)

        return recognition_model

    def z_nn_fn(self):

        l_in = InputLayer((None, int(self.z_dim/2)))

        l_means_and_covs = DenseLayer(l_in, self.z_dim, nonlinearity=linear, b=T.zeros)

        return l_means_and_covs

    def get_means_and_covs_gen(self, z_forward, z_backward):

        SN = z_forward.shape[0]
        L = z_forward.shape[1]

        z_forward_in = T.concatenate((T.zeros((SN, 1, int(self.z_dim/2))), z_forward), axis=1)[:, :-1]
        z_backward_in = T.concatenate((T.zeros((SN, 1, int(self.z_dim/2))), z_backward[:, ::-1]), axis=1)[:, :-1]

        means_and_covs_gen_forward = get_output(self.z_nn_forward, z_forward_in.reshape((SN * L, int(self.z_dim/2))))
        means_and_covs_gen_forward = means_and_covs_gen_forward.reshape((SN, L, self.z_dim))  # (S*N) * max(L) * dim(z)

        means_and_covs_gen_backward = get_output(self.z_nn_backward, z_backward_in.reshape((SN * L, int(self.z_dim/2))))
        means_and_covs_gen_backward = means_and_covs_gen_backward.reshape((SN, L, self.z_dim))  # (S*N) * max(L) *
        # dim(z)
        means_and_covs_gen_backward = means_and_covs_gen_backward[:, ::-1]

        means_gen_forward = means_and_covs_gen_forward[:, :, :int(self.z_dim/2)]
        covs_gen_forward = means_and_covs_gen_forward[:, :, -int(self.z_dim/2):]

        means_gen_backward = means_and_covs_gen_backward[:, :, :int(self.z_dim/2)]
        covs_gen_backward = means_and_covs_gen_backward[:, :, -int(self.z_dim/2):]

        means_gen = T.concatenate((means_gen_forward, means_gen_backward), axis=-1)  # (S*N) * max(L) * dim(z)
        covs_gen = T.concatenate((covs_gen_forward, covs_gen_backward), axis=-1)  # (S*N) * max(L) * dim(z)
        covs_gen = elu_plus_one(covs_gen)

        return means_gen, covs_gen

    def log_p_z(self, z_forward, z_backward):

        means_gen, covs_gen = self.get_means_and_covs_gen(z_forward, z_backward)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # (S*N) * max(L) * dim(z)

        log_p_z = self.dist_z_gen().log_density(z, (means_gen, covs_gen))  # (S*N)

        return log_p_z

    def kl(self, z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward):

        means_rec = T.concatenate((means_rec_forward, means_rec_backward), axis=-1)  # (S*N) * max(L) * dim(z)
        covs_rec = T.concatenate((covs_rec_forward, covs_rec_backward), axis=-1)  # (S*N) * max(L) * dim(z)

        means_gen, covs_gen = self.get_means_and_covs_gen(z_forward, z_backward)

        kl = 0.5 * T.sum((covs_rec/covs_gen) + (((means_gen - means_rec)**2)/covs_gen) - T.ones_like(means_gen) +
                         T.log(covs_gen/covs_rec), axis=range(1, means_rec.ndim))

        return kl

    def symbolic_elbo_only(self, l, x, num_samples, beta=None, drop_mask=None):

        x_embedded = embedder(x, self.all_embeddings[l])  # N * max(L) * E

        z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward = \
            self.recognition_model.get_samples_only(x, x_embedded, l, num_samples)

        kl = self.kl(z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # (S*N) * max(L) * dim(z)

        if drop_mask is None:
            x_embedded_dropped = x_embedded
        else:
            x_embedded_dropped = x_embedded * T.shape_padright(drop_mask)

        log_p_x = self.generative_models[l].log_p_x(x, x_embedded, x_embedded_dropped, z, self.all_embeddings[l])
        # (S*N)

        if beta is None:
            elbo = log_p_x - kl
        else:
            elbo = log_p_x - (beta * kl)

        return T.mean(elbo), T.mean(kl)

    def symbolic_elbo_both(self, l_0, l_1, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings[l_0])  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])  # N * max(L) * E

        z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward = \
            self.recognition_model.get_samples_both(x_0, x_0_embedded, x_1, x_1_embedded, l_0, l_1, num_samples)

        kl = self.kl(z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # (S*N) * max(L) * dim(z)

        if drop_mask_0 is None:
            x_0_embedded_dropped = x_0_embedded
        else:
            x_0_embedded_dropped = x_0_embedded * T.shape_padright(drop_mask_0)

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_0 = self.generative_models[l_0].log_p_x(x_0, x_0_embedded, x_0_embedded_dropped, z,
                                                        self.all_embeddings[l_0])  # (S*N)

        log_p_x_1 = self.generative_models[l_1].log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z,
                                                        self.all_embeddings[l_1])  # (S*N)

        if beta is None:
            elbo = log_p_x_0 + log_p_x_1 - kl
        else:
            elbo = log_p_x_0 + log_p_x_1 - (beta * kl)

        return T.mean(elbo), T.mean(kl)

    def symbolic_elbo_all(self, xs_only, xs_both, num_samples, beta=None, drop_masks_only=None, drop_masks_both=None):

        if drop_masks_only is None:
            drop_masks_only = {l: None for l in range(self.num_langs)}

        if drop_masks_both is None:
            drop_masks_both = {l: [None, None] for l in combinations(range(self.num_langs), 2)}

        elbos_only = []

        for l in range(self.num_langs):
            elbo_l_only, _ = self.symbolic_elbo_only(l, xs_only[l], num_samples, beta, drop_masks_only[l])
            elbos_only.append(elbo_l_only)

        elbos_both = []
        kls_both = []

        for pair in combinations(range(self.num_langs), 2):
            l_0 = pair[0]
            l_1 = pair[1]

            elbo_both, kl_both = self.symbolic_elbo_both(l_0, l_1, xs_both[(l_0, l_1)][0], xs_both[(l_0, l_1)][1],
                                                         num_samples, beta, drop_masks_both[(l_0, l_1)][0],
                                                         drop_masks_both[(l_0, l_1)][1])

            elbos_both.append(elbo_both)
            kls_both.append(kl_both)

        return T.sum(elbos_only) + T.sum(elbos_both), T.sum(kls_both)

    def elbo_fn(self, l_0, l_1, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo_both(l_0, l_1, x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        xs_only = {l: T.imatrix('x_' + str(l) + '_only') for l in range(self.num_langs)}
        xs_both = {l: [T.imatrix('x_' + str(l[0]) + '_both'), T.imatrix('x_' + str(l[1]) + '_both')]
                   for l in combinations(range(self.num_langs), 2)}

        beta = T.scalar('beta')

        drop_masks_only = {l: T.matrix('drop_mask_' + str(l) + '_only') for l in range(self.num_langs)}
        drop_masks_both = {l: [T.matrix('drop_mask_' + str(l[0]) + '_both'), T.matrix('drop_mask_' + str(l[1]) +
                                                                                      '_both')]
                           for l in combinations(range(self.num_langs), 2)}

        elbo, kls_both = self.symbolic_elbo_all(xs_only, xs_both, num_samples, beta, drop_masks_only, drop_masks_both)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        xs_both_flat = [item for sublist in xs_both.values() for item in sublist]
        drop_masks_both_flat = [item for sublist in drop_masks_both.values() for item in sublist]

        inputs = list(xs_only.values()) + xs_both_flat + [beta] + list(drop_masks_only.values()) + drop_masks_both_flat

        optimiser = theano.function(inputs=inputs,
                                    outputs=[elbo, kls_both],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_outputs_z(self, z, beam_size):

        outputs = []

        for l in range(self.num_langs):

            x_beam_l = self.generative_models[l].beam_search(z, self.all_embeddings[l], beam_size)  # S
            # * max(L)

            x_beam_l = cut_off(x_beam_l, self.eos_ind)

            outputs.append(x_beam_l)

        return outputs

    def generate_output_posterior_fn_only(self, l, beam_size, num_time_steps=None):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = embedder(x, self.all_embeddings[l])

        z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward = \
            self.recognition_model.get_samples_only(x, x_embedded, l, 1, means_only=True)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # N * max(L) * dim(z)

        outputs = self.generate_outputs_z(z, beam_size)

        return theano.function(inputs=[x],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_both(self, l_0, l_1, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings[l_0])
        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])

        z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward = \
            self.recognition_model.get_samples_both(x_0, x_0_embedded, x_1, x_1_embedded, l_0, l_1, 1, means_only=True)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # N * max(L) * dim(z)

        outputs = self.generate_outputs_z(z, beam_size)

        return theano.function(inputs=[x_0, x_1],
                               outputs=outputs,
                               allow_input_downcast=True,
                               )

    def translate_fn(self, l_from, l_to, beam_size, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        all_embeddings_from = self.all_embeddings[l_from]
        all_embeddings_to = self.all_embeddings[l_to]

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward = \
            self.recognition_model.get_samples_both(x_from, x_from_embedded, x_to_best_guess, x_to_best_guess_embedded,
                                                    l_from, l_to, 1, means_only=True)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # N * max(L) * dim(z)

        x_to = self.generative_models[l_to].beam_search(z, self.all_embeddings[l_to], beam_size)

        x_to = cut_off(x_to, self.eos_ind)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, l_from, l_to, beam_size, num_samples, num_time_steps=None):

        x_from = T.imatrix('x_from')  # N * max(L)
        x_to_best_guess = T.imatrix('x_to_best_guess')  # N * max(L)

        all_embeddings_from = self.all_embeddings[l_from]
        all_embeddings_to = self.all_embeddings[l_to]

        x_from_embedded = embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        z_forward, z_backward, means_rec_forward, covs_rec_forward, means_rec_backward, covs_rec_backward = \
            self.recognition_model.get_samples_both(x_from, x_from_embedded, x_to_best_guess, x_to_best_guess_embedded,
                                                    l_from, l_to, num_samples)

        z = T.concatenate((z_forward, z_backward), axis=-1)  # (S*N) * max(L) * dim(z)

        log_p_z = self.log_p_z(z_forward, z_backward)

        x_to = self.generative_models[l_to].beam_search_samples(z, log_p_z, self.all_embeddings[l_to], beam_size,
                                                                num_samples)

        x_to = cut_off(x_to, self.eos_ind)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )

    def set_embedding_vals(self, embedding_values):

        for l in range(self.num_langs):

            self.all_embeddings[l].set_value(embedding_values[l])

    def set_generative_models_params_vals(self, generative_models_params_vals):

        for l in range(self.num_langs):

            self.generative_models[l].set_param_values(generative_models_params_vals[l])


class SGVBWordsVNMT(object):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, dist_z_gen,
                 dist_x_1_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings_0 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
        self.all_embeddings_1 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_1_gen = dist_x_1_gen

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model = self.init_generative_model(generative_model)

        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

        self.params = self.generative_model.get_params() + self.recognition_model.get_params() + \
            [self.all_embeddings_0, self.all_embeddings_1]

    def init_generative_model(self, generative_model):

        generative_model = generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                            embedder, self.dist_z_gen, self.dist_x_1_gen, self.gen_nn_kwargs)

        return generative_model

    def init_recognition_model(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec, 
                                              self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo(self, x_0, x_1, num_samples, beta=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

        z, means_rec, covs_rec = self.recognition_model.get_samples_and_means_and_covs(x_0, x_0_embedded, x_1,
                                                                                       x_1_embedded, num_samples)

        kl = self.generative_model.kl(x_0, x_0_embedded, means_rec, covs_rec)  # N

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_1 = self.generative_model.log_p_x_1(x_0, x_0_embedded, z, x_1, x_1_embedded, x_1_embedded_dropped,
                                                    self.all_embeddings_1)  # (S*N)

        if beta is None:
            elbo = T.mean(log_p_x_1) - T.mean(kl)
        else:
            elbo = T.mean(log_p_x_1) - T.mean(beta * kl)

        return elbo, T.mean(kl)

    def elbo_fn(self, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        beta = T.scalar('beta')

        drop_mask_1 = T.matrix('drop_mask_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x_0, x_1, num_samples, beta, drop_mask_1)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x_0, x_1, beta, drop_mask_1],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def translate_fn(self, beam_size):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

        z, _ = self.generative_model.get_z_means_and_covs(x_0, x_0_embedded)  # N * dim(z)

        x_1 = self.generative_model.beam_search(x_0, x_0_embedded, z, self.all_embeddings_1, beam_size)  # N * max(L)

        x_1 = cut_off(x_1, self.eos_ind)  # N * max(L)

        return theano.function(inputs=[x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, beam_size, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

        z = self.generative_model.get_samples_z(x_0, x_0_embedded, num_samples)  # (S*N) * dim(z)

        log_p_z = self.generative_model.log_p_z(z, x_0, x_0_embedded)  # (S*N)

        x_1 = self.generative_model.beam_search_samples(x_0, x_0_embedded, z, log_p_z, self.all_embeddings_1, beam_size,
                                                        num_samples)  # N * max(L)

        x_1 = cut_off(x_1, self.eos_ind)  # N * max(L)

        return theano.function(inputs=[x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )


class SGVBWordsVNMTMultiIndicator(object):

    def __init__(self, generative_model, recognition_model, num_langs, z_dim, max_length, vocab_size, embedding_dim,
                 dist_z_gen, dist_x_1_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.num_langs = num_langs
        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings = theano.shared(np.float32(np.random.normal(0., 0.1, (num_langs, vocab_size,
                                                                                  embedding_dim))))

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_1_gen = dist_x_1_gen

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model = self.init_generative_model(generative_model)

        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

        self.params = self.generative_model.get_params() + self.recognition_model.get_params() + [self.all_embeddings]

    def init_generative_model(self, generative_model):

        generative_model = generative_model(self.num_langs, self.z_dim, self.max_length, self.vocab_size,
                                            self.embedding_dim, embedder, self.dist_z_gen, self.dist_x_1_gen,
                                            self.gen_nn_kwargs)

        return generative_model

    def init_recognition_model(self, recognition_model):

        recognition_model = recognition_model(self.num_langs, self.z_dim, self.max_length, self.embedding_dim,
                                              self.dist_z_rec, self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo(self, l_0, l_1, x_0, x_1, num_samples, beta=None, drop_mask_1=None):

        x_0_embedded, _ = theano.scan(lambda x, l: embedder(x, self.all_embeddings[l]),
                                      sequences=[x_0, l_0],
                                      )

        # x_0_embedded = embedder(x_0, self.all_embeddings[l_0])  # N * max(L) * E

        x_1_embedded = embedder(x_1, self.all_embeddings[l_1])  # N * max(L) * E

        z, means_rec, covs_rec = self.recognition_model.get_samples_and_means_and_covs(l_0, l_1, x_0, x_0_embedded, x_1,
                                                                                       x_1_embedded, num_samples)

        h = self.generative_model.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)

        kl = self.generative_model.kl(h, x_0, means_rec, covs_rec)  # N

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_1 = self.generative_model.log_p_x_1(l_1, h, z, x_1, x_1_embedded, x_1_embedded_dropped,
                                                    self.all_embeddings[l_1])  # (S*N)

        if beta is None:
            elbo = T.mean(log_p_x_1) - T.mean(kl)
        else:
            elbo = T.mean(log_p_x_1) - T.mean(beta * kl)

        return elbo, T.mean(kl)

    def elbo_fn(self, num_samples):

        l_0 = T.ivector('l_0')  # N
        l_1 = T.iscalar('l_1')  #
        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(l_0, l_1, x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[l_0, l_1, x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        l_0 = T.ivector('l_0')  # N
        l_1 = T.iscalar('l_1')  #
        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        beta = T.scalar('beta')

        drop_mask_1 = T.matrix('drop_mask_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(l_0, l_1, x_0, x_1, num_samples, beta, drop_mask_1)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[l_0, l_1, x_0, x_1, beta, drop_mask_1],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def translate_fn(self, beam_size):

        l_0 = T.ivector('l_0')  # N
        l_1 = T.iscalar('l_1')  #
        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded, _ = theano.scan(lambda x, l: embedder(x, self.all_embeddings[l]),
                                      sequences=[x_0, l_0],
                                      )

        # x_0_embedded = embedder(x_0, self.all_embeddings[l_0])  # N * max(L) * E

        h = self.generative_model.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)

        z, _ = self.generative_model.get_z_means_and_covs(h, x_0)  # N * dim(z)

        x_1 = self.generative_model.beam_search(l_0, l_1, x_0, x_0_embedded, z, self.all_embeddings[l_1], beam_size)
        # N * max(L)

        x_1 = cut_off(x_1, self.eos_ind)  # N * max(L)

        return theano.function(inputs=[l_0, l_1, x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, beam_size, num_samples):

        l_0 = T.ivector('l_0')  # N
        l_1 = T.iscalar('l_1')  #
        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded, _ = theano.scan(lambda x, l: embedder(x, self.all_embeddings[l]),
                                      sequences=[x_0, l_0],
                                      )

        # x_0_embedded = embedder(x_0, self.all_embeddings[l_0])  # N * max(L) * E

        h = self.generative_model.get_h(l_0, x_0, x_0_embedded)  # N * max(L) * dim(hid)

        z = self.generative_model.get_samples_z(h, x_0, num_samples)  # (S*N) * dim(z)

        log_p_z = self.generative_model.log_p_z(z, l_0, x_0, x_0_embedded)  # (S*N)

        x_1 = self.generative_model.beam_search_samples(l_0, l_1, x_0, x_0_embedded, z, log_p_z,
                                                        self.all_embeddings[l_1], beam_size, num_samples)  # N * max(L)

        x_1 = cut_off(x_1, self.eos_ind)  # N * max(L)

        return theano.function(inputs=[l_0, l_1, x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )


class SGVBWordsVNMTJoint(object):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, dist_z_gen,
                 dist_x_1_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings_0 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
        self.all_embeddings_1 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))

        self.dist_z_gen = dist_z_gen
        self.dist_z_rec = dist_z_rec
        self.dist_x_1_gen = dist_x_1_gen

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model = self.init_generative_model(generative_model)

        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

        self.params = self.generative_model.get_params() + self.recognition_model.get_params() + \
                      [self.all_embeddings_0, self.all_embeddings_1]

    def init_generative_model(self, generative_model):

        generative_model = generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                            embedder, self.dist_z_gen, self.dist_x_1_gen, self.gen_nn_kwargs)

        return generative_model

    def init_recognition_model(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec,
                                              self.rec_nn_kwargs)

        return recognition_model

    def symbolic_elbo(self, x_0, x_1, num_samples, beta=None, drop_mask_1=None):

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

        z, means_rec, covs_rec = self.recognition_model.get_samples_and_means_and_covs(x_0, x_0_embedded, x_1,
                                                                                       x_1_embedded, num_samples)

        kl = self.generative_model.kl(means_rec, covs_rec)  # N

        x_0_embedded_dropped = x_0_embedded

        log_p_x_0 = self.generative_model.log_p_x_0(z, x_0, x_0_embedded, x_0_embedded_dropped, self.all_embeddings_0)
        # (S*N)

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        log_p_x_1 = self.generative_model.log_p_x_1(x_0, x_0_embedded, z, x_1, x_1_embedded, x_1_embedded_dropped,
                                                    self.all_embeddings_1)  # (S*N)

        if beta is None:
            elbo = T.mean(log_p_x_0 + log_p_x_1) - T.mean(kl)
        else:
            elbo = T.mean(log_p_x_0 + log_p_x_1) - T.mean(beta * kl)

        return elbo, T.mean(kl)

    def elbo_fn(self, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x_0, x_1, num_samples)

        elbo_fn = theano.function(inputs=[x_0, x_1],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        beta = T.scalar('beta')

        drop_mask_1 = T.matrix('drop_mask_1')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x_0, x_1, num_samples, beta, drop_mask_1)

        grads = T.grad(-elbo, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x_0, x_1, beta, drop_mask_1],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def translate_fn(self, beam_size):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

        z = T.zeros((x_0.shape[0], self.z_dim))  # N * dim(z)

        x_1 = self.generative_model.beam_search(x_0, x_0_embedded, z, self.all_embeddings_1, beam_size)  # N * max(L)

        x_1 = cut_off(x_1, self.eos_ind)  # N * max(L)

        return theano.function(inputs=[x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )

    def translate_fn_sampling(self, beam_size, num_samples):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

        z = self.generative_model.get_samples_z(x_0, x_0_embedded, num_samples)  # (S*N) * dim(z)

        log_p_z = self.generative_model.log_p_z(z, x_0, x_0_embedded)  # (S*N)

        x_1 = self.generative_model.beam_search_samples(x_0, x_0_embedded, z, log_p_z, self.all_embeddings_1, beam_size,
                                                        num_samples)  # N * max(L)

        x_1 = cut_off(x_1, self.eos_ind)  # N * max(L)

        return theano.function(inputs=[x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )
