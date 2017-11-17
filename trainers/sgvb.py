import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import norm_constraint


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
                                                self.embedder, self.dist_z_gen, self.dist_x_0_gen, self.gen_nn_0_kwargs)

        generative_model_1 = generative_model_1(self.z_dim, self.max_length_1, self.vocab_size_1, self.embedding_dim_1,
                                                self.embedder, self.dist_z_gen, self.dist_x_1_gen, self.gen_nn_1_kwargs)

        return generative_model_0, generative_model_1

    def init_recognition_models(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length_0, self.max_length_1, self.embedding_dim_0,
                                              self.embedding_dim_1, self.dist_z_rec, self.rec_nn_kwargs)

        return recognition_model

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

        return all_embeddings[x]

    def cut_off(self, x, eos_ind):

        def step(x_l, x_lm1):

            x_l = T.switch(T.eq(x_lm1, eos_ind), -1, x_l)
            x_l = T.switch(T.eq(x_lm1, -1), -1, x_l)

            return T.cast(x_l, 'int32')

        x_cut_off, _ = theano.scan(step,
                                   sequences=x.T,
                                   outputs_info=T.zeros((x.shape[0],), 'int32'),
                                   )

        return x_cut_off.T

    def symbolic_elbo(self, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

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

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)
        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)

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

        x_from_embedded = self.embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = self.embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

        if from_index == 0:

            z = self.recognition_model.get_samples(x_from, x_from_embedded, x_to_best_guess, x_to_best_guess_embedded,
                                                   1, means_only=True)  # N * dim(z)

            x_to = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)

        else:

            z = self.recognition_model.get_samples(x_to_best_guess, x_to_best_guess_embedded, x_from, x_from_embedded,
                                                   1, means_only=True)  # N * dim(z)

            x_to = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)

        x_to = self.cut_off(x_to, eos_ind_to)

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

        x_from_embedded = self.embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = self.embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

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

        x_to = self.cut_off(x_to, eos_ind_to)

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
                                                self.embedder, self.dist_z_gen, self.dist_x_0_gen, self.gen_nn_0_kwargs)

        generative_model_1 = generative_model_1(self.z_dim, self.max_length_1, self.vocab_size_1, self.embedding_dim_1,
                                                self.embedder, self.dist_z_gen, self.dist_x_1_gen, self.gen_nn_1_kwargs)

        return generative_model_0, generative_model_1

    def init_recognition_models(self, recognition_model):

        recognition_model = recognition_model(self.z_dim, self.max_length_0, self.max_length_1, self.embedding_dim_0,
                                              self.embedding_dim_1, self.dist_z_rec, self.rec_nn_kwargs)

        return recognition_model

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

        return all_embeddings[x]

    def cut_off(self, x, eos_ind):

        def step(x_l, x_lm1):

            x_l = T.switch(T.eq(x_lm1, eos_ind), -1, x_l)
            x_l = T.switch(T.eq(x_lm1, -1), -1, x_l)

            return T.cast(x_l, 'int32')

        x_cut_off, _ = theano.scan(step,
                                   sequences=x.T,
                                   outputs_info=T.zeros((x.shape[0],), 'int32'),
                                   )

        return x_cut_off.T

    def symbolic_elbo_both(self, x_0, x_1, num_samples, beta=None, drop_mask_0=None, drop_mask_1=None):

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

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

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

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

        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

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

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)
        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_both(x_0, x_0_embedded, x_1, x_1_embedded, 1,
                                                                           means_only=True)  # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        x_beam_0 = self.cut_off(x_beam_0, self.eos_ind_0)
        x_beam_1 = self.cut_off(x_beam_1, self.eos_ind_1)

        return theano.function(inputs=[x_0, x_1],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_0_only(self, beam_size, num_time_steps=None):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_0_only(x_0, x_0_embedded, 1, means_only=True)
        # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        x_beam_0 = self.cut_off(x_beam_0, self.eos_ind_0)
        x_beam_1 = self.cut_off(x_beam_1, self.eos_ind_1)

        return theano.function(inputs=[x_0],
                               outputs=[x_beam_0, x_beam_1],
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn_1_only(self, beam_size, num_time_steps=None):

        x_1 = T.imatrix('x_1')  # N * max(L)

        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)

        z, _ = self.recognition_model.get_samples_and_kl_std_gaussian_1_only(x_1, x_1_embedded, 1, means_only=True)
        # N * dim(z)

        x_beam_0 = self.generative_model_0.beam_search(z, self.all_embeddings_0, beam_size, num_time_steps)  # S *
        # max(L)
        x_beam_1 = self.generative_model_1.beam_search(z, self.all_embeddings_1, beam_size, num_time_steps)  # S *
        # max(L)

        x_beam_0 = self.cut_off(x_beam_0, self.eos_ind_0)
        x_beam_1 = self.cut_off(x_beam_1, self.eos_ind_1)

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

        x_from_embedded = self.embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = self.embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

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

        x_to = self.cut_off(x_to, eos_ind_to)

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

        x_from_embedded = self.embedder(x_from, all_embeddings_from)  # N * max(L) * E
        x_to_best_guess_embedded = self.embedder(x_to_best_guess, all_embeddings_to)  # N * max(L) * E

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

        x_to = self.cut_off(x_to, eos_ind_to)

        return theano.function(inputs=[x_from, x_to_best_guess],
                               outputs=x_to,
                               allow_input_downcast=True,
                               )
