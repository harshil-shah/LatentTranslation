from itertools import combinations, chain
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import norm_constraint


class DiscriminativeTrainer(object):

    def __init__(self, encoder, decoder, z_dim, max_length, vocab_size, embedding_dim, encoder_nn_kwargs,
                 decoder_nn_kwargs, eos_ind):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings_0 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
        self.all_embeddings_1 = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))

        self.encoder_nn_kwargs = encoder_nn_kwargs
        self.decoder_nn_kwargs = decoder_nn_kwargs

        self.encoder = self.init_encoder(encoder)
        self.decoder = self.init_decoder(decoder)

        self.params = self.encoder.get_params() + self.decoder.get_params() + [self.all_embeddings_0,
                                                                               self.all_embeddings_1]

        self.eos_ind = eos_ind

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

        return all_embeddings[x]

    def init_encoder(self, encoder):

        return encoder(self.z_dim, self.max_length, self.embedding_dim, self.encoder_nn_kwargs)

    def init_decoder(self, decoder):

        return decoder(self.z_dim, self.max_length, self.embedding_dim, self.embedder, self.decoder_nn_kwargs)

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

    def log_p_x(self, x_0, x_1, drop_mask_1=None):

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)  # N * max(L) * E
        x_1_embedded = self.embedder(x_1, self.all_embeddings_1)  # N * max(L) * E

        if drop_mask_1 is None:
            x_1_embedded_dropped = x_1_embedded
        else:
            x_1_embedded_dropped = x_1_embedded * T.shape_padright(drop_mask_1)

        z = self.encoder.get_z(x_0, x_0_embedded)

        log_p_x_1 = self.decoder.log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, z, self.all_embeddings_1)

        return T.mean(log_p_x_1)

    def log_p_x_fn(self):

        x_0 = T.imatrix('x_0')
        x_1 = T.imatrix('x_1')

        log_p_x = self.log_p_x(x_0, x_1)

        log_p_x_fn = theano.function(inputs=[x_0, x_1],
                                     outputs=log_p_x,
                                     allow_input_downcast=True,
                                     )

        return log_p_x_fn

    def optimiser(self, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x_0 = T.imatrix('x_0')
        x_1 = T.imatrix('x_1')

        drop_mask_1 = T.matrix('drop_mask_1')

        log_p_x = self.log_p_x(x_0, x_1, drop_mask_1)

        grads = T.grad(-log_p_x, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x_0, x_1, drop_mask_1],
                                    outputs=log_p_x,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def translate_fn(self, beam_size):

        x_0 = T.imatrix('x_0')  # N * max(L)

        x_0_embedded = self.embedder(x_0, self.all_embeddings_0)  # N * max(L) * E

        z = self.encoder.get_z(x_0, x_0_embedded)

        x_1 = self.decoder.beam_search(z, self.all_embeddings_1, beam_size)

        x_1 = self.cut_off(x_1, self.eos_ind)

        return theano.function(inputs=[x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )


class DiscriminativeTrainerSSLMulti(object):

    def __init__(self, num_langs, encoder, decoder, z_dim, max_length, vocab_size, embedding_dim, encoder_nn_kwargs,
                 decoder_nn_kwargs, eos_ind):

        self.num_langs = num_langs
        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings = [theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))
                               for l in range(self.num_langs)]

        self.encoder_nn_kwargs = encoder_nn_kwargs
        self.decoder_nn_kwargs = decoder_nn_kwargs

        self.encoders = [self.init_encoder(encoder) for l in range(self.num_langs)]
        self.decoders = [self.init_decoder(decoder) for l in range(self.num_langs)]

        self.encoders_params = list(chain.from_iterable([encoder.get_params() for encoder in self.encoders]))
        self.decoders_params = list(chain.from_iterable([decoder.get_params() for decoder in self.decoders]))

        self.params = self.encoders_params + self.decoders_params + self.all_embeddings

        self.eos_ind = eos_ind

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, all_embeddings.shape[1]))], axis=0)

        return all_embeddings[x]

    def init_encoder(self, encoder):

        return encoder(self.z_dim, self.max_length, self.embedding_dim, self.encoder_nn_kwargs)

    def init_decoder(self, decoder):

        return decoder(self.z_dim, self.max_length, self.embedding_dim, self.embedder, self.decoder_nn_kwargs)

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

    def log_p_x_only(self, l, x, drop_mask=None):

        x_embedded = self.embedder(x, self.all_embeddings[l])  # N * max(L) * E

        if drop_mask is None:
            x_embedded_dropped = x_embedded
        else:
            x_embedded_dropped = x_embedded * T.shape_padright(drop_mask)

        z = self.encoders[l].get_z(x, x_embedded)

        log_p_x = self.decoders[l].log_p_x(x, x_embedded, x_embedded_dropped, z, self.all_embeddings[l])

        return T.mean(log_p_x)

    def log_p_x_to_y(self, l_x, l_y, x, y, drop_mask_y=None):

        x_embedded = self.embedder(x, self.all_embeddings[l_x])  # N * max(L) * E
        y_embedded = self.embedder(y, self.all_embeddings[l_y])  # N * max(L) * E

        if drop_mask_y is None:
            y_embedded_dropped = y_embedded
        else:
            y_embedded_dropped = y_embedded * T.shape_padright(drop_mask_y)

        z = self.encoders[l_x].get_z(x, x_embedded)

        log_p_y = self.decoders[l_y].log_p_x(y, y_embedded, y_embedded_dropped, z, self.all_embeddings[l_y])

        return T.mean(log_p_y)

    def log_p_x_both(self, l_0, l_1, x_0, x_1, drop_mask_0=None, drop_mask_1=None):

        log_p_x_0_to_x_1 = self.log_p_x_to_y(l_0, l_1, x_0, x_1, drop_mask_1)
        log_p_x_1_to_x_0 = self.log_p_x_to_y(l_1, l_0, x_1, x_0, drop_mask_0)

        return T.mean(log_p_x_0_to_x_1 + log_p_x_1_to_x_0)

    def log_p_x_all(self, xs_only, xs_both, drop_masks_only=None, drop_masks_both=None):

        if drop_masks_only is None:
            drop_masks_only = {l: None for l in range(self.num_langs)}

        if drop_masks_both is None:
            drop_masks_both = {l: [None, None] for l in combinations(range(self.num_langs), 2)}

        log_p_xs_only = []

        for l in range(self.num_langs):
            log_p_xs_only.append(self.log_p_x_only(l, xs_only[l], drop_masks_only[l]))

        log_p_xs_both = []

        for pair in combinations(range(self.num_langs), 2):
            l_0 = pair[0]
            l_1 = pair[1]

            log_p_xs_both.append(self.log_p_x_both(l_0, l_1, xs_both[(l_0, l_1)][0], xs_both[(l_0, l_1)][1],
                                                   drop_masks_both[(l_0, l_1)][0], drop_masks_both[(l_0, l_1)][1]))

        return T.sum(log_p_xs_only) + T.sum(log_p_xs_both)

    def log_p_x_fn(self, l_0, l_1):

        x_0 = T.imatrix('x_0')  # N * max(L)
        x_1 = T.imatrix('x_1')  # N * max(L)

        log_p_x = self.log_p_x_both(l_0, l_1, x_0, x_1)

        log_p_x_fn = theano.function(inputs=[x_0, x_1],
                                     outputs=log_p_x,
                                     allow_input_downcast=True,
                                     )

        return log_p_x_fn

    def optimiser(self, grad_norm_constraint, update, update_kwargs, saved_update=None):

        xs_only = {l: T.imatrix('x_' + str(l) + '_only') for l in range(self.num_langs)}
        xs_both = {pair: [T.imatrix('x_' + str(pair[0]) + '_both'), T.imatrix('x_' + str(pair[1]) + '_both')]
                   for pair in combinations(range(self.num_langs), 2)}

        drop_masks_only = {l: T.matrix('drop_mask_' + str(l) + '_only') for l in range(self.num_langs)}
        drop_masks_both = {l: [T.matrix('drop_mask_' + str(l[0]) + '_both'), T.matrix('drop_mask_' + str(l[1]) +
                                                                                      '_both')]
                           for l in combinations(range(self.num_langs), 2)}

        log_p_x_all = self.log_p_x_all(xs_only, xs_both, drop_masks_only, drop_masks_both)

        grads = T.grad(-log_p_x_all, self.params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = self.params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        inputs = list(xs_only.values()) + list(chain.from_iterable(xs_both.values())) + list(drop_masks_only.values()) \
            + list(chain.from_iterable(drop_masks_both.values()))

        optimiser = theano.function(inputs=inputs,
                                    outputs=log_p_x_all,
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def translate_x_to_y_fn(self, l_x, l_y, beam_size):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = self.embedder(x, self.all_embeddings[l_x])  # N * max(L) * E

        z = self.encoders[l_x].get_z(x, x_embedded)

        y = self.decoders[l_y].beam_search(z, self.all_embeddings[l_y], beam_size)
        y = self.cut_off(y, self.eos_ind)

        return theano.function(inputs=[x],
                               outputs=y,
                               allow_input_downcast=True,
                               )
