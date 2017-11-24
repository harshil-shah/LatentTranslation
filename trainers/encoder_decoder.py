import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import norm_constraint


class EncoderDecoderTrainer(object):

    def __init__(self, encoder, decoder, h_dim, max_length, vocab_size, embedding_dim, encoder_nn_kwargs,
                 decoder_nn_kwargs, eos_ind):

        self.h_dim = h_dim
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

        return encoder(self.h_dim, self.max_length, self.embedding_dim, self.encoder_nn_kwargs)

    def init_decoder(self, decoder):

        return decoder(self.h_dim, self.max_length, self.embedding_dim, self.embedder, self.decoder_nn_kwargs)

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

        h = self.encoder.get_hid(x_0, x_0_embedded)

        log_p_x_1 = self.decoder.log_p_x(x_1, x_1_embedded, x_1_embedded_dropped, h, self.all_embeddings_1)

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

        h = self.encoder.get_hid(x_0, x_0_embedded)

        x_1 = self.decoder.beam_search(h, self.all_embeddings_1, beam_size)

        x_1 = self.cut_off(x_1, self.eos_ind)

        return theano.function(inputs=[x_0],
                               outputs=x_1,
                               allow_input_downcast=True,
                               )
