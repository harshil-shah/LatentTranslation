from collections import OrderedDict
import os
import pickle as cPickle
import time
import numpy as np
import json
from lasagne.updates import adam
# from data_processing.utilities import chunker


class RunWords(object):

    def __init__(self, solver, solver_kwargs, valid_vocab_0, valid_vocab_1, main_dir, out_dir, dataset,
                 load_param_dir=None, pre_trained=False, train_prop=0.95):

        self.valid_vocab_0 = valid_vocab_0
        self.valid_vocab_1 = valid_vocab_1

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.vocab_size_0 = solver_kwargs['vocab_size_0']
        self.vocab_size_1 = solver_kwargs['vocab_size_1']

        print('loading data')

        start = time.clock()

        self.x_0_train, self.x_0_test, self.x_1_train, self.x_1_test, self.L_0_train, self.L_0_test, self.L_1_train, \
            self.L_1_test = self.load_data(dataset, train_prop)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        print('# training sentences = ' + str(len(self.L_0_train)))
        print('# test sentences = ' + str(len(self.L_0_test)))

        self.max_length_0 = np.concatenate((self.x_0_train, self.x_0_test), axis=0).shape[1]
        self.max_length_1 = np.concatenate((self.x_1_train, self.x_1_test), axis=0).shape[1]

        self.max_length = max((self.max_length_0, self.max_length_1))

        self.vb = solver(max_length_0=self.max_length_0, max_length_1=self.max_length_1, **self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings_0.save'), 'rb') as f:
                self.vb.all_embeddings_0.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'all_embeddings_1.save'), 'rb') as f:
                self.vb.all_embeddings_1.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params_0.save'), 'rb') as f:
                self.vb.generative_model_0.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params_1.save'), 'rb') as f:
                self.vb.generative_model_1.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.vb.recognition_model.set_param_values(cPickle.load(f))

    def load_data(self, dataset, train_prop, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        files_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0')])
        files_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1')])

        words_0 = []
        words_1 = []

        for f in files_0:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words_0 += json.loads(s.read())

        for f in files_1:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words_1 += json.loads(s.read())

        L_0 = np.array([len(s) for s in words_0])
        L_1 = np.array([len(s) for s in words_1])

        max_L = max(np.concatenate((L_0, L_1)))

        word_arrays_0 = []

        start = time.clock()

        batches_0_loaded = 0

        for i in range(0, len(L_0), load_batch_size):

            L_i = L_0[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max(L_0))] = \
                np.concatenate(words_0[i: i+load_batch_size])

            word_arrays_0.append(word_array)

            del L_i, word_array

            batches_0_loaded += 1

            print(str(batches_0_loaded) + ' batches x_0 loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words_0

        words_to_return_0 = np.concatenate(word_arrays_0)

        word_arrays_1 = []

        start = time.clock()

        batches_1_loaded = 0

        for i in range(0, len(L_1), load_batch_size):

            L_i = L_1[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max(L_1))] = \
                np.concatenate(words_1[i: i+load_batch_size])

            word_arrays_1.append(word_array)

            del L_i, word_array

            batches_1_loaded += 1

            print(str(batches_1_loaded) + ' batches x_1 loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words_1

        words_to_return_1 = np.concatenate(word_arrays_1)

        np.random.seed(1234)

        training_mask = np.random.rand(len(words_to_return_0)) < train_prop

        return words_to_return_0[training_mask], words_to_return_0[~training_mask], words_to_return_1[training_mask], \
               words_to_return_1[~training_mask], L_0[training_mask], L_0[~training_mask], L_1[training_mask], \
               L_1[~training_mask]

    def call_elbo_fn(self, elbo_fn, x_0, x_1):

        return elbo_fn(x_0, x_1)

    def call_optimiser(self, optimiser, x_0, x_1, beta, drop_mask_0, drop_mask_1):

        return optimiser(x_0, x_1, beta, drop_mask_0, drop_mask_1)

    def get_generate_output_prior(self, num_outputs, beam_size):

        return self.vb.generate_output_prior_fn(num_outputs, beam_size)

    def call_generate_output_prior(self, generate_output_prior):

        x_0_gen_beam, x_1_gen_beam = generate_output_prior()

        out = OrderedDict()

        out['generated_x_0_beam_prior'] = x_0_gen_beam
        out['generated_x_1_beam_prior'] = x_1_gen_beam

        return out

    def print_output_prior(self, output_prior):

        x_0_gen_beam = output_prior['generated_x_0_beam_prior']
        x_1_gen_beam = output_prior['generated_x_1_beam_prior']

        print('='*10)
        print('samples')
        print('='*10)

        for n in range(x_0_gen_beam.shape[0]):

            print('gen x_0 beam: ' + ' '.join([self.valid_vocab_0[int(i)] for i in x_0_gen_beam[n]]))
            print('gen x_1 beam: ' + ' '.join([self.valid_vocab_1[int(i)] for i in x_1_gen_beam[n]]))

            print('-'*10)

        print('='*10)

    def get_generate_output_posterior(self, beam_size):

        return self.vb.generate_output_posterior_fn(beam_size)

    def call_generate_output_posterior(self, generate_output_posterior, x_0, x_1):

        x_0_gen_beam, x_1_gen_beam = generate_output_posterior(x_0, x_1)

        out = OrderedDict()

        out['true_x_0_for_posterior'] = x_0
        out['true_x_1_for_posterior'] = x_1
        out['generated_x_0_beam_posterior'] = x_0_gen_beam
        out['generated_x_1_beam_posterior'] = x_1_gen_beam

        return out

    def print_output_posterior(self, output_posterior):

        x_0 = output_posterior['true_x_0_for_posterior']
        x_1 = output_posterior['true_x_1_for_posterior']

        x_0_gen_beam = output_posterior['generated_x_0_beam_posterior']
        x_1_gen_beam = output_posterior['generated_x_1_beam_posterior']

        valid_vocab_0 = self.valid_vocab_0 + ['']
        valid_vocab_1 = self.valid_vocab_1 + ['']

        print('='*10)
        print('reconstructions')
        print('='*10)

        for n in range(x_0.shape[0]):

            print('    true x_0: ' + ' '.join([valid_vocab_0[i] for i in x_0[n]]).strip())
            print('    true x_1: ' + ' '.join([valid_vocab_1[i] for i in x_1[n]]).strip())
            print('gen x_0 beam: ' + ' '.join([valid_vocab_0[int(i)] for i in x_0_gen_beam[n]]))
            print('gen x_1 beam: ' + ' '.join([valid_vocab_1[int(i)] for i in x_1_gen_beam[n]]))

            print('-'*10)

        print('='*10)

    def get_translation_fn(self, from_index, beam_size):

        return self.vb.translate_fn_sampling(from_index, beam_size, num_samples=50)

    def call_translation_fn(self, translation_fn, from_index, num_outputs, num_iterations):

        if from_index == 0:
            x_test_from = self.x_0_test
            x_test_to = self.x_1_test
            valid_vocab_from = self.valid_vocab_0
            valid_vocab_to = self.valid_vocab_1
        else:
            x_test_from = self.x_1_test
            x_test_to = self.x_0_test
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

        batch_indices = np.random.choice(len(x_test_from), num_outputs, replace=False)
        batch_in = np.array([x_test_from[ind] for ind in batch_indices])
        true_batch_out = np.array([x_test_to[ind] for ind in batch_indices])

        best_guess = -1. * np.ones((num_outputs, self.max_length))

        print('')

        for i in range(num_iterations):

            start = time.clock()

            best_guess = translation_fn(batch_in, best_guess)

            print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            print(' ')
            print('  in: ' + ' '.join([valid_vocab_from[i] for i in batch_in[0]]))
            print('true: ' + ' '.join([valid_vocab_to[i] for i in true_batch_out[0]]))
            print(' gen: ' + ' '.join([valid_vocab_to[i] for i in best_guess[0]]))
            print(' ')

        print('')

        out = OrderedDict()

        out['true_x_from_for_translation_from_' + str(from_index)] = batch_in
        out['true_x_to_for_translation_from_' + str(from_index)] = true_batch_out
        out['gen_x_to_for_translation_from_' + str(from_index)] = best_guess

        return out

    def print_translations(self, from_index, translations):

        if from_index == 0:
            valid_vocab_from = self.valid_vocab_0
            valid_vocab_to = self.valid_vocab_1
        else:
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

        x_in = translations['true_x_from_for_translation_from_' + str(from_index)]
        x_out_true = translations['true_x_to_for_translation_from_' + str(from_index)]
        x_out_gen = translations['gen_x_to_for_translation_from_' + str(from_index)]

        print('='*10)
        print('translations')
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([valid_vocab_from[j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([valid_vocab_to[j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([valid_vocab_to[j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def train(self, n_iter, batch_size, num_samples, word_drop=None, grad_norm_constraint=None, update=adam,
              update_kwargs=None, warm_up=None, val_freq=None, val_batch_size=0, val_num_samples=0, val_print_gen=5,
              val_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
        else:
            saved_update = None

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                               update=update, update_kwargs=update_kwargs, saved_update=saved_update)

        elbo_fn = self.vb.elbo_fn(val_num_samples)

        generate_output_prior = self.get_generate_output_prior(val_print_gen, val_beam_size)
        generate_output_posterior = self.get_generate_output_posterior(val_beam_size)
        generate_translation_0 = self.get_translation_fn(0, val_beam_size)
        generate_translation_1 = self.get_translation_fn(1, val_beam_size)

        for i in range(n_iter):

            start = time.clock()

            batch_indices = np.random.choice(len(self.x_0_train), batch_size)
            batch_0 = np.array([self.x_0_train[ind] for ind in batch_indices])
            batch_1 = np.array([self.x_1_train[ind] for ind in batch_indices])

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            if word_drop is not None:

                L_0 = np.array([self.L_0_train[ind] for ind in batch_indices])
                L_1 = np.array([self.L_1_train[ind] for ind in batch_indices])

                drop_indices_0 = np.array([np.random.permutation(np.arange(i))[:int(np.floor(word_drop * i))]
                                           for i in L_0])

                drop_mask_0 = np.ones_like(batch_0)

                for n in range(len(drop_indices_0)):
                    drop_mask_0[n][drop_indices_0[n]] = 0.

                drop_indices_1 = np.array([np.random.permutation(np.arange(i))[:int(np.floor(word_drop * i))]
                                           for i in L_1])

                drop_mask_1 = np.ones_like(batch_1)

                for n in range(len(drop_indices_1)):
                    drop_mask_1[n][drop_indices_1[n]] = 0.

            else:

                drop_mask_0 = np.ones_like(batch_0)
                drop_mask_1 = np.ones_like(batch_1)

            elbo, kl = self.call_optimiser(optimiser, batch_0, batch_1, beta, drop_mask_0, drop_mask_1)

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo/batch_size) + ' (KL = ' + str(kl/batch_size) +
                  ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                val_batch_indices = np.random.choice(len(self.x_0_test), val_batch_size)
                val_batch_0 = np.array([self.x_0_test[ind] for ind in val_batch_indices])
                val_batch_1 = np.array([self.x_1_test[ind] for ind in val_batch_indices])

                val_elbo, val_kl = self.call_elbo_fn(elbo_fn, val_batch_0, val_batch_1)

                print('Test set ELBO = ' + str(val_elbo/val_batch_size) + ' (KL = ' + str(kl/batch_size) +
                      ') per data point')

                output_prior = self.call_generate_output_prior(generate_output_prior)

                self.print_output_prior(output_prior)

                post_batch_indices = np.random.choice(len(self.x_0_test), val_print_gen, replace=False)
                post_batch_0 = np.array([self.x_0_test[ind] for ind in post_batch_indices])
                post_batch_1 = np.array([self.x_1_test[ind] for ind in post_batch_indices])

                output_posterior = self.call_generate_output_posterior(generate_output_posterior, post_batch_0,
                                                                       post_batch_1)

                self.print_output_posterior(output_posterior)

                translations_0 = self.call_translation_fn(generate_translation_0, 0, val_print_gen, 5)
                self.print_translations(0, translations_0)

                translations_1 = self.call_translation_fn(generate_translation_1, 1, val_print_gen, 5)
                self.print_translations(1, translations_1)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                with open(os.path.join(self.out_dir, 'all_embeddings_0.save'), 'wb') as f:
                    cPickle.dump(self.vb.all_embeddings_0.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'all_embeddings_1.save'), 'wb') as f:
                    cPickle.dump(self.vb.all_embeddings_1.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'gen_params_0.save'), 'wb') as f:
                    cPickle.dump(self.vb.generative_model_0.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'gen_params_1.save'), 'wb') as f:
                    cPickle.dump(self.vb.generative_model_1.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                    cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                    cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings_0.save'), 'wb') as f:
            cPickle.dump(self.vb.all_embeddings_0.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings_1.save'), 'wb') as f:
            cPickle.dump(self.vb.all_embeddings_1.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params_0.save'), 'wb') as f:
            cPickle.dump(self.vb.generative_model_0.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params_1.save'), 'wb') as f:
            cPickle.dump(self.vb.generative_model_1.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # def test(self, batch_size, num_samples, sub_sample_size=None):
    #
    #     elbo_fn = self.vb.elbo_fn(num_samples) if sub_sample_size is None else self.vb.elbo_fn(sub_sample_size)
    #
    #     elbo = 0
    #     kl = 0
    #     pp_0 = 0
    #     pp_1 = 0
    #
    #     batches_complete = 0
    #
    #     for batch_X in chunker([self.X_test], batch_size):
    #
    #         start = time.clock()
    #
    #         if sub_sample_size is None:
    #
    #             elbo_batch, kl_batch, pp_batch = self.call_elbo_fn(elbo_fn, batch_X[0])
    #
    #         else:
    #
    #             elbo_batch = 0
    #             kl_batch = 0
    #             pp_batch = 0
    #
    #             for sub_sample in range(1, int(num_samples / sub_sample_size) + 1):
    #
    #                 elbo_sub_batch, kl_sub_batch, pp_sub_batch = self.call_elbo_fn(elbo_fn, batch_X[0])
    #
    #                 elbo_batch = (elbo_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
    #                                             float(sub_sample * sub_sample_size))) + \
    #                              (elbo_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
    #
    #                 kl_batch = (kl_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
    #                                         float(sub_sample * sub_sample_size))) + \
    #                            (kl_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
    #
    #                 pp_batch = (pp_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
    #                                         float(sub_sample * sub_sample_size))) + \
    #                            (pp_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
    #
    #         elbo += elbo_batch
    #         kl += kl_batch
    #         pp += pp_batch
    #
    #         batches_complete += 1
    #
    #         print('Tested batches ' + str(batches_complete) + ' of ' + str(round(self.X_test.shape[0] / batch_size))
    #               + 'so far; test set ELBO = ' + str(elbo) + ', test set KL = ' + str(kl) + ', test set perplexity = '
    #               + str(pp) + ' / ' + str(elbo / (batches_complete * batch_size)) + ', '
    #               + str(kl / (batches_complete * batch_size)) + ', ' + str(pp / (batches_complete * batch_size))
    #               + ' per obs. (time taken = ' + str(time.clock() - start) + ' seconds)')
    #
    #     print('Test set ELBO = ' + str(elbo))
    #
    def generate_output(self, prior, posterior, num_outputs, beam_size=15):

        if prior:

            generate_output_prior = self.vb.generate_output_prior_fn(num_outputs, beam_size)

            output_prior = self.call_generate_output_prior(generate_output_prior)

            for key, value in output_prior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

        if posterior:

            generate_output_posterior = self.vb.generate_output_posterior_fn(beam_size)

            np.random.seed(1234)

            batch_indices = np.random.choice(len(self.x_0_test), num_outputs, replace=False)
            batch_0 = np.array([self.x_0_test[ind] for ind in batch_indices])
            batch_1 = np.array([self.x_1_test[ind] for ind in batch_indices])

            output_posterior = self.call_generate_output_posterior(generate_output_posterior, batch_0, batch_1)

            for key, value in output_posterior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def translate(self, from_index, num_outputs, num_iterations, num_samples_per_output, sampling=True, beam_size=15):

        np.random.seed(1234)

        if sampling:
            translate_fn = self.vb.translate_fn_sampling(from_index, beam_size, num_samples=num_samples_per_output)
        else:
            translate_fn = self.vb.translate_fn(from_index, beam_size)

        if from_index == 0:
            x_test_from = self.x_0_test
            x_test_to = self.x_1_test
            valid_vocab_from = self.valid_vocab_0
            valid_vocab_to = self.valid_vocab_1
        else:
            x_test_from = self.x_1_test
            x_test_to = self.x_0_test
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

        batch_indices = np.random.choice(len(x_test_from), num_outputs, replace=False)
        batch_in = np.array([x_test_from[ind] for ind in batch_indices])
        true_batch_out = np.array([x_test_to[ind] for ind in batch_indices])

        best_guess = -1. * np.ones((num_outputs, self.max_length))

        for i in range(num_iterations):

            start = time.clock()

            best_guess = translate_fn(batch_in, best_guess)

            print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            print(' ')

            for ind_to_print in range(10):
                print('  in: ' + ' '.join([valid_vocab_from[j] for j in batch_in[ind_to_print] if j >= 0]))
                print('true: ' + ' '.join([valid_vocab_to[j] for j in true_batch_out[ind_to_print] if j >= 0]))
                print(' gen: ' + ' '.join([valid_vocab_to[j] for j in best_guess[ind_to_print] if j >= 0]))
                print(' ')

        out = OrderedDict()

        if sampling:
            suffix = str(from_index) + '_sampling'
        else:
            suffix = str(from_index)

        out['true_x_from_for_translation_from_' + suffix] = batch_in
        out['true_x_to_for_translation_from_' + suffix] = true_batch_out
        out['gen_x_to_for_translation_from_' + suffix] = best_guess

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)


class RunWordsSSL(object):

    def __init__(self, solver, solver_kwargs, valid_vocab_0, valid_vocab_1, main_dir, out_dir, dataset,
                 load_param_dir=None, pre_trained=False, train_prop=0.9):

        self.valid_vocab_0 = valid_vocab_0
        self.valid_vocab_1 = valid_vocab_1

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.vocab_size_0 = solver_kwargs['vocab_size_0']
        self.vocab_size_1 = solver_kwargs['vocab_size_1']

        print('loading data')

        start = time.clock()

        self.x_0_only_train, self.x_0_both_train, self.x_0_test, self.x_1_only_train, self.x_1_both_train, \
            self.x_1_test, self.L_0_only_train, self.L_0_both_train, self.L_0_test, self.L_1_only_train, \
            self.L_1_both_train, self.L_1_test = self.load_data(dataset, train_prop)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        print('# 0 only training sentences = ' + str(len(self.L_0_only_train)))
        print('# 1 only training sentences = ' + str(len(self.L_1_only_train)))
        print('# joint training sentences = ' + str(len(self.L_0_both_train)))
        print('# test sentences = ' + str(len(self.L_0_test)))

        self.max_length_0 = np.concatenate((self.x_0_only_train, self.x_0_both_train, self.x_0_test), axis=0).shape[1]
        self.max_length_1 = np.concatenate((self.x_1_only_train, self.x_1_both_train, self.x_1_test), axis=0).shape[1]

        self.max_length = max((self.max_length_0, self.max_length_1))

        self.vb = solver(max_length_0=self.max_length_0, max_length_1=self.max_length_1, **self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings_0.save'), 'rb') as f:
                self.vb.all_embeddings_0.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'all_embeddings_1.save'), 'rb') as f:
                self.vb.all_embeddings_1.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params_0.save'), 'rb') as f:
                self.vb.generative_model_0.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params_1.save'), 'rb') as f:
                self.vb.generative_model_1.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.vb.recognition_model.set_param_values(cPickle.load(f))

    def load_data(self, dataset, train_prop, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        files_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0')])
        files_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1')])

        words_0 = []
        words_1 = []

        for f in files_0:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words_0 += json.loads(s.read())

        for f in files_1:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words_1 += json.loads(s.read())

        L_0 = np.array([len(s) for s in words_0])
        L_1 = np.array([len(s) for s in words_1])

        max_L = max(np.concatenate((L_0, L_1)))

        word_arrays_0 = []

        start = time.clock()

        batches_0_loaded = 0

        for i in range(0, len(L_0), load_batch_size):

            L_i = L_0[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max(L_0))] = \
                np.concatenate(words_0[i: i+load_batch_size])

            word_arrays_0.append(word_array)

            del L_i, word_array

            batches_0_loaded += 1

            print(str(batches_0_loaded) + ' batches x_0 loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words_0

        words_to_return_0 = np.concatenate(word_arrays_0)

        word_arrays_1 = []

        start = time.clock()

        batches_1_loaded = 0

        for i in range(0, len(L_1), load_batch_size):

            L_i = L_1[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max(L_1))] = \
                np.concatenate(words_1[i: i+load_batch_size])

            word_arrays_1.append(word_array)

            del L_i, word_array

            batches_1_loaded += 1

            print(str(batches_1_loaded) + ' batches x_1 loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words_1

        words_to_return_1 = np.concatenate(word_arrays_1)

        np.random.seed(1234)

        training_mask = np.random.rand(len(words_to_return_0)) < train_prop

        x_0_train = words_to_return_0[training_mask]
        x_0_test = words_to_return_0[~training_mask]
        x_1_train = words_to_return_1[training_mask]
        x_1_test = words_to_return_1[~training_mask]

        L_0_train = L_0[training_mask]
        L_0_test = L_0[~training_mask]
        L_1_train = L_1[training_mask]
        L_1_test = L_1[~training_mask]

        n_training = len(L_0_train)
        n_only = int(train_prop*n_training)

        x_0_only_train = x_0_train[:n_only]
        x_0_both_train = x_0_train[n_only:]
        x_1_only_train = x_1_train[:n_only]
        x_1_both_train = x_1_train[n_only:]

        L_0_only_train = L_0_train[:n_only]
        L_0_both_train = L_0_train[n_only:]
        L_1_only_train = L_1_train[:n_only]
        L_1_both_train = L_1_train[n_only:]

        return x_0_only_train, x_0_both_train, x_0_test, x_1_only_train, x_1_both_train, x_1_test, L_0_only_train, \
            L_0_both_train, L_0_test, L_1_only_train, L_1_both_train, L_1_test

    def call_generate_output_prior(self, generate_output_prior):

        x_0_gen_beam, x_1_gen_beam = generate_output_prior()

        out = OrderedDict()

        out['generated_x_0_beam_prior'] = x_0_gen_beam
        out['generated_x_1_beam_prior'] = x_1_gen_beam

        return out

    def print_output_prior(self, output_prior):

        x_0_gen_beam = output_prior['generated_x_0_beam_prior']
        x_1_gen_beam = output_prior['generated_x_1_beam_prior']

        print('='*10)
        print('samples')
        print('='*10)

        for n in range(x_0_gen_beam.shape[0]):

            print('gen x_0 beam: ' + ' '.join([self.valid_vocab_0[int(i)] for i in x_0_gen_beam[n]]))
            print('gen x_1 beam: ' + ' '.join([self.valid_vocab_1[int(i)] for i in x_1_gen_beam[n]]))

            print('-'*10)

        print('='*10)

    def call_generate_output_posterior_only(self, generate_output_posterior, x_in, x_tgt, from_index):

        x_0_gen_beam, x_1_gen_beam = generate_output_posterior(x_in)

        out = OrderedDict()

        out['true_x_' + str(from_index) + '_for_posterior_' + str(from_index) + '_only'] = x_in
        out['true_x_' + str(1 - from_index) + '_for_posterior_' + str(from_index) + '_only'] = x_tgt
        out['generated_x_0_beam_posterior_' + str(from_index) + '_only'] = x_0_gen_beam
        out['generated_x_1_beam_posterior_' + str(from_index) + '_only'] = x_1_gen_beam

        return out

    def print_output_posterior_only(self, output_posterior, from_index):

        x_0 = output_posterior['true_x_0_for_posterior_' + str(from_index) + '_only']
        x_1 = output_posterior['true_x_1_for_posterior_' + str(from_index) + '_only']

        x_0_gen_beam = output_posterior['generated_x_0_beam_posterior_' + str(from_index) + '_only']
        x_1_gen_beam = output_posterior['generated_x_1_beam_posterior_' + str(from_index) + '_only']

        valid_vocab_0 = self.valid_vocab_0 + ['']
        valid_vocab_1 = self.valid_vocab_1 + ['']

        print('='*10)
        print('reconstructions ' + str(from_index) + ' only')
        print('='*10)

        for n in range(x_0.shape[0]):

            print('    true x_0: ' + ' '.join([valid_vocab_0[i] for i in x_0[n] if i >= 0]).strip())
            print('    true x_1: ' + ' '.join([valid_vocab_1[i] for i in x_1[n] if i >= 0]).strip())
            print('gen x_0 beam: ' + ' '.join([valid_vocab_0[int(i)] for i in x_0_gen_beam[n] if i >= 0]))
            print('gen x_1 beam: ' + ' '.join([valid_vocab_1[int(i)] for i in x_1_gen_beam[n] if i >= 0]))

            print('-'*10)

        print('='*10)

    def call_generate_output_posterior_both(self, generate_output_posterior, x_0, x_1):

        x_0_gen_beam, x_1_gen_beam = generate_output_posterior(x_0, x_1)

        out = OrderedDict()

        out['true_x_0_for_posterior'] = x_0
        out['true_x_1_for_posterior'] = x_1
        out['generated_x_0_beam_posterior'] = x_0_gen_beam
        out['generated_x_1_beam_posterior'] = x_1_gen_beam

        return out

    def print_output_posterior_both(self, output_posterior):

        x_0 = output_posterior['true_x_0_for_posterior']
        x_1 = output_posterior['true_x_1_for_posterior']

        x_0_gen_beam = output_posterior['generated_x_0_beam_posterior']
        x_1_gen_beam = output_posterior['generated_x_1_beam_posterior']

        valid_vocab_0 = self.valid_vocab_0 + ['']
        valid_vocab_1 = self.valid_vocab_1 + ['']

        print('='*10)
        print('reconstructions')
        print('='*10)

        for n in range(x_0.shape[0]):

            print('    true x_0: ' + ' '.join([valid_vocab_0[i] for i in x_0[n]]).strip())
            print('    true x_1: ' + ' '.join([valid_vocab_1[i] for i in x_1[n]]).strip())
            print('gen x_0 beam: ' + ' '.join([valid_vocab_0[int(i)] for i in x_0_gen_beam[n]]))
            print('gen x_1 beam: ' + ' '.join([valid_vocab_1[int(i)] for i in x_1_gen_beam[n]]))

            print('-'*10)

        print('='*10)

    def get_translation_fn(self, from_index, beam_size):

        return self.vb.translate_fn_sampling(from_index, beam_size, num_samples=10)

    def call_translation_fn(self, translation_fn, x_in, x_out_true, best_guess, from_index, num_iterations):

        if from_index == 0:
            valid_vocab_from = self.valid_vocab_0
            valid_vocab_to = self.valid_vocab_1

            # best_guess = self.call_generate_output_posterior_only(generate_output_posterior_fn,
            #                                                       x_test_from,
            #                                                       0)['generated_x_1_beam_posterior']
        else:
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

            # best_guess = self.call_generate_output_posterior_only(generate_output_posterior_fn,
            #                                                       x_test_from,
            #                                                       1)['generated_x_0_beam_posterior']


        # best_guess = -1. * np.ones((num_outputs, self.max_length))

        print('')

        for i in range(num_iterations):

            start = time.clock()

            best_guess = translation_fn(x_in, best_guess)

            print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            print(' ')
            print('  in: ' + ' '.join([valid_vocab_from[j] for j in x_in[0] if j >= 0]))
            print('true: ' + ' '.join([valid_vocab_to[j] for j in x_out_true[0] if j >= 0]))
            print(' gen: ' + ' '.join([valid_vocab_to[j] for j in best_guess[0] if j >= 0]))
            print(' ')

        print('')

        out = OrderedDict()

        out['true_x_from_for_translation_from_' + str(from_index)] = x_in
        out['true_x_to_for_translation_from_' + str(from_index)] = x_out_true
        out['gen_x_to_for_translation_from_' + str(from_index)] = best_guess

        return out

    def print_translations(self, from_index, translations):

        if from_index == 0:
            valid_vocab_from = self.valid_vocab_0
            valid_vocab_to = self.valid_vocab_1
        else:
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

        x_in = translations['true_x_from_for_translation_from_' + str(from_index)]
        x_out_true = translations['true_x_to_for_translation_from_' + str(from_index)]
        x_out_gen = translations['gen_x_to_for_translation_from_' + str(from_index)]

        print('='*10)
        print('translations ' + str(from_index) + ' to ' + str(1 - from_index))
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([valid_vocab_from[j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([valid_vocab_to[j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([valid_vocab_to[j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def train(self, n_iter, batch_size, num_samples, word_drop=None, grad_norm_constraint=None, update=adam,
              update_kwargs=None, warm_up=None, val_freq=None, val_batch_size=0, val_num_samples=0, val_print_gen=5,
              val_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                               update=update, update_kwargs=update_kwargs, saved_update=saved_update)

        elbo_fn = self.vb.elbo_fn_both(val_num_samples)

        # generate_output_prior = self.vb.generate_output_prior_fn(val_print_gen, val_beam_size)
        generate_output_posterior_0_only = self.vb.generate_output_posterior_fn_0_only(val_beam_size)
        generate_output_posterior_1_only = self.vb.generate_output_posterior_fn_1_only(val_beam_size)
        # generate_output_posterior_both = self.vb.generate_output_posterior_fn_both(val_beam_size)
        generate_translation_0 = self.get_translation_fn(0, val_beam_size)
        generate_translation_1 = self.get_translation_fn(1, val_beam_size)

        for i in range(n_iter):

            start = time.clock()

            batch_indices_0_only = np.random.choice(len(self.x_0_only_train), batch_size)
            batch_0_only = np.array([self.x_0_only_train[ind] for ind in batch_indices_0_only])

            batch_indices_1_only = np.random.choice(len(self.x_1_only_train), batch_size)
            batch_1_only = np.array([self.x_1_only_train[ind] for ind in batch_indices_1_only])

            batch_indices_both = np.random.choice(len(self.x_0_both_train), batch_size)
            batch_0_both = np.array([self.x_0_both_train[ind] for ind in batch_indices_both])
            batch_1_both = np.array([self.x_1_both_train[ind] for ind in batch_indices_both])

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            if word_drop is not None:

                L_0_only = np.array([self.L_0_only_train[ind] for ind in batch_indices_0_only])

                drop_indices_0_only = np.array([np.random.permutation(np.arange(i))[:int(np.floor(word_drop * i))]
                                                for i in L_0_only])

                drop_mask_0_only = np.ones_like(batch_0_only)

                for n in range(len(drop_indices_0_only)):
                    drop_mask_0_only[n][drop_indices_0_only[n]] = 0.

                L_1_only = np.array([self.L_1_only_train[ind] for ind in batch_indices_1_only])

                drop_indices_1_only = np.array([np.random.permutation(np.arange(i))[:int(np.floor(word_drop * i))]
                                                for i in L_1_only])

                drop_mask_1_only = np.ones_like(batch_1_only)

                for n in range(len(drop_indices_1_only)):
                    drop_mask_1_only[n][drop_indices_1_only[n]] = 0.

                L_0_both = np.array([self.L_0_both_train[ind] for ind in batch_indices_both])

                drop_indices_0_both = np.array([np.random.permutation(np.arange(i))[:int(np.floor(word_drop * i))]
                                                for i in L_0_both])

                drop_mask_0_both = np.ones_like(batch_0_both)

                for n in range(len(drop_indices_0_both)):
                    drop_mask_0_both[n][drop_indices_0_both[n]] = 0.

                L_1_both = np.array([self.L_1_both_train[ind] for ind in batch_indices_both])

                drop_indices_1_both = np.array([np.random.permutation(np.arange(i))[:int(np.floor(word_drop * i))]
                                                for i in L_1_both])

                drop_mask_1_both = np.ones_like(batch_1_both)

                for n in range(len(drop_indices_1_both)):
                    drop_mask_1_both[n][drop_indices_1_both[n]] = 0.

            else:

                drop_mask_0_only = np.ones_like(batch_0_only)
                drop_mask_1_only = np.ones_like(batch_1_only)
                drop_mask_0_both = np.ones_like(batch_0_both)
                drop_mask_1_both = np.ones_like(batch_1_both)

            elbo, kl_both = optimiser(batch_0_only, batch_1_only, batch_0_both, batch_1_both, beta, drop_mask_0_only,
                                      drop_mask_1_only, drop_mask_0_both, drop_mask_1_both)

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo/batch_size) + ' (KL (both) = ' +
                  str(kl_both/batch_size) + ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                val_batch_indices = np.random.choice(len(self.x_0_test), val_batch_size)
                val_batch_0 = np.array([self.x_0_test[ind] for ind in val_batch_indices])
                val_batch_1 = np.array([self.x_1_test[ind] for ind in val_batch_indices])

                val_elbo, val_kl_both = elbo_fn(val_batch_0, val_batch_1)

                print('Test set ELBO = ' + str(val_elbo/val_batch_size) + ' (KL (both) = ' + str(val_kl_both/batch_size)
                      + ') per data point')

                # output_prior = self.call_generate_output_prior(generate_output_prior)
                #
                # self.print_output_prior(output_prior)

                post_batch_indices = np.random.choice(len(self.x_0_test), val_print_gen, replace=False)
                post_batch_0 = np.array([self.x_0_test[ind] for ind in post_batch_indices])
                post_batch_1 = np.array([self.x_1_test[ind] for ind in post_batch_indices])

                output_posterior_0_only = self.call_generate_output_posterior_only(generate_output_posterior_0_only,
                                                                                   post_batch_0, post_batch_1, 0)
                self.print_output_posterior_only(output_posterior_0_only, 0)

                output_posterior_1_only = self.call_generate_output_posterior_only(generate_output_posterior_1_only,
                                                                                   post_batch_1, post_batch_0, 1)
                self.print_output_posterior_only(output_posterior_1_only, 1)

                # output_posterior_both = self.call_generate_output_posterior_both(generate_output_posterior_both,
                #                                                                  post_batch_0, post_batch_1)
                # self.print_output_posterior_both(output_posterior_both)

                translations_0 = self.call_translation_fn(generate_translation_0, post_batch_0, post_batch_1,
                                                          output_posterior_0_only['generated_x_1_beam_posterior_0_only'],
                                                          0, 5)
                self.print_translations(0, translations_0)

                translations_1 = self.call_translation_fn(generate_translation_1, post_batch_1, post_batch_0,
                                                          output_posterior_1_only['generated_x_0_beam_posterior_1_only'],
                                                          1, 5)
                self.print_translations(1, translations_1)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                with open(os.path.join(self.out_dir, 'all_embeddings_0.save'), 'wb') as f:
                    cPickle.dump(self.vb.all_embeddings_0.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'all_embeddings_1.save'), 'wb') as f:
                    cPickle.dump(self.vb.all_embeddings_1.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'gen_params_0.save'), 'wb') as f:
                    cPickle.dump(self.vb.generative_model_0.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'gen_params_1.save'), 'wb') as f:
                    cPickle.dump(self.vb.generative_model_1.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                    cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                    cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings_0.save'), 'wb') as f:
            cPickle.dump(self.vb.all_embeddings_0.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings_1.save'), 'wb') as f:
            cPickle.dump(self.vb.all_embeddings_1.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params_0.save'), 'wb') as f:
            cPickle.dump(self.vb.generative_model_0.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params_1.save'), 'wb') as f:
            cPickle.dump(self.vb.generative_model_1.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # def test(self, batch_size, num_samples, sub_sample_size=None):
    #
    #     elbo_fn = self.vb.elbo_fn(num_samples) if sub_sample_size is None else self.vb.elbo_fn(sub_sample_size)
    #
    #     elbo = 0
    #     kl = 0
    #     pp_0 = 0
    #     pp_1 = 0
    #
    #     batches_complete = 0
    #
    #     for batch_X in chunker([self.X_test], batch_size):
    #
    #         start = time.clock()
    #
    #         if sub_sample_size is None:
    #
    #             elbo_batch, kl_batch, pp_batch = self.call_elbo_fn(elbo_fn, batch_X[0])
    #
    #         else:
    #
    #             elbo_batch = 0
    #             kl_batch = 0
    #             pp_batch = 0
    #
    #             for sub_sample in range(1, int(num_samples / sub_sample_size) + 1):
    #
    #                 elbo_sub_batch, kl_sub_batch, pp_sub_batch = self.call_elbo_fn(elbo_fn, batch_X[0])
    #
    #                 elbo_batch = (elbo_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
    #                                             float(sub_sample * sub_sample_size))) + \
    #                              (elbo_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
    #
    #                 kl_batch = (kl_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
    #                                         float(sub_sample * sub_sample_size))) + \
    #                            (kl_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
    #
    #                 pp_batch = (pp_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
    #                                         float(sub_sample * sub_sample_size))) + \
    #                            (pp_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
    #
    #         elbo += elbo_batch
    #         kl += kl_batch
    #         pp += pp_batch
    #
    #         batches_complete += 1
    #
    #         print('Tested batches ' + str(batches_complete) + ' of ' + str(round(self.X_test.shape[0] / batch_size))
    #               + 'so far; test set ELBO = ' + str(elbo) + ', test set KL = ' + str(kl) + ', test set perplexity = '
    #               + str(pp) + ' / ' + str(elbo / (batches_complete * batch_size)) + ', '
    #               + str(kl / (batches_complete * batch_size)) + ', ' + str(pp / (batches_complete * batch_size))
    #               + ' per obs. (time taken = ' + str(time.clock() - start) + ' seconds)')
    #
    #     print('Test set ELBO = ' + str(elbo))
    #
    def generate_output(self, prior, posterior, num_outputs, beam_size=15):

        if prior:

            generate_output_prior = self.vb.generate_output_prior_fn(num_outputs, beam_size)

            output_prior = self.call_generate_output_prior(generate_output_prior)

            for key, value in output_prior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

        if posterior:

            np.random.seed(1234)

            generate_output_posterior_0_only = self.vb.generate_output_posterior_fn_0_only(beam_size)
            generate_output_posterior_1_only = self.vb.generate_output_posterior_fn_1_only(beam_size)
            generate_output_posterior_both = self.vb.generate_output_posterior_fn_both(beam_size)

            batch_indices = np.random.choice(len(self.x_0_test), num_outputs, replace=False)
            batch_0 = np.array([self.x_0_test[ind] for ind in batch_indices])
            batch_1 = np.array([self.x_1_test[ind] for ind in batch_indices])

            output_posterior_0_only = self.call_generate_output_posterior_only(generate_output_posterior_0_only,
                                                                               batch_0, batch_1, 0)

            for key, value in output_posterior_0_only.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

            output_posterior_1_only = self.call_generate_output_posterior_only(generate_output_posterior_1_only,
                                                                               batch_1, batch_0, 1)

            for key, value in output_posterior_1_only.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

            output_posterior_both = self.call_generate_output_posterior_both(generate_output_posterior_both, batch_0,
                                                                             batch_1)

            for key, value in output_posterior_both.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def translate(self, from_index, num_outputs, num_iterations, num_samples_per_output, sampling=True, beam_size=15):

        np.random.seed(1234)

        if sampling:
            translate_fn = self.vb.translate_fn_sampling(from_index, beam_size, num_samples=num_samples_per_output)
        else:
            translate_fn = self.vb.translate_fn(from_index, beam_size)

        if from_index == 0:
            x_test_from = self.x_0_test
            x_test_to = self.x_1_test
            valid_vocab_from = self.valid_vocab_0
            valid_vocab_to = self.valid_vocab_1

            generate_output_posterior_fn = self.vb.generate_output_posterior_fn_0_only(beam_size)

        else:
            x_test_from = self.x_1_test
            x_test_to = self.x_0_test
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

            generate_output_posterior_fn = self.vb.generate_output_posterior_fn_1_only(beam_size)

        batch_indices = np.random.choice(len(x_test_from), num_outputs, replace=False)
        batch_in = np.array([x_test_from[ind] for ind in batch_indices])
        true_batch_out = np.array([x_test_to[ind] for ind in batch_indices])

        best_guess = self.call_generate_output_posterior_only(generate_output_posterior_fn, batch_in, true_batch_out,
                                                              from_index)['generated_x_' + str(1 - from_index) +
                                                                          '_beam_posterior_' + str(from_index) +
                                                                          '_only']

        for i in range(num_iterations):

            start = time.clock()

            best_guess = translate_fn(batch_in, best_guess)

            print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            print(' ')

            for ind_to_print in range(10):
                print('  in: ' + ' '.join([valid_vocab_from[j] for j in batch_in[ind_to_print] if j >= 0]))
                print('true: ' + ' '.join([valid_vocab_to[j] for j in true_batch_out[ind_to_print] if j >= 0]))
                print(' gen: ' + ' '.join([valid_vocab_to[j] for j in best_guess[ind_to_print] if j >= 0]))
                print(' ')

        out = OrderedDict()

        if sampling:
            suffix = str(from_index) + '_sampling'
        else:
            suffix = str(from_index)

        out['true_x_from_for_translation_from_' + suffix] = batch_in
        out['true_x_to_for_translation_from_' + suffix] = true_batch_out
        out['gen_x_to_for_translation_from_' + suffix] = best_guess

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)