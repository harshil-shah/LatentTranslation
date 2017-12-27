from itertools import combinations, permutations
from collections import OrderedDict
import os
import pickle as cPickle
import time
import numpy as np
import json
from lasagne.updates import adam
from data_processing.utilities import chunker


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

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, train_prop, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        files_0_both = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0_both')])
        files_0_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0_test')])
        files_1_both = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1_both')])
        files_1_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1_test')])

        print('loading 0 both')
        x_0_both_train, L_0_both_train = self.load_words(folder, files_0_both, load_batch_size)
        print('loading 0 test')
        x_0_test, L_0_test = self.load_words(folder, files_0_test, load_batch_size)
        print('loading 1 both')
        x_1_both_train, L_1_both_train = self.load_words(folder, files_1_both, load_batch_size)
        print('loading 1 test')
        x_1_test, L_1_test = self.load_words(folder, files_1_test, load_batch_size)

        return x_0_both_train, x_0_test, x_1_both_train, x_1_test, L_0_both_train, L_0_test, L_1_both_train, L_1_test

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

        return self.vb.translate_fn_sampling(from_index, beam_size, num_samples=10)

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

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, train_prop, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        files_0_only = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0_only')])
        files_0_both = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0_both')])
        files_0_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_0_test')])
        files_1_only = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1_only')])
        files_1_both = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1_both')])
        files_1_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_1_test')])

        print('loading 0 only')
        x_0_only_train, L_0_only_train = self.load_words(folder, files_0_only, load_batch_size)
        print('loading 0 both')
        x_0_both_train, L_0_both_train = self.load_words(folder, files_0_both, load_batch_size)
        print('loading 0 test')
        x_0_test, L_0_test = self.load_words(folder, files_0_test, load_batch_size)
        print('loading 1 only')
        x_1_only_train, L_1_only_train = self.load_words(folder, files_1_only, load_batch_size)
        print('loading 1 both')
        x_1_both_train, L_1_both_train = self.load_words(folder, files_1_both, load_batch_size)
        print('loading 1 test')
        x_1_test, L_1_test = self.load_words(folder, files_1_test, load_batch_size)

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
        else:
            valid_vocab_from = self.valid_vocab_1
            valid_vocab_to = self.valid_vocab_0

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

    def train(self, n_iter, only_batch_size, both_batch_size, num_samples, word_drop=None, grad_norm_constraint=None,
              update=adam, update_kwargs=None, warm_up=None, val_freq=None, val_batch_size=0, val_num_samples=0,
              val_print_gen=5, val_beam_size=15, save_params_every=None):

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

            batch_indices_0_only = np.random.choice(len(self.x_0_only_train), only_batch_size)
            batch_0_only = np.array([self.x_0_only_train[ind] for ind in batch_indices_0_only])

            batch_indices_1_only = np.random.choice(len(self.x_1_only_train), only_batch_size)
            batch_1_only = np.array([self.x_1_only_train[ind] for ind in batch_indices_1_only])

            batch_indices_both = np.random.choice(len(self.x_0_both_train), both_batch_size)
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

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo) + ' (KL (both) = ' + str(kl_both)
                  + ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                val_batch_indices = np.random.choice(len(self.x_0_test), val_batch_size)
                val_batch_0 = np.array([self.x_0_test[ind] for ind in val_batch_indices])
                val_batch_1 = np.array([self.x_1_test[ind] for ind in val_batch_indices])

                val_elbo, val_kl_both = elbo_fn(val_batch_0, val_batch_1)

                print('Test set ELBO = ' + str(val_elbo) + ' (KL (both) = ' + str(val_kl_both) + ') per data point')

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


class RunWordsMulti(object):

    def __init__(self, solver, solver_kwargs, langs, valid_vocabs, main_dir, out_dir, dataset, load_param_dir=None,
                 pre_trained=False):

        self.langs = langs  # e.g. {0: 'en', 1: 'es', 2: 'fr'}
        self.num_langs = len(langs)
        self.valid_vocabs = valid_vocabs  # e.g. {0: [...], 1: [...], 2:[...]}

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.max_length = solver_kwargs['max_length']
        self.vocab_size = solver_kwargs['vocab_size']

        print('loading data')

        start = time.clock()

        self.x_both, self.x_test, self.L_both, self.L_test = self.load_data(dataset)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' = ' + str(len(self.L_both[pair][0])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' test = ' + str(len(self.L_test[pair][0])))

        self.vb = solver(**self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings.save'), 'rb') as f:
                self.vb.set_embedding_vals(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
                self.vb.set_generative_models_params_vals(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.vb.recognition_model.set_param_values(cPickle.load(f))

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        x_both = {}
        L_both = {}

        x_test = {}
        L_test = {}

        for pair in combinations(range(self.num_langs), 2):

            print('loading ' + str(pair))

            l_0 = self.langs[pair[0]]
            l_1 = self.langs[pair[1]]

            files_both_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and not f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_0, L_both_l_0 = self.load_words(folder, files_both_l_0, load_batch_size)

            files_both_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and not f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_1, L_both_l_1 = self.load_words(folder, files_both_l_1, load_batch_size)

            x_both[pair] = (x_both_l_0, x_both_l_1)
            L_both[pair] = (L_both_l_0, L_both_l_1)

            files_test_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_0, L_test_l_0 = self.load_words(folder, files_test_l_0, load_batch_size)

            files_test_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_1, L_test_l_1 = self.load_words(folder, files_test_l_1, load_batch_size)

            x_test[pair] = (x_test_l_0, x_test_l_1)
            L_test[pair] = (L_test_l_0, L_test_l_1)

        return x_both, x_test, L_both, L_test

    def call_generate_output_prior(self, generate_output_prior):

        outputs = generate_output_prior()

        out = OrderedDict()

        for l in range(self.num_langs):

            out['generated_x_' + self.langs[l] + '_beam_prior'] = outputs[l]

        return out

    def print_output_prior(self, output_prior):

        print('='*10)
        print('samples')
        print('='*10)

        for n in range(output_prior['generated_x_' + self.langs[0] + '_beam_prior'].shape[0]):

            for l in range(self.num_langs):

                x_l = output_prior['generated_x_' + self.langs[l] + '_beam_prior']

                print('gen x_' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][int(i)] for i in x_l[n]
                                                                  if i >= 0]))

            print('-'*10)

        print('='*10)

    def call_generate_output_posterior_both(self, generate_output_posterior, x, l_0, l_1):

        outputs = generate_output_posterior(x[l_0], x[l_1])

        out = OrderedDict()

        for l in range(self.num_langs):

            out['true_x_' + self.langs[l] + '_for_posterior_' + self.langs[l_0] + '_' + self.langs[l_1] +
                '_only'] = x[l]
            out['generated_x_' + self.langs[l] + '_beam_posterior_' + self.langs[l_0] + '_' + self.langs[l_1] +
                '_only'] = outputs[l]

        return out

    def print_output_posterior_both(self, output_posterior, l_0, l_1):

        print('='*10)
        print('reconstructions ' + str(l_0) + '_' + str(l_1) + '_only')
        print('='*10)

        for n in range(output_posterior['true_x_' + self.langs[0] + '_for_posterior_' + self.langs[l_0] + '_' +
                self.langs[l_1] + '_only'].shape[0]):

            for l in range(self.num_langs):

                true_x_l = output_posterior['true_x_' + self.langs[l] + '_for_posterior_' + self.langs[l_0] + '_' +
                                            self.langs[l_1] + '_only']
                gen_x_l = output_posterior['generated_x_' + self.langs[l] + '_beam_posterior_' + self.langs[l_0] + '_' +
                                           self.langs[l_1] + '_only']

                print('true x ' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][i] for i in true_x_l[n]
                                                                   if i >= 0]))
                print(' gen x ' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][int(i)] for i in gen_x_l[n]
                                                                   if i >= 0]))

            print('-'*10)

        print('='*10)

    def get_translation_fn(self, l_from, l_to, beam_size):

        return self.vb.translate_fn_sampling(l_from, l_to, beam_size, num_samples=10)

    def call_translation_fn(self, translation_fn, l_from, l_to, x_in, x_out_true, best_guess, num_iterations):

        # print('='*10)
        # print('translating from  ' + self.langs[l_from] + ' to ' + self.langs[l_to])
        # print('='*10)

        for i in range(num_iterations):

            # start = time.clock()

            best_guess = translation_fn(x_in, best_guess)

            # print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            # print(' ')
            # print('  in: ' + ' '.join([self.valid_vocabs[l_from][j] for j in x_in[0] if j >= 0]))
            # print('true: ' + ' '.join([self.valid_vocabs[l_to][j] for j in x_out_true[0] if j >= 0]))
            # print(' gen: ' + ' '.join([self.valid_vocabs[l_to][j] for j in best_guess[0] if j >= 0]))
            # print(' ')

        # print('')

        out = OrderedDict()

        out['true_x_from_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]] = x_in
        out['true_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]] = x_out_true
        out['gen_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]] = best_guess

        return out

    def print_translations(self, l_from, l_to, translations):

        x_in = translations['true_x_from_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]]
        x_out_true = translations['true_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]]
        x_out_gen = translations['gen_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]]

        print('='*10)
        print('translations ' + self.langs[l_from] + ' to ' + self.langs[l_to])
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([self.valid_vocabs[l_from][j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([self.valid_vocabs[l_to][j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([self.valid_vocabs[l_to][j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def train(self, n_iter, only_batch_size, both_batch_size, num_samples, word_drop=None, grad_norm_constraint=None,
              update=adam, update_kwargs=None, warm_up=None, val_freq=None, val_batch_size=0, val_num_samples=0,
              val_print_gen=5, val_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                               update=update, update_kwargs=update_kwargs, saved_update=saved_update)

        elbo_fns = {l: self.vb.elbo_fn(l[0], l[1], val_num_samples) for l in combinations(range(self.num_langs), 2)}

        generate_output_prior_fn = self.vb.generate_output_prior_fn(val_print_gen, val_beam_size)
        generate_translation_fns = {l: self.get_translation_fn(l[0], l[1], val_beam_size)
                                    for l in permutations(range(self.num_langs), 2)}

        for i in range(n_iter):

            start = time.clock()

            batches_both = []
            drop_masks_both = []

            for l in combinations(range(self.num_langs), 2):

                batch_indices_l_both = np.random.choice(len(self.x_both[l][0]), both_batch_size)
                batch_both_l_0 = self.x_both[l][0][batch_indices_l_both]
                batch_both_l_1 = self.x_both[l][1][batch_indices_l_both]
                batches_both.append(batch_both_l_0)
                batches_both.append(batch_both_l_1)

                if word_drop is not None:
                    L_both_l_0 = self.L_both[l][0][batch_indices_l_both]

                    drop_indices_both_l_0 = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))]
                                                      for j in L_both_l_0])

                    drop_mask_both_l_0 = np.ones_like(batch_both_l_0)

                    for n in range(len(drop_indices_both_l_0)):
                        drop_mask_both_l_0[n][drop_indices_both_l_0[n]] = 0.

                    L_both_l_1 = self.L_both[l][1][batch_indices_l_both]

                    drop_indices_both_l_1 = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))]
                                                      for j in L_both_l_1])

                    drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                    for n in range(len(drop_indices_both_l_1)):
                        drop_mask_both_l_1[n][drop_indices_both_l_1[n]] = 0.
                else:
                    drop_mask_both_l_0 = np.ones_like(batch_both_l_0)
                    drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                drop_masks_both.append(drop_mask_both_l_0)
                drop_masks_both.append(drop_mask_both_l_1)

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            optimiser_args = batches_both + [beta] + drop_masks_both

            elbo, kl_both = optimiser(*optimiser_args)

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo) + ' (KL (both) = ' + str(kl_both)
                  + ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                for l in combinations(range(self.num_langs), 2):

                    val_batch_indices_l = np.random.choice(len(self.x_test[l][0]), val_batch_size)
                    val_batch_l_0 = self.x_test[l][0][val_batch_indices_l]
                    val_batch_l_1 = self.x_test[l][1][val_batch_indices_l]

                    val_elbo_l, val_kl_both_l = elbo_fns[l](val_batch_l_0, val_batch_l_1)

                    print('Test set ELBO ' + self.langs[l[0]] + ' and ' + self.langs[l[1]] + ' = ' + str(val_elbo_l) +
                          ' (KL (both) = ' + str(val_kl_both_l) + ') per data point')

                output_prior = self.call_generate_output_prior(generate_output_prior_fn)

                self.print_output_prior(output_prior)

                post_batches_both = {}

                for l in combinations(range(self.num_langs), 2):

                    post_batch_indices_l = np.random.choice(len(self.x_test[l][0]), val_print_gen)
                    post_batch_l_0 = self.x_test[l][0][post_batch_indices_l]
                    post_batch_l_1 = self.x_test[l][1][post_batch_indices_l]

                    post_batches_both[l] = [post_batch_l_0, post_batch_l_1]

                for l in combinations(range(self.num_langs), 2):

                    translations_l_0_to_l_1 = self.call_translation_fn(
                        generate_translation_fns[(l[0], l[1])], l[0], l[1], post_batches_both[(l[0], l[1])][0],
                        post_batches_both[(l[0], l[1])][1], -np.ones_like(post_batches_both[(l[0], l[1])][1]), 5
                    )

                    self.print_translations(l[0], l[1], translations_l_0_to_l_1)

                    translations_l_1_to_l_0 = self.call_translation_fn(
                        generate_translation_fns[(l[1], l[0])], l[1], l[0], post_batches_both[(l[0], l[1])][1],
                        post_batches_both[(l[0], l[1])][0], -np.ones_like(post_batches_both[(l[0], l[1])][0]), 5
                    )

                    self.print_translations(l[1], l[0], translations_l_1_to_l_0)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                with open(os.path.join(self.out_dir, 'all_embeddings.save'), 'wb') as f:
                    cPickle.dump([self.vb.all_embeddings[l].get_value() for l in range(self.num_langs)], f,
                                 protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
                    cPickle.dump([self.vb.generative_models[l].get_param_values() for l in range(self.num_langs)], f,
                                 protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                    cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                    cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings.save'), 'wb') as f:
            cPickle.dump([self.vb.all_embeddings[l].get_value() for l in range(self.num_langs)], f,
                         protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            cPickle.dump([self.vb.generative_models[l].get_param_values() for l in range(self.num_langs)], f,
                         protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def translate(self, num_outputs, num_iterations, num_samples_per_output, sampling=True, beam_size=15):

        np.random.seed(1234)

        batches = {}

        for l in combinations(range(self.num_langs), 2):

            batch_indices_l = np.random.choice(len(self.x_test[l][0]), num_outputs)
            batch_l_0 = self.x_test[l][0][batch_indices_l]
            batch_l_1 = self.x_test[l][1][batch_indices_l]

            batches[l] = [batch_l_0, batch_l_1]

        for l in permutations(range(self.num_langs), 2):

            if sampling:
                generate_translation_fn = self.vb.translate_fn_sampling(l[0], l[1], beam_size,
                                                                        num_samples=num_samples_per_output)
            else:
                generate_translation_fn = self.vb.translate_fn(l[0], l[1], beam_size)

            batch_in = batches[(min(l), max(l))][0]
            true_batch_out = batches[(min(l), max(l))][1]

            best_guess = -np.ones_like(true_batch_out)

            for i in range(num_iterations):

                start = time.clock()

                best_guess = generate_translation_fn(batch_in, best_guess)

                print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
                print(' ')

                for ind_to_print in range(10):
                    print('  in: ' + ' '.join([self.valid_vocabs[l[0]][j] for j in batch_in[ind_to_print] if j >= 0]))
                    print('true: ' + ' '.join([self.valid_vocabs[l[1]][j] for j in true_batch_out[ind_to_print]
                                               if j >= 0]))
                    print(' gen: ' + ' '.join([self.valid_vocabs[l[1]][j] for j in best_guess[ind_to_print] if j >= 0]))
                    print(' ')

            out = OrderedDict()

            if sampling:
                suffix = '_for_translation_from_' + self.langs[l[0]] + '_to_' + self.langs[l[1]] + '_sampling'
            else:
                suffix = '_for_translation_from_' + self.langs[l[0]] + '_to_' + self.langs[l[1]]

            out['true_x_from' + suffix] = batch_in
            out['true_x_to' + suffix] = true_batch_out
            out['gen_x_to' + suffix] = best_guess

            for key, value in out.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)


class RunWordsSSLMulti(object):

    def __init__(self, solver, solver_kwargs, langs, valid_vocabs, main_dir, out_dir, dataset, load_param_dir=None,
                 pre_trained=False):

        self.langs = langs  # e.g. {0: 'en', 1: 'es', 2: 'fr'}
        self.num_langs = len(langs)
        self.valid_vocabs = valid_vocabs  # e.g. {0: [...], 1: [...], 2:[...]}

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.max_length = solver_kwargs['max_length']
        self.vocab_size = solver_kwargs['vocab_size']

        print('loading data')

        start = time.clock()

        self.x_only, self.x_both, self.x_test, self.L_only, self.L_both, self.L_test = self.load_data(dataset)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        for l in range(self.num_langs):
            print('num sentences ' + self.langs[l] + ' only = ' + str(len(self.L_only[l])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' = ' + str(len(self.L_both[pair][0])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' test = ' + str(len(self.L_test[pair][0])))

        self.vb = solver(**self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings.save'), 'rb') as f:
                self.vb.set_embedding_vals(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
                self.vb.set_generative_models_params_vals(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.vb.recognition_model.set_param_values(cPickle.load(f))

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        x_only = {}
        L_only = {}

        for l in range(self.num_langs):

            print('loading ' + self.langs[l] + ' only')

            files_only_l = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.langs[l] +
                                                                                 '_only')])

            x_only_l, L_only_l = self.load_words(folder, files_only_l, load_batch_size)

            x_only[l] = x_only_l
            L_only[l] = L_only_l

        x_both = {}
        L_both = {}

        x_test = {}
        L_test = {}

        for pair in combinations(range(self.num_langs), 2):

            print('loading ' + str(pair))

            l_0 = self.langs[pair[0]]
            l_1 = self.langs[pair[1]]

            files_both_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and not f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_0, L_both_l_0 = self.load_words(folder, files_both_l_0, load_batch_size)

            files_both_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and not f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_1, L_both_l_1 = self.load_words(folder, files_both_l_1, load_batch_size)

            x_both[pair] = (x_both_l_0, x_both_l_1)
            L_both[pair] = (L_both_l_0, L_both_l_1)

            files_test_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_0, L_test_l_0 = self.load_words(folder, files_test_l_0, load_batch_size)

            files_test_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_1, L_test_l_1 = self.load_words(folder, files_test_l_1, load_batch_size)

            x_test[pair] = (x_test_l_0, x_test_l_1)
            L_test[pair] = (L_test_l_0, L_test_l_1)

        return x_only, x_both, x_test, L_only, L_both, L_test

    # def call_generate_output_prior(self, generate_output_prior):
    #
    #     outputs = generate_output_prior()
    #
    #     out = OrderedDict()
    #
    #     for l in range(self.num_langs):
    #
    #         out['generated_x_' + self.langs[l] + '_beam_prior'] = outputs[l]
    #
    #     return out
    #
    # def print_output_prior(self, output_prior):
    #
    #     print('='*10)
    #     print('samples')
    #     print('='*10)
    #
    #     for n in range(output_prior['generated_x_' + self.langs[0] + '_beam_prior'].shape[0]):
    #
    #         for l in range(self.num_langs):
    #
    #             x_l = output_prior['generated_x_' + self.langs[l] + '_beam_prior']
    #
    #             print('gen x_' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][int(i)] for i in x_l[n]
    #                                                               if i >= 0]))
    #
    #         print('-'*10)
    #
    #     print('='*10)

    def call_generate_output_posterior_only(self, generate_output_posterior, x, l_in):

        outputs = generate_output_posterior(x)

        out = OrderedDict()

        out['true_x_for_posterior_' + self.langs[l_in] + '_only'] = x

        for l in range(self.num_langs):

            out['generated_x_' + self.langs[l] + '_beam_posterior_' + self.langs[l_in] + '_only'] = outputs[l]

        return out

    def print_output_posterior_only(self, output_posterior, l_in):

        print('='*10)
        print('reconstructions ' + self.langs[l_in] + ' only')
        print('='*10)

        for n in range(output_posterior['true_x_for_posterior_' + self.langs[l_in] + '_only'].shape[0]):

            true_x = output_posterior['true_x_for_posterior_' + self.langs[l_in] + '_only']

            print('true x ' + self.langs[l_in] + ': ' + ' '.join([self.valid_vocabs[l_in][i] for i in true_x[n]
                                                                  if i >= 0]))

            for l in range(self.num_langs):

                gen_x_l = output_posterior['generated_x_' + self.langs[l] + '_beam_posterior_' + self.langs[l_in] +
                                           '_only']

                print(' gen x ' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][int(i)] for i in gen_x_l[n]
                                                                   if i >= 0]))

            print('-'*10)

        print('='*10)

    def call_generate_output_posterior_both(self, generate_output_posterior, x, l_0, l_1):

        outputs = generate_output_posterior(x[l_0], x[l_1])

        out = OrderedDict()

        for l in range(self.num_langs):

            out['true_x_' + self.langs[l] + '_for_posterior_' + self.langs[l_0] + '_' + self.langs[l_1] +
                '_only'] = x[l]
            out['generated_x_' + self.langs[l] + '_beam_posterior_' + self.langs[l_0] + '_' + self.langs[l_1] +
                '_only'] = outputs[l]

        return out

    def print_output_posterior_both(self, output_posterior, l_0, l_1):

        print('='*10)
        print('reconstructions ' + str(l_0) + '_' + str(l_1) + '_only')
        print('='*10)

        for n in range(output_posterior['true_x_' + self.langs[0] + '_for_posterior_' + self.langs[l_0] + '_' +
                                        self.langs[l_1] + '_only'].shape[0]):

            for l in range(self.num_langs):

                true_x_l = output_posterior['true_x_' + self.langs[l] + '_for_posterior_' + self.langs[l_0] + '_' +
                                            self.langs[l_1] + '_only']
                gen_x_l = output_posterior['generated_x_' + self.langs[l] + '_beam_posterior_' + self.langs[l_0] + '_' +
                                           self.langs[l_1] + '_only']

                print('true x ' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][i] for i in true_x_l[n]
                                                                   if i >= 0]))
                print(' gen x ' + self.langs[l] + ': ' + ' '.join([self.valid_vocabs[l][int(i)] for i in gen_x_l[n]
                                                                   if i >= 0]))

            print('-'*10)

        print('='*10)

    def get_translation_fn(self, l_from, l_to, beam_size):

        return self.vb.translate_fn_sampling(l_from, l_to, beam_size, num_samples=10)

    def call_translation_fn(self, translation_fn, l_from, l_to, x_in, x_out_true, best_guess, num_iterations):

        # print('='*10)
        # print('translating from  ' + self.langs[l_from] + ' to ' + self.langs[l_to])
        # print('='*10)

        for i in range(num_iterations):

            # start = time.clock()

            best_guess = translation_fn(x_in, best_guess)

            # print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            # print(' ')
            # print('  in: ' + ' '.join([self.valid_vocabs[l_from][j] for j in x_in[0] if j >= 0]))
            # print('true: ' + ' '.join([self.valid_vocabs[l_to][j] for j in x_out_true[0] if j >= 0]))
            # print(' gen: ' + ' '.join([self.valid_vocabs[l_to][j] for j in best_guess[0] if j >= 0]))
            # print(' ')

        # print('')

        out = OrderedDict()

        out['true_x_from_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]] = x_in
        out['true_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]] = x_out_true
        out['gen_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]] = best_guess

        return out

    def print_translations(self, l_from, l_to, translations):

        x_in = translations['true_x_from_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]]
        x_out_true = translations['true_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]]
        x_out_gen = translations['gen_x_to_for_translation_from_' + self.langs[l_from] + ' to ' + self.langs[l_to]]

        print('='*10)
        print('translations ' + self.langs[l_from] + ' to ' + self.langs[l_to])
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([self.valid_vocabs[l_from][j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([self.valid_vocabs[l_to][j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([self.valid_vocabs[l_to][j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def train(self, n_iter, only_batch_size, both_batch_size, num_samples, word_drop=None, grad_norm_constraint=None,
              update=adam, update_kwargs=None, warm_up=None, val_freq=None, val_batch_size=0, val_num_samples=0,
              val_print_gen=5, val_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                               update=update, update_kwargs=update_kwargs, saved_update=saved_update)

        elbo_fns = {l: self.vb.elbo_fn(l[0], l[1], val_num_samples) for l in combinations(range(self.num_langs), 2)}

        # generate_output_prior_fn = self.vb.generate_output_prior_fn(val_print_gen, val_beam_size)
        generate_output_posterior_only_fns = {l: self.vb.generate_output_posterior_fn_only(l, val_beam_size)
                                              for l in range(self.num_langs)}
        generate_output_posterior_both_fns = {l: self.vb.generate_output_posterior_fn_both(l[0], l[1], val_beam_size)
                                              for l in combinations(range(self.num_langs), 2)}
        generate_translation_fns = {l: self.get_translation_fn(l[0], l[1], val_beam_size)
                                    for l in permutations(range(self.num_langs), 2)}

        for i in range(n_iter):

            start = time.clock()

            batches_only = []
            drop_masks_only = []

            for l in range(self.num_langs):

                batch_indices_l = np.random.choice(len(self.x_only[l]), only_batch_size)
                batch_only_l = self.x_only[l][batch_indices_l]
                batches_only.append(batch_only_l)

                if word_drop is not None:
                    L_only_l = self.L_only[l][batch_indices_l]

                    drop_indices_only_l = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))]
                                                    for j in L_only_l])

                    drop_mask_only_l = np.ones_like(batch_only_l)

                    for n in range(len(drop_indices_only_l)):
                        drop_mask_only_l[n][drop_indices_only_l[n]] = 0.
                else:
                    drop_mask_only_l = np.ones_like(batch_only_l)

                drop_masks_only.append(drop_mask_only_l)

            batches_both = []
            drop_masks_both = []

            for l in combinations(range(self.num_langs), 2):

                batch_indices_l_both = np.random.choice(len(self.x_both[l][0]), both_batch_size)
                batch_both_l_0 = self.x_both[l][0][batch_indices_l_both]
                batch_both_l_1 = self.x_both[l][1][batch_indices_l_both]
                batches_both.append(batch_both_l_0)
                batches_both.append(batch_both_l_1)

                if word_drop is not None:
                    L_both_l_0 = self.L_both[l][0][batch_indices_l_both]

                    drop_indices_both_l_0 = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))]
                                                    for j in L_both_l_0])

                    drop_mask_both_l_0 = np.ones_like(batch_both_l_0)

                    for n in range(len(drop_indices_both_l_0)):
                        drop_mask_both_l_0[n][drop_indices_both_l_0[n]] = 0.

                    L_both_l_1 = self.L_both[l][1][batch_indices_l_both]

                    drop_indices_both_l_1 = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))]
                                                      for j in L_both_l_1])

                    drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                    for n in range(len(drop_indices_both_l_1)):
                        drop_mask_both_l_1[n][drop_indices_both_l_1[n]] = 0.
                else:
                    drop_mask_both_l_0 = np.ones_like(batch_both_l_0)
                    drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                drop_masks_both.append(drop_mask_both_l_0)
                drop_masks_both.append(drop_mask_both_l_1)

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            optimiser_args = batches_only + batches_both + [beta] + drop_masks_only + drop_masks_both

            elbo, kl_both = optimiser(*optimiser_args)

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo) + ' (KL (both) = ' + str(kl_both)
                  + ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                for l in combinations(range(self.num_langs), 2):

                    val_batch_indices_l = np.random.choice(len(self.x_test[l][0]), val_batch_size)
                    val_batch_l_0 = self.x_test[l][0][val_batch_indices_l]
                    val_batch_l_1 = self.x_test[l][1][val_batch_indices_l]

                    val_elbo_l, val_kl_both_l = elbo_fns[l](val_batch_l_0, val_batch_l_1)

                    print('Test set ELBO ' + self.langs[l[0]] + ' and ' + self.langs[l[1]] + ' = ' + str(val_elbo_l) +
                          ' (KL (both) = ' + str(val_kl_both_l) + ') per data point')

                # output_prior = self.call_generate_output_prior(generate_output_prior_fn)
                #
                # self.print_output_prior(output_prior)

                post_batches_only = {}
                post_batches_both = {}

                for l in combinations(range(self.num_langs), 2):

                    post_batch_indices_l = np.random.choice(len(self.x_test[l][0]), val_print_gen)
                    post_batch_l_0 = self.x_test[l][0][post_batch_indices_l]
                    post_batch_l_1 = self.x_test[l][1][post_batch_indices_l]

                    post_batches_both[l] = [post_batch_l_0, post_batch_l_1]

                    if l[0] not in post_batches_only.keys():
                        post_batches_only[l[0]] = post_batch_l_0

                    if l[1] not in post_batches_only.keys():
                        post_batches_only[l[1]] = post_batch_l_1

                output_posterior_only = {}

                for l in range(self.num_langs):

                    output_posterior_only_l = self.call_generate_output_posterior_only(
                        generate_output_posterior_only_fns[l], post_batches_only[l], l
                    )

                    output_posterior_only[l] = output_posterior_only_l

                    self.print_output_posterior_only(output_posterior_only_l, l)

                for l in combinations(range(self.num_langs), 2):

                    translations_l_0_to_l_1 = self.call_translation_fn(
                        generate_translation_fns[(l[0], l[1])], l[0], l[1], post_batches_both[(l[0], l[1])][0],
                        post_batches_both[(l[0], l[1])][1],
                        output_posterior_only[l[0]]['generated_x_' + self.langs[l[1]] + '_beam_posterior_' +
                                                    self.langs[l[0]] + '_only'], 5
                    )

                    self.print_translations(l[0], l[1], translations_l_0_to_l_1)

                    translations_l_1_to_l_0 = self.call_translation_fn(
                        generate_translation_fns[(l[1], l[0])], l[1], l[0], post_batches_both[(l[0], l[1])][1],
                        post_batches_both[(l[0], l[1])][0],
                        output_posterior_only[l[1]]['generated_x_' + self.langs[l[0]] + '_beam_posterior_' +
                                                    self.langs[l[1]] + '_only'], 5
                    )

                    self.print_translations(l[1], l[0], translations_l_1_to_l_0)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                with open(os.path.join(self.out_dir, 'all_embeddings.save'), 'wb') as f:
                    cPickle.dump([self.vb.all_embeddings[l].get_value() for l in range(self.num_langs)], f,
                                 protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
                    cPickle.dump([self.vb.generative_models[l].get_param_values() for l in range(self.num_langs)], f,
                                 protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                    cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                    cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings.save'), 'wb') as f:
            cPickle.dump([self.vb.all_embeddings[l].get_value() for l in range(self.num_langs)], f,
                         protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            cPickle.dump([self.vb.generative_models[l].get_param_values() for l in range(self.num_langs)], f,
                         protocol=cPickle.HIGHEST_PROTOCOL)

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

    def generate_output(self, prior, posterior, num_outputs, beam_size=15):

        # if prior:
        #
        #     generate_output_prior = self.vb.generate_output_prior_fn(num_outputs, beam_size)
        #
        #     output_prior = self.call_generate_output_prior(generate_output_prior)
        #
        #     for key, value in output_prior.items():
        #         np.save(os.path.join(self.out_dir, key + '.npy'), value)

        if posterior:

            np.random.seed(1234)

            generate_output_posterior_only_fns = {l: self.vb.generate_output_posterior_fn_only(l, beam_size)
                                                  for l in range(self.num_langs)}

            post_batches_only = {}

            for l in combinations(range(self.num_langs), 2):

                post_batch_indices_l = np.random.choice(len(self.x_test[l][0]), num_outputs, replace=False)
                post_batch_l_0 = self.x_test[l][0][post_batch_indices_l]
                post_batch_l_1 = self.x_test[l][1][post_batch_indices_l]

                if l[0] not in post_batches_only.keys():
                    post_batches_only[l[0]] = post_batch_l_0

                if l[1] not in post_batches_only.keys():
                    post_batches_only[l[1]] = post_batch_l_1

            for l in range(self.num_langs):

                output_posterior_only_l = self.call_generate_output_posterior_only(
                    generate_output_posterior_only_fns[l], post_batches_only[l], l
                )

                for key, value in output_posterior_only_l.items():
                    np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def translate(self, num_outputs, num_iterations, num_samples_per_output, sampling=True, beam_size=15):

        np.random.seed(1234)

        batches = {}

        for l in combinations(range(self.num_langs), 2):

            batch_indices_l = np.random.choice(len(self.x_test[l][0]), num_outputs)
            batch_l_0 = self.x_test[l][0][batch_indices_l]
            batch_l_1 = self.x_test[l][1][batch_indices_l]

            batches[l] = [batch_l_0, batch_l_1]

        generate_output_posterior_fns = {l: self.vb.generate_output_posterior_fn_only(l, beam_size)
                                         for l in range(self.num_langs)}

        for l in permutations(range(self.num_langs), 2):

            generate_output_posterior_fn = generate_output_posterior_fns[l[0]]

            if sampling:
                generate_translation_fn = self.vb.translate_fn_sampling(l[0], l[1], beam_size,
                                                                        num_samples=num_samples_per_output)
            else:
                generate_translation_fn = self.vb.translate_fn(l[0], l[1], beam_size)

            batch_in = batches[(min(l), max(l))][0]
            true_batch_out = batches[(min(l), max(l))][1]

            output_posterior = self.call_generate_output_posterior_only(generate_output_posterior_fn, batch_in, l[0])

            best_guess = output_posterior['generated_x_' + self.langs[l[1]] + '_beam_posterior_' + self.langs[l[0]] +
                                          '_only']

            for i in range(num_iterations):

                start = time.clock()

                best_guess = generate_translation_fn(batch_in, best_guess)

                print('Translation iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
                print(' ')

                for ind_to_print in range(10):
                    print('  in: ' + ' '.join([self.valid_vocabs[l[0]][j] for j in batch_in[ind_to_print] if j >= 0]))
                    print('true: ' + ' '.join([self.valid_vocabs[l[1]][j] for j in true_batch_out[ind_to_print]
                                               if j >= 0]))
                    print(' gen: ' + ' '.join([self.valid_vocabs[l[1]][j] for j in best_guess[ind_to_print] if j >= 0]))
                    print(' ')

            out = OrderedDict()

            if sampling:
                suffix = '_for_translation_from_' + self.langs[l[0]] + '_to_' + self.langs[l[1]] + '_sampling'
            else:
                suffix = '_for_translation_from_' + self.langs[l[0]] + '_to_' + self.langs[l[1]]

            out['true_x_from' + suffix] = batch_in
            out['true_x_to' + suffix] = true_batch_out
            out['gen_x_to' + suffix] = best_guess

            for key, value in out.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)


class RunDiscriminative(object):

    def __init__(self, solver, solver_kwargs, l_0, l_1, valid_vocab_0, valid_vocab_1, main_dir, out_dir, dataset,
                 load_param_dir=None, pre_trained=False):

        self.l_0 = l_0
        self.l_1 = l_1

        self.valid_vocab_0 = valid_vocab_0
        self.valid_vocab_1 = valid_vocab_1

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.max_length = solver_kwargs['max_length']
        self.vocab_size = solver_kwargs['vocab_size']

        print('loading data')

        start = time.clock()

        self.x_0_train, self.x_1_train, self.x_0_test, self.x_1_test, self.L_0_train, self.L_1_train, self.L_0_test, \
            self.L_1_test = self.load_data(dataset)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        print('num sentences train = ' + str(len(self.L_0_train)))
        print('num sentences test = ' + str(len(self.L_0_test)))

        self.solver = solver(**self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings_0.save'), 'rb') as f:
                self.solver.all_embeddings_0.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'all_embeddings_1.save'), 'rb') as f:
                self.solver.all_embeddings_1.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'decoder_params.save'), 'rb') as f:
                self.solver.deocder.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'encoder_params.save'), 'rb') as f:
                self.solver.encoder.set_param_values(cPickle.load(f))

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'all_embeddings_0.save'), 'wb') as f:
            cPickle.dump(self.solver.all_embeddings_0.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings_1.save'), 'wb') as f:
            cPickle.dump(self.solver.all_embeddings_1.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'decoder_params.save'), 'wb') as f:
            cPickle.dump(self.solver.decoder.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'encoder_params.save'), 'wb') as f:
            cPickle.dump(self.solver.encoder.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        files_0_train = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_0)
                                and not f.endswith('test.txt')
                                and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_1_train = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_1)
                                and not f.endswith('test.txt')
                                and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_0_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_0)
                               and f.endswith('test.txt') and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_1_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_1)
                               and f.endswith('test.txt') and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        x_0_train, L_0_train = self.load_words(folder, files_0_train, load_batch_size)
        x_1_train, L_1_train = self.load_words(folder, files_1_train, load_batch_size)
        x_0_test, L_0_test = self.load_words(folder, files_0_test, load_batch_size)
        x_1_test, L_1_test = self.load_words(folder, files_1_test, load_batch_size)

        return x_0_train, x_1_train, x_0_test, x_1_test, L_0_train, L_1_train, L_0_test, L_1_test

    def get_translation_fn(self, beam_size):

        return self.solver.translate_fn(beam_size)

    def call_translation_fn(self, translation_fn, x_in, x_out_true):

        guess = translation_fn(x_in)

        out = OrderedDict()

        out['true_x_from_for_translation_from_' + self.l_0 + ' to ' + self.l_1] = x_in
        out['true_x_to_for_translation_from_' + self.l_0 + ' to ' + self.l_1] = x_out_true
        out['gen_x_to_for_translation_from_' + self.l_0 + ' to ' + self.l_1] = guess

        return out

    def print_translations(self, translations):

        x_in = translations['true_x_from_for_translation_from_' + self.l_0 + ' to ' + self.l_1]
        x_out_true = translations['true_x_to_for_translation_from_' + self.l_0 + ' to ' + self.l_1]
        x_out_gen = translations['gen_x_to_for_translation_from_' + self.l_0 + ' to ' + self.l_1]

        print('='*10)
        print('translations ' + self.l_0 + ' to ' + self.l_1)
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([self.valid_vocab_0[j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([self.valid_vocab_1[j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([self.valid_vocab_1[j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def train(self, n_iter, batch_size, word_drop=None, grad_norm_constraint=None, update=adam, update_kwargs=None,
              val_freq=None, val_batch_size=0, val_print_gen=5, val_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.solver.optimiser(grad_norm_constraint=grad_norm_constraint, update=update,
                                                   update_kwargs=update_kwargs, saved_update=saved_update)

        log_p_x_fn = self.solver.log_p_x_fn()

        translation_fn = self.get_translation_fn(val_beam_size)

        for i in range(n_iter):

            start = time.clock()

            batch_indices = np.random.choice(len(self.x_0_train), batch_size)
            batch_0 = self.x_0_train[batch_indices]
            batch_1 = self.x_1_train[batch_indices]

            if word_drop is not None:

                L_1 = self.L_1_train[batch_indices]

                drop_indices_1 = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))]
                                           for j in L_1])

                drop_mask_1 = np.ones_like(batch_1, dtype='float32')

                for n in range(len(drop_indices_1)):
                    drop_mask_1[n][drop_indices_1[n]] = 0.

            else:

                drop_mask_1 = np.ones_like(batch_1, dtype='float32')

            log_p_x = optimiser(batch_0, batch_1, drop_mask_1)

            print('Iteration ' + str(i + 1) + ': log p x = ' + str(log_p_x) + ' per data point (time taken = ' +
                  str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                val_batch_indices = np.random.choice(len(self.x_0_test), val_batch_size)
                val_batch_l_0 = self.x_0_test[val_batch_indices]
                val_batch_l_1 = self.x_1_test[val_batch_indices]

                val_log_p_x = log_p_x_fn(val_batch_l_0, val_batch_l_1)

                print('Test set log p x = ' + str(val_log_p_x) + ' per data point')

                translation_batch_indices = np.random.choice(len(self.x_0_test), val_print_gen)
                translation_batch_l_0 = self.x_0_test[translation_batch_indices]
                translation_batch_l_1 = self.x_1_test[translation_batch_indices]

                translations = self.call_translation_fn(translation_fn, translation_batch_l_0, translation_batch_l_1)

                self.print_translations(translations)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def translate(self, num_outputs, beam_size=15):

        np.random.seed(1234)

        batch_indices = np.random.choice(len(self.x_0_test), num_outputs)
        batch_0 = self.x_0_test[batch_indices]
        batch_1 = self.x_1_test[batch_indices]

        translation_fn = self.get_translation_fn(beam_size)

        translations = self.call_translation_fn(translation_fn, batch_0, batch_1)

        for key, value in translations.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)


class RunDiscriminativeSSLMulti(object):

    def __init__(self, solver, solver_kwargs, langs, valid_vocabs, main_dir, out_dir, dataset, load_param_dir=None,
                 pre_trained=False):

        self.langs = langs  # e.g. {0: 'en', 1: 'es', 2: 'fr'}
        self.num_langs = len(langs)
        self.valid_vocabs = valid_vocabs  # e.g. {0: [...], 1: [...], 2:[...]}

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.max_length = solver_kwargs['max_length']
        self.vocab_size = solver_kwargs['vocab_size']

        print('loading data')

        start = time.clock()

        self.x_only, self.x_both, self.x_test, self.L_only, self.L_both, self.L_test = self.load_data(dataset)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        for l in range(self.num_langs):
            print('num sentences ' + self.langs[l] + ' only = ' + str(len(self.L_only[l])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' = ' + str(len(self.L_both[pair][0])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' test = ' + str(len(self.L_test[pair][0])))

        self.solver = solver(**self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings.save'), 'rb') as f:
                all_embeddings = cPickle.load(f)
                for l in range(self.num_langs):
                    self.solver.all_embeddings[l].set_value(all_embeddings[l])
                del all_embeddings

            with open(os.path.join(self.load_param_dir, 'decoders_params.save'), 'rb') as f:
                decoders_params = cPickle.load(f)
                for l in range(self.num_langs):
                    self.solver.decoders[l].set_param_values(decoders_params[l])
                del decoders_params

            with open(os.path.join(self.load_param_dir, 'encoders_params.save'), 'rb') as f:
                encoders_params = cPickle.load(f)
                for l in range(self.num_langs):
                    self.solver.encoders[l].set_param_values(encoders_params[l])
                del encoders_params

    def save_params(self, updates):

        with open(os.path.join(self.out_dir, 'all_embeddings.save'), 'wb') as f:
            cPickle.dump([self.solver.all_embeddings[l].get_value() for l in range(self.num_langs)],
                         f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'encoders_params.save'), 'wb') as f:
            cPickle.dump([self.solver.encoders[l].get_param_values() for l in range(self.num_langs)],
                         f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'decoders_params.save'), 'wb') as f:
            cPickle.dump([self.solver.decoders[l].get_param_values() for l in range(self.num_langs)],
                         f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        x_only = {}
        L_only = {}

        for l in range(self.num_langs):

            print('loading ' + self.langs[l] + ' only')

            files_only_l = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.langs[l] +
                                                                                 '_only')])

            x_only_l, L_only_l = self.load_words(folder, files_only_l, load_batch_size)

            x_only[l] = x_only_l
            L_only[l] = L_only_l

        x_both = {}
        L_both = {}

        x_test = {}
        L_test = {}

        for pair in combinations(range(self.num_langs), 2):

            print('loading ' + str(pair))

            l_0 = self.langs[pair[0]]
            l_1 = self.langs[pair[1]]

            files_both_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and not f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_0, L_both_l_0 = self.load_words(folder, files_both_l_0, load_batch_size)

            files_both_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and not f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_1, L_both_l_1 = self.load_words(folder, files_both_l_1, load_batch_size)

            x_both[pair] = (x_both_l_0, x_both_l_1)
            L_both[pair] = (L_both_l_0, L_both_l_1)

            files_test_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_0, L_test_l_0 = self.load_words(folder, files_test_l_0, load_batch_size)

            files_test_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_1, L_test_l_1 = self.load_words(folder, files_test_l_1, load_batch_size)

            x_test[pair] = (x_test_l_0, x_test_l_1)
            L_test[pair] = (L_test_l_0, L_test_l_1)

        return x_only, x_both, x_test, L_only, L_both, L_test

    def get_translation_fn(self, l_in, l_out, beam_size):

        return self.solver.translate_fn(l_in, l_out, beam_size)

    def call_translation_fn(self, translation_fn, l_in, l_out, x_in, x_out_true):

        guess = translation_fn(x_in)

        out = OrderedDict()

        suffix = '_for_translation_from_' + self.langs[l_in] + ' to ' + self.langs[l_out]

        out['true_x_from' + suffix] = x_in
        out['true_x_to' + suffix] = x_out_true
        out['gen_x_to' + suffix] = guess

        return out

    def print_translations(self, translations, l_in, l_out):

        suffix = '_for_translation_from_' + self.langs[l_in] + ' to ' + self.langs[l_out]

        x_in = translations['true_x_from' + suffix]
        x_out_true = translations['true_x_to' + suffix]
        x_out_gen = translations['gen_x_to' + suffix]

        print('='*10)
        print('translations ' + self.langs[l_in] + ' to ' + self.langs[l_out])
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([self.valid_vocabs[l_in][j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([self.valid_vocabs[l_out][j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([self.valid_vocabs[l_out][j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def make_drop_mask(self, word_drop, L):

        drop_indices = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))] for j in L])

        drop_mask = np.ones((L.shape[0], self.max_length))

        for n in range(len(drop_indices)):
            drop_mask[n][drop_indices[n]] = 0.

        return drop_mask

    def train(self, n_iter, only_batch_size, both_batch_size, word_drop=None, grad_norm_constraint=None, update=adam,
              update_kwargs=None, val_freq=None, val_batch_size=0, val_print_gen=5, val_beam_size=15,
              save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        print('compiling optimiser')

        optimiser, updates = self.solver.optimiser(grad_norm_constraint=grad_norm_constraint, update=update,
                                                   update_kwargs=update_kwargs, saved_update=saved_update)

        print('compiling optimiser done')

        print('compiling log_p_x functions')

        log_p_x_fns = {pair: self.solver.log_p_x_fn(pair[0], pair[1])
                       for pair in combinations(range(self.num_langs), 2)}

        print('compiling log_p_x functions done')

        print('compiling translation functions')

        translation_fns = {pair: self.get_translation_fn(pair[0], pair[1], val_beam_size)
                           for pair in permutations(range(self.num_langs), 2)}

        print('compiling translation functions done')

        for i in range(n_iter):

            start = time.clock()

            batches_only = []
            drop_masks_only = []

            for pair in range(self.num_langs):

                batch_indices_l = np.random.choice(len(self.x_only[pair]), only_batch_size)
                batch_only_l = self.x_only[pair][batch_indices_l]
                batches_only.append(batch_only_l)

                if word_drop is not None:
                    L_only_l = self.L_only[pair][batch_indices_l]
                    drop_mask_only_l = self.make_drop_mask(word_drop, L_only_l)
                else:
                    drop_mask_only_l = np.ones_like(batch_only_l)

                drop_masks_only.append(drop_mask_only_l)

            batches_both = []
            drop_masks_both = []

            for pair in combinations(range(self.num_langs), 2):

                batch_indices_l_both = np.random.choice(len(self.x_both[pair][0]), both_batch_size)
                batch_both_l_0 = self.x_both[pair][0][batch_indices_l_both]
                batch_both_l_1 = self.x_both[pair][1][batch_indices_l_both]
                batches_both.append(batch_both_l_0)
                batches_both.append(batch_both_l_1)

                if word_drop is not None:
                    L_both_l_0 = self.L_both[pair][0][batch_indices_l_both]
                    drop_mask_both_l_0 = self.make_drop_mask(word_drop, L_both_l_0)

                    L_both_l_1 = self.L_both[pair][1][batch_indices_l_both]
                    drop_mask_both_l_1 = self.make_drop_mask(word_drop, L_both_l_1)
                else:
                    drop_mask_both_l_0 = np.ones_like(batch_both_l_0)
                    drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                drop_masks_both.append(drop_mask_both_l_0)
                drop_masks_both.append(drop_mask_both_l_1)

            optimiser_args = batches_only + batches_both + drop_masks_only + drop_masks_both

            log_p_x = optimiser(*optimiser_args)

            print('Iteration ' + str(i + 1) + ': log p x = ' + str(log_p_x) + ' per data point (time taken = ' +
                  str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                for pair in combinations(range(self.num_langs), 2):

                    val_batch_indices_l = np.random.choice(len(self.x_test[pair][0]), val_batch_size)
                    val_batch_l_0 = self.x_test[pair][0][val_batch_indices_l]
                    val_batch_l_1 = self.x_test[pair][1][val_batch_indices_l]

                    val_log_p_x = log_p_x_fns[pair](val_batch_l_0, val_batch_l_1)

                    print('Test set ELBO ' + self.langs[pair[0]] + ' and ' + self.langs[pair[1]] + ' = ' +
                          str(val_log_p_x) + ' per data point')

                for pair in combinations(range(self.num_langs), 2):

                    translation_batch_indices_l = np.random.choice(len(self.x_test[pair][0]), val_print_gen)
                    translation_batch_l_0 = self.x_test[pair][0][translation_batch_indices_l]
                    translation_batch_l_1 = self.x_test[pair][1][translation_batch_indices_l]

                    translations_0_to_1 = self.call_translation_fn(translation_fns[(pair[0], pair[1])], pair[0],
                                                                   pair[1], translation_batch_l_0,
                                                                   translation_batch_l_1)

                    self.print_translations(translations_0_to_1, pair[0], pair[1])

                    translations_1_to_0 = self.call_translation_fn(translation_fns[(pair[1], pair[0])], pair[1],
                                                                   pair[0], translation_batch_l_1,
                                                                   translation_batch_l_0)

                    self.print_translations(translations_1_to_0, pair[1], pair[0])

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def translate(self, num_outputs, beam_size=15):

        np.random.seed(1234)

        for pair in permutations(range(self.num_langs), 2):

            batch_indices_l = np.random.choice(len(self.x_test[pair][0]), num_outputs)
            batch_l_0 = self.x_test[pair][0][batch_indices_l]
            batch_l_1 = self.x_test[pair][1][batch_indices_l]

            translation_fn = self.get_translation_fn(pair[0], pair[1], beam_size)

            translations = self.call_translation_fn(translation_fn, pair[0], pair[1], batch_l_0, batch_l_1)

            for key, value in translations.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)


class RunVNMT(object):

    def __init__(self, solver, solver_kwargs, l_0, l_1, valid_vocab_0, valid_vocab_1, main_dir, out_dir, dataset,
                 load_param_dir=None, pre_trained=False):

        self.l_0 = l_0
        self.l_1 = l_1

        self.valid_vocab_0 = valid_vocab_0
        self.valid_vocab_1 = valid_vocab_1

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.max_length = solver_kwargs['max_length']
        self.vocab_size = solver_kwargs['vocab_size']

        print('loading data')

        start = time.clock()

        self.x_0_train, self.x_1_train, self.x_0_val, self.x_1_val, self.x_0_test, self.x_1_test, self.L_0_train, \
            self.L_1_train, self.L_0_val, self.L_1_val, self.L_0_test, self.L_1_test = self.load_data(dataset)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        print('num sentences train = ' + str(len(self.L_0_train)))
        print('num sentences test = ' + str(len(self.L_0_test)))

        self.solver = solver(**self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            all_embeddings_0_files = sorted([f for f in os.listdir(self.load_param_dir) if
                                             f.startswith('all_embeddings_0')])

            all_embeddings_1_files = sorted([f for f in os.listdir(self.load_param_dir) if
                                             f.startswith('all_embeddings_1')])

            decoder_params_files = sorted([f for f in os.listdir(self.load_param_dir) if
                                             f.startswith('decoder_params')])

            encoder_params_files = sorted([f for f in os.listdir(self.load_param_dir) if
                                           f.startswith('encoder_params')])

            with open(os.path.join(self.load_param_dir, all_embeddings_0_files[-1]), 'rb') as f:
                self.solver.all_embeddings_0.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, all_embeddings_1_files[-1]), 'rb') as f:
                self.solver.all_embeddings_1.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, decoder_params_files[-1]), 'rb') as f:
                self.solver.generative_model.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, encoder_params_files[-1]), 'rb') as f:
                self.solver.recognition_model.set_param_values(cPickle.load(f))

    def save_params(self, updates, suffix=''):

        with open(os.path.join(self.out_dir, 'all_embeddings_0' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.all_embeddings_0.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'all_embeddings_1' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.all_embeddings_1.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'decoder_params' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.generative_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'encoder_params' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates' + suffix + '.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = max(L)

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        files_0_train = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_0)
                                and not f.endswith('test.txt') and not f.endswith('val.txt')
                                and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_1_train = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_1)
                                and not f.endswith('test.txt') and not f.endswith('val.txt')
                                and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_0_val = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_0)
                              and f.endswith('val.txt') and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_1_val = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_1)
                              and f.endswith('val.txt') and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_0_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_0)
                               and f.endswith('test.txt') and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        files_1_test = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.l_1)
                               and f.endswith('test.txt') and (self.l_0 + self.l_1 in f or self.l_1 + self.l_0 in f)])

        x_0_train, L_0_train = self.load_words(folder, files_0_train, load_batch_size)
        x_1_train, L_1_train = self.load_words(folder, files_1_train, load_batch_size)
        x_0_val, L_0_val = self.load_words(folder, files_0_val, load_batch_size)
        x_1_val, L_1_val = self.load_words(folder, files_1_val, load_batch_size)
        x_0_test, L_0_test = self.load_words(folder, files_0_test, load_batch_size)
        x_1_test, L_1_test = self.load_words(folder, files_1_test, load_batch_size)

        return x_0_train, x_1_train, x_0_val, x_1_val, x_0_test, x_1_test, L_0_train, L_1_train, L_0_val, L_1_val, \
            L_0_test, L_1_test

    def call_translation_fn(self, translation_fn, x_in, x_out_true, sub_batch_size=50):

        guesses = []

        for x_in_batch, x_out_batch in chunker([x_in, x_out_true], sub_batch_size):
            start = time.clock()

            guess_batch = translation_fn(x_in_batch)
            guesses.append(guess_batch)

            print('batch translated (time taken = ' + str(time.clock() - start) + ' seconds)')

        guess = np.concatenate(guesses, axis=0)

        out = OrderedDict()

        out['true_x_from_for_translation_from_' + self.l_0 + '_to_' + self.l_1] = x_in
        out['true_x_to_for_translation_from_' + self.l_0 + '_to_' + self.l_1] = x_out_true
        out['gen_x_to_for_translation_from_' + self.l_0 + '_to_' + self.l_1] = guess

        return out

    def print_translations(self, translations):

        x_in = translations['true_x_from_for_translation_from_' + self.l_0 + '_to_' + self.l_1]
        x_out_true = translations['true_x_to_for_translation_from_' + self.l_0 + '_to_' + self.l_1]
        x_out_gen = translations['gen_x_to_for_translation_from_' + self.l_0 + '_to_' + self.l_1]

        print('='*10)
        print('translations ' + self.l_0 + ' to ' + self.l_1)
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([self.valid_vocab_0[j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([self.valid_vocab_1[j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([self.valid_vocab_1[j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def make_drop_mask(self, word_drop, L):

        drop_indices = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))] for j in L])

        drop_mask = np.ones((L.shape[0], self.max_length))

        for n in range(len(drop_indices)):
            drop_mask[n][drop_indices[n]] = 0.

        return drop_mask

    def train(self, n_iter, batch_size, num_samples, word_drop=None, grad_norm_constraint=None, update=adam,
              update_kwargs=None, warm_up=None, early_stopping_freq=None, early_stopping_num_samples=5,
              early_stopping_patience=5, print_freq=None, print_batch_size=5, print_beam_size=15,
              save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.solver.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                                   update=update, update_kwargs=update_kwargs,
                                                   saved_update=saved_update)

        elbo_fn = self.solver.elbo_fn(early_stopping_num_samples)

        translation_fn = self.solver.translate_fn(print_beam_size)

        val_elbo_best = -np.inf
        early_stopping_done = False
        early_stopping_evals_wo_improv = 0

        for i in range(n_iter):

            start = time.clock()

            batch_indices = np.random.choice(len(self.x_0_train), batch_size)
            batch_0 = self.x_0_train[batch_indices]
            batch_1 = self.x_1_train[batch_indices]

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            if word_drop is not None:
                L_0 = self.L_0_train[batch_indices]
                drop_mask_0 = self.make_drop_mask(word_drop, L_0)
                L_1 = self.L_1_train[batch_indices]
                drop_mask_1 = self.make_drop_mask(word_drop, L_1)
            else:
                drop_mask_0 = np.ones_like(batch_0, dtype='float32')
                drop_mask_1 = np.ones_like(batch_1, dtype='float32')

            elbo, kl = optimiser(batch_0, batch_1, beta, drop_mask_0, drop_mask_1)

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo) + ' (KL = ' + str(kl) +
                  ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if print_freq is not None and i % print_freq == 0:

                translation_batch_indices = np.random.choice(len(self.x_0_val), print_batch_size)
                translation_batch_l_0 = self.x_0_val[translation_batch_indices]
                translation_batch_l_1 = self.x_1_val[translation_batch_indices]

                translations = self.call_translation_fn(translation_fn, translation_batch_l_0, translation_batch_l_1)

                self.print_translations(translations)

            if early_stopping_freq is not None and i % early_stopping_freq == 0:

                val_elbo = 0
                val_kl = 0

                for val_batch_l_0, val_batch_l_1 in chunker([self.x_0_val, self.x_1_val], 200):

                    val_elbo_batch, val_kl_batch = elbo_fn(val_batch_l_0, val_batch_l_1)

                    val_elbo += val_elbo_batch * len(val_batch_l_0)
                    val_kl += val_kl_batch * len(val_batch_l_0)

                val_elbo /= len(self.x_0_val)
                val_kl /= len(self.x_0_val)

                print('Test set ELBO = ' + str(val_elbo) + ' (KL = ' + str(val_kl) + ') per data point')

                if val_elbo > val_elbo_best:
                    val_elbo_best = val_elbo
                    early_stopping_evals_wo_improv = 0
                else:
                    early_stopping_evals_wo_improv += 1
                    if not early_stopping_done and early_stopping_evals_wo_improv > early_stopping_patience:
                        self.save_params(updates, suffix='_early_stopping_' + str(i))
                        early_stopping_done = True

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def translate(self, num_outputs, beam_size=15, sampling=False, num_samples=0, sub_batch_size=50):

        np.random.seed(1234)

        batch_indices = np.random.choice(len(self.x_0_test), num_outputs)
        batch_0 = self.x_0_test[batch_indices]
        batch_1 = self.x_1_test[batch_indices]

        if sampling:
            translation_fn = self.solver.translate_fn_sampling(beam_size, num_samples)
            suffix = '_sampling.npy'
        else:
            translation_fn = self.solver.translate_fn(beam_size)
            suffix = '.npy'

        translations = self.call_translation_fn(translation_fn, batch_0, batch_1, sub_batch_size)

        for key, value in translations.items():
            np.save(os.path.join(self.out_dir, key + suffix), value)


class RunVNMTMultiIndicator(object):

    def __init__(self, solver, solver_kwargs, langs, valid_vocabs, main_dir, out_dir, dataset, load_param_dir=None,
                 pre_trained=False):

        self.langs = langs  # e.g. {0: 'en', 1: 'es', 2: 'fr'}
        self.num_langs = len(langs)
        self.valid_vocabs = valid_vocabs  # e.g. {0: [...], 1: [...], 2:[...]}

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.max_length = solver_kwargs['max_length']
        self.vocab_size = solver_kwargs['vocab_size']

        print('loading data')

        start = time.clock()

        self.x_only, self.x_both, self.x_val, self.x_test, self.L_only, self.L_both, self.L_val, self.L_test = \
            self.load_data(dataset)

        print('data loaded; time taken = ' + str(time.clock() - start) + ' seconds')

        for l in range(self.num_langs):
            print('num sentences ' + self.langs[l] + ' only = ' + str(len(self.L_only[l])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' = ' + str(len(self.L_both[pair][0])))

        for pair in combinations(range(self.num_langs), 2):
            print('num sentences ' + str(pair) + ' test = ' + str(len(self.L_test[pair][0])))

        self.solver = solver(**self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'all_embeddings.save'), 'rb') as f:
                self.solver.all_embeddings.set_value(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
                self.solver.generative_model.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.solver.recognition_model.set_param_values(cPickle.load(f))

    def save_params(self, updates, suffix=''):

        with open(os.path.join(self.out_dir, 'all_embeddings' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.all_embeddings.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.generative_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'recog_params' + suffix + '.save'), 'wb') as f:
            cPickle.dump(self.solver.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates' + suffix + '.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_words(self, folder, files, load_batch_size):

        words = []

        for f in files:
            with open(os.path.join(self.main_dir, folder, f), 'r') as s:
                words += json.loads(s.read())

        L = np.array([len(w) for w in words])

        max_L = self.max_length

        word_arrays = []

        start = time.clock()

        batches_loaded = 0

        for i in range(0, len(L), load_batch_size):

            L_i = L[i: i+load_batch_size]

            word_array = np.full((len(L_i), max_L), -1, dtype='int32')
            word_array[L_i.reshape((L_i.shape[0], 1)) > np.arange(max_L)] = \
                np.concatenate(words[i: i+load_batch_size])

            word_arrays.append(word_array)

            batches_loaded += 1

            print(str(batches_loaded) + ' batches loaded (time taken = ' + str(time.clock() - start) +
                  ' seconds)')

        del words

        return np.concatenate(word_arrays), L

    def load_data(self, dataset, load_batch_size=500000):

        folder = '../_datasets/' + dataset

        x_only = {}
        L_only = {}

        for l in range(self.num_langs):

            print('loading ' + self.langs[l] + ' only')

            files_only_l = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + self.langs[l] +
                                                                                 '_only')])

            x_only_l, L_only_l = self.load_words(folder, files_only_l, load_batch_size)

            x_only[l] = x_only_l
            L_only[l] = L_only_l

        x_both = {}
        L_both = {}

        x_val = {}
        L_val = {}

        x_test = {}
        L_test = {}

        for pair in combinations(range(self.num_langs), 2):

            print('loading ' + str(pair))

            l_0 = self.langs[pair[0]]
            l_1 = self.langs[pair[1]]

            files_both_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and not f.endswith('test.txt') and not f.endswith('val.txt')
                                     and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_0, L_both_l_0 = self.load_words(folder, files_both_l_0, load_batch_size)

            files_both_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and not f.endswith('test.txt') and not f.endswith('val.txt')
                                     and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_both_l_1, L_both_l_1 = self.load_words(folder, files_both_l_1, load_batch_size)

            x_both[pair] = (x_both_l_0, x_both_l_1)
            L_both[pair] = (L_both_l_0, L_both_l_1)

            files_val_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                    and f.endswith('val.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_val_l_0, L_val_l_0 = self.load_words(folder, files_val_l_0, load_batch_size)

            files_val_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                    and f.endswith('val.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_val_l_1, L_val_l_1 = self.load_words(folder, files_val_l_1, load_batch_size)

            x_val[pair] = (x_val_l_0, x_val_l_1)
            L_val[pair] = (L_val_l_0, L_val_l_1)

            files_test_l_0 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_0)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_0, L_test_l_0 = self.load_words(folder, files_test_l_0, load_batch_size)

            files_test_l_1 = sorted([f for f in os.listdir(folder) if f.startswith('sentences_' + l_1)
                                     and f.endswith('test.txt') and (l_0 + l_1 in f or l_1 + l_0 in f)])

            x_test_l_1, L_test_l_1 = self.load_words(folder, files_test_l_1, load_batch_size)

            x_test[pair] = (x_test_l_0, x_test_l_1)
            L_test[pair] = (L_test_l_0, L_test_l_1)

        return x_only, x_both, x_val, x_test, L_only, L_both, L_val, L_test

    def call_translation_fn(self, translation_fn, l_in, l_out, x_in, x_out_true, sub_batch_size=50):

        guesses = []

        for x_in_batch, x_out_batch in chunker([x_in, x_out_true], sub_batch_size):

            start = time.clock()

            l_in_rep = np.array([l_in]*x_in_batch.shape[0])

            guess_batch = translation_fn(l_in_rep, l_out, x_in_batch)
            guesses.append(guess_batch)

            print('batch translated (time taken = ' + str(time.clock() - start) + ' seconds)')

        guess = np.concatenate(guesses, axis=0)

        out = OrderedDict()

        suffix = '_for_translation_from_' + self.langs[l_in] + '_to_' + self.langs[l_out]

        out['true_x_from' + suffix] = x_in
        out['true_x_to' + suffix] = x_out_true
        out['gen_x_to' + suffix] = guess

        return out

    def print_translations(self, translations, l_in, l_out):

        suffix = '_for_translation_from_' + self.langs[l_in] + '_to_' + self.langs[l_out]

        x_in = translations['true_x_from' + suffix]
        x_out_true = translations['true_x_to' + suffix]
        x_out_gen = translations['gen_x_to' + suffix]

        print('='*10)
        print('translations ' + self.langs[l_in] + ' to ' + self.langs[l_out])
        print('='*10)

        for n in range(x_in.shape[0]):

            print('  in: ' + ' '.join([self.valid_vocabs[l_in][j] for j in x_in[n] if j >= 0]))
            print('true: ' + ' '.join([self.valid_vocabs[l_out][j] for j in x_out_true[n] if j >= 0]))
            print(' gen: ' + ' '.join([self.valid_vocabs[l_out][j] for j in x_out_gen[n] if j >= 0]))

            print('-'*10)

        print('='*10)

    def make_drop_mask(self, word_drop, L):

        drop_indices = np.array([np.random.permutation(np.arange(j))[:int(np.floor(word_drop * j))] for j in L])

        drop_mask = np.ones((L.shape[0], self.max_length))

        for n in range(len(drop_indices)):
            drop_mask[n][drop_indices[n]] = 0.

        return drop_mask

    def train(self, n_iter, only_batch_size, both_batch_size, num_samples, ssl=True, word_drop=None,
              grad_norm_constraint=None, update=adam, update_kwargs=None, warm_up=None, exclude_lang_pair=(),
              early_stopping_freq=None, early_stopping_num_samples=5, early_stopping_patience=5, print_freq=None,
              print_batch_size=5, print_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.solver.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                                   update=update, update_kwargs=update_kwargs,
                                                   saved_update=saved_update)

        elbo_fn = self.solver.elbo_fn(early_stopping_num_samples)

        translation_fn = self.solver.translate_fn(print_beam_size)

        val_elbo_best = -np.inf
        early_stopping_done = False
        early_stopping_evals_wo_improv = 0

        for i in range(n_iter):

            start = time.clock()

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            for l_1 in range(self.num_langs):

                ls_l_0 = []
                batches_l_0 = []
                batches_l_1 = []
                drop_masks_l_0 = []
                drop_masks_l_1 = []

                for l_0 in range(self.num_langs):

                    if l_0 == l_1:

                        if ssl:

                            ls_l_0.append(np.array([l_0]*only_batch_size))

                            batch_indices_only = np.random.choice(len(self.x_only[l_1]), only_batch_size)
                            batch_only_l = self.x_only[l_1][batch_indices_only]

                            batches_l_0.append(batch_only_l)
                            batches_l_1.append(batch_only_l)

                            if word_drop is not None:
                                L_only_l = self.L_only[l_1][batch_indices_only]
                                drop_mask_only_l = self.make_drop_mask(word_drop, L_only_l)
                            else:
                                drop_mask_only_l = np.ones_like(batch_only_l)

                            drop_masks_l_0.append(drop_mask_only_l)
                            drop_masks_l_1.append(drop_mask_only_l)

                    else:

                        if l_0 not in exclude_lang_pair or l_1 not in exclude_lang_pair:

                            ls_l_0.append(np.array([l_0]*both_batch_size))

                            min_l = min((l_0, l_1))
                            max_l = max((l_0, l_1))

                            pair = (min_l, max_l)

                            batch_indices_both = np.random.choice(len(self.x_both[pair][0]), both_batch_size)

                            if min_l == l_0:
                                batch_both_l_0 = self.x_both[pair][0][batch_indices_both]
                                batch_both_l_1 = self.x_both[pair][1][batch_indices_both]
                            else:
                                batch_both_l_0 = self.x_both[pair][1][batch_indices_both]
                                batch_both_l_1 = self.x_both[pair][0][batch_indices_both]

                            batches_l_0.append(batch_both_l_0)
                            batches_l_1.append(batch_both_l_1)

                            if word_drop is not None:
                                if min_l == l_0:
                                    L_both_l_0 = self.L_both[pair][0][batch_indices_both]
                                    L_both_l_1 = self.L_both[pair][1][batch_indices_both]
                                else:
                                    L_both_l_0 = self.L_both[pair][1][batch_indices_both]
                                    L_both_l_1 = self.L_both[pair][0][batch_indices_both]
                                drop_mask_both_l_0 = self.make_drop_mask(word_drop, L_both_l_0)
                                drop_mask_both_l_1 = self.make_drop_mask(word_drop, L_both_l_1)
                            else:
                                drop_mask_both_l_0 = np.ones_like(batch_both_l_0)
                                drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                            drop_masks_l_0.append(drop_mask_both_l_0)
                            drop_masks_l_1.append(drop_mask_both_l_1)

                ls_l_0 = np.concatenate(ls_l_0, axis=0)

                batches_l_0 = np.concatenate(batches_l_0, axis=0)
                batches_l_1 = np.concatenate(batches_l_1, axis=0)
                drop_masks_l_0 = np.concatenate(drop_masks_l_0, axis=0)
                drop_masks_l_1 = np.concatenate(drop_masks_l_1, axis=0)

                elbo_only_l, kl_only_l = optimiser(ls_l_0, l_1, batches_l_0, batches_l_1, beta, drop_masks_l_0,
                                                   drop_masks_l_1)

                print('Iteration ' + str(i + 1) + ' all to ' + self.langs[l_1] + ': ELBO = ' + str(elbo_only_l) +
                      ' (KL = ' + str(kl_only_l) + ') per data point (time taken = ' + str(time.clock() - start) +
                      ' seconds)')

            if print_freq is not None and i % print_freq == 0:

                for pair in combinations(range(self.num_langs), 2):

                    translation_batch_indices_l = np.random.choice(len(self.x_test[pair][0]), print_batch_size)
                    translation_batch_l_0 = self.x_test[pair][0][translation_batch_indices_l]
                    translation_batch_l_1 = self.x_test[pair][1][translation_batch_indices_l]

                    translations_0_to_1 = self.call_translation_fn(translation_fn, pair[0], pair[1],
                                                                   translation_batch_l_0, translation_batch_l_1)

                    self.print_translations(translations_0_to_1, pair[0], pair[1])

                    translations_1_to_0 = self.call_translation_fn(translation_fn, pair[1], pair[0],
                                                                   translation_batch_l_1, translation_batch_l_0)

                    self.print_translations(translations_1_to_0, pair[1], pair[0])

            if early_stopping_freq is not None and i % early_stopping_freq == 0:

                val_elbo = 0
                val_kl = 0

                for pair in combinations(range(self.num_langs), 2):

                    val_elbo_0_to_1 = 0
                    val_kl_0_to_1 = 0

                    val_elbo_1_to_0 = 0
                    val_kl_1_to_0 = 0

                    for val_batch_l_0, val_batch_l_1 in chunker([self.x_val[pair][0], self.x_val[pair][1]], 200):

                        val_elbo_0_to_1_batch, val_kl_0_to_1_batch = elbo_fn(np.array([pair[0]]*len(val_batch_l_0)),
                                                                             pair[1], val_batch_l_0, val_batch_l_1)

                        val_elbo_0_to_1 += val_elbo_0_to_1_batch * len(val_batch_l_0)
                        val_kl_0_to_1 += val_kl_0_to_1_batch * len(val_batch_l_0)

                        val_elbo_1_to_0_batch, val_kl_1_to_0_batch = elbo_fn(np.array([pair[1]]*len(val_batch_l_1)),
                                                                             pair[0], val_batch_l_1, val_batch_l_0)

                        val_elbo_1_to_0 += val_elbo_1_to_0_batch * len(val_batch_l_0)
                        val_kl_1_to_0 += val_kl_1_to_0_batch * len(val_batch_l_0)

                    val_elbo_0_to_1 /= len(self.x_val[pair][0])
                    val_kl_0_to_1 /= len(self.x_val[pair][0])
                    val_elbo_1_to_0 /= len(self.x_val[pair][0])
                    val_kl_1_to_0 /= len(self.x_val[pair][0])

                    print('Test set ELBO ' + self.langs[pair[0]] + ' to ' + self.langs[pair[1]] + ' = ' +
                          str(val_elbo_0_to_1) + ' (KL = ' + str(val_kl_0_to_1) + ') per data point')

                    val_elbo += val_elbo_0_to_1
                    val_kl += val_kl_0_to_1

                    print('Test set ELBO ' + self.langs[pair[1]] + ' to ' + self.langs[pair[0]] + ' = ' +
                          str(val_elbo_1_to_0) + ' (KL = ' + str(val_kl_1_to_0) + ') per data point')

                    val_elbo += val_elbo_1_to_0
                    val_kl += val_kl_1_to_0

                val_elbo /= len(list(combinations(range(self.num_langs), 2))) * 2
                val_kl /= len(list(combinations(range(self.num_langs), 2))) * 2

                print('Test set ELBO average = ' + str(val_elbo) + ' (KL = ' + str(val_kl) + ') per data point')

                if val_elbo > val_elbo_best:
                    val_elbo_best = val_elbo
                    early_stopping_evals_wo_improv = 0
                else:
                    early_stopping_evals_wo_improv += 1
                    if not early_stopping_done and early_stopping_evals_wo_improv > early_stopping_patience:
                        self.save_params(updates, suffix='_early_stopping_' + str(i))
                        early_stopping_done = True

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)

    def translate(self, num_outputs, beam_size=15, sampling=False, num_samples=0, sub_batch_size=50):

        np.random.seed(1234)

        if sampling:
            translation_fn = self.solver.translate_fn_sampling(beam_size, num_samples)
            suffix = '_sampling.npy'
        else:
            translation_fn = self.solver.translate_fn(beam_size)
            suffix = '.npy'

        for pair in combinations(range(self.num_langs), 2):

            batch_indices = np.random.choice(len(self.x_test[pair][0]), num_outputs)
            batch_l_0 = self.x_test[pair][0][batch_indices]
            batch_l_1 = self.x_test[pair][1][batch_indices]

            translations_0_to_1 = self.call_translation_fn(translation_fn, pair[0], pair[1], batch_l_0, batch_l_1,
                                                           sub_batch_size)

            for key, value in translations_0_to_1.items():
                np.save(os.path.join(self.out_dir, key + suffix), value)

            translations_1_to_0 = self.call_translation_fn(translation_fn, pair[1], pair[0], batch_l_1, batch_l_0,
                                                           sub_batch_size)

            for key, value in translations_1_to_0.items():
                np.save(os.path.join(self.out_dir, key + suffix), value)


class RunVNMTJointMultiIndicator(RunVNMTMultiIndicator):

    def call_translation_fn(self, translation_fn, l_in, l_out, x_in, x_out_true, sub_batch_size=50):

        guesses = []

        for x_in_batch, x_out_batch in chunker([x_in, x_out_true], sub_batch_size):

            start = time.clock()

            guess_batch = translation_fn(l_in, l_out, x_in_batch)
            guesses.append(guess_batch)

            print('batch translated (time taken = ' + str(time.clock() - start) + ' seconds)')

        guess = np.concatenate(guesses, axis=0)

        out = OrderedDict()

        suffix = '_for_translation_from_' + self.langs[l_in] + '_to_' + self.langs[l_out]

        out['true_x_from' + suffix] = x_in
        out['true_x_to' + suffix] = x_out_true
        out['gen_x_to' + suffix] = guess

        return out

    def train(self, n_iter, only_batch_size, both_batch_size, num_samples, ssl=True, word_drop=None,
              grad_norm_constraint=None, update=adam, update_kwargs=None, warm_up=None, exclude_lang_pair=(),
              early_stopping_freq=None, early_stopping_num_samples=5, early_stopping_patience=5, print_freq=None,
              print_batch_size=5, print_beam_size=15, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
            np.random.seed()
        else:
            saved_update = None

        optimiser, updates = self.solver.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                                   update=update, update_kwargs=update_kwargs,
                                                   saved_update=saved_update)

        elbo_fn = self.solver.elbo_fn(early_stopping_num_samples)

        translation_fn = self.solver.translate_fn(print_beam_size)

        val_elbo_best = -np.inf
        early_stopping_done = False
        early_stopping_evals_wo_improv = 0

        for i in range(n_iter):

            start = time.clock()

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            for l_1 in range(self.num_langs):

                for l_0 in range(self.num_langs):

                    if l_0 == l_1:

                        if ssl:

                            batch_indices_only = np.random.choice(len(self.x_only[l_1]), only_batch_size)
                            batch_only_l = self.x_only[l_1][batch_indices_only]

                            if word_drop is not None:
                                L_only_l = self.L_only[l_1][batch_indices_only]
                                drop_mask_only_l = self.make_drop_mask(word_drop, L_only_l)
                            else:
                                drop_mask_only_l = np.ones_like(batch_only_l)

                            elbo, kl = optimiser(l_0, l_1, batch_only_l, batch_only_l, beta, drop_mask_only_l,
                                                 drop_mask_only_l)

                            print('Iteration ' + str(i + 1) + ' ' + self.langs[l_0] + ' to ' + self.langs[l_1] +
                                  ': ELBO = ' + str(elbo) + ' (KL = ' + str(kl) + ') per data point (time taken = ' +
                                  str(time.clock() - start) + ' seconds)')

                    else:

                        if l_0 not in exclude_lang_pair or l_1 not in exclude_lang_pair:

                            min_l = min((l_0, l_1))
                            max_l = max((l_0, l_1))

                            pair = (min_l, max_l)

                            batch_indices_both = np.random.choice(len(self.x_both[pair][0]), both_batch_size)

                            if min_l == l_0:
                                batch_both_l_0 = self.x_both[pair][0][batch_indices_both]
                                batch_both_l_1 = self.x_both[pair][1][batch_indices_both]
                            else:
                                batch_both_l_0 = self.x_both[pair][1][batch_indices_both]
                                batch_both_l_1 = self.x_both[pair][0][batch_indices_both]

                            if word_drop is not None:
                                if min_l == l_0:
                                    L_both_l_0 = self.L_both[pair][0][batch_indices_both]
                                    L_both_l_1 = self.L_both[pair][1][batch_indices_both]
                                else:
                                    L_both_l_0 = self.L_both[pair][1][batch_indices_both]
                                    L_both_l_1 = self.L_both[pair][0][batch_indices_both]
                                drop_mask_both_l_0 = self.make_drop_mask(word_drop, L_both_l_0)
                                drop_mask_both_l_1 = self.make_drop_mask(word_drop, L_both_l_1)
                            else:
                                drop_mask_both_l_0 = np.ones_like(batch_both_l_0)
                                drop_mask_both_l_1 = np.ones_like(batch_both_l_1)

                            elbo, kl = optimiser(l_0, l_1, batch_both_l_0, batch_both_l_1, beta, drop_mask_both_l_0,
                                                 drop_mask_both_l_1)

                            print('Iteration ' + str(i + 1) + ' ' + self.langs[l_0] + ' to ' + self.langs[l_1] +
                                  ': ELBO = ' + str(elbo) + ' (KL = ' + str(kl) + ') per data point (time taken = ' +
                                  str(time.clock() - start) + ' seconds)')

            if print_freq is not None and i % print_freq == 0:

                for pair in combinations(range(self.num_langs), 2):

                    translation_batch_indices_l = np.random.choice(len(self.x_test[pair][0]), print_batch_size)
                    translation_batch_l_0 = self.x_test[pair][0][translation_batch_indices_l]
                    translation_batch_l_1 = self.x_test[pair][1][translation_batch_indices_l]

                    translations_0_to_1 = self.call_translation_fn(translation_fn, pair[0], pair[1],
                                                                   translation_batch_l_0, translation_batch_l_1)

                    self.print_translations(translations_0_to_1, pair[0], pair[1])

                    translations_1_to_0 = self.call_translation_fn(translation_fn, pair[1], pair[0],
                                                                   translation_batch_l_1, translation_batch_l_0)

                    self.print_translations(translations_1_to_0, pair[1], pair[0])

            if early_stopping_freq is not None and i % early_stopping_freq == 0:

                val_elbo = 0
                val_kl = 0

                for pair in combinations(range(self.num_langs), 2):

                    val_elbo_0_to_1 = 0
                    val_kl_0_to_1 = 0

                    val_elbo_1_to_0 = 0
                    val_kl_1_to_0 = 0

                    for val_batch_l_0, val_batch_l_1 in chunker([self.x_val[pair][0], self.x_val[pair][1]], 100):

                        val_elbo_0_to_1_batch, val_kl_0_to_1_batch = elbo_fn(pair[0], pair[1], val_batch_l_0,
                                                                             val_batch_l_1)

                        val_elbo_0_to_1 += val_elbo_0_to_1_batch * len(val_batch_l_0)
                        val_kl_0_to_1 += val_kl_0_to_1_batch * len(val_batch_l_0)

                        val_elbo_1_to_0_batch, val_kl_1_to_0_batch = elbo_fn(pair[1], pair[0], val_batch_l_1,
                                                                             val_batch_l_0)

                        val_elbo_1_to_0 += val_elbo_1_to_0_batch * len(val_batch_l_0)
                        val_kl_1_to_0 += val_kl_1_to_0_batch * len(val_batch_l_0)

                    val_elbo_0_to_1 /= len(self.x_val[pair][0])
                    val_kl_0_to_1 /= len(self.x_val[pair][0])
                    val_elbo_1_to_0 /= len(self.x_val[pair][0])
                    val_kl_1_to_0 /= len(self.x_val[pair][0])

                    print('Test set ELBO ' + self.langs[pair[0]] + ' to ' + self.langs[pair[1]] + ' = ' +
                          str(val_elbo_0_to_1) + ' (KL = ' + str(val_kl_0_to_1) + ') per data point')

                    val_elbo += val_elbo_0_to_1
                    val_kl += val_kl_0_to_1

                    print('Test set ELBO ' + self.langs[pair[1]] + ' to ' + self.langs[pair[0]] + ' = ' +
                          str(val_elbo_1_to_0) + ' (KL = ' + str(val_kl_1_to_0) + ') per data point')

                    val_elbo += val_elbo_1_to_0
                    val_kl += val_kl_1_to_0

                val_elbo /= len(list(combinations(range(self.num_langs), 2))) * 2
                val_kl /= len(list(combinations(range(self.num_langs), 2))) * 2

                print('Test set ELBO average = ' + str(val_elbo) + ' (KL = ' + str(val_kl) + ') per data point')

                if val_elbo > val_elbo_best:
                    val_elbo_best = val_elbo
                    early_stopping_evals_wo_improv = 0
                else:
                    early_stopping_evals_wo_improv += 1
                    if not early_stopping_done and early_stopping_evals_wo_improv > early_stopping_patience:
                        self.save_params(updates, suffix='_early_stopping_' + str(i))
                        early_stopping_done = True

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                self.save_params(updates)

        self.save_params(updates)
