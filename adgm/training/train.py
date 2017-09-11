import numpy as np
from ..utils import env_paths as paths
from base import Train
import time


class TrainModel(Train):
    def __init__(self, model, model_logger, model_directory, output_freq=1, pickle_f_custom_freq=None,
                 f_custom_eval=None):
        super(TrainModel, self).__init__(model, model_logger, model_directory, pickle_f_custom_freq, f_custom_eval)
        self.output_freq = output_freq

    def train_model(self, f_train, train_args, f_test, test_args, f_validate, validation_args,
                    n_train_batches=600, n_valid_batches=1, n_test_batches=1, n_epochs=100, anneal=None):
        self.write_to_logger("### MODEL PARAMS ###")
        self.write_to_logger(self.model.model_info())
        self.write_to_logger("### TRAINING PARAMS ###")
        self.write_to_logger(
            "Train -> %s: %s" % (";".join(train_args['inputs'].keys()), str(train_args['inputs'].values())))
        self.write_to_logger(
            "Test -> %s: %s" % (";".join(test_args['inputs'].keys()), str(test_args['inputs'].values())))
        if anneal is not None:
            for t in anneal:
                key, freq, rate, min_val = t
                self.write_to_logger(
                    "Anneal %s %0.4f after %i epochs with minimum value %f." % (key, rate, int(freq), min_val))

        self.write_to_logger("### TRAINING MODEL ###")

        if True:  
            if self.custom_eval_func is not None:
                test_auprc = self.custom_eval_func(self.model)
                self.test_auprc_list.append((0,test_auprc))
                print >>self.testeval_csv, '0, {}'.format(test_auprc)
                self.testeval_csv.flush()

        done_looping = False
        epoch = 0
        prev_epoch_elbo = None

        def is_stopping_criterion_met(prev_epoch_elbo, this_epoch_elbo):
            if prev_epoch_elbo is None:
                print 'First epoch'
                return False
            else:
                self.model_logger.info('prev epoch elbo = {}, this epoch elbo = {}'.format(prev_epoch_elbo, this_epoch_elbo))
                rel_elbo_change = (prev_epoch_elbo - this_epoch_elbo)/prev_epoch_elbo
                self.model_logger.info('relative elbo change = {:.4e}'.format(rel_elbo_change))
                return rel_elbo_change < 1e-6

        print 'Training...'
        while (epoch < n_epochs):
            epoch += 1
            start_time = time.time()
            train_outputs = []
            for i in xrange(n_train_batches):
                train_output = f_train(i, *train_args['inputs'].values())
                train_outputs.append(train_output)
            
            self.eval_train[epoch] = np.mean(np.array(train_outputs), axis=0)
            self.model.after_epoch()
            end_time = time.time() - start_time

            if anneal is not None:
                for t in anneal:
                    key, freq, rate, min_val = t
                    new_val = train_args['inputs'][key] * rate
                    if new_val < min_val:
                        train_args['inputs'][key] = min_val
                    elif epoch % freq == 0:
                        train_args['inputs'][key] = new_val

            if epoch % self.output_freq == 0:
                if n_test_batches == 1:
                    #self.eval_test[epoch] = f_test(*test_args['inputs'].values())
                    self.eval_test[epoch] = 0.0
                else:
                    test_outputs = []
                    for i in xrange(n_test_batches):
                        test_output = f_test(i, *test_args['inputs'].values())
                        test_outputs.append(test_output)
                    self.eval_test[epoch] = np.mean(np.array(test_outputs), axis=0)

                if f_validate is not None:
                    if n_valid_batches == 1:
                        self.eval_validation[epoch] = f_validate(*validation_args['inputs'].values())
                    else:
                        valid_outputs = []
                        for i in xrange(n_valid_batches):
                            valid_output = f_validate(i, *validation_args['inputs'].values())
                            valid_outputs.append(valid_output)
                        self.eval_validation[epoch] = np.mean(np.array(valid_outputs), axis=0)#
                else:
                    self.eval_validation[epoch] = [0.] * len(validation_args['outputs'].keys())

                # Formatting the output string from the generic and the user-defined values.
                output_str = "epoch=%0" + str(len(str(n_epochs))) + "i; time=%0.2f;"
                output_str %= (epoch, end_time)

                def concatenate_output_str(out_str, d):
                    for k, v in zip(d.keys(), d.values()):
                        out_str += " %s=%s;" % (k, v)
                    return out_str

                output_str = concatenate_output_str(output_str, train_args['outputs'])
                output_str = concatenate_output_str(output_str, test_args['outputs'])
                output_str = concatenate_output_str(output_str, validation_args['outputs'])

                outputs = [float(o) for o in self.eval_train[epoch]]
                #outputs += [float(o) for o in self.eval_test[epoch]]
                #outputs += [float(o) for o in self.eval_validation[epoch]]

                output_str = tuple(str(f) for f in outputs)
                self.write_to_logger(output_str)

                #
                # Write in CSV format
                csv_outputs = [epoch, end_time] + outputs
                format_str = ','.join(['{}'] * len(csv_outputs))
                print >>self.learning_csv, format_str.format(*csv_outputs)
                self.learning_csv.flush()


            if self.pickle_f_custom_freq is not None and epoch % self.pickle_f_custom_freq == 0:
                if self.custom_eval_func is not None:
                    test_auprc = self.custom_eval_func(self.model)
                    self.test_auprc_list.append((epoch,test_auprc))
                    print >>self.testeval_csv, '{}, {}'.format(epoch, test_auprc)
                    self.testeval_csv.flush()
                #self.plot_eval(self.eval_train, train_args['outputs'].keys(), "_train")
                #self.plot_eval(self.eval_test, test_args['outputs'].keys(), "_test")
                #self.plot_eval(self.eval_validation, validation_args['outputs'].keys(), "_validation")
                #self.dump_dicts()
                #self.model.dump_model()

            #
            # Check if stopping criterion is met
            #
            this_elbo = np.mean(np.asarray(train_outputs)[:,0])
            done_looping = is_stopping_criterion_met(prev_epoch_elbo, this_elbo)

            if not done_looping:
                prev_epoch_elbo = this_elbo
            else:
                break

        #if self.pickle_f_custom_freq is not None:
        #    self.model.dump_model()

        # Log final test AUPRC
        if self.custom_eval_func is not None:
            test_auprc = self.custom_eval_func(self.model)
            self.test_auprc_list.append((epoch,test_auprc))
