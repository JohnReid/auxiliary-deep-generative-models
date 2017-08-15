#!/usr/bin/env python2

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J RUNMNIST
#! Which project should be charged (NB Wilkes projects end in '-GPU'):
#SBATCH -A MRC-BSU-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<=nodes*12)
#SBATCH --ntasks=1
##SBATCH --array=0-47%5
#! How much wallclock time will be required? [nrg] if unknown set to 12h (12:00:00) 
#SBATCH --time=1:00:00
#! What types of email messages do you wish to receive?
##SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH --mem=29000
#SBATCH --exclusive
#! Where to put output and error
#SBATCH -o output/run-mnist-%j.out
#SBATCH -e output/run-mnist-%j.err
#! Do not change:
#SBATCH -p mrc-bsu-tesla
#! sbatch directives end here (put any additional directives above this line)

"""
Train a skip deep generative model on the mnist dataset with 100 evenly distributed labels.
"""

import theano
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify
from data_loaders import mnist
from models.sdgmssl import SDGMSSL
import numpy as np


seed = np.random.randint(1, 2147462579)
n_labeled = 100  # The total number of labeled data points.
mnist_data = mnist.load_semi_supervised(n_labeled=n_labeled, filter_std=0.0, seed=seed, train_valid_combine=True)

n, n_x = mnist_data[0][0].shape  # Datapoints in the dataset, input features.
n_samples = 100  # The number of sampled labeled data points for each batch.
n_batches = n / 100  # The number of batches.
bs = n / n_batches  # The batchsize.
hidspec = [50]

# Initialize the auxiliary deep generative model.
model = SDGMSSL(n_x=n_x, n_a=100, n_z=100, n_y=10, qa_hid=hidspec,
                qz_hid=hidspec, qy_hid=hidspec, px_hid=hidspec, pa_hid=hidspec,
                nonlinearity=rectify, batchnorm=True, x_dist='bernoulli')

# Get the training functions.
f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(*mnist_data)
# Update the default function arguments.
train_args['inputs']['batchsize_unlabeled'] = bs
train_args['inputs']['batchsize_labeled'] = n_samples
train_args['inputs']['beta'] = .1
train_args['inputs']['learningrate'] = 3e-4
train_args['inputs']['beta1'] = 0.9
train_args['inputs']['beta2'] = 0.999
train_args['inputs']['samples'] = 5
test_args['inputs']['samples'] = 5
validate_args['inputs']['samples'] = 5

# Evaluate the approximated classification error with 100 MC samples for a good estimate
def custom_evaluation(model, path):
    mean_evals = model.get_output(mnist_data[2][0], 100)
    t_class = np.argmax(mnist_data[2][1], axis=1)
    y_class = np.argmax(mean_evals, axis=1)
    missclass = (np.sum(y_class != t_class, dtype='float32') / len(y_class)) * 100.
    train.write_to_logger("test 100-samples: %0.2f%%." % missclass)

# Define training loop. Output training evaluations every 1 epoch
# and the custom evaluation method every 10 epochs.
train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=custom_evaluation)
train.add_initial_training_notes("Training the skip deep generative model with %i labels. bn %s. seed %i." % (
    n_labeled, str(model.batchnorm), seed))
train.train_model(f_train, train_args,
                  f_test, test_args,
                  f_validate, validate_args,
                  n_train_batches=n_batches,
                  n_epochs=1000,
                  # Any symbolic model variable can be annealed during
                  # training with a tuple of (var_name, every, scale constant, minimum value).
                  anneal=[("learningrate", 200, 0.75, 3e-5)])
