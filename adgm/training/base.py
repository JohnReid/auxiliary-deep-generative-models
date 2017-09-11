import logging
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
from ..utils import env_paths as paths
import seaborn as sns
import numpy as np
import cPickle as pkl

class Train(object):
    """
    The :class:'Train' class general training functions.
    It should be subclassed when implementing new types of training loops.
    """

    def __init__(self, model, model_logger, model_directory, pickle_f_custom_freq=None, custom_eval_func=None):
        """
        Initialisation of the basic architecture and programmatic settings of any training procedure.
        This method should be called from any subsequent inheriting training procedure.
        :param model: The model to train on.
        :param pickle_f_custom_freq: The number of epochs between each serialization, plotting etc.
        :param custom_eval_func: The custom evaluation function taking (model, output_path) as arguments.
        """
        self.model = model
        self.model_logger = model_logger
        self.open_csv_files(model_directory)
        self.x_dist = None
        self.custom_eval_func = custom_eval_func
        self.eval_train = {}
        self.eval_test = {}
        self.eval_validation = {}
        self.pickle_f_custom_freq = pickle_f_custom_freq
        self.test_auprc_list = []

    def open_csv_files(self, model_directory):
        #
        # Open a file to save learning curve information
        learningcsvfile = os.path.join(model_directory, 'learning.csv')
        self.model_logger.info('Will write learning curves to: %s', learningcsvfile)
        self.learning_csv = open(learningcsvfile, 'w')
        # These output columns can change, don't hardcode them
        # self.learning_csv.write('epoch,time,lb,lb-labeled,lb-unlabeled,test,validation')
        # Csv file for test evaluation log
        testevalcsvfile = os.path.join(model_directory, 'testeval.csv')
        self.model_logger.info('Will write test eval values to: %s', testevalcsvfile)
        self.testeval_csv = open(testevalcsvfile, 'w')

    def train_model(self, *args):
        """
        This is where the training of the model is performed.
        :param n_epochs: The number of epochs to train for.
        """
        raise NotImplementedError

    def dump_dicts(self):
        """
        Dump the model evaluation dictionaries
        """
        p_train = paths.get_plot_evaluation_path_for_model(self.model.get_root_path(), "train_dict.pkl")
        pkl.dump(self.eval_train, open(p_train, "wb"))
        p_test = paths.get_plot_evaluation_path_for_model(self.model.get_root_path(), "test_dict.pkl")
        pkl.dump(self.eval_test, open(p_test, "wb"))
        p_val = paths.get_plot_evaluation_path_for_model(self.model.get_root_path(), "validation_dict.pkl")
        pkl.dump(self.eval_validation, open(p_val, "wb"))

    def plot_eval(self, eval_dict, labels, path_extension=""):
        """
        Plot the loss function in a overall plot and a zoomed plot.
        :param path_extension: If the plot should be saved in an incremental way.
        """

        def plot(x, y, fit, label):
            sns.regplot(np.array(x), np.array(y), fit_reg=fit, label=label, scatter_kws={"s": 5})

        plt.clf()
        plt.subplot(211)
        idx = np.array(eval_dict.values()[0]).shape[0]
        x = np.array(eval_dict.values())
        for i in range(idx):
            plot(eval_dict.keys(), x[:, i], False, labels[i])
        plt.legend()
        plt.subplot(212)
        for i in range(idx):
            plot(eval_dict.keys()[-int(len(x) * 0.25):], x[-int(len(x) * 0.25):][:, i], True, labels[i])
        plt.xlabel('Epochs')
        plt.savefig(paths.get_plot_evaluation_path_for_model(self.model.get_root_path(), path_extension+".png"))

    def write_to_logger(self, s):
        """
        Write a string to the logger and the console.
        :param s: A string with the text to print.
        """
        self.model_logger.info(s)


    def add_initial_training_notes(self, s):
        """
        Add an initial text for the model as a personal
        note, to keep an order of experimental testing.
        :param s: A string with the text to print.
        """
        if len(s) == 0:
            return
        line_length = 10
        self.write_to_logger("### INITIAL TRAINING NOTES ###")
        w_lst = s.split(" ")
        new_s = ""
        for i in range(len(w_lst)):
            if (not i == 0) and (i % line_length == 0):
                new_s += "\n"
            new_s += " " + w_lst[i]
        self.write_to_logger(new_s)