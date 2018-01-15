import os
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
import matplotlib.pyplot as plt


def extract_logs(log_folder_name, train_or_validation='train'):
    # assumes the event log is in a folder names train or validation, under the folder passed in.
    event_folder = os.path.join(log_folder_name, train_or_validation)
    files = glob.glob(event_folder + "/*")
    files.sort(key=os.path.getmtime)
    f = files[-1]  # get the last modified file from the vents folder
    x = [e.summary.value for e in tf.train.summary_iterator(f)]
    cross_entropy = [y[0].simple_value for y in x if len(y) > 0]
    accuracy = [y[1].simple_value for y in x if len(y) > 0]
    num_iterations = len(cross_entropy)
    out = pd.DataFrame(np.vstack((cross_entropy, accuracy)).transpose(), columns=("Cross_Entropy", "Accuracy"),
                       index=np.linspace(1, num_iterations, num_iterations))
    return out


def plot_loss_and_accuracy(log_name=None, train_or_validation=None):
    if train_or_validation == None:
        train = extract_logs(log_name, "train")
        validation = extract_logs(log_name, "validation")

        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].plot(train.index, train.Cross_Entropy)
        axarr[0, 0].set_title('Training')
        axarr[0, 0].set_ylabel('Cross Entropy Loss')
        axarr[0, 0].grid()
        axarr[0, 1].plot(validation.index * 50, validation.Cross_Entropy)
        axarr[0, 1].set_ylim(axarr[0, 0].get_ylim())
        axarr[0, 1].set_title('Validation')
        axarr[0, 1].grid()
        axarr[1, 0].plot(train.index, train.Accuracy)
        # axarr[1, 0].set_title('Axis [1,0]')
        axarr[1, 0].grid()
        axarr[1, 0].set_ylabel('Accuracy')
        axarr[1, 1].plot(validation.index * 50, validation.Accuracy)
        axarr[1, 1].set_ylim(axarr[1, 0].get_ylim())
        # axarr[1, 1].set_title('Axis [1,1]')
        axarr[1, 1].grid()
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
        # plt.grid()
        #plt.show()
        plt.savefig(log_name+'/graph.png')
    else:
        stats = extract_logs(log_name, train_or_validation)
        plt.plot(stats)
        plt.grid()
        #plt.show()
        plt.savefig(log_name + '/graph'+train_or_validation+'.png')


def plot_loss_and_accuracy_rolling(log_name=None, train_or_validation=None, window=20, fname=None):
    if train_or_validation == None:
        train = extract_logs(log_name, "train")
        validation = extract_logs(log_name, "validation")

        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].plot(train.index, train.Cross_Entropy.rolling(window).mean())
        axarr[0, 0].set_title('Training')
        axarr[0, 0].set_ylabel('Cross Entropy Loss')
        axarr[0, 0].grid()
        axarr[0, 0].set_ylim([0, axarr[0, 0].get_ylim()[1]])
        axarr[0, 1].plot(validation.index * 50, validation.Cross_Entropy.rolling(int(window / 50)).mean())
        # axarr[0, 1].plot(train_index, validation.Cross_Entropy.rolling(window).mean())
        axarr[0, 1].set_ylim(axarr[0, 0].get_ylim())
        axarr[0, 1].set_title('Validation')
        axarr[0, 1].grid()
        axarr[1, 0].plot(train.index, train.Accuracy.rolling(window).mean())
        axarr[1, 0].set_ylim([0, 1])
        # axarr[1, 0].set_title('Axis [1,0]')
        axarr[1, 0].grid()
        axarr[1, 0].set_ylabel('Accuracy')
        axarr[1, 1].plot(validation.index * 50, validation.Accuracy.rolling(int(window / 50)).mean())
        axarr[1, 1].set_ylim(axarr[1, 0].get_ylim())
        # axarr[1, 1].set_title('Axis [1,1]')
        axarr[1, 1].grid()
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
        # plt.grid()
        f.text(0.5, 0.04, 'Training Step', ha='center')
        plt.show()

        if fname:
            print('saving to ',fname)
            f.savefig(fname)
            # return (train, validation)

    else:
        stats = extract_logs(log_name, train_or_validation)

        plt.plot(stats)
        plt.grid()
        plt.show()

        # return stats


