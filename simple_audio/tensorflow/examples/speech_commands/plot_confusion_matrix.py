import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def get_key_values_from_confusion_matrix(conf_arr):
    largest_correct = 0
    largest_mistake = 0
    largest_incorrect_ratio = 0.0
    for x in range(len(conf_arr)):
        for y in range(len(conf_arr)):
            if x != y:
                if conf_arr[x][y] > largest_mistake:
                    largest_mistake = conf_arr[x][y]
                if conf_arr[x][x] > 0 and conf_arr[x][y] / conf_arr[x][x] < 1.0 and conf_arr[x][y] / conf_arr[x][x] > largest_incorrect_ratio:
                    largest_incorrect_ratio = conf_arr[x][y] / conf_arr[x][x]
            else:
                if conf_arr[x][y] > largest_correct:
                    largest_correct = conf_arr[x][y]


    return largest_correct, largest_mistake, largest_incorrect_ratio


def get_precision_and_recall_from_confusion_matrix(conf_arr, index):
    precision_sum = np.sum(conf_arr, axis=1)
    recall_sum = np.sum(conf_arr, axis=0)
    precision = 0.0
    recall = 0.0
    if precision_sum[index] > 0.0:
        precision = conf_arr[index][index]/precision_sum[index]
    if recall_sum[index] > 0.0:
        recall = conf_arr[index][index]/recall_sum[index]
    return precision, recall

def calc_nomalised_array(conf_arr):
    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
    return norm_conf

def create_pretty_conf_matrix_and_save(labels, conf_arr, filename):
    c = mcolors.ColorConverter().to_rgb
    largest_correct, largest_mistake, largest_incorrect_ratio = get_key_values_from_confusion_matrix(conf_arr)
    rvb = make_colormap(
        [c('white'), (1.0 / largest_correct) / 2, c('orange'), c('red'), largest_incorrect_ratio,
         c('green')])

    norm_conf = calc_nomalised_array(conf_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=rvb,  # plt.cm.RdYlGn,
                    interpolation='nearest')

    width = len(labels)
    height = len(labels)

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    plt.savefig(filename, format='png')



def produce_charts():
    conf_arr = [[12, 0, 0, 0, 0, 0],
                [0, 27, 0, 2, 1, 6],
                [2, 2, 11, 1, 0, 1],
                [1, 1, 4, 17, 2, 3],
                [0, 0, 1, 3, 28, 0],
                [2, 2, 4, 4, 0, 31]]

    conf_arr_100_it = [[12, 0, 0, 0, 0, 0],
                       [26, 1, 0, 0, 0, 9],
                       [3, 5, 0, 0, 0, 9],
                       [0, 9, 0, 3, 0, 16],
                       [0, 12, 0, 1, 0, 19],
                       [0, 12, 0, 1, 0, 30]]

    conf_arr_6000_it = [[12, 0, 0, 0, 0, 0],
                        [0, 27, 0, 1, 2, 6],
                        [2, 2, 10, 0, 1, 2],
                        [1, 1, 4, 17, 2, 3],
                        [0, 0, 1, 3, 28, 0],
                        [2, 3, 3, 4, 0, 31]]

    conf_arr_9200_it = [[13, 0, 0, 0, 0, 0],
                        [1, 34, 0, 3, 0, 1],
                        [0, 1, 13, 0, 0, 2],
                        [1, 0, 4, 41, 3, 6],
                        [0, 0, 1, 3, 11, 0],
                        [3, 2, 0, 6, 0, 31]]

    conf_arr_14200_it = [[13, 0, 0, 0, 0, 0],
                         [0, 33, 3, 1, 1, 1],
                         [0, 1, 14, 0, 0, 1],
                         [1, 0, 3, 41, 1, 9],
                         [0, 0, 0, 3, 12, 0],
                         [4, 1, 2, 5, 0, 30]]

    conf_arr_19200_it = [[13, 0, 0, 0, 0, 0],
                         [0, 33, 1, 4, 0, 1],
                         [0, 1, 15, 0, 0, 0],
                         [1, 0, 6, 41, 1, 6],
                         [0, 0, 1, 4, 10, 0],
                         [2, 1, 1, 7, 0, 31]]

    conf_convnet = [[12, 0, 0, 0, 0, 0],
                    [1, 30, 0, 2, 1, 2],
                    [0, 0, 13, 3, 0, 1],
                    [0, 0, 1, 25, 1, 1],
                    [0, 1, 2, 1, 28, 0],
                    [0, 2, 0, 3, 1, 37]]

    conf_alexnet = [[13, 0, 0, 0, 0, 0],
                    [0, 9, 0, 2, 1, 1],
                    [0, 0, 16, 0, 0, 0],
                    [0, 0, 0, 46, 1, 8],
                    [0, 0, 0, 1, 14, 0],
                    [1, 2, 0, 4, 0, 37]]
    labels = ['silence', 'unknown', 'possum', 'cat', 'dog', 'bird']
    create_pretty_conf_matrix_and_save(labels, conf_arr, 'deep_ear_.png')
    create_pretty_conf_matrix_and_save(labels, conf_arr_100_it, 'deep_ear_100_it.png')
    create_pretty_conf_matrix_and_save(labels, conf_arr_6000_it, 'deep_ear_6000_it.png')
    create_pretty_conf_matrix_and_save(labels, conf_arr_9200_it, 'deep_ear_9200_it.png')
    create_pretty_conf_matrix_and_save(labels, conf_arr_14200_it, 'deep_ear_14200_it.png')
    create_pretty_conf_matrix_and_save(labels, conf_arr_19200_it, 'deep_ear_19200_it.png')
    create_pretty_conf_matrix_and_save(labels, conf_alexnet, 'alexnet.png')
    create_pretty_conf_matrix_and_save(labels, conf_convnet, 'convnet_.png')


def run_test(index = 2):
    conf_arr =  [[12,  0,  0,  0,  0,  0],
 [ 1, 27,  0,  1,  0,  7],
 [ 3,  0, 14,  0,  0,  0],
 [ 0,  2,  3, 20,  0,  3],
 [ 0,  0,  1,  4, 27,  0],
 [ 0,  5,  4,  0,  1, 33]]
    precision, recall  = get_precision_and_recall_from_confusion_matrix(conf_arr, index)
    print(precision)
    print(recall)

if __name__ == '__main__':
    run_test()