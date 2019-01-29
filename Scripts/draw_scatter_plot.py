import numpy as np
import os
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys


basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Scripts.RNATracker import preprocess_data
plt.style.use('ggplot')
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'font.size': 36, 'font.weight': 'light', 'figure.dpi': 350})

def label_dist(dist):
    '''
    dummy function
    :param dist:
    :return:
    '''
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)

def plot_scatter(expr_path, dataset, randomization_test=False):

    if not os.path.isabs(expr_path):
        expr_path = os.path.join(basedir, expr_path)
    print('Loading experiments at', expr_path)
    '''load kfolds data'''
    # X, y, folds, encoding_keys, encoding_vectors = preprocess_data(0, 4000, False, dataset,
    #                                                                4000, randomization_test=randomization_test)
    # test_length = []
    # for i, (train_indices, test_indices) in enumerate(folds):
    #     lens = []
    #     for sample in X[test_indices]:
    #         for i, val in enumerate(sample):
    #             if val != 0:
    #                 break
    #         lens.append(4000 - i)
    #     test_length.append(lens)
    #
    # test_length = np.concatenate(test_length)
    #
    # '''length splits of interest'''
    # length_split_points = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    # indices = []
    # for i, split_point in enumerate(length_split_points):
    #     if i == 0:
    #         indices.append(np.where(test_length <= split_point)[0])
    #     else:
    #         indices.append(
    #             np.intersect1d(np.where(test_length > length_split_points[i - 1]),
    #                            np.where(test_length <= split_point)))
    #
    # if not os.path.exists(os.path.join(expr_path, 'length-effect')):
    #     os.makedirs(os.path.join(expr_path, 'length-effect'))

    if not os.path.exists(os.path.join(expr_path, 'scatter')):
        os.makedirs(os.path.join(expr_path, 'scatter'))

    '''load predictions made for different folds'''
    y_test = []
    y_pred = []
    for kfold_index in range(10):
        if os.path.exists(os.path.join(expr_path, 'y_label_fold_{}.npy'.format(kfold_index))):
            y_test.append(np.load(os.path.join(expr_path, 'y_label_fold_{}.npy'.format(kfold_index))))
            y_pred.append(np.load(os.path.join(expr_path, 'y_predict_fold_{}.npy'.format(kfold_index))))
        else:
            break
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)

    if dataset == 'apex-rip':
        locations = ['KDEL', 'Mito', 'NES', 'NLS']
    elif dataset == 'cefra-seq':
        locations = ["cytosol", "insoluble", "membrane", "nucleus"]
    else:
        raise RuntimeError('No such dataset.')
    figures = []
    for i, loc in enumerate(locations):
        '''True label - predicted label scatter'''
        plt.figure(figsize=(12, 12))
        plt.title(loc)
        plt.xlabel('True localization value')
        plt.ylabel('Predicted localization value')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        corr, pval = stats.pearsonr(y_pred[:, i], y_test[:, i])
        plt.scatter(y_test[:, i], y_pred[:, i], label='Pearson corr: {0:0.3f}'.format(corr))
        plt.legend()
        plt.setp(plt.gca().get_legend().get_texts())  # legend 'list' fontsize
        plt.savefig(os.path.join(expr_path, 'scatter', loc + '_all_scatter.png'))

        '''short and long mRNA samples difference'''
        # for j, len_split in enumerate(length_split_points[:-1]):
        #     plt.close()
        #     plt.figure(figsize=(10, 7))
        #     plt.subplot(121, aspect='equal')
        #     plt.title('Shorter than {0} in {1}'.format(len_split, loc))
        #     plt.xlabel('True label')
        #     plt.ylabel('Predicted label')
        #     plt.plot([0, 1], [0, 1], 'k--')
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     indices_shorter = np.concatenate(indices[:j + 1])
        #     corr, pval = stats.pearsonr(y_pred[:, i][indices_shorter], y_test[:, i][indices_shorter])
        #     plt.scatter(y_test[:, i][indices_shorter], y_pred[:, i][indices_shorter],
        #                 label='pearson corr: {0:0.3f}'.format(corr))
        #     plt.legend()
        #
        #     plt.subplot(122, aspect='equal')
        #     plt.title('Longer than {0} in {1}'.format(len_split, loc))
        #     plt.xlabel('True label')
        #     plt.ylabel('Predicted label')
        #     plt.plot([0, 1], [0, 1], 'k--')
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     indices_longer = np.concatenate(indices[j + 1:])
        #     corr, pval = stats.pearsonr(y_pred[:, i][indices_longer], y_test[:, i][indices_longer])
        #     plt.scatter(y_test[:, i][indices_longer], y_pred[:, i][indices_longer],
        #                 label='pearson corr: {0:0.3f}'.format(corr))
        #     plt.legend()
        #     plt.savefig(os.path.join(expr_path, 'scatter', loc + '_' + str(len_split) + '_scatter.png'))
        #
        # '''length scatter'''
        # plt.figure(figsize=(10, 10))
        # plt.title(loc)
        # plt.xlabel('Length')
        # plt.ylabel('True label')
        # corr, pval = stats.pearsonr(test_length, y_test[:, i])
        # plt.scatter(test_length, y_test[:, i], label='pearson corr: {0:0.3f}'.format(corr))
        # plt.legend()
        # plt.savefig(os.path.join(expr_path, 'length-effect', loc + '_truelabel_length.png'))
        #
        # plt.figure(figsize=(10, 10))
        # plt.title(loc)
        # plt.xlabel('Length')
        # plt.ylabel('Predicted label')
        # corr, pval = stats.pearsonr(test_length, y_pred[:, i])
        # plt.scatter(test_length, y_pred[:, i], label='pearson corr: {0:0.3f}'.format(corr))
        # plt.legend()
        # plt.savefig(os.path.join(expr_path, 'length-effect', loc + '_predictedlabel_length.png'))

        # '''length factor on mRNA sequences'''
        # for j, len_split in enumerate(length_split_points[:-1]):
        #     plt.close()
        #     plt.figure()
        #     plt.subplot(121, aspect='equal')
        #     plt.title('Shorter than {0} in {1}'.format(len_split, loc))
        #     plt.xlabel('True label')
        #     plt.ylabel('Length')
        #     indices_shorter = np.concatenate(indices[:j + 1])
        #     corr, pval = stats.pearsonr(test_length[indices_shorter], y_test[:, i][indices_shorter])
        #     plt.scatter(y_test[:, i][indices_shorter], test_length[indices_shorter],
        #                 label='pearson corr: {0:0.3f}'.format(corr))
        #     plt.legend()
        #
        #     plt.subplot(122, aspect='equal')
        #     plt.title('Longer than {0} in {1}'.format(len_split, loc))
        #     plt.xlabel('True label')
        #     plt.ylabel('Length')
        #     indices_longer = np.concatenate(indices[j + 1:])
        #     corr, pval = stats.pearsonr(test_length[indices_longer], y_test[:, i][indices_longer])
        #     plt.scatter(y_test[:, i][indices_longer], test_length[indices_longer],
        #                 label='pearson corr: {0:0.3f}'.format(corr))
        #     plt.legend()
        #     plt.savefig(OUTPATH + 'length-effect/' + loc + '_' + str(len_split) + '_length.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_path", type=str, default="",
                        help="Specify saved experiment folder. If path is relative, please make sure it's relative to the root folder.")
    parser.add_argument("--randomization", type=int, default=None,
                        help="Running randomization test with three settings - {1,2,3}.")
    parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'apex-rip'],
                        help='choose from cefra-seq and apex-rip')
    args = parser.parse_args()
    print(args.randomization)

    plot_scatter(args.expr_path, args.dataset, args.randomization)

