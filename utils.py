import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from transcript_gene_data import Gene_Wrapper
import pandas as pd


def data_stats_length_effect(data):
    stat_norm = {}
    stat_discri = {}
    for gene in data:
        length = int(len(gene.seq) / 1000)  # 0 is between 0 and 1000, 1 is between 1000 and 2000 and etc.
        norm = np.sum(gene.distA)
        discri_ratio = np.max(gene.distA) / np.sum(gene.distA)
        if length in stat_norm:
            stat_norm[length].append(norm)
        else:
            stat_norm[length] = [norm]
        if length in stat_discri:
            stat_discri[length].append(discri_ratio)
        else:
            stat_discri[length] = [discri_ratio]
    # plt.ion()
    plt.subplot(2, 1, 1)
    plt.bar(np.array(list(stat_norm.keys())), [np.sum(norm) / len(norm) for norm in list(stat_norm.values())])
    plt.xlabel(int(length / 1000))
    plt.ylabel('l1 norm')
    plt.subplot(2, 1, 2)
    plt.bar(np.array(list(stat_discri.keys())), [np.sum(ratio) / len(ratio) for ratio in list(stat_discri.values())])
    plt.xlabel(int(length / 1000))
    plt.ylabel('discriminative ratio')
    plt.show()


def draw_heatmap(ids, truth, preds):
    '''
    only draw 40 instances -- 10 for each majority located position
    :param ids: 
    :param truth: 
    :param preds: 
    :return: 
    '''
    selected = []
    bound = [5, 5, 5, 5]
    for id, true_dist, pred_dist in zip(ids, truth, preds):
        mode = np.argmax(true_dist)
        if bound[mode] != 0:
            bound[mode] -= 1
            selected.append(np.concatenate((np.array([id]), true_dist, pred_dist), axis=0))
    selected = np.array(selected)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    im = ax1.pcolor(selected[:, 1:5], vmin=0.0, vmax=1.0, cmap=plt.cm.Blues, edgecolors='k')  # , aspect='auto')
    im = ax2.pcolor(selected[:, 5:], vmin=0.0, vmax=1.0, cmap=plt.cm.Blues, edgecolors='k')  # , aspect='auto')
    fig.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.yticks(np.arange(len(selected)), selected[:, 0])
    # ax1.pcolor()
    # ax2.pcolor(selected[:, 5:], cmap='Blues', vmin=0.0, vmax=1.0)

    plt.pause(20)


def screen_cefra_seq_from_ensembl_file(path_data_folder='./Data/'):
    '''
    Select cDNA with some specific criteria, such as:
    0. gene_biotype and transcipt_biotype must be protein_coding
    1. longest isoform
    :param path_data_folder: base Data folder of this project
    :return: screened cDNAs are saved to project root
    '''
    genes = {}
    all_genes_id = []
    flag = False
    with open(path_data_folder + 'Homo_sapiens.GRCh38.cdna.all.fa', "r") as cdna:
        for line in cdna:
            if line[0] == '>':
                '''
                0: transcript_id
                1: 'cdna'
                2: coordinates
                3: gene_id -> "gene:ID.xxx"
                4: gene_biotype
                5: transcript_biotype
                6: gene_symbol
                7; ....
                NOT INTER
                '''
                if flag:
                    # only retain protein_coding mRNAs
                    if gene_biotype == 'gene_biotype:protein_coding' and transcript_biotype == 'transcript_biotype:protein_coding':
                        if id in genes:
                            if len(genes[id][0]) < len(seq):  # take the longest isoform
                                # print('update longest isoform')
                                genes[id] = [seq, gene_biotype, transcript_biotype]
                        else:
                            genes[id] = [seq, gene_biotype, transcript_biotype]

                    if id not in all_genes_id:
                        all_genes_id.append(id)
                else:
                    flag = True

                seq = ""
                tokens = line.split()
                id = tokens[3].split(':')[1].split('.')[0]
                gene_biotype = tokens[4]
                transcript_biotype = tokens[5]
            else:
                seq += line[:-1]

    output = open("./cefra_seq_cDNA_screened.fa", "w")
    missing = open('./missing_cDNA.txt', "w")
    screened = open('./screened_cDNA.txt', 'w')
    mode_change = 0
    with open(path_data_folder + 'cefra-seq/Supplemental_File_3.tsv', "r") as sup:
        reader = csv.reader(sup, delimiter='\t')
        next(reader)
        for line in reader:
            id = line[0]
            type = line[2]
            longest_isoform = line[4]
            distA = [float(line[5]), float(line[7]), float(line[9]), float(line[11])]
            distB = [float(line[6]), float(line[8]), float(line[10]), float(line[12])]
            dist = [(distA[i] + distB[i]) / 2 for i in range(0, 4)]
            if np.sum(dist) < 1:
                screened.write('{} screened, target too small\n'.format(id))
                continue
            if id in genes:
                if np.argmax(distA) != np.argmax(distB):
                    mode_change += 1
                output.write(
                    '> {0} {1} {2} distA:{3}_{4}_{5}_{6} distB:{7}_{8}_{9}_{10}\n{11}\n'.
                        format(id, genes[id][1], genes[id][2], distA[0], distA[1], distA[2], distA[3],
                               distB[0], distB[1], distB[2], distB[3], genes[id][0]))
                output.flush()
            else:
                if id in all_genes_id:
                    screened.write('{} screened, not mRNA\n'.format(id))
                else:
                    missing.write("{} missing\n".format(id))
    print('total mode change:', mode_change)


def screen_apex_rip_from_ensembl_file(path_data_folder='./Data/'):
    all_genes_id = []
    genes = {}
    flag = False
    with open(path_data_folder + 'Homo_sapiens.GRCh38.cdna.all.fa', "r") as cdna:
        for line in cdna:
            if line[0] == '>':
                '''
                0: transcript_id
                1: 'cdna'
                2: coordinates
                3: gene_id -> "gene:ID.xxx"
                4: gene_biotype
                5: transcript_biotype
                6: gene_symbol
                7; ....
                NOT INTER
                '''
                if flag:  # only protein_coding mRNAs
                    if gene_biotype == 'gene_biotype:protein_coding' and transcript_biotype == 'transcript_biotype:protein_coding':
                        if id in genes:
                            if len(genes[id][0]) < len(seq):  # take the longest isoform
                                genes[id] = [seq, gene_biotype, transcript_biotype]
                        else:
                            genes[id] = [seq, gene_biotype, transcript_biotype]
                    # debug
                    if id not in all_genes_id:
                        all_genes_id.append(id)
                else:
                    flag = True

                seq = ""
                tokens = line.split()
                id = tokens[3].split(':')[1].split('.')[0]
                gene_biotype = tokens[4]
                transcript_biotype = tokens[5]
            else:
                seq += line[:-1]

    files = ['GSE106493_KDEL-HRP.gene_exp.diff', 'GSE106493_Mito-APEX2.gene_exp.diff',
             'GSE106493_NES-APEX2.gene_exp.diff', 'GSE106493_NLS-APEX2.gene_exp.diff']
    frames = []
    for fname in files:
        loc = fname.split('-')[0].split('_')[1]
        frame = pd.read_csv(os.path.join(path_data_folder, 'apex-rip', fname), sep='\t').set_index('gene_id')[
            ['value_2']]
        frame.rename(columns={'value_2': loc}, inplace=True)
        frames.append(frame)
    frame = pd.concat(frames, axis=1)

    output = open("./apex_rip_cDNA_screened.fa", "w")
    missing = open('./missing_cDNA.txt', "w")
    screened = open('./screened_cDNA.txt', 'w')

    for index, val in frame.iterrows():
        dist = list(val)
        id = index.split('.')[0]
        if 0.0 in dist or np.sum(dist) < 1:
            screened.write('{} screened, target too small\n'.format(id))
            continue
        if id in genes:
            output.write(
                '> {0} {1} {2} dist:{3}_{4}_{5}_{6}\n{7}\n'.
                    format(id, genes[id][1], genes[id][2], dist[0], dist[1], dist[2], dist[3],
                           genes[id][0]))
            output.flush()
        else:
            if id in all_genes_id:
                screened.write('{} screened, not mRNA\n'.format(id))
            else:
                missing.write("{} missing\n".format(id))


def count_mode_change(path_data_folder='./Data/'):
    '''
    Select cDNA with some specific criteria, such as:
    0. gene_biotype and transcipt_biotype must be protein_coding
    1. longest isoform
    :param path_data_folder: base Data folder of this project
    :return: screened cDNAs are saved to project root
    '''
    genes = {}
    flag = False
    with open(path_data_folder + 'raw/Homo_sapiens.GRCh38.cdna.all.fa', "r") as cdna:
        for line in cdna:
            if line[0] == '>':
                '''
                0: transcript_id
                1: 'cdna'
                2: coordinates
                3: gene_id -> "gene:ID.xxx"
                4: gene_biotype
                5: transcript_biotype
                6: gene_symbol
                7; ....
                NOT INTER
                '''
                if flag:
                    # only retain protein_coding mRNAs
                    if gene_biotype == 'gene_biotype:protein_coding' and transcript_biotype == 'transcript_biotype:protein_coding':
                        if id in genes:
                            if len(genes[id][0]) < len(seq):  # take the longest isoform
                                # print('update longest isoform')
                                genes[id] = [seq, gene_biotype, transcript_biotype]
                        else:
                            genes[id] = [seq, gene_biotype, transcript_biotype]
                else:
                    flag = True

                seq = ""
                tokens = line.split()
                id = tokens[3].split(':')[1].split('.')[0]
                gene_biotype = tokens[4]
                transcript_biotype = tokens[5]
            else:
                seq += line[:-1]

    mode_change = 0
    mag_a = 0
    mag_b = 0
    total = 0
    with open(path_data_folder + 'Supplemental_File_3.tsv', "r") as sup:
        reader = csv.reader(sup, delimiter='\t')
        next(reader)
        for line in reader:
            id = line[0]
            type = line[2]
            longest_isoform = line[4]
            distA = [float(line[5]), float(line[7]), float(line[9]), float(line[11])]
            distB = [float(line[6]), float(line[8]), float(line[10]), float(line[12])]
            dist = [(distA[i] + distB[i]) / 2 for i in range(0, 4)]
            if np.sum(dist) < 1:
                continue
            if id in genes:
                total += 1
                if np.argmax(distA) != np.argmax(distB):
                    print(id)
                    mode_change += 1
                    mag_a += np.sum(distA)
                    mag_b += np.sum(distB)
    print('mode change {0} out of {1} in total'.format(mode_change, total))
    print('magnitude of A run {0}, magnitude of B run {1}'.format(mag_a / mode_change, mag_b / mode_change))


def compare_roc(label_and_pred_dirs, legends, savetofile, title='Title'):
    """
    Compare baselines from different directories, which are all evaluated with 10 folds cross validation.
    Only micro-averaged ROC curves are shown, where different classes are unravelled to a single dimension vector.
    :param label_and_pred_dirs: A list to include all dirs of models you want to compare, each of which contains all the saved labels and predictions for different folds.
    :param legends: Usually what models you are presenting, in the order you put their dirs in the list
    :param savetofile: Where you would like to save the figures
    :param title: Title for the figure
    :return:
    """
    assert (type(label_and_pred_dirs) is list)
    assert (type(legends) is list)
    assert (len(legends) == len(label_and_pred_dirs))

    '''assemble labels from different folds'''
    labels = []
    for dir in label_and_pred_dirs:
        dir_label = []
        for kfold_index in range(10):
            dir_label.append(np.load(dir + 'y_label_fold_{}.npy'.format(kfold_index)))
        dir_label = np.concatenate(dir_label)
        labels.append(dir_label)

    '''assemble predicts from different folds'''
    predicts = []
    for dir in label_and_pred_dirs:  # assemble results from different folds
        dir_predict = []
        for kfold_index in range(10):
            dir_predict.append(np.load(dir + 'y_predict_fold_{}.npy'.format(kfold_index)))
        dir_predict = np.concatenate(dir_predict)
        predicts.append(dir_predict)

    # # load npy saved arrays
    # labels = [np.load(label_file) for label_file in label_dirs]
    # predicts = [np.load(predict_file) for predict_file in predicts]

    plt.figure(figsize=(12, 12))
    plt.title(title)
    plt.plot([0, 1], [0, 1], 'k--')

    for y_label, y_predict, legend in zip(labels, predicts, legends):
        y_label_ = list()
        for label in y_label:
            mode = np.argmax(label)
            fill = [0, 0, 0, 0]
            fill[mode] = 1
            y_label_.append(fill)
        y_label = np.array(y_label_)

        # micro-averaging
        fpr, tpr, _ = roc_curve(y_label.ravel(), y_predict.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='{0} ({1:0.3f})'.format(legend, roc_auc))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right", fontsize='26')
    plt.savefig(savetofile)


def compare_pr(label_and_pred_dirs, legends, savetofile, title='Title'):
    assert (type(label_and_pred_dirs) is list)
    assert (type(legends) is list)
    assert (len(legends) == len(label_and_pred_dirs))

    labels = []
    for dir in label_and_pred_dirs:  # assemble results from different folds
        dir_label = []
        for kfold_index in range(10):
            dir_label.append(np.load(dir + 'y_label_fold_{}.npy'.format(kfold_index)))
        dir_label = np.concatenate(dir_label)
        labels.append(dir_label)

    predicts = []
    for dir in label_and_pred_dirs:  # assemble results from different folds
        dir_predict = []
        for kfold_index in range(10):
            dir_predict.append(np.load(dir + 'y_predict_fold_{}.npy'.format(kfold_index)))
        dir_predict = np.concatenate(dir_predict)
        predicts.append(dir_predict)

    # # load npy saved arrays
    # labels = [np.load(label_file) for label_file in labels]
    # predicts = [np.load(predict_file) for predict_file in predicts]

    plt.figure(figsize=(12, 12))
    plt.title(title)

    for y_label, y_predict, legend in zip(labels, predicts, legends):
        y_label_ = list()
        for label in y_label:
            mode = np.argmax(label)
            fill = [0, 0, 0, 0]
            fill[mode] = 1
            y_label_.append(fill)
        y_label = np.array(y_label_)
        # micro-averaging
        precision, recall, _ = precision_recall_curve(y_label.ravel(),
                                                      y_predict.ravel())
        average_precision = average_precision_score(y_label, y_predict,
                                                    average="micro")

        plt.plot(recall, precision, label='{0} ({1:0.3f})'.format(legend, average_precision))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right", fontsize='26')
    plt.savefig(savetofile)


def multiclass_roc(model_out_dir, savetofile, title='', locations=("cytosol", "insoluble", "membrane", "nucleus")):
    """
    To evaluate a single model's performance on four different locations, i.e.
    to compare ROC curves of different locations of a single model.
    :param model_out_dir: usually the RNATracker's output dir where all folds results are stored.
    :param savetofile: where you want to save the figure
    :param title: title for the figure
    :return: saved figure of above properties
    """
    '''assemble labels from different folds'''
    y_label = []
    for kfold_index in range(10):
        y_label.append(np.load(model_out_dir + 'y_label_fold_{}.npy'.format(kfold_index)))
    y_label = np.concatenate(y_label)

    y_label_ = list()
    for label in y_label:
        mode = np.argmax(label)
        fill = [0, 0, 0, 0]
        fill[mode] = 1
        y_label_.append(fill)
    y_label = np.array(y_label_)

    '''assemble predicts from different folds'''
    y_predict = []
    for kfold_index in range(10):
        y_predict.append(np.load(model_out_dir + 'y_predict_fold_{}.npy'.format(kfold_index)))
    y_predict = np.concatenate(y_predict)

    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 12))
    from itertools import cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='{0} ({1:0.3f})'
                       ''.format(locations[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.title(title)
    plt.savefig(savetofile)


def multiclass_pr(model_out_dir, savetofile, title='', locations=("cytosol", "insoluble", "membrane", "nucleus")):
    '''assemble labels from different folds'''
    y_label = []
    for kfold_index in range(10):
        y_label.append(np.load(model_out_dir + 'y_label_fold_{}.npy'.format(kfold_index)))
    y_label = np.concatenate(y_label)

    y_label_ = list()
    for label in y_label:
        mode = np.argmax(label)
        fill = [0, 0, 0, 0]
        fill[mode] = 1
        y_label_.append(fill)
    y_label = np.array(y_label_)

    '''assemble predicts from different folds'''
    y_predict = []
    for kfold_index in range(10):
        y_predict.append(np.load(model_out_dir + 'y_predict_fold_{}.npy'.format(kfold_index)))
    y_predict = np.concatenate(y_predict)

    n_classes = 4
    from itertools import cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_label[:, i],
                                                            y_predict[:, i])
        average_precision[i] = average_precision_score(y_label[:, i], y_predict[:, i])

    plt.figure(figsize=(12, 12))

    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color,
                 label='{0} ({1:0.3f})'.format(locations[i], average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(savetofile)


def get_kmer_features(fasta_file, output_csv, kmer_length_top=7):
    '''
    a simple function to extract kmer features for baseline algorithms.
    '''
    kmers = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line[0] == '>':
                tokens = line[1:].split()
                id = tokens[0]
                gene_biotype = tokens[1].split(':')[1]
                dist = tokens[3].split(':')[1]
            else:
                seq = line[:-1]
                kmer_dict = {}
                kmer_dict['id'] = id
                kmer_dict['gene_biotype'] = gene_biotype
                kmer_dict['dist'] = dist
                kmer_dict['length'] = len(seq)
                for kmer_len in range(1, kmer_length_top + 1):
                    for i in range(len(seq)):
                        if i + kmer_len <= len(seq):
                            if seq[i:i + kmer_len] not in kmer_dict:
                                kmer_dict[seq[i:i + kmer_len]] = 1
                            else:
                                kmer_dict[seq[i:i + kmer_len]] += 1
                kmers.append(kmer_dict)
        fieldnames = set().union(*[list(kmer_dict.keys()) for kmer_dict in kmers])

    with open(output_csv, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
        writer.writeheader()
        writer.writerows(kmers)


def violin_plot(dataset):
    data = Gene_Wrapper.seq_data_loader(False, dataset)
    X = [len(gene.seq) for gene in data]
    y = np.array([np.array(gene.dist) / sum(gene.dist) for gene in data])

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_ylabel('Localization values')

    fig = plt.figure(figsize=(12, 12))
    plt.violinplot([y[:, i] for i in range(4)], showmeans=True)
    ax = fig.axes[0]
    if dataset == "cefra-seq":
        locations = ["cytosol", "insoluble", "membrane", "nucleus"]
    elif dataset == "apex-rip":
        locations = ['KDEL', 'Mito', 'NES', 'NLS']
    else:
        raise RuntimeError('No such dataset')
    set_axis_style(ax, locations)
    plt.xticks(rotation=-20)
    plt.savefig('Graph/violin_{}.png'.format(dataset))


if __name__ == "__main__":
    # screen_cefra_seq_from_ensembl_file()
    # exit()
    # get_kmer_features('./Data/apex-rip/apex_rip_cDNA_screened.fa', 'kmer_apex.csv', 5)
    # get_kmer_features('./data/cDNA/ensembl_cDNA_screened.fa', 'kmer_feature.csv', 5)
    # count_mode_change()

    plt.style.use('ggplot')
    matplotlib.rcParams.update(
        {'font.family': 'Times New Roman', 'font.size': 36, 'font.weight': 'light', 'figure.dpi': 350})
    # violin_plot('cefra-seq')
    # violin_plot('apex-rip')

    '''cefra-seq'''
    compare_roc([
        './Results/SGDModel-10foldcv/cefra-seq/2019-01-08-11-14-51-cnn_bilstm-adam/',
        './Results/SGDModel-10foldcv/cefra-seq/2018-12-28-18-03-18-cnn-four-cnn/',
        './Results/RNATracker-10foldcv/cefra-seq/2018-08-17-18-20-51-new_split/',
        './Results/RNATracker-10foldcv/cefra-seq/2018-08-18-19-08-14-new_split_no_attention/',
        './Results/SeparateEncoding-10foldcv/cefra-seq/2018-08-17-00-30-17-no-bn/',
        './Results/RNATracker-10foldcv/cefra-seq/2018-08-18-18-52-51-new_split_ann_joint_encoding/',
        './Results/kmer-baseline-10foldcv/cefra-seq/2018-12-21-16-09-27-NN-5Mer-cefra-seq/',
        './Results/kmer-baseline-10foldcv/cefra-seq/2018-12-21-16-59-37-LR-5Mer-cefra-seq/',],
        ['RNATracker-FullLength', 'NoLSTM', 'RNATracker-FixedLength', 'NoAttention', 'Seq+Struct',
         'SeqxStruct', 'NN-5Mer', 'LR-5Mer']
        , './Graph/cefra-seq-10foldcv-compare-roc.png',
        title='Micro-averaged ROC')

    compare_pr([
        './Results/SGDModel-10foldcv/cefra-seq/2019-01-08-11-14-51-cnn_bilstm-adam/',
        './Results/SGDModel-10foldcv/cefra-seq/2018-12-28-18-03-18-cnn-four-cnn/',
        './Results/RNATracker-10foldcv/cefra-seq/2018-08-17-18-20-51-new_split/',
        './Results/RNATracker-10foldcv/cefra-seq/2018-08-18-19-08-14-new_split_no_attention/',
        './Results/SeparateEncoding-10foldcv/cefra-seq/2018-08-17-00-30-17-no-bn/',
        './Results/RNATracker-10foldcv/cefra-seq/2018-08-18-18-52-51-new_split_ann_joint_encoding/',
        './Results/kmer-baseline-10foldcv/cefra-seq/2018-12-21-16-09-27-NN-5Mer-cefra-seq/',
        './Results/kmer-baseline-10foldcv/cefra-seq/2018-12-21-16-59-37-LR-5Mer-cefra-seq/',],
        ['RNATracker-FullLength', 'NoLSTM', 'RNATracker-FixedLength', 'NoAttention', 'Seq+Struct',
         'SeqxStruct', 'NN-5Mer', 'LR-5Mer']
        , './Graph/cefra-seq-10foldcv-compare-pr.png',
        title='Micro-averaged PR')

    multiclass_roc('./Results/SGDModel-10foldcv/cefra-seq/2019-01-08-11-14-51-cnn_bilstm-adam/',
                   './Graph/RNATracker-cefra-seq-roc.png', title='ROC by fraction')

    multiclass_pr('./Results/SGDModel-10foldcv/cefra-seq/2019-01-08-11-14-51-cnn_bilstm-adam/',
                  './Graph/RNATracker-cefra-seq-pr.png', title='PR by fraction')

    '''apex-rip'''
    compare_roc([
        './Results/SGDModel-10foldcv/apex-rip/2019-01-04-16-36-23-cnn_bilstm-adam/',
        './Results/SGDModel-10foldcv/apex-rip/2018-12-22-22-00-37-cnn-four-cnn/',
        './Results/RNATracker-10foldcv/apex-rip/2018-12-20-19-31-47-apex-rip-no-reg/',
        './Results/RNATracker-10foldcv/apex-rip/2018-12-21-17-32-22-no_att/',
        './Results/SeparateEncoding-10foldcv/apex-rip/2018-12-23-23-12-54/',
        './Results/RNATracker-10foldcv/apex-rip/2018-12-22-11-35-50-apex-rip-with-annotations/',
        './Results/kmer-baseline-10foldcv/apex-rip/2018-12-21-16-25-32-NN-5Mer-apex-rip/',
        './Results/kmer-baseline-10foldcv/apex-rip/2018-12-21-16-57-54-LR-5Mer-apex-rip/',],
        ['RNATracker-FullLength', 'NoLSTM', 'RNATracker-FixedLength', 'NoAttention', 'Seq+Struct',
         'SeqxStruct', 'NN-5Mer', 'LR-5Mer']
        , './Graph/apex-rip-10foldcv-compare-roc.png',
        title='Micro-averaged ROC')

    compare_pr([
        './Results/SGDModel-10foldcv/apex-rip/2019-01-04-16-36-23-cnn_bilstm-adam/',
        './Results/SGDModel-10foldcv/apex-rip/2018-12-22-22-00-37-cnn-four-cnn/',
        './Results/RNATracker-10foldcv/apex-rip/2018-12-20-19-31-47-apex-rip-no-reg/',
        './Results/RNATracker-10foldcv/apex-rip/2018-12-21-17-32-22-no_att/',
        './Results/SeparateEncoding-10foldcv/apex-rip/2018-12-23-23-12-54/',
        './Results/RNATracker-10foldcv/apex-rip/2018-12-22-11-35-50-apex-rip-with-annotations/',
        './Results/kmer-baseline-10foldcv/apex-rip/2018-12-21-16-25-32-NN-5Mer-apex-rip/',
        './Results/kmer-baseline-10foldcv/apex-rip/2018-12-21-16-57-54-LR-5Mer-apex-rip/',],
        ['RNATracker-FullLength', 'NoLSTM', 'RNATracker-FixedLength', 'NoAttention', 'Seq+Struct',
         'SeqxStruct', 'NN-5Mer', 'LR-5Mer']
        , './Graph/apex-rip-10foldcv-compare-pr.png',
        title='Micro-averaged PR')

    multiclass_roc('./Results/SGDModel-10foldcv/apex-rip/2019-01-04-16-36-23-cnn_bilstm-adam/',
                   './Graph/RNATracker-apex-rip-roc.png', title='ROC by fraction',
                   locations=['KDEL', 'Mito', 'NES', 'NCL'])

    multiclass_pr('./Results/SGDModel-10foldcv/apex-rip/2019-01-04-16-36-23-cnn_bilstm-adam/',
                  './Graph/RNATracker-apex-rip-pr.png', title='PR by fraction',
                  locations=['KDEL', 'Mito', 'NES', 'NCL'])

    '''new cDNA data batch, a little bit better'''
    #
    # compare_roc(['./Results/RNATracker-10foldcv/new_mrna_file-2018-08-15-15-09-28/',
    #              './Results/RNATracker-10foldcv/new_mrna_file-2018-08-16-13-01-34/',
    #              './Results/RNATracker-10foldcv/new_data_smaller_net-2018-08-16-20-39-43/',
    #              './Results/RNATracker-10foldcv/new_data_bn-2018-08-16-20-06-53/',
    #              './Results/RNATracker-10foldcv/new_data_small_l1_reg-2018-08-16-20-00-11/',
    #              './Results/RNATracker-10foldcv/slice_3utr_200bps_smaller_net-2018-08-16-00-16-38/',
    #              './Results/RNATracker-10foldcv/slice_3utr_1000bps-2018-08-16-00-16-48/',
    #              './Results/SeparateEncoding-10foldcv/no-bn-2018-08-17-00-30-17/',
    #              './Results/RNATracker-10foldcv/new_ann_joint_encoding-2018-08-17-00-23-59/',
    #              ],
    #             ['4000bps', '4000bps_filter_len_3', '4000bps_smaller_net', '4000bps_batchnorm', '4000bps_l1_reg', '200bps', '1000bps', '2d-struct-separate-encoding', '2d-struct-joint-encoding']
    #             , './Graph/new_data_compare_roc.png',
    #             title='Micro-averaged ROC')

    '''pretrianed test results, which is not really impressive'''
    # multiclass_roc('./Results/RNATracker-10foldcv/expr_nontrainable-pretrained_weights-2018-06-06-16-10-34/',
    #                './Graph/pretrained_roc.png', title='Pretrained test')
    # multiclass_pr('./Results/RNATracker-10foldcv/permute-test-no-annotation-2018-05-23-11-38-31/',
    #                './Graph/permute_pr.png', title='Permutating test')

    '''permutation test results'''
    # multiclass_roc('./Results/RNATracker-10foldcv/permute-test-no-annotation-2018-05-23-11-38-31/',
    #                './Graph/permute_roc.png', title='Permutating test')
    # multiclass_pr('./Results/RNATracker-10foldcv/permute-test-no-annotation-2018-05-23-11-38-31/',
    #                './Graph/permute_pr.png', title='Permutating test')

    ''' versions using batch-normalization; corresponding to the build_model() type model'''
    # compare_roc(['./Results/SeparateEncoding-10foldcv/2018-05-17-01-05-38/',
    #              './Results/RNATracker-10foldcv/RNATracker-10foldcv-2018-05-15-16-30-24/',
    #              './Results/RNATracker-10foldcv/without-annotation-10foldcv-2018-05-15-18-46-55/',
    #              './Results/RNATracker-10foldcv/without-attention-10foldcv-2018-05-15-17-54-22/',
    #              './Results/kmer-baseline-10foldcv/LR-5Mer-2018-05-16-15-46-13/',
    #              './Results/kmer-baseline-10foldcv/NN-5Mer-2018-05-16-15-47-50/'
    #              ],
    #             ['RNATracker separate encoding', 'RNATracker unified encoding', 'Without Annotation',
    #              'Without Attention', 'LR-5Mer', 'NN-5Mer']
    #             # ['RNATracker', 'Without Annotation', 'Without Attention']
    #             , './Graph/10foldcv_compare_roc.png',
    #             title='Micro-averaged ROC')

    '''without batchnormalization ones, these are presented on the paper'''
    # compare_roc(['./Results/RNATracker-10foldcv/without-annotation-no-bn-10foldcv-2018-05-16-15-04-11/',
    #              './Results/SeparateEncoding-10foldcv/no-bn-2018-05-17-13-44-55/',
    #              './Results/RNATracker-10foldcv/RNATracker-no-bn-new-annotation-2018-05-16-16-23-14/',
    #              './Results/RNATracker-10foldcv/without-attention-10foldcv-2018-05-15-17-54-22/',
    #              './Results/kmer-baseline-10foldcv/LR-5Mer-2018-05-16-15-46-13/',
    #              './Results/kmer-baseline-10foldcv/NN-5Mer-2018-05-16-15-47-50/'
    #              ],
    #             ['RNATracker', 'RNATracker-struct-1', 'RNATracker-struct-2',
    #              'RNATracker-without-attention', 'LR-5Mer', 'NN-5Mer']
    #             , './Graph/10foldcv_nobn_compare_roc.png',
    #             title='Micro-averaged ROC')
    #
    # compare_pr(['./Results/RNATracker-10foldcv/without-annotation-no-bn-10foldcv-2018-05-16-15-04-11/',
    #             './Results/SeparateEncoding-10foldcv/no-bn-2018-05-17-13-44-55/',
    #             './Results/RNATracker-10foldcv/RNATracker-no-bn-new-annotation-2018-05-16-16-23-14/',
    #             './Results/RNATracker-10foldcv/without-attention-10foldcv-2018-05-15-17-54-22/',
    #             './Results/kmer-baseline-10foldcv/LR-5Mer-2018-05-16-15-46-13/',
    #             './Results/kmer-baseline-10foldcv/NN-5Mer-2018-05-16-15-47-50/'
    #             ],
    #            ['RNATracker', 'RNATracker-struct-1', 'RNATracker-struct-2',
    #             'RNATracker-without-attention', 'LR-5Mer', 'NN-5Mer']
    #            , './Graph/10foldcv_nobn_compare_pr.png',
    #            title='Micro-averaged PR')
    #
    # multiclass_roc('./Results/RNATracker-10foldcv/without-annotation-no-bn-10foldcv-2018-05-16-15-04-11/',
    #                './Graph/10foldcv_multiclass_roc.png', title='ROC per location for RNATracker')
    #
    # multiclass_pr('./Results/RNATracker-10foldcv/without-annotation-no-bn-10foldcv-2018-05-16-15-04-11/',
    #               './Graph/10foldcv_multiclass_pr.png', title='PR per location for RNATracker')
