# observe that the inefficiency mainly comes from calculating kl divergence
import datetime
from collections import OrderedDict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--saved_expr", type=str,
                    default='',
                    help="If specified, this program will try to load cached results from a previous experiment")
parser.add_argument("--mask_window", required=True, type=int, help="An integer denoting the size of the mask window")
parser.add_argument("--stride", type=int, default=1, help="")
parser.add_argument("--permute_mask", action="store_true", help="Mask window uses 100 randomly permuted sequences; uses zero masking by default.")
parser.add_argument("--export_zipcodes", action="store_true", help="Enable exporting zipcodes")
parser.add_argument("--kl_sig", type=float, default=0.0106, help="Extract zipcodes identified by a single kl cutoff")
args = parser.parse_args()

plt.style.use('ggplot')
# matplotlib.rcParams.update({'figure.dpi': 350})
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18, 'font.weight': 'light', 'figure.dpi': 350})

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session=sess)

from Scripts.SGDModel import cnn_bilstm_model
from transcript_gene_data import Gene_Wrapper
from keras.preprocessing.sequence import pad_sequences

batch_size = 256
nb_classes = 4
seq_dim = 4
ann_dim = 6

np.random.seed(1234)

class CSVLogger:

    def __init__(self, name, path, fieldnames):
        self.log_file = open(os.path.join(path, name), 'w', newline='')
        self.writer = csv.DictWriter(self.log_file, fieldnames, restval=',')
        self.writer.writeheader()

    def update_with_dict(self, dict_entries):
        self.writer.writerow(dict_entries)
        self.log_file.flush()

    def close(self):
        del self.writer
        self.log_file.close()


def label_dist(dist):
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)

locs = ['cyto', 'insol', 'membr', 'nucl']

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

encoding_annotation = OrderedDict([
    ('UNK', [0, 0, 0, 0, 0, 0]),  # for padding use
    ('f', [1, 0, 0, 0, 0, 0]),  # 'dangling start',
    ('t', [0, 1, 0, 0, 0, 0]),  # dangling end',
    ('i', [0, 0, 1, 0, 0, 0]),  # 'internal loop',
    ('h', [0, 0, 0, 1, 0, 0]),  # 'hairpin loop',
    ('m', [0, 0, 0, 0, 1, 0]),  # 'multi loop',
    ('s', [0, 0, 0, 0, 0, 1])  # 'stem'
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))
annotation_encoding_keys = list(encoding_annotation.keys())
annotation_encoding_vectors = np.array(list(encoding_annotation.values()))

gene_data = Gene_Wrapper.seq_data_loader(False, False, 0, 4000)
encoding_keys = seq_encoding_keys
encoding_vectors = seq_encoding_vectors
X = pad_sequences([[encoding_keys.index(c) for c in gene.seq] for gene in gene_data],
                  maxlen=4000,
                  dtype=np.int8, value=encoding_keys.index('UNK'))  # , truncating='post')
y = np.array([label_dist(gene.dist) for gene in gene_data])
ids = np.array([gene.id for gene in gene_data])
true_length = np.array([len(gene.seq) for gene in gene_data])

if args.saved_expr == "":
    print('New experiment')
    OUTPATH = os.path.join(basedir, 'Results', 'cefra-seq', 'SGDModel-10foldcv',
                           'mask-test-{}-'.format(args.mask_window) + str(datetime.datetime.now()).
                           split('.')[0].replace(':', '-').replace(' ', '-') + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
else:
    OUTPATH = args.saved_expr
    if not os.path.exists(OUTPATH):
        raise Exception('{} doesn\'t exists'.format(OUTPATH))
    if 'mask-test' in OUTPATH:
        s = OUTPATH[OUTPATH.index('mask-test-') + len('mask-test-'):].split('-')[0]
        s = float(s)
        if s == 2018:
            raise Warning(
                'Mask window size for saved experiment is not clear. Are you sure the mask window size agrees?')
        elif s != args.mask_window:
            raise Exception(
                'Mask window size specified to this program is different from what has been used in the saved experiment.')
    else:
        raise Exception('Saved experiment folder not reocognized.')

MAT_OUTPATH = os.path.join(OUTPATH, 'matrices')
SG_OUTPATH = os.path.join(OUTPATH, 'summary_graphs_' + str(datetime.datetime.now()).
                          split('.')[0].replace(':', '-').replace(' ', '-') + '/')

if not os.path.exists(MAT_OUTPATH):
    os.makedirs(MAT_OUTPATH)
if not os.path.exists(SG_OUTPATH):
    os.makedirs(SG_OUTPATH)
if not os.path.exists(os.path.join(OUTPATH, 'ind_plots')):
    os.makedirs(os.path.join(OUTPATH, 'ind_plots'))

print('OUTPATH:', OUTPATH)

'''load model'''
model = cnn_bilstm_model()
model.load_weights(os.path.join(basedir, 'Results/SGDModel-10foldcv/cefra-seq/2019-01-08-11-14-51-cnn_bilstm-adam/weights_fold_0.h5'))

'''loading conservation scores'''
print('Loading conservation scores')
path_to_conservation_scores = os.path.join(basedir, 'Transcript_Coordinates_Mapping/all_conservation_scores.txt')
conserv_scores_dict = {}
genome_to_trans_id_mapping = {}
with open(path_to_conservation_scores, 'r') as conserv_scores_f:
    for line in conserv_scores_f:
        if line[0] == '>':
            tokens = line.rstrip().split()
            genome_id = tokens[1].split(':')[1]
            trans_id = tokens[2].split(':')[1]
        else:
            scores = line.rstrip().split(',')[-4000:]
            scores = [float(score) if score != 'n/a' else 0. for score in scores]
            act_len = len(scores)  # maximum set to 4000
            scores = np.concatenate([[0.] * (4000 - act_len), scores])
            conserv_scores_dict[genome_id] = scores
            genome_to_trans_id_mapping[genome_id] = trans_id
print('Finished')

'''function to get the length of 3'UTR'''
exon_ref_path = os.path.join(basedir, 'Transcript_Coordinates_Mapping/lib/exon_reference_human_processed.tsv')
import pandas as pd

exon_ref_pd = pd.read_csv(exon_ref_path, sep="\t",
                          dtype={"Transcript stable ID": str,
                                 "Exon rank in transcript": int,
                                 "Chromosome/scaffold name": str,
                                 "cDNA coding start": float,
                                 "cDNA coding end": float,
                                 "Exon region start (bp)": int,
                                 "Exon region end (bp)": int,
                                 "5' UTR start": float,
                                 "5' UTR end": float,
                                 "3' UTR start": float,
                                 "3' UTR end": float,
                                 "CDS start": float,
                                 "CDS end": float,
                                 "Strand": float,
                                 "cDNA exon start": int,
                                 "cDNA exon end": int
                                 })


def get_utr3_length(transcript_id):
    transcripts = exon_ref_pd.loc[exon_ref_pd['Transcript stable ID'] == transcript_id]
    # assume 3'UTR is always contingent
    diff = transcripts["3' UTR end"] - transcripts["3' UTR start"]
    import math
    length = int(sum([i for i in diff if not math.isnan(i)]))
    return length


'''find out a distinctive sample for masking test'''
from scipy.stats import entropy, pearsonr, ks_2samp
import scipy.interpolate as interp

mask_window_size = args.mask_window
kl_threshold = np.linspace(0.1, 500, 100) / 10000
out_f = open(os.path.join(OUTPATH, 'outputs.txt'), 'a')

pr_att_con = []
pr_kl_con = []
'''store conservation scores above or below some threshold'''
'''singular conservation scores are actually no good'''
'''singular conservation scores should still be maintained nonetheless, since they are i.i.d. for real'''
above_kl_threshold = {}
below_kl_threshold = {}
# above_kl_threshold_sg = {}
# below_kl_threshold_sg = {}
non_overlapping_counts = {}
unioned_counts = {}
positive_zipcodes_np = {}
negative_zipcodes_np = {}
positive_zipcodes_unioned = {}
negative_zipcodes_unioned = {}

for thres in kl_threshold:
    above_kl_threshold[thres] = []
    below_kl_threshold[thres] = []
    # above_kl_threshold_sg[thres] = []
    # below_kl_threshold_sg[thres] = []
    non_overlapping_counts[thres] = 0
    unioned_counts[thres] = 0
    positive_zipcodes_np[thres] = []
    negative_zipcodes_np[thres] = []
    positive_zipcodes_unioned[thres] = []
    negative_zipcodes_unioned[thres] = []

if args.export_zipcodes:  # we only export zipcodes at a particular threshold
    non_overlapping_zipcodes_file = open(
        os.path.join(SG_OUTPATH, 'non_overlapping_zipcodes_{:4f}.fa'.format(args.kl_sig)), 'w')
    unioned_zipcodes_file = open(os.path.join(SG_OUTPATH, 'unioned_zipcodes_{:4f}.fa'.format(args.kl_sig)), 'w')

for i, (id, true_length, x, y) in enumerate(zip(reversed(ids), reversed(true_length), reversed(X), reversed(y))):

    original_predict = model.predict(x.reshape(1, -1)).reshape(-1, )
    utr3_length = get_utr3_length(genome_to_trans_id_mapping[id])
    # naive filtering rules
    if np.max(y) > 0.5 and np.argmax(original_predict) == np.argmax(y) and utr3_length != 0:
        '''
        true length is the actual length used in the training process;
        if true length is strictly smaller than 4000, then utr'3 should be smaller than true length;
        if true legnth is 4000, then it's probable that utr'3 legnth is higher than 4000.
        '''
        if true_length < utr3_length:
            indicator_length = true_length  # limit the 3'utr length within the 4000 interval
        else:
            indicator_length = utr3_length

        true_starting_index = 4000 - indicator_length

        if os.path.exists(os.path.join(MAT_OUTPATH, '{0}_{1}_{2}.npy'.format(id, true_length, utr3_length))):
            print('Loading existing {0}_{1}_{2}.npy'.format(id, true_length, utr3_length))
            if args.permute_mask:
                print('Warning, make sure the cached experiment uses permuted mask')
            '''load results'''
            mat = np.load(os.path.join(MAT_OUTPATH, '{0}_{1}_{2}.npy'.format(id, true_length, utr3_length)))
            indicated_attention = mat[0]
            indicated_scores = mat[1]
            kl_distance = mat[2]
            averaged_scores = mat[4]
            '''pearson correlations between attention/kl divergence and conservation scores'''
            pr_k = pearsonr(kl_distance, averaged_scores)[0]
            pr_a = pearsonr(indicated_attention, indicated_scores)[0]

            if not np.isnan(pr_k):
                pr_kl_con.append(pr_k)
            if not np.isnan(pr_a):
                pr_att_con.append(pr_a)

        else:
            # the model's chance of getting a corrected prediction
            # on sample with a predominant localization is higher
            print(id)
            print('true target:', y)
            print('original predicted target:', original_predict)
            print('true length', true_length)
            print('utr3 length', utr3_length, '\n')

            out_f.write('> ' + id + '\n')
            out_f.write('true target: {}\n'.format(y.__str__()))
            out_f.write('original predicted target: {}\n'.format(original_predict.tolist().__str__()))
            out_f.write('true length: {}\n'.format(true_length))
            out_f.write('3\'utr length: {}\n\n'.format(utr3_length))
            out_f.flush()

            fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 20), sharex='all')

            '''interpolate the attention of time steps 440 back to the original sequence of length 4000'''
            attention = model.get_attention(x.reshape(1, -1))[0].reshape(-1)
            attention_steps = [int(i * 3999 / 439) for i in range(len(attention))]
            interpolated_attention = interp.InterpolatedUnivariateSpline(attention_steps, attention)(np.arange(0, 4000))
            indicated_attention = interpolated_attention[-indicator_length:]
            axes[0].set_title(id)
            axes[0].set_ylabel('Singular ATT')
            axes[0].bar(np.arange(indicator_length), indicated_attention)

            '''plot conservation scores'''
            all_scores = np.array(conserv_scores_dict[id])
            indicated_scores = all_scores[-indicator_length:]
            indices_pos = np.where(indicated_scores > 0)[0]
            indices_neg = np.delete(np.arange(indicator_length), indices_pos)
            axes[1].bar(indices_pos, indicated_scores[indices_pos], color='mediumslateblue')
            axes[1].bar(indices_neg, indicated_scores[indices_neg], color='maroon')
            axes[1].set_ylabel('Singular CS')

            '''begin mask test'''
            '''optimize these codes to form a batch'''
            true_starting_index = 4000 - indicator_length
            averaged_scores = []
            averaged_att = []
            kl_distance = []
            tmp = []
            for i in range(4000 - true_starting_index):
                x_ = x.copy()
                if true_starting_index + i + mask_window_size // 2 < 4000:
                    effective_right = mask_window_size // 2
                else:
                    effective_right = 4000 - true_starting_index - i
                if true_starting_index + i - mask_window_size // 2 > 0:
                    effective_left = mask_window_size // 2
                else:
                    effective_left = true_starting_index + i
                batch = []
                if args.permute_mask:
                    for _ in range(100):
                        # permute version
                        x_[true_starting_index + i - effective_left:
                           true_starting_index + i + effective_right] = np.random.choice([1,2,3,4], effective_right+effective_left, p=[0.25,0.25,0.25,0.25])
                        batch.append(x_[None,:])
                    preds = model.predict(np.concatenate(batch, axis=0))
                    kl_distance.append(np.mean([entropy(original_predict, pred) for pred in preds], axis=0))
                else:
                    # zero masking
                    x_[true_starting_index + i - effective_left:
                       true_starting_index + i + effective_right] = 0
                    tmp.append(x_.reshape(1, -1))

                averaged_scores.append(
                    np.mean(all_scores
                            [true_starting_index + i - effective_left:
                             true_starting_index + i + effective_right]))
                averaged_att.append(
                    np.mean(interpolated_attention
                            [true_starting_index + i - effective_left:
                             true_starting_index + i + effective_right]))
            averaged_scores = np.array(averaged_scores)
            averaged_att = np.array(averaged_att)
            if args.permute_mask:
                kl_distance = np.array(kl_distance)
            else:
                preds = model.predict(np.concatenate(tmp, axis=0))
                # preds = np.concatenate(batch, axis=0)
                kl_distance = np.array([entropy(original_predict, pred) for pred in preds])
            axes[2].bar(np.arange(indicator_length), kl_distance)
            axes[2].set_ylabel('KL MASK')

            '''averaged and aligned attention'''
            axes[3].bar(np.arange(indicator_length), averaged_att)
            axes[3].set_ylabel('Averaged Aligned ATT')

            '''averaged and aligned conservation score'''
            indices_pos = np.where(averaged_scores > 0)[0]
            indices_neg = np.delete(np.arange(indicator_length), indices_pos)
            axes[4].bar(indices_pos, averaged_scores[indices_pos], color='mediumslateblue')
            axes[4].bar(indices_neg, averaged_scores[indices_neg], color='maroon')
            axes[4].set_ylabel('Averaged Aligned CS')
            axes[4].set_xlabel('3\'UTR')

            fig.tight_layout()
            plt.savefig(os.path.join(OUTPATH, 'ind_plots', '{}.png'.format(id)))
            plt.close(fig)

            '''save intermediate vectors for future reference'''
            np.save(
                os.path.join(MAT_OUTPATH, '{0}_{1}_{2}.npy'.format(id, true_length, utr3_length)),
                np.concatenate([
                    indicated_attention[None, :],
                    indicated_scores[None, :],
                    kl_distance[None, :],
                    averaged_att[None, :],
                    averaged_scores[None, :]
                ])
            )

            '''pearson correlations between attention/kl divergence and conservation scores'''
            pr_k = pearsonr(kl_distance, averaged_scores)[0]
            pr_a = pearsonr(indicated_attention, indicated_scores)[0]

            if not np.isnan(pr_k):
                pr_kl_con.append(pr_k)
            if not np.isnan(pr_a):
                pr_att_con.append(pr_a)

        ind_mode = np.argmax(y)

        '''threshold of kl'''
        for thresholding in kl_threshold:
            int_kl_indices = np.where(kl_distance >= thresholding)[0]

            above_kl_threshold[thresholding] = np.concatenate(
                [above_kl_threshold[thresholding], averaged_scores[int_kl_indices]])
            # above_kl_threshold_sg[thresholding] = np.concatenate(
            #     [above_kl_threshold_sg[thresholding], indicated_scores[int_kl_indices]])

            non_int_kl_indices = np.delete(np.arange(indicator_length), int_kl_indices)
            below_kl_threshold[thresholding] = np.concatenate(
                [below_kl_threshold[thresholding], averaged_scores[non_int_kl_indices]])
            # below_kl_threshold_sg[thresholding] = np.concatenate(
            #     [below_kl_threshold_sg[thresholding], indicated_scores[non_int_kl_indices]])

            '''subject to new filtering rules, these are all correct predictions'''
            # if args.export_zipcodes and round(thresholding, 3) == args.kl_sig:# and\
            # np.argmax(y) == np.argmax(original_predict):


            '''count non-overlapping indices'''
            non_overlapping_indices = []
            for index in int_kl_indices:
                if len(non_overlapping_indices) == 0:
                    non_overlapping_indices.append(index)
                elif index >= np.max(non_overlapping_indices) + mask_window_size:
                    non_overlapping_indices.append(index)
            non_overlapping_counts[thresholding] += len(non_overlapping_indices)


            for i in non_overlapping_indices:
                if true_starting_index + i + mask_window_size // 2 < 4000:
                    effective_right = mask_window_size // 2
                else:
                    effective_right = 4000 - true_starting_index - i
                if true_starting_index + i - mask_window_size // 2 > 0:
                    effective_left = mask_window_size // 2
                else:
                    effective_left = true_starting_index + i

                x_ = x.copy()
                x_[true_starting_index + i - effective_left:
                   true_starting_index + i + effective_right] = 0
                masked_predict = model.predict(x_.reshape(1, -1)).reshape(-1, )

                '''export non overlapping zipcodes'''
                if args.export_zipcodes and np.round(thresholding, 4) == args.kl_sig:
                    if masked_predict[ind_mode] > original_predict[ind_mode]:
                        eff = 'neg'
                    else:
                        eff = 'pos'
                    non_overlapping_zipcodes_file.write('> genome_id:{0} position:{1}:{2} {3} '.
                                                        format(id, true_starting_index + i - effective_left,
                                                               true_starting_index + i + effective_right,
                                                               eff + '_' + locs[ind_mode]
                                                               ))
                    non_overlapping_zipcodes_file.write('true_target:{0} pred_target:{1} masked_target:{2}\n'.
                                                        format('_'.join([str(val) for val in y]),
                                                               '_'.join([str(val) for val in original_predict]),
                                                               '_'.join([str(val) for val in masked_predict])))
                    non_overlapping_zipcodes_file.write('{}\n'.format(
                        ''.join([encoding_keys[char] for char in
                                 x[true_starting_index + i - effective_left:
                                   true_starting_index + i + effective_right]])))
                    non_overlapping_zipcodes_file.flush()

                if masked_predict[ind_mode] > original_predict[ind_mode]:
                    '''negative zipcodes'''
                    negative_zipcodes_np[thresholding].append(averaged_scores[i])
                else:
                    positive_zipcodes_np[thresholding].append(averaged_scores[i])

            all_scores = np.array(conserv_scores_dict[id])
            ''' we take the union of contingent indices'''
            if len(int_kl_indices) > 0:
                # this ensures that last position is always looked at
                int_kl_indices = np.append(int_kl_indices, int(1e10))
                pre = post = int_kl_indices[0]
                for index in int_kl_indices[1:]:
                    if index > post + mask_window_size:
                        # consecutive from pre to post
                        if true_starting_index + post + mask_window_size // 2 < 4000:
                            effective_right = mask_window_size // 2
                        else:
                            effective_right = 4000 - true_starting_index - post
                        if true_starting_index + pre - mask_window_size // 2 > 0:
                            effective_left = mask_window_size // 2
                        else:
                            effective_left = true_starting_index + pre
                        x_ = x.copy()
                        x_[true_starting_index + pre - effective_left:
                           true_starting_index + post + effective_right] = 0
                        masked_predict = model.predict(x_.reshape(1, -1)).reshape(-1, )

                        if args.export_zipcodes and np.round(thresholding, 4) == args.kl_sig:
                            if masked_predict[ind_mode] > original_predict[ind_mode]:
                                eff = 'neg'
                            else:
                                eff = 'pos'
                            unioned_zipcodes_file.write('> genome_id:{0} position:{1}:{2} {3} '.
                                                        format(id, true_starting_index + pre - effective_left,
                                                               true_starting_index + post + effective_right,
                                                               eff+'_'+locs[ind_mode]
                                                               ))
                            unioned_zipcodes_file.write('true_target:{0} pred_target:{1} masked_target:{2}\n'.
                                                        format('_'.join([str(val) for val in y]),
                                                               '_'.join([str(val) for val in original_predict]),
                                                               '_'.join([str(val) for val in masked_predict])))
                            unioned_zipcodes_file.write('{}\n'.format(
                                ''.join([encoding_keys[char] for char in
                                         x[true_starting_index + pre - effective_left:
                                           true_starting_index + post + effective_right]])))
                            unioned_zipcodes_file.flush()

                        unioned_counts[thresholding] += 1

                        averaged_cs_union = np.mean(all_scores[true_starting_index + pre - effective_left:
                                                            true_starting_index + post + effective_right])
                        if masked_predict[ind_mode] > original_predict[ind_mode]:
                            '''negative zipcodes'''
                            negative_zipcodes_unioned[thresholding].append(averaged_cs_union)
                        else:
                            positive_zipcodes_unioned[thresholding].append(averaged_cs_union)

                        pre = post = index
                    else:
                        post = index
out_f.close()

'''histograms of all samples correslation, between kl and conservation scores'''
fig = plt.figure(figsize=(10, 10))
plt.title('KL-CONS pcorr')
plt.hist(pr_kl_con, bins=500,
         weights=np.ones(len(pr_kl_con)) / len(pr_kl_con))
plt.axvline(np.mean(pr_kl_con), color='k', linestyle='dashed', linewidth=1)
plt.text(np.mean(pr_kl_con), 0.1, 'mean {:.2f}'.format(np.mean(pr_kl_con)))
plt.xlabel('pcorr')
plt.grid(True)
plt.savefig(os.path.join(SG_OUTPATH, 'pr_kl_cons.png'))
plt.close(fig)

'''same format, but between attention and conservation scores'''
fig = plt.figure(figsize=(10, 10))
plt.title('ATT-CONS pcorr')
plt.hist(pr_att_con, bins=500,
         weights=np.ones(len(pr_att_con)) / len(pr_att_con))
plt.axvline(np.mean(pr_att_con), color='k', linestyle='dashed', linewidth=1)
plt.text(np.mean(pr_att_con), 0.1, 'mean {:.2f}'.format(np.mean(pr_att_con)))
plt.xlabel('pcorr')
plt.grid(True)
plt.savefig(os.path.join(SG_OUTPATH, 'pr_att_cons.png'))
plt.close(fig)

logger = CSVLogger('log.csv', SG_OUTPATH,
                   ['kl_cutoff', 'above_mean', 'above_total_count', 'above_nonoverlapping', 'above_unioned',
                    'below_mean', 'below_total_count', 'ks_total', 'pval_total', 'pos_np_count', 'ks_pos_np', 'pval_pos_np', 'neg_np_count', 'ks_neg_np', 'pval_neg_np',
                    'pos_unioned_count', 'ks_pos_unioned', 'pval_pos_unioned', 'neg_unioned_count', 'ks_neg_unioned', 'pval_neg_unioned',
                    # 'sg_above_mean', 'sg_above_count', 'sg_below_mean', 'sg_below_count', 'sg_ks_statics',
                    # 'sg_p_value'
                    ])

'''kl thresholding histograms'''
for thresholding in sorted(kl_threshold):
    above_mean, below_mean = np.mean(above_kl_threshold[thresholding]), np.mean(below_kl_threshold[thresholding])
    pos_np_mean, neg_np_mean = np.mean(positive_zipcodes_np[thresholding]), np.mean(negative_zipcodes_np[thresholding])
    pos_unioned_mean, neg_unioned_mean = np.mean(positive_zipcodes_unioned[thresholding]), np.mean(negative_zipcodes_unioned[thresholding])

    '''ALL cs above and below kl cutoff'''
    fig = plt.figure(figsize=(10, 10))
    plt.title('KL cutoff at {:4f}'.format(thresholding))
    plt.xlabel('Averaged Conservation Score')
    plt.ylabel('Density')
    plt.hist(above_kl_threshold[thresholding], bins=500,
             weights=np.ones(len(above_kl_threshold[thresholding])) / len(above_kl_threshold[thresholding]),
             color='mediumslateblue')
    plt.axvline(above_mean, color='mediumslateblue', linestyle='dashed', linewidth=1)
    plt.text(above_mean, 0.2, '$\geq$ {:.4f}'.format(thresholding))
    plt.hist(below_kl_threshold[thresholding], bins=500,
             weights=np.ones(len(below_kl_threshold[thresholding])) / len(below_kl_threshold[thresholding]),
             color='maroon', alpha=0.5)
    plt.axvline(below_mean, color='maroon', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.text(below_mean, 0.1, '$\leq$ {:.4f}'.format(thresholding))
    plt.savefig(os.path.join(SG_OUTPATH, 'kl_thres_{:.4f}.png'.format(thresholding)))
    ks_all, pval_all = ks_2samp(above_kl_threshold[thresholding], below_kl_threshold[thresholding])
    plt.close(fig)

    '''NONOVERLAPPING positive and negative zipcodes, with cs below cutoff'''
    fig = plt.figure(figsize=(10, 10))
    plt.title('KL cutoff at {:4f}'.format(thresholding))
    plt.xlabel('Averaged Conservation Score')
    plt.ylabel('Density')
    plt.hist(positive_zipcodes_np[thresholding], bins=500,
             weights=np.ones(len(positive_zipcodes_np[thresholding])) / len(positive_zipcodes_np[thresholding]),
             color='mediumslateblue')
    plt.axvline(pos_np_mean, color='mediumslateblue', linestyle='dashed', linewidth=1)
    plt.text(pos_np_mean, 0.1, 'positive zipcodes')

    plt.hist(negative_zipcodes_np[thresholding], bins=500,
             weights=np.ones(len(negative_zipcodes_np[thresholding])) / len(negative_zipcodes_np[thresholding]),
             color='green')
    plt.axvline(neg_np_mean, color='green', linestyle='dashed', linewidth=1)
    plt.text(neg_np_mean, 0.2, 'negative zipcodes')

    plt.hist(below_kl_threshold[thresholding], bins=500,
             weights=np.ones(len(below_kl_threshold[thresholding])) / len(below_kl_threshold[thresholding]),
             color='maroon', alpha=0.5)
    plt.axvline(below_mean, color='maroon', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.text(below_mean, 0.1, 'below kl cutoff')
    plt.savefig(os.path.join(SG_OUTPATH, 'kl_thres_{:.4f}_np.png'.format(thresholding)))
    ks_pos_np, pval_pos_np = ks_2samp(positive_zipcodes_np[thresholding], below_kl_threshold[thresholding])
    ks_neg_np, pval_neg_np = ks_2samp(negative_zipcodes_np[thresholding], below_kl_threshold[thresholding])
    plt.close(fig)

    '''UNIONED positive and negative zipcodes, with cs below cutoff'''
    fig = plt.figure(figsize=(10, 10))
    plt.title('KL cutoff at {:4f}'.format(thresholding))
    plt.xlabel('Averaged Conservation Score')
    plt.ylabel('Density')
    plt.hist(positive_zipcodes_unioned[thresholding], bins=500,
             weights=np.ones(len(positive_zipcodes_unioned[thresholding])) / len(positive_zipcodes_unioned[thresholding]),
             color='mediumslateblue')
    plt.axvline(pos_unioned_mean, color='mediumslateblue', linestyle='dashed', linewidth=1)
    plt.text(pos_unioned_mean, 0.1, 'positive zipcodes')

    plt.hist(negative_zipcodes_unioned[thresholding], bins=500,
             weights=np.ones(len(negative_zipcodes_unioned[thresholding])) / len(negative_zipcodes_unioned[thresholding]),
             color='green')
    plt.axvline(neg_unioned_mean, color='green', linestyle='dashed', linewidth=1)
    plt.text(neg_unioned_mean, 0.2, 'negative zipcodes')

    plt.hist(below_kl_threshold[thresholding], bins=500,
             weights=np.ones(len(below_kl_threshold[thresholding])) / len(below_kl_threshold[thresholding]),
             color='maroon', alpha=0.5)
    plt.axvline(below_mean, color='maroon', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.text(below_mean, 0.1, 'below kl cutoff')
    plt.savefig(os.path.join(SG_OUTPATH, 'kl_thres_{:.4f}_unioned.png'.format(thresholding)))
    ks_pos_unioned, pval_pos_unioned = ks_2samp(positive_zipcodes_unioned[thresholding], below_kl_threshold[thresholding])
    ks_neg_unioned, pval_neg_unioned = ks_2samp(negative_zipcodes_unioned[thresholding], below_kl_threshold[thresholding])
    plt.close(fig)

    # fig = plt.figure(figsize=(10, 10))
    # plt.title('KL thresholding at {:4f}'.format(thresholding))
    # plt.xlabel('Singular Conservation Score')
    # plt.ylabel('Frequency')
    # plt.hist(above_kl_threshold_sg[thresholding], bins=500,
    #          weights=np.ones(len(above_kl_threshold_sg[thresholding])) / len(above_kl_threshold_sg[thresholding]),
    #          color='mediumslateblue')
    # plt.hist(below_kl_threshold_sg[thresholding], bins=500,
    #          weights=np.ones(len(below_kl_threshold_sg[thresholding])) / len(below_kl_threshold_sg[thresholding]),
    #          color='maroon', alpha=0.5)
    # plt.savefig(os.path.join(SG_OUTPATH, 'kl_thres_{:.4f}_sg.png'.format(thresholding)))
    # sg_above_mean, sg_below_mean = np.mean(above_kl_threshold_sg[thresholding]), np.mean(
    #     below_kl_threshold_sg[thresholding])
    # sg_ks_stat, sg_p_value = ks_2samp(above_kl_threshold_sg[thresholding], below_kl_threshold_sg[thresholding])
    # plt.close(fig)

    logger.update_with_dict({
        'kl_cutoff': thresholding,
        'above_mean': above_mean,
        'above_total_count': len(above_kl_threshold[thresholding]),
        'above_nonoverlapping': non_overlapping_counts[thresholding],
        'above_unioned': unioned_counts[thresholding],
        'below_mean': below_mean,
        'below_total_count': len(below_kl_threshold[thresholding]),
        'ks_total': ks_all,
        'pval_total': pval_all,
        'pos_np_count': len(positive_zipcodes_np[thresholding]),
        'ks_pos_np': ks_pos_np,
        'pval_pos_np': pval_pos_np,
        'neg_np_count': len(negative_zipcodes_np[thresholding]),
        'ks_neg_np': ks_neg_np,
        'pval_neg_np': pval_neg_np,
        'pos_unioned_count': len(positive_zipcodes_unioned[thresholding]),
        'ks_pos_unioned': ks_pos_unioned,
        'pval_pos_unioned': pval_pos_unioned,
        'neg_unioned_count': len(negative_zipcodes_unioned[thresholding]),
        'ks_neg_unioned': ks_neg_unioned,
        'pval_neg_unioned': pval_neg_unioned,
        # 'sg_above_mean': sg_above_mean,
        # 'sg_above_count': len(above_kl_threshold_sg[thresholding]),
        # 'sg_below_mean': sg_below_mean,
        # 'sg_below_count': len(below_kl_threshold_sg[thresholding]),
        # 'sg_ks_statics': sg_ks_stat,
        # 'sg_p_value': sg_p_value
    })

logger.close()
