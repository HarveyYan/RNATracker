# download genome transcript from the ensembl.org
import csv
import os
import urllib
import urllib.request as request
from bs4 import BeautifulSoup
import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(1234)


class transcript_gene(object):
    '''
    Note: Obsolete!!
    '''
    supp_file_path = './Supplemental_File_3.tsv'
    gene_file_path = './genes/'
    base_link = 'https://www.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g='
    download_link = 'https://www.ensembl.org/Homo_sapiens/Export/Output/Transcript?db=core;flank3_display=0;flank5_display=0;g=g_id;output=fasta;r_value;strand=feature;t=t_value;param=cdna;genomic=off;_format=Text'
    train_test_split_ratio = 0.3

    # 0: ensembl id
    # 1: unique identifier
    # 4: longest isoform
    # 5 ~ 12: distributions

    def __download__(self):
        # hasn't been downloaded yet
        if not os.path.exists(self.gene_file_path + self.id):
            # make sure the folder exists
            if not os.path.exists(self.gene_file_path):
                os.mkdir(self.gene_file_path)
            # start download from website
            res = request.urlopen(self.base_link + self.id)
            r_value = res.geturl().split(';')[-1]
            t_value = None
            site = BeautifulSoup(res.read())
            if site.find('table', {'id': 'transcripts_table'}) is None:
                # gene not in ensembl database
                print(self.id + " not found")
                return -1
            transcripts = site.find('table', {'id': 'transcripts_table'}).find_all('tr')[1:]
            for t in transcripts:
                if int(t.find_all('td')[2].getText()) == self.longest_isoform:
                    t_value = t.find('a').getText().split('.')[0]
                    break
            # supplemental file isoform length not found
            if t_value == None:
                longest = 0
                for t in transcripts:
                    if int(t.find_all('td')[2].getText()) > longest:
                        longest = int(t.find_all('td')[2].getText())
                        t_value = t.find('a').getText().split('.')[0]

            # print(self.id)
            # print(r_value)
            # print(t_value)
            # now assemble r_value and t_value
            link = self.download_link.replace('g_id', self.id)
            link = link.replace('r_value', r_value)
            link = link.replace('t_value', t_value)
            # now go for the download!

            # if self.id == 'ENSG00000078328':
            #     pass

            try:
                seq = request.urlopen(link).read()
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print('http error, consider the gene not exist')
                return -1

            out = open(self.gene_file_path + self.id, 'w')
            out.writelines(seq.decode('utf-8'))
            out.flush()
            out.close()
        print('finished downloading ' + self.id)

    @classmethod
    def load_from_file(cls, not_found):
        file = open('./Supplemental_File_3.tsv')
        reader = csv.reader(file, delimiter='\t')
        genes = []
        # throw away header
        next(reader)
        for line in reader:
            id = line[0]
            longest_isoform = int(line[4])
            distA = [line[5], line[7], line[9], line[11]]
            distB = [line[6], line[8], line[10], line[12]]
            gene = transcript_gene(id, longest_isoform, distA, distB)
            if gene.__download__() == -1:
                not_found.write(id + '\n')
                not_found.flush()
            genes.append(gene)

    def __init__(self, id, longest_isoform, distA, distB):
        assert type(distA) is list and len(distA) == 4
        assert type(distB) is list and len(distB) == 4
        assert type(id) is str and id.startswith('ENSG')
        assert type(longest_isoform) is int and longest_isoform > 0
        self.id = id
        self.longest_isoform = longest_isoform
        self.distA = distA
        self.distB = distB
        self.seq = None
        self.annotation = None

    @staticmethod
    def read_gene_file(id):
        file_id = open(transcript_gene.gene_file_path + id)
        gene_seq = ''
        for line in file_id:
            if line is None or len(line) == 0 or line == '\n':
                continue
            elif line.startswith('>'):
                continue
            else:
                # delete '\n' at the end
                gene_seq += line[:-1]
        return gene_seq

    '''
    Too slow in actual experiment settings
    Rather, just load all data into memory.
    '''

    @classmethod
    def data_generator(cls):
        # do the shuffling here
        existing_files = os.listdir(cls.gene_file_path)
        permute = np.random.permutation(np.linspace(0, len(existing_files) - 1, len(existing_files), dtype=int))
        genes = []
        with open('./Supplemental_File_3.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for i, line in enumerate(reader):
                id = line[0]
                # not yet downloaded, todo check from the notfound file
                if id not in existing_files:
                    continue
                longest_isoform = int(line[4])
                distA = [float(line[5]), float(line[7]), float(line[9]), float(line[11])]
                distB = [float(line[6]), float(line[8]), float(line[10]), float(line[12])]
                if int(np.sum(distA)) == 0 or int(np.sum(distB)) == 0:
                    # ditch failure data
                    continue
                gene = transcript_gene(id, longest_isoform, distA, distB)
                genes.append(gene)
        for ind in permute:
            genes[ind].seq = cls.read_gene_file(genes[ind].id)
            yield genes[ind]

    @classmethod
    def get_data(cls, ditch_UNK=False, lower_bound=0, upper_bound=np.inf):
        # count = 0
        # do the shuffling here
        longest = 0
        existing_files = os.listdir(cls.gene_file_path)
        genes = []
        with open('./Supplemental_File_3.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for i, line in enumerate(reader):
                # count += 1
                # # if count > 1000:
                # #     break
                # if count%10000 == 0:
                #     print(count)
                id = line[0]
                # not yet downloaded, todo check from the notfound file
                if id not in existing_files:
                    continue
                longest_isoform = int(line[4])
                distA = [float(line[5]), float(line[7]), float(line[9]), float(line[11])]
                distB = [float(line[6]), float(line[8]), float(line[10]), float(line[12])]
                if ditch_UNK:  # (int(np.sum(distA)) == 0 or int(np.sum(distB)) == 0):
                    if np.max(distA) < 5:  # or np.max(distA) < 0.4 * np.sum(distA):
                        continue
                gene = transcript_gene(id, longest_isoform, distA, distB)
                gene.seq = cls.read_gene_file(gene.id)
                if not (len(gene.seq) > lower_bound and len(gene.seq) < upper_bound):
                    continue
                if len(gene.seq) > longest:
                    longest = len(gene.seq)
                genes.append(gene)
        # split training and testing data
        size_genes = len(genes)
        print('longest seq', longest)
        print('test samples: ', int(size_genes * cls.train_test_split_ratio))
        permute = np.random.permutation(np.linspace(0, size_genes - 1, size_genes, dtype=int))
        return longest, np.array(genes)[permute[int(size_genes * cls.train_test_split_ratio):]], \
               np.array(genes)[permute[:int(size_genes * cls.train_test_split_ratio)]]

    @classmethod
    def get_sequence_and_annotation(cls, ditch_UNK=True, lower_bound=0, upper_bound=np.inf, partial=False,
                                    filter_sum=1):
        genes_path = './ensembl_cDNA'
        annotations_path = './annotations_RNAplfold_forgi'
        expr_path = './Supplemental_File_3.tsv'
        longest = 0

        # load gene sequence
        gene_seqs = {}
        with open(genes_path) as f:
            for line in f:
                if line.startswith('>'):
                    id = line[1:-1]
                elif line == '\n':
                    continue
                else:
                    gene_seqs[id] = line[:-1]

        # load annotations
        annotation_seqs = {}
        with open(annotations_path) as f:
            for line in f:
                if line.startswith('>'):
                    id = line[1:-1]
                elif line == '\n':
                    continue
                else:
                    annotation_seqs[id] = line[:-1]

        # load samples
        genes = []
        with open(expr_path) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for i, line in enumerate(reader):
                id = line[0]
                # not yet downloaded, todo check from the notfound file
                if id not in gene_seqs:
                    continue
                if not line[2] == 'protein_coding':
                    continue
                longest_isoform = int(line[4])
                distA = [float(line[5]), float(line[7]), float(line[9]), float(line[11])]
                distB = [float(line[6]), float(line[8]), float(line[10]), float(line[12])]
                dist = [(distA[i] + distB[i]) / 2 for i in range(0, 4)]
                if ditch_UNK:
                    if np.sum(dist) < filter_sum:
                        continue
                    pass
                if not (len(gene_seqs[id]) > lower_bound and len(gene_seqs[id]) < upper_bound):
                    continue
                gene = transcript_gene(id, longest_isoform, distA, distB)
                # do some funny thing
                if partial:
                    gene.seq1 = gene_seqs[id][:300]
                    gene.seq2 = gene_seqs[id][-300:]
                    gene.ann1 = annotation_seqs[id][:300]
                    gene.ann2 = annotation_seqs[id][-300:]
                else:
                    gene.seq = gene_seqs[id]
                    gene.annotation = annotation_seqs[id]
                    if len(gene.seq) > longest:
                        longest = len(gene.seq)
                genes.append(gene)
        genes = np.array(genes)
        # shuffle, split training and testing data
        size_genes = len(genes)
        print('total samples', size_genes)
        print('longest sequence', longest)
        # print('test samples: ', int(size_genes * cls.train_test_split_ratio))
        permute = np.random.permutation(np.linspace(0, size_genes - 1, size_genes, dtype=int))
        return longest, np.array(genes)[permute[int(size_genes * cls.train_test_split_ratio):]], \
               np.array(genes)[permute[:int(size_genes * cls.train_test_split_ratio)]]


class Gene_Wrapper:
    basedir = os.path.dirname(os.path.abspath(__file__))
    path_to_cefra_cDNA = os.path.join(basedir, 'Data/cefra-seq/cefra_seq_cDNA_screened.fa')
    path_to_cefra_ann = os.path.join(basedir, 'Data/cefra-seq/cefra_seq_cDNA_ann_screened.fa')

    path_to_apex_cDNA = os.path.join(basedir, 'Data/apex-rip/apex_rip_cDNA_screened.fa')
    path_to_apex_ann = os.path.join(basedir, 'Data/apex-rip/apex_rip_cDNA_ann_screened.fa')

    train_test_split_ratio = 0.1

    def __init__(self, id, type, dist):
        self.id = id
        self.type = type
        self.dist = dist
        self.seq = None
        self.ann = None
        np.random.seed(1234)

    @classmethod
    def seq_data_loader(cls, use_ann, dataset, lower_bound=0, upper_bound=np.inf, permute=None):
        """
        permute option is for randanmization test, with three types;
        For conventional data fitting please don't toggle this option on.
        """
        longest = 0
        genes = []
        count = 0

        if dataset == 'cefra-seq':
            path = cls.path_to_cefra_cDNA
        elif dataset == 'apex-rip':
            path = cls.path_to_apex_cDNA
        else:
            raise RuntimeError('No dataset named {}. Available dataset are "cefra-seq" and "apex-rip"'.format(dataset))

        print('Importing dataset {0}, at {1}'.format(dataset, path))

        with open(path, 'r') as f:
            for line in f:
                if line[0] == '>':
                    tokens = line[1:].split()
                    id = tokens[0]
                    gene_biotype = tokens[1].split(':')[1]
                    transcript_biotype = tokens[2].split(':')[1]
                    dist = [float(c) for c in tokens[3].split(':')[1].split('_')]
                else:
                    if len(line[:-1]) <= lower_bound: # normally not active
                        continue
                    if len(line[:-1]) >= upper_bound:
                        if len(line[:-1]) > longest:
                            longest = len(line[:-1])
                        line = line[:-1][-upper_bound:] # trying to keep (at least) the 3'UTR part
                        # continue
                    if gene_biotype != "protein_coding":
                        # mRNA only
                        continue
                    gene = Gene_Wrapper(id, gene_biotype, dist)
                    gene.seq = line.rstrip().upper()

                    if transcript_biotype != 'protein_coding':
                        count += 1
                    genes.append(gene)
        # print(len(genes))
        # count = 0
        # for gene in genes:
        #     if len(gene.seq) < 4000:
        #         count+=1
        print('non protein coding transcipt gene:', count)
        print('longest sequence: {0}'.format(longest))

        # see if we need annotations
        if use_ann:
            annotations = {}
            if dataset == 'cefra-seq':
                path = cls.path_to_cefra_ann
            elif dataset == 'apex-rip':
                path = cls.path_to_apex_ann
            else:
                raise RuntimeError(
                    'No dataset named {}. Available dataset are "cefra-seq" and "apex-rip"'.format(dataset))
            # load all annotations
            with open(path, 'r') as f:
                for line in f:
                    if line[0] == '>':
                        id = line[1:].split()[1]
                    else:
                        annotations[id] = line.rstrip()
            for gene in genes:
                if gene.id in annotations:
                    gene.ann = annotations[gene.id].upper()
                else:
                    print('Gene id {} not found in annotation file!'.format(gene.id))
                    gene.ann = None


        # do some permutations
        genes = np.array(genes)
        genes = genes[np.random.permutation(np.arange(len(genes)))]

        print('Total number of samples:', genes.shape[0])
        if permute:
            print('Warning: permuting mRNA samples!')
            if permute == 1:
                '''preserving the length and nucleotide contents'''
                print('Type 1')
                for gene in genes:
                    gene.seq = ''.join(np.random.permutation(list(gene.seq)))
                    if use_ann:
                        gene.ann = ''.join(np.random.permutation(list(gene.ann)))
            elif permute == 2:
                '''preserving length but altering the actual nucleotide contents'''
                print('Type 2')
                for gene in genes:
                    gene.seq = ''.join(np.random.choice(['A','C','G','T'], len(gene.seq)))
                    if use_ann:
                        gene.ann = ''.join(np.random.choice(['F', 'T', 'I', 'H', 'M', 'S'], len(gene.seq)))
            elif permute == 3:
                '''same length: 3000, and altering the nucleotides content'''
                print('Type 3')
                for gene in genes:
                    gene.seq = ''.join(np.random.choice(['A','C','G','T'], 3000))
                    if use_ann:
                        gene.ann = ''.join(np.random.choice(['F', 'T', 'I', 'H', 'M', 'S'], 3000))
            else:
                raise RuntimeError('Permute option only takes {1,2,3}.')
        return genes


if __name__ == "__main__":
    data = Gene_Wrapper.seq_data_loader(True, 'apex-rip')

    # print('shortest:', min([len(gene.seq) for gene in data]))
    # print('longest:', max([len(gene.seq) for gene in data]))
    #
    import Scripts.RNATracker
    Scripts.RNATracker.OUTPATH = './Graph/'
    train_data_label = [gene.dist for gene in data]
    Scripts.RNATracker.mode_frequency(data, train_data_label, 'mode-freq-apex-rip', ['KDEL', 'Mito', 'NES', 'NCL'])

    data = Gene_Wrapper.seq_data_loader(True, 'cefra-seq')
    import Scripts.RNATracker

    Scripts.RNATracker.OUTPATH = './Graph/'
    train_data_label = [gene.dist for gene in data]
    Scripts.RNATracker.mode_frequency(data, train_data_label, 'mode-freq-cefra-seq', ['cytosol', 'insoluble', 'membrane', 'nucleus'])
