import os
import csv
import numpy as np
import pandas as pd
import requests
import gzip
import subprocess as sp

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
current_dir = os.path.dirname(os.path.abspath(__file__))
wig_download_template = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.100way.phyloP100way/chr{}.phyloP100way.wigFix.gz"
bw_download_url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
bw_summary_download_url = "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigSummary"


def get_transcripts_list():
    genes = {}
    flag = False
    with open(os.path.join(basedir, 'Data/Homo_sapiens.GRCh38.cdna.all.fa'), "r") as cdna:
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
                '''
                if flag:
                    # only retain protein_coding mRNAs
                    if gene_biotype == 'gene_biotype:protein_coding' and transcript_biotype == 'transcript_biotype:protein_coding':
                        if gene_id in genes:
                            if len(genes[gene_id][0]) < len(seq):  # take the longest isoform
                                # print('update longest isoform')
                                genes[gene_id] = [seq, transcript_id]
                        else:
                            genes[gene_id] = [seq, transcript_id]
                else:
                    flag = True

                seq = ""
                tokens = line.split()
                transcript_id = tokens[0][1:]
                gene_id = tokens[3].split(':')[1].split('.')[0]
                gene_biotype = tokens[4]
                transcript_biotype = tokens[5]
            else:
                seq += line[:-1]

    output = open(os.path.join(current_dir, "important_transcripts.txt"), "w")
    with open(os.path.join(basedir, 'Data/cefra-seq/Supplemental_File_3.tsv'), "r") as sup:
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
                output.write(
                    '{0} {1}\n'.format(id, genes[id][1]))
                output.flush()


# def _download_all_wig():
#     '''download all chromozone wig files and convert them to bed format'''
#     def _download(chrom, save_dir):
#         sp.call(['wget', download_template.format(chrom), '-P', save_dir])
#
#     conserv_dir = os.path.join(current_dir, 'conservation_files')
#     if not os.path.exists(conserv_dir):
#         os.makedirs(conserv_dir)
#     for chrom in list(range(1, 23))+['X']:
#         if not os.path.exists(os.path.join(conserv_dir, 'chr{}.phyloP100way.wigFix.gz'.format(chrom))):
#             # have to download it...
#             print('Downloading hg38.phyloP100way.bw')
#             sp.call(['wget', download_template.format(chrom), '-P', conserv_dir])

def _download_all_bw(download_path):
    if not os.path.exists(os.path.join(download_path, 'hg38.phyloP100way.bw')):
        print('Downloading hg38.phyloP100way.bw')
        sp.call(['wget', bw_download_url, '-P', download_path])  # this will block this thread, but it's fine..
    if not os.path.exists(os.path.join(download_path, 'bigWigSummary')):
        print('Downloading bigWigSummary')
        sp.call(['wget', bw_summary_download_url, '-P', download_path])
        sp.call(['chmod', 'u+x', os.path.join(download_path, 'bigWigSummary')])


# get all the transcript id used in this experiment
# get_transcripts_list()

def bw_get_score():
    exon_ref_pd = pd.read_csv(os.path.join(current_dir, 'lib/exon_reference_human_processed.zip'), compression='zip', sep="\t",
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

    conserv_dir = os.path.join(current_dir, 'conservation_files')
    if not os.path.exists(conserv_dir):
        os.makedirs(conserv_dir)

    # conserv_dir = '/Users/HarveyYan/Desktop/transcript2DNA mapping/bigwigs'

    _download_all_bw(conserv_dir)

    cmd_template = [os.path.join(conserv_dir, 'bigWigSummary.dms'), os.path.join(conserv_dir, 'hg38.phyloP100way.bw')]
    with open(os.path.join(current_dir, 'important_transcripts.txt'), 'r') as trans_f:
        with open(os.path.join(current_dir, 'all_conservation_scores_new.txt'), 'w') as out_f:
            for line in trans_f:
                genome_id = line.rstrip().split()[0]
                trans_id = line.rstrip().split()[2].split('.')[0]
                transcript = exon_ref_pd.loc[exon_ref_pd["Transcript stable ID"] == trans_id]
                chrom = transcript['Chromosome/scaffold name']
                strand = transcript['Strand']

                assert (len(set(chrom)) == 1)
                chrom = set(chrom).pop()  # where it belongs
                assert (len(set(strand)) == 1)
                strand = set(strand).pop()  # used in finding conservation scores
                t_len = transcript.tail(1)['cDNA exon end'].item()

                transcript.sort_values(by=['Exon rank in transcript'])  # should be in ascending order already...
                transcript_scores = ['n/a'] * t_len
                last_batch = 0
                for row_index, row in transcript.iterrows():
                    # if strand is -1 (suggesting the gene is on the reverse strand), we fill the exons from right to left;
                    # transcript_scores[row['cDNA exon start'] - 1: row['cDNA exon end']] =
                    exon_start, exon_end = row['Exon region start (bp)'] - 1, row['Exon region end (bp)']
                    full_cmd = cmd_template + ['chr' + chrom, str(exon_start), str(exon_end),
                                               str(exon_end - exon_start)]
                    p = sp.Popen(full_cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
                    print('Processing {0} exon {1}'.format(trans_id, row['Exon rank in transcript']))
                    res, err = p.communicate(
                        'Processing {0} exon {1}'.format(trans_id, row['Exon rank in transcript']).encode())

                    if p.returncode != 0:
                        print(err)
                    else:
                        exon_scores = [token for token in res.decode().rstrip().split('\t')]
                        if strand == 1.0:
                            transcript_scores[last_batch:last_batch + len(exon_scores)] = exon_scores
                        elif strand == -1.0:
                            transcript_scores[t_len - last_batch - len(exon_scores):t_len - last_batch] = exon_scores
                        else:
                            raise Exception('Unkown strand!')
                        last_batch += len(exon_scores)

                if strand == -1:  # always putting 5'UTR at the front
                    transcript_scores = transcript_scores[::-1]
                out_f.write('> genome_id:{0} transcrip_id:{1} chrom:{2} strand:{3}\n'.format(genome_id, trans_id, chrom,
                                                                                             strand))
                out_f.write(','.join(transcript_scores) + '\n')
                out_f.flush()


if __name__ == "__main__":
    if not os.path.exists(os.path.join(current_dir, 'important_transcripts.txt')):
        get_transcripts_list()

    bw_get_score()
