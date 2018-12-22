# RNATracker

RNATracker is a deep learning approach aimed at inferring mRNA localization patterns.
It operates on the complemetary DNA sequence of a gene with or without its corresponding secondary structure annnotations.
The learning targets are distribution ratios (probability mass) of the localization to a fixed set of subcellular compartments of interest, e.g. ctyoplasm, insoluble, membrane and nucleus.

Our work provides computational centric insights into the understanding of mRNA localization, by the simplfied assumption of identifying cis-acting zipcodes in the sequences. 

For what's exactly the RNA trafficking mechanism and its role in the broader gene regulatory network,
I find this survey extremely helpful.

[RNA localization: Making its way to the center stage](https://www.ncbi.nlm.nih.gov/pubmed/28630007)

## Dataset

[Cefra-Seq](https://www.ncbi.nlm.nih.gov/pubmed/28579403)

[APEX-RIP](https://www.ncbi.nlm.nih.gov/pubmed/29239719) 

Other emerging APEX technologies such as the APEX-Seq will be included into the analysis once their results become accessible in the GEO dataset repository. 

## Software dependencies

[Keras](https://github.com/keras-team/keras) The deep learning framework for humans. The idea
 can also be easily adapted to other deep leaing frameworks such as Tensorflow or PyTorch. 

RNAplfold and forgi libraries from the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/) and their python wrapper [Eden](https://github.com/fabriziocosta/EDeN) for acquiring RNA secondary annotations.

[TOMTOM](http://meme-suite.org) for comparing similarity between motifs.

[Weblogo](https://weblogo.berkeley.edu/logo.cgi) and its python wrapper [Basset](https://github.com/davek44/Basset) for visualizing learned motifs.


## Running the RNATracker

Placeholder

## Remarks

The raw localization targets measured in FPKM units, tend to be very noisy.

To process all natural mRNA sequences is capped the computational bottleneck of LSTMs.

We cannot acquire reliable mRNA secondary structures, due to the insufficient understanding of this matter. 