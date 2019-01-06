# RNATracker

RNATracker is a deep learning approach to learn mRNA subcellular localization patterns and to infer its outcome.
It operates on the cDNA of the longest isoformic protein-coding transcript of a gene with or without its corresponding secondary structure annnotations.
The learning targets are fractions/percentage of the transcripts being localized to a fixed set of subcellular compartments of interest.

Our method provides computational-centric insights into the the mRNA trafficking mechanism with identication to the cis-acting zipcodes elements from the transcript sequences. 

For what's exactly the RNA trafficking mechanism and its role in the broader gene regulatory network,
I find this survey extremely helpful.

[RNA localization: Making its way to the center stage](https://www.ncbi.nlm.nih.gov/pubmed/28630007)

## Dataset

- [Cefra-Seq](https://www.ncbi.nlm.nih.gov/pubmed/28579403) which provides localization targets for cytoplasm, insoluble, membrane and nucleus.

- [APEX-RIP](https://www.ncbi.nlm.nih.gov/pubmed/29239719) on KDEL(endoplasmic reticulum), Mito(Mitochdrial), NES (cytosol) and NCL (Nucleus)

Other emerging read-mapping technologies investigating subcellular zipcode proximity might provide additional dataset. 

## Software dependencies

[Keras](https://github.com/keras-team/keras) Version 2.0.9 is recommeneded. The idea
 can be easily adapted to other deep leaing frameworks such as Tensorflow or PyTorch. 

RNAplfold and forgi libraries from the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/) and their python wrapper [Eden](https://github.com/fabriziocosta/EDeN) for acquiring RNA secondary annotations.

[TOMTOM](http://meme-suite.org) for comparing similarity between motifs.

[Weblogo](https://weblogo.berkeley.edu/logo.cgi) and its python wrapper [Basset](https://github.com/davek44/Basset) for visualizing learned motifs.


## Notes

The secondary structures are acquired using the [codes](https://github.com/HarveyYan/mRNA-secondary-structure-annotater) 