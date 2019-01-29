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

## Software dependency

[Keras](https://github.com/keras-team/keras) version 2.0.9 is recommeneded. The idea
 can be easily adapted to other deep leaing frameworks such as Tensorflow and PyTorch. 

RNAplfold and forgi libraries from the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/) and their python wrapper [Eden](https://github.com/fabriziocosta/EDeN) for acquiring RNA secondary annotations.

[TOMTOM](http://meme-suite.org) for comparing similarity between motifs.

[Weblogo](https://weblogo.berkeley.edu/logo.cgi) and its python wrapper [Basset](https://github.com/davek44/Basset) for visualizing learned motifs.

## Running the codes

- Scripts/RNATracker.py<br></br>
    - Main experiment entry<br></br>
    - Use `python3 Scripts/RNATracker.py -h` to get a comprehensive list of experiment parameters<br></br>
    - For model definitions refer to Models/cnn_bilstm_attention.py
- Scripts/SGDModel.py<br></br>
    - Experiment without padding or truncation
- Scripts/mask_test.py<br></br>
    - Mask test to identify zipcodes with a sufficiently trained RNATracker model
- Transcript_Coordinates_Mapping/get_conservation_scores.py<br></br>
    - A script to prepare conseration scores for the downstream mask test
    - Highly recommend downloading **Homo_sapiens.GRCh38.cdna.all.fa** from the ensembl website, to be further saved under the **Data** directory
    

## Notes

For secondary structures refer to this customized [annotator](https://github.com/HarveyYan/mRNA-secondary-structure-annotater) 