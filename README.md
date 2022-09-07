# Tempo

This is the Pytorch implementation of TEMPO in the paper: [TEMPO: A Transformer-based Mutation Prediction Framework for SARS-CoV-2 Evolution] . The  phylogenetic tree data can be found at [here](https://ngdc.cncb.ac.cn/ncov/variation/tree) for reference.

## Requirements
- pytorch
- sklearn

## Usage
To run the code
1. Download data: data.zip shuold be downloaded **separately**， you can [click here](https://github.com/ZJUDataIntelligence/Tempo/raw/main/data.zip) to download the data for convenience.
2. cd to *data* folder:
`unzip data.zip`
3. you may need to change the dataset path defined in training.py(Corresponds to your *data* directory)
4. cd to project root folder:
`python training.py`

The files contained in data.zip
1. Preprocessed data used to reproduce the paper， including SARS-COV-2, H1N1, H3N2 and H5N1 dataset.
2. Phylogenetic tree data for SARS-COV-2, named as "tree.txt".
3. COV-19 s-protein sequence data aligned by mafft, named as "spike_prot_processed.csv".
