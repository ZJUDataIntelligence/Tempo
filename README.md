# Tempo

This is the Pytorch implementation of TEMPO in the paper: [TEMPO: A Transformer-based Mutation Prediction Framework for SARS-CoV-2 Evolution]. 

## Requirements
- pytorch
- sklearn

## Data preparation

### Protein sequence data:

This data file contains the original and preprossed protein sequence data for SARS-COV-2, H1N1, H3N2 and H5N1, which is necessary to run the code. Before running the code, data.zip shuold be downloaded **separately**， you can [click here](https://github.com/ZJUDataIntelligence/Tempo/raw/main/data.zip) to download the data for convenience.

The files contained in data.zip
1. Preprocessed data used to reproduce the paper， including SARS-COV-2, H1N1, H3N2 and H5N1 dataset.
2. Phylogenetic tree data for SARS-COV-2, named as "tree.txt".
3. COV-19 s-protein sequence data aligned by mafft, named as "spike_prot_processed.csv".

### Phylogenetic tree data:
This is a supplementary data which is not necessary to run the code, while it could be helpful for others to understand our paper in more depth and to do further work based on it. The  phylogenetic tree data for SARS-COV-2 can be found at [here](https://ngdc.cncb.ac.cn/ncov/variation/tree).



## Usage
To run the code
1. add the *"data.zip"* to the root directory of the project(at the same level as *"train.py"*)
2. decompress the data and you will get a folder named *data*.

  `unzip data.zip`

3. modify the dataset path defined in training.py(line 14 to line 31), corresponding to your *data* folder's path in your enviroment.
4. train the model which the folllowing command:


  `python training.py`


