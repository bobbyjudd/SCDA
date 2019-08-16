# Sparse Convolutional Denoising Autoencoders for Genotype Imputation
Genotype imputation, where missing genotypes can be computationally imputed, is an essential tool in genomic analysis ranging from genome wide associations to phenotype prediction. Traditional genotype imputation methods are typically based on Hidden Markov Models (HMMs) and are usually computationally expensive. Deep learning-based methods have been recently reported to nicely address the missing data problems in various fields. To explore the performance of deep learning for genotype imputation, in this study we propose a deep model called a Sparse Convolutional Denoising Autoencoder (SCDA) to impute missing genotypes. We constructed the SCDA model by using a convolutional layer that can extract various correlation or linkage patterns in the genotype data and applying a sparse weight matrix resulted from the L1 regularization to handle high dimensional data. We comprehensively evaluated the performance of the SCDA model in different scenarios for genotype imputation on the yeast and human genotype data respectively. Our results showed that SCDA has strong robustness and significantly outperforms three popular reference-free imputation methods including row average, k-nearest neighbors and singular-value decomposition. This study thus points to another novel application of deep learning models for missing data imputation in genomic studies. 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The code is performed in a Jupyter Notebook, because Jupyter Notebook provides an interactivly develop and test manner, which is very convient for practical purpose. 

### Prerequisites
```
Python3.6 
virtualenv >= 16.4.3
```

### Setup
Create virtual environment
```
git clone https://github.com/shilab/SCDA.git
cd SCDA
mkdir venv
python3 -m venv venv/
source venv/bin/activate
```

Install requirment dependents
```
pip install tensorflow sklearn pandas jupyter matplotlib
```

### Launch Jupyter
Launch Jupyter notebook  by 
``` 
jupyter notebook 
```
which will launch a browser. If you have any problem, please reference Jupyter notebook [offical document](https://jupyter-notebook.readthedocs.io/en/stable/)

## Case study 
To demonstrate the SCDA's usage, we provide a case study for yeast genotype imputation, which achieves above 99% accuracy.

### Input file format 
The input file should be CSV format with 'tab' delimiter. The header is SNPs' position, and the first column is sample ID, as following table shown. This format can be easily generated from VCF file by removing the first 8 lines and transferring the row and column.

| SAMID | chrI_33070 | chrI_33147 | chrI_33152 | chrI_33200 |
|:-----:|:----------:|:----------:|:----------:|:----------:|
| 01_01 |      1     |      2     |      0     |      1     |
| 01_02 |      1     |      1     |      2     |      1     |
| 01_03 |      2     |      0     |      1     |      1     |
| 01_04 |      1     |      2     |      0     |      2     |
| 01_06 |      0     |      1     |      2     |      0     |

where 1s and 2s reprensent the variants from laboratory strain (BY) and an isolate from a vineyard (RM), respectively. Os reprensent the mssing SNPs. 

### For the training process 
1. Unzip the `training_data` file in data folder 
2. Click `SCDA_train.ipynb` in brower. Run the cells one by one, or run the whole notebook in once. It will load training data, train a SCDA model and save the trained model.


### For the imputation process
1. Unzip the `testin_data` file in data folder 
2. Click `SCDA_test.ipynb` in brower. Run the cells one by one, or run the whole notebook in once. It will load training data, train a SCDA model and save the trained model.


## Citation
Junjie Chen, Xinghua Shi\*. Sparse Convolutional Denoising Autoencoders for Genotype Imputation. *Genes*, 2019.