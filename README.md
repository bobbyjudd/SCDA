NOTE: This is a fork, go to https://github.com/work-hard-play-harder/SCDA to see the original work.

### Prerequisites
```
Python3.6 
Download 
```

### Setup
Create virtual environment
```
git clone https://github.com/bobbyjudd/SCDA.git
```
Install requirment dependents
```
pip install tensorflow sklearn pandas matplotlib
```

### To Run
- Download ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf from 1000 Genomes Project and put in data/
- Run:
```
python convert_data.py
```
- Wait for it to finish preprocessing the raw data
```
python SCDA_train.py
```
- Training can take a long time to complpete depending on compute capabilaties
- Create a directory for plots and results and run the test script
```
mkdir plots results
python SCDA_test.py
```


