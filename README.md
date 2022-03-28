# Capstone-TabulardataGeneration

## Source codes

Tablegan(modified by Minshen): https://github.com/mahmoodm2/tableGAN (tf:1.x)

Daisy: https://github.com/ruclty/Daisy

Selnet: https://github.com/ppo2020/SIGMOD2021ID73 (tf:1.x)

## Data:
Adult: https://github.com/ruclty/Daisy/blob/master/dataset/adult_train.csv


## envir
python 3.7.4
tensorflow 1.15
pytorch 1.10

## How to Run

### Data preprocess


```
python data_preprocess_full.py [filename from dataset/origin]
python data_preprocess_full.py adult

```

### Pre-trained Sel model


```
python pretrained_sel.py adult
```

### Generate data

```
python generate.py adult False False KL
python generate.py adult True False mean
python generate.py adult True True sel_mean
python generate.py adult False True sel

```


## Access Spartan
login to Spartan
```
cd /data/gpfs/projects/punim1578/Capstone-TabulardataGeneration
```
all the slurm files store in the slurm folders
to send file to spartan
```
sbatch slurms/xxxxx.slurm
```
The out put file will generated in  the slumrs folder as well


If you need some software in particular, you may try installing that version in your home dir using pip.

Start an interactive session a gpgpu node
```
$ module load fosscuda/2019b python/3.7.4
$ pip install --user torch==1.10.0
```

Sample slurm file
Note that you can change the outputfile name, email address for get notification,
job name and memory

```

#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=full
#SBATCH --time=48:00:00
#SBATCH --mail-user=youremail@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=10G
#SBATCH --output=outs/sel.out

module load fosscuda/2019b
module load python/3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load scikit-learn/0.23.1-python-3.7.4
module load tensorflow/1.15.0-python-3.7.4
module load pytorch/1.5.1-python-3.7.4
module load tqdm/4.41.1


python onlysel.py Adult False False True only_sel
```


Old instructions
===================================================================================================
### To run this model (Adult example) 

(Fixed, tf 1.0 will work.)
We need a tensorflow 2.x version to generate high dimension data
```
conda activate daisy
```


** generated pre-trained functional dependiencies models. (Not neccessary)
```
python pretrained_fd.py 
```


** preprocess data, generate high dim representation
```
python data_preprocess.py [filename from dataset/origin]
python data_preprocess.py adult

```
preprocessed high dimension data will store at dataset/train filename_preprocessed.npy



We need a tensorflow 1.x version to train model
```
conda activate tanlegan
```
** pretrained selectivity model
```
python pretrained_sel.py adult
```






