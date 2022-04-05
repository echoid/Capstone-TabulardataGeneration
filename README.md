# Capstone-TabulardataGeneration

## Source codes

Tablegan(modified by Minshen): https://github.com/mahmoodm2/tableGAN (tf:1.x)

Daisy: https://github.com/ruclty/Daisy

Selnet: https://github.com/ppo2020/SIGMOD2021ID73 (tf:1.x)

CTGAN: https://github.com/sdv-dev/CTGAN

## Data:
Adult: https://github.com/ruclty/Daisy/blob/master/dataset/adult_train.csv
Adult: http://archive.ics.uci.edu/ml/datasets/adult
Covertype: http://archive.ics.uci.edu/ml/datasets/covertype
Ticket: https://www.transtats.bts.gov/DataIndex.asp. From tablegan
News: https://archive.ics.uci.edu/ml/datasets/online+news+popularity
Credit: https://www.kaggle.com/mlg-ulb/creditcardfraud



## envir
For selnet, daisy, VAE
python/3.7.4 
tensorflow/1.15
pytorch/1.10

For sel-gan, ctgan, octgan(haven't test)
python/3.8.6
tensorflow/2.6.0-python-3.8.6
pytorch/1.9.0-python-3.8.6

## How to Run

## Version 3

### Data transformation


python selgan/data_transformer.py [filename from dataset/origin]
python selgan/data_transformer.py adult


### Pre-trained Sel model

python selgan/pretrain_selnet.py [filename from dataset/origin]
python selgan/pretrain_selnet.py adult

### Generate

python selgan/selgan_generate.py [filename from dataset/origin] [# Epoches] [path/name]
python selgan/selgan_generate.py adult 300 sel_gan


### Spartan slurm


```
#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=sel-news
#SBATCH --time=150:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=30G
#SBATCH --output=outs/selgan_news_300.out

# module load fosscuda/2019b
# module load python/3.7.4



module load fosscuda/2020b 
module load python/3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
module load tqdm/4.60.0
module load scipy-bundle/2020.11

python selgan/selgan_generate.py news 300 sel_gan

```



=================================================================================
## Version 2

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



===================================================================================================
## Version 1

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






