# Capstone-TabulardataGeneration

## Baseline Source codes

TableGAN: https://github.com/mahmoodm2/tableGAN (required tf:1.x)

Daisy: https://github.com/ruclty/Daisy

Selnet: https://github.com/ppo2020/SIGMOD2021ID73 (required tf:1.x)

CTGAN: https://github.com/sdv-dev/CTGAN

OCTGAN: https://github.com/bigdyl-yonsei/OCTGAN 

## Data:
Adult: https://github.com/ruclty/Daisy/blob/master/dataset/adult_train.csv

Adult: http://archive.ics.uci.edu/ml/datasets/adult

Covertype: http://archive.ics.uci.edu/ml/datasets/covertype

Ticket: https://www.transtats.bts.gov/DataIndex.asp. (Could access from TableGAN)

News: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

Credit: https://www.kaggle.com/mlg-ulb/creditcardfraud



## Requirements:
For selnet, daisy, VAE
```
python/3.7.4 
tensorflow/1.15
pytorch/1.10
```

For sel-gan, ctgan, octgan
```
python/3.8.6
tensorflow/2.6.0-python-3.8.6
pytorch/1.9.0-python-3.8.6
```

Additional configure.json are stored under dataset/configreation/[dataname_config.json].
The json files are required to represent the preprocessing type for each data column.


## How to Run

## Sel-GAN

### 1. Data transformation(mRDT)

```
python selgan/data_transformer.py [filename from dataset/origin]
python selgan/data_transformer.py adult
```

### 2. Pre-trained Sel model

```
python selgan/pretrain_selnet.py [filename from dataset/origin]
python selgan/pretrain_selnet.py adult
```

### 3. GAN-training

```
python selgan/selgan_generate.py [filename from dataset/origin] [# Epoches] [path/name]
python selgan/selgan_generate.py adult 300 sel_gan
```

## Daisy-Sel


### 1. Data transformation(mRDT)


```
python daisy/data_preprocess_full.py [filename from dataset/origin]
python daisy/data_preprocess_full.py adult

```

### 2. Pre-trained Sel model


```
python daisy/pretrained_sel.py [dataname]
```

### 3. GAN-training

The first binary value indicates weather they add Mean_Loss to G_Loss of not.
The second binary value indicates weather they add Sel_Loss to G_Loss of not.
```
python daisy/generate.py [dataname] [False/True] [False/True] [generated_path]

```

## Outputs

All outputs files are stored in dataset/generated/datasetname/modeltype/synthetic_data.csv


## Spartan
All the experiments are done through Spartan.
See Spartan Documentation https://dashboard.hpc.unimelb.edu.au/ for more informations.

### How to Run
All the slurms we used are stored in slurms folders.
To submit jobs to Spartan, run
```
sbatch slurms/xxxxx.slurm
```

### Spartan slurm Example
```
#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=[job_name]
#SBATCH --time=150:00:00
#SBATCH --mail-user=[email_address]
#SBATCH --mail-type=ALL
#SBATCH --mem=30G
#SBATCH --output=outs/outname.out

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






