#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=ctgan
#SBATCH --time=150:00:00
#SBATCH --mem=10G

# module load fosscuda/2019b
# module load python/3.7.4



module load fosscuda/2020b 
module load python/3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
module load tqdm/4.60.0
module load scipy-bundle/2020.11

python selgan/selgan_generate.py adult 2 sel_gan_test


