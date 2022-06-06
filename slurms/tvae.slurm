#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=tvae
#SBATCH --time=150:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=10G
#SBATCH --output=outs/tvae_all.out

# module load fosscuda/2019b
# module load python/3.7.4


module load fosscuda/2020b 
module load python/3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
module load tqdm/4.60.0
module load scipy-bundle/2020.11


python CTGAN/tvae.py adult 200
python CTGAN/tvae.py covertype 200
python CTGAN/tvae.py credit 200
python CTGAN/tvae.py news 200
python CTGAN/tvae.py ticket 200

