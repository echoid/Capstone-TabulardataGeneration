#! /bin/bash


#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn

#SBATCH -A punim1578
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name=sel_5
#SBATCH --time=200:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=30G
#SBATCH --output=outs/sel_co_full2.out

# module load fosscuda/2019b
# module load python/3.7.4



module load fosscuda/2020b 
module load python/3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
module load tqdm/4.60.0
module load scipy-bundle/2020.11

python selgan/selgan_generate.py covertype 300 full_batch2

