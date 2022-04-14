#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=octgan_t
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=10G
#SBATCH --output=outs/octgan_t.out


module load fosscuda/2020b 
module load python/3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
module load tqdm/4.60.0
module load scipy-bundle/2020.11


# python OCTGAN/main.py --dataset_name=adult
# python OCTGAN/main.py --dataset_name=news
#python OCTGAN/main.py --dataset_name=covertype --epochs=200
#python OCTGAN/main.py --dataset_name=credit --epochs=200
python OCTGAN/main.py --dataset_name=ticket --epochs=200 --batch_size=20