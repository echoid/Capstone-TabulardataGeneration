#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=all_for_covertype
#SBATCH --time=90:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=10G
#SBATCH --output=outs/all_for_covertype.out

module load fosscuda/2019b
module load python/3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load scikit-learn/0.23.1-python-3.7.4
module load tensorflow/1.15.0-python-3.7.4
module load pytorch/1.5.1-python-3.7.4
module load tqdm/4.41.1

#python data_preprocess_full.py news
#python pretrained_sel.py news
#python generate.py news False False KL
#python generate.py news True False mean
#python generate.py covertype True True sel_mean
python generate.py covertype False True sel