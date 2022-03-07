#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=full
#SBATCH --time=24:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --output=full.out

module load fosscuda/2019b
module load python/3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load scikit-learn/0.23.1-python-3.7.4
module load tensorflow/1.15.0-python-3.7.4
module load pytorch/1.5.1-python-3.7.4
module load tqdm/4.41.1



python full.py strong_cate full
python full.py weak_cate full
python full.py strong_num full
python full.py weak_num full

python full.py strong_cate ITS
python full.py weak_cate ITS
python full.py strong_num ITS
python full.py weak_num ITS



