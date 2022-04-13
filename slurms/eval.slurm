#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_a
#SBATCH --time=150:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=60G
#SBATCH --output=outs/eval_a_final.out



module load fosscuda/2020b 
module load python/3.8.6
module load tensorflow/2.6.0-python-3.8.6
module load pytorch/1.9.0-python-3.8.6
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
module load tqdm/4.60.0
module load matplotlib/3.3.3-python-3.8.6
module load scipy-bundle/2020.11

python evaluation_sel_1.py adult
# python evaluation.py news
# python evaluation.py credit
# python evaluation.py ticket
# python evaluation_sel_1.py covertype
