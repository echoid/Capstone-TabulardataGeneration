#! /bin/bash



#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=data_preprocess_ticket
#SBATCH --time=48:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=30G
#SBATCH --output=outs/data_preprocess_ticket.out

module load fosscuda/2019b
module load python/3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load scikit-learn/0.23.1-python-3.7.4
module load tensorflow/1.15.0-python-3.7.4
module load pytorch/1.5.1-python-3.7.4
module load tqdm/4.41.1

#python data_preprocess_full.py covertype
#python data_preprocess_full.py credit
#python data_preprocess_full.py news
python data_preprocess_full.py ticket
