#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=transform
#SBATCH --time=150:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=10G
#SBATCH --output=outs/sel_gan_transform.out

# module load fosscuda/2019b
# module load python/3.7.4



module load fosscuda/2019b python/3.7.4
module load tqdm/4.41.1

# python selgan/data_transformer.py adult
# python selgan/data_transformer.py covertype
# python selgan/data_transformer.py credit
# python selgan/data_transformer.py news
python selgan/data_transformer.py ticket2

