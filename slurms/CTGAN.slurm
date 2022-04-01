#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=ctgan
#SBATCH --time=150:00:00
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --mem=10G
#SBATCH --output=outs/ctgan_all.out

# module load fosscuda/2019b
# module load python/3.7.4



module load fosscuda/2019b python/3.7.4
module load tqdm/4.41.1

python CTGAN/ctgan_generate.py adult 300
python CTGAN/ctgan_generate.py covertype 300
python CTGAN/ctgan_generate.py credit 300
python CTGAN/ctgan_generate.py news 300
python CTGAN/ctgan_generate.py ticket 300

python CTGAN/ctgan_generate.py adult 1000
python CTGAN/ctgan_generate.py covertype 1000
python CTGAN/ctgan_generate.py credit 1000
python CTGAN/ctgan_generate.py news 1000
python CTGAN/ctgan_generate.py ticket 1000
