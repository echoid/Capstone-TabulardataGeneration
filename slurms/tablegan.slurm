#! /bin/bash


#! /bin/bash

#SBATCH -p deeplearn
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --job-name=tablegan
#SBATCH --mail-user=youran@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --time=150:00:00
#SBATCH --mem=10G
#SBATCH --output=outs/tablegan.out


module load fosscuda/2019b
module load python/3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load scikit-learn/0.23.1-python-3.7.4
module load tensorflow/1.15.0-python-3.7.4
module load pytorch/1.5.1-python-3.7.4
module load tqdm/4.41.1



python tableGAN/main.py --train --dataset=adult --epoch=300 --test_id=OI_11_00
python tableGAN/main.py --dataset=adult  --test_id=OI_11_00


# python tableGAN/main.py --train --dataset=covertype --epoch=5 --test_id=OI_11_00 --y_dim=7
# python tableGAN/main.py --dataset=covertype  --test_id=OI_11_00 --y_dim=7

python tableGAN/main.py --train --dataset=credit --epoch=300 --test_id=OI_11_00
python tableGAN/main.py --dataset=credit  --test_id=OI_11_00