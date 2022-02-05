# Capstone-TabulardataGeneration

## Source codes

Tablegan(modified by Minshen): https://github.com/mahmoodm2/tableGAN (tf:1.x)

Daisy: https://github.com/ruclty/Daisy

Selnet: https://github.com/ppo2020/SIGMOD2021ID73 (tf:1.x)

## Data:
Adult: https://github.com/ruclty/Daisy/blob/master/dataset/adult_train.csv



To run this model (Adult example) 
1. python pretrained_fd.py generated fd pretrained models.
2. python selectivity_generation.py dataset/train/adult_train.csv
3. python pretrained_sel pretrained the selectivity model