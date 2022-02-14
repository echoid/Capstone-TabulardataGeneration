# Capstone-TabulardataGeneration

## Source codes

Tablegan(modified by Minshen): https://github.com/mahmoodm2/tableGAN (tf:1.x)

Daisy: https://github.com/ruclty/Daisy

Selnet: https://github.com/ppo2020/SIGMOD2021ID73 (tf:1.x)

## Data:
Adult: https://github.com/ruclty/Daisy/blob/master/dataset/adult_train.csv


### To run this model (Adult example) 

** generated pre-trained functional dependiencies models. 
python pretrained_fd.py 

** preprocess data, generate high dim representation
python selectivity_generation.py

** pretrained selectivity model
python pretrained_sel.py 

