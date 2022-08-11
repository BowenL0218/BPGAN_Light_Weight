#!/bin/bash
mkdir datasets
mkdir tmp 
cd tmp
wget https://data.deepai.org/timit.zip
unzip timit.zip
cd ../Data_Processing
python Generate_Dataset_8k.py --phase='train'
python Generate_Dataset_8k.py --phase='test'
cd ../
rm -r tmp
