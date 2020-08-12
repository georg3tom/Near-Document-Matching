#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

pdfPath="./data/pdf/"
imgPath="./data/img/"
for i in $(ls $pdfPath);
do
    name=$(echo $i |sed 's/\./\n/g'  | head -n1)
    pdftoppm $pdfPath$i $imgPath$name -png
    echo $pdfPath$i
done

python3 data_gen.py
