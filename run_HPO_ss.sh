#!/bin/bash
#Highlight: Semisupervised HPO
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset17 -lt semisupervised -predefined-partitions True -train True -validate True -test False -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo True;
