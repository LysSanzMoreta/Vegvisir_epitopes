#!/bin/bash




CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -save-all False -st Icore;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore -save-all False;



CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding blosum -plot-all True -st Icore -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding onehot -plot-all True -st Icore -save-all False;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore -save-all False;
