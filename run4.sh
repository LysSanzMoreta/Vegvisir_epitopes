#!/bin/bash

#Highlight: Semi supervised with old test dataset as unobserved



CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 0 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 0 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 5000 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 5000 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 0 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 0 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 5000 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-unobserved 5000 -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;




#Highlight: Old dataset is not mixed in

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#Highlight: Filter kmers
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#Highlight: Shuffle -variable len

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#highlight: Shuffle & filter kmers

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#TODO: repeat with num-unobserved == 0

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;

#Highlight: Filter kmers
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;

#Highlight: Shuffle -variable len

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;

#highlight: Shuffle & filter kmers

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;
