#!/bin/bash


#Highlight: Semi supervised with old test dataset as unobserved

#CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding blosum -plot-all True -st Icore -save-all False -num-unobserved 0;
#CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding onehot -plot-all True -st Icore -save-all False -num-unobserved 0;
#
#CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 0;
#CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 0;


##Highlight: Old dataset out
#
#CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
#CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
#
#
#
##Highlight: SemiSupervised blosum
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
#Highlight: SemiSupervised onehot
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;


#Highlight: SemiSupervised blosum filter kmers
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
#Highlight: SemiSupervised onehot filter kmers
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#Highlight: Single fold

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;
