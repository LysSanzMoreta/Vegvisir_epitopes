#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 5000;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 5000;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 5000;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset8 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 5000;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 0;

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 0;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 0;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 5000;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 5000;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset11 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 0;

sh run5.sh