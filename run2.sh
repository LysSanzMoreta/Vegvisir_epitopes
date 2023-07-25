#!/bin/bash

#Highlight: old test!!

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;



CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#Highlight: New test!!!

#Highlight: Supervised blosum
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
#Highlight: Supervised onehot
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

#Highlight: Supervised blosum-filter kmers
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
#Highlight: Supervised onehot - filter-kmers
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -random True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;
#Highlight: With learning representations
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers False -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding blosum -plot-all True -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test False -validate True -num-samples 30 -shuffle False -k-folds 1 -filter-kmers True -encoding onehot -plot-all True -st Icore_non_anchor -save-all False;
