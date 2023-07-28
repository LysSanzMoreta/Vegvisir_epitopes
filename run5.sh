#!/bin/bash


#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -train False;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -save-all False -st Icore_non_anchor;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -save-all False -st Icore_non_anchor;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore_non_anchor -save-all False;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False;

sh run6.sh