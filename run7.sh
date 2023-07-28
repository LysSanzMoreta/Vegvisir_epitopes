










CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 5000;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset10 -lt semisupervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 5000;

CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore_non_anchor -save-all False -num-unobserved 5000;



CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 5000;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 5000;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 5000;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 5000;


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -st Icore -save-all False -num-unobserved 5000;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -st Icore -save-all False -num-unobserved 5000;

