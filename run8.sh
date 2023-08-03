


CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -likelihood-scale 30 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -likelihood-scale 40 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -likelihood-scale 50 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore;
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset3 -lt supervised -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore;
