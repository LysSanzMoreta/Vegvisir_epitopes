#Highlight: random stratified partitions: Icore

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;



#Highlight: random stratified partitions: Icore_non_anchor

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;

CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding onehot -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -num-samples 30 -likelihood-scale 60 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding onehot -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


