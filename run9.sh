#Highlight:Predefined_partitions: Icore_non_anchor


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle True -shuffle-labels False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle True -shuffle-labels False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;

#needed to repeat the experiment with predicting the random sequences
CUDA_VISIBLE_DEVICES=0 python Vegvisir_example.py -n 60 -name viral_dataset15 -lt supervised -predefined-partitions True -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -shuffle-labels False -random False -k-folds 1 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;



#Highlight: random stratified partitions: Icore
#
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
#
#
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;


#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -shuffle-labels True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -shuffle-labels True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;



#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore -pretrained-model "None" -hpo False;


#Highlight: random stratified partitions: Icore_non_anchor

#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -random True -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -random True -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -shuffle-labels True -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 30 -shuffle False -shuffle-labels True -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -random False -k-folds 5 -filter-kmers False -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;
#CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -n 60 -name viral_dataset9 -lt supervised -predefined-partitions False -train True -test True -validate True -generate False -immunomodulate False -num-samples 60 -shuffle False -random False -k-folds 5 -filter-kmers True -encoding blosum -plot-all False -save-all False -st Icore_non_anchor -pretrained-model "None" -hpo False;


