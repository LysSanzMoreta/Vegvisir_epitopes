#Highlight: First will length mask
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -name viral_dataset15 -train True -validate False -test True -k-folds 1 -generate True -immunomodulate True -num-synthetic-peptides 50 -num-generate-loops 3 -plot-all True -generate-sampling-type independent -generate-argmax False
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -name viral_dataset15  -train True -validate False -test True -k-folds 1 -generate True -immunomodulate True -num-synthetic-peptides 50 -num-generate-loops 3 -plot-all True -generate-sampling-type conditional -generate-argmax False
