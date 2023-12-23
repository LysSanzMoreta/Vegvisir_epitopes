#Highlight: First will length mask
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -train True -generate True -immunomodulate True -generate-argmax True -num-synthetic-peptides 80 -num-immunomodulate-peptides 80
CUDA_VISIBLE_DEVICES=1 python Vegvisir_example.py -train True -generate True -immunomodulate True -generate-argmax False -num-synthetic-peptides 80 -num-immunomodulate-peptides 80