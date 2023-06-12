#!/usr/bin/env python3
"""
=======================
2023: Lys Sanz Moreta
nnalignpy :
=======================
"""
import pyro
import torch
import argparse
import os,sys,ntpath
import datetime
import json
from argparse import RawTextHelpFormatter
local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__))
if local_repository: #TODO: The local imports are extremely slow
     sys.path.insert(1, "{}/vegvisir/src".format(script_dir))
     import vegvisir
else:#pip installed module
     import vegvisir
from vegvisir import str2bool,str2None
import vegvisir.utils as VegvisirUtils
print("Loading Vegvisir module from {}".format(vegvisir.__file__))
now = datetime.datetime.now()
def main():
    """Executes nnalignpy:
    1) Select the train/validation/test dataset
    2) Execute Vegvisir"""

    results_dir = "{}/PLOTS_Vegvisir_{}_{}_{}epochs_{}_{}".format(script_dir, args.dataset_name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs,args.learning_type,args.sequence_type)
    VegvisirUtils.folders(ntpath.basename(results_dir), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Train"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Valid"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Test"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Vegvisir_checkpoints"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Scripts"), script_dir)



    vegvisir_dataset = vegvisir.select_dataset(args.dataset_name, script_dir,args,results_dir, update=False)

    json.dump(args.__dict__, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)

    vegvisir.run(vegvisir_dataset,results_dir,args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    parser.add_argument('-name','--dataset-name', type=str, nargs='?',
                        default="viral_dataset8",
                        help='Dataset project name, look at nnalignpy.available_datasets(). The data should be always located at nnalignpy/src/nnalignpy/data \n'
                             'viral_dataset3 : Only sequences, partitioned into train,validation and test \n'
                             'viral_dataset4 : Sequences + Features \n '
                             'viral_dataset5: Contains additional artificially generated negative data points \n'
                             'viral_dataset6: Contains additional artificially generated negative and positive data points for semi supervised learning. Train, validation and test are mixed \n'
                             'viral_dataset7: Same dataset as viral_dataset3, where the test dataset is mixed with the train and validation datasets \n'
                             'viral_dataset8: Same dataset as viral_dataset6, where the original test dataset is leaft out of the training')
    parser.add_argument('-subset_data', type=str, default="no",
                        help="Pick only the first <n> datapoints (epitopes) for testing the pipeline\n"
                             "<no>: Keep all \n"
                             "<insert_number>: Keep first <n> data points")
    parser.add_argument('--run-nnalign', type=bool, nargs='?', default=False, help='Executes NNAlign 2.1 as in https://services.healthtech.dtu.dk/service.php?NNAlign-2.1')
    parser.add_argument('-n', '--num-epochs', type=int, nargs='?', default=1, help='Number of epochs + 1  (number of times that the model is run through the entire dataset (all batches) ')
    parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=False, help='True: Use GPU; False: Use CPU')

    #TODO: include more blosum matrix types?
    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str,
                        help='blosum matrix to create blosum embeddings, choose one from /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')

    parser.add_argument('-k-folds', type=int, nargs='?', default=1, help='Number of k-fold for k-fold cross validation')
    parser.add_argument('-batch-size', type=int, nargs='?', default=100, help='Batch size')
    parser.add_argument('-optimizer_name', type=str, nargs='?', default="Adam", help='Gradient optimizer name \n '
                                                                                     '<ClippedAdam>'
                                                                                     '<Adam>')
    parser.add_argument('-loss-func', type=str, nargs='?', default="bcelogits", help="Error loss function to be optimized, options are: \n"
                                                                                         "<bcelogits>: Binary Cross Entropy with logits (no activation in last layer) \n "
                                                                                         "<bceprobs>: Binary Cross Entropy with probabilities (sigmoid activation)\n"
                                                                                         "<weighted_bce>: Weighted Binary Cross Entropy \n"
                                                                                         "<ae_loss>: Uses a reconstruction and a classification error loss \n"
                                                                                         "<softloss>: Label smoothing + Taylorsoftmax \n "
                                                                                         "<elbo>: Evidence Lower Bound objective used for SVI")
    parser.add_argument('-clip-gradients', type=bool, nargs='?', default=True, help='Computes the 2D Euclidean norm of the gradient to normalize the gradient by that value and \n '
                                                                                    ' prevent exploding gradients (small gradients that lead to abscence of training) ')

    parser.add_argument('-guide', type=str, nargs='?', default="custom", help='<custom>: See guides.py \n'
                                                                              '<autodelta> : Automatic guide for amortized inference in Pyro see pyro.autoguides. Does not work with mini-batching, (perhaps subsampling in the plate)')
    parser.add_argument('-test', type=str2bool, nargs='?', default=False, help='Evaluate the model on the external test dataset')

    parser.add_argument('-aa-types', type=int, nargs='?', default=20, help='Define the number of unique amino acid types. It determines the blosum matrix to be used. If the sequence contains gaps, the script will use 20 aa + 1 gap character ')
    parser.add_argument('-st','--sequence-type', type=str, nargs='?', default="Icore", help='Define the type of peptide sequence to use:\n'
                                                                                'Icore: Full peptide '
                                                                                'Icore_non_anchor: Peptide without the anchoring points marked from NetMHCPan 4.1')
    parser.add_argument('-p','--seq-padding', type=str, nargs='?', default="ends", help='Controls how the sequences are padded to the length of the longest sequence \n'
                                                                                    '<ends>: The sequences are padded at the end'
                                                                                    '<borders>: The sequences are padded at the beginning and the end. Random choice when the pad is an even number'
                                                                                    '<replicated_borders>: Padds by replicating the borders of the sequence'
                                                                                    '<random>: random insertion of 0 along the sequence')
    parser.add_argument('-shuffle','--shuffle_sequence', type=str2bool, nargs='?', default=False, help='Shuffling the sequence prior to padding for model stress-testing')
    parser.add_argument('-random','--random_sequences', type=str2bool, nargs='?', default=False, help='Create completely random peptide sequences for model stress-testing')


    parser.add_argument('-z-dim','--z-dim', type=int, nargs='?', default=30, help='Latent space dimension')
    parser.add_argument('-beta-scale', type=int, nargs='?', default=1, help='Scaling the KL (p(z) | p(z \mid x)) of the variational autoencoder')
    parser.add_argument('-hidden-dim', type=int, nargs='?', default=40, help='Dimensions of fully connected networks')
    parser.add_argument('-embedding-dim', type=int, nargs='?', default=40, help='Embedding dimensions, use with self-attention')
    parser.add_argument('-lt','--learning-type', type=str, nargs='?', default="semisupervised", help='<unsupervised> Unsupervised learning. The class is inferred directly from the latent representation and via amortized inference \n'
                                                                                        '<semisupervised> Semi-supervised model/learning. The likelihood of the class (p(c | z)) is only computed and maximized using the most confident scores. \n '
                                                                                                            'The non confident data points are inferred by the guide \n'
                                                                                        '<supervised> Supervised model. All target observations are used to compute the likelihood of the class given the latent representation')

    parser.add_argument('-glitch','--glitch', type=str2bool, nargs='?', default=True, help='<True>: Applies a random noise distortion (via rotations) to the encoded vector within the conserved positions of the sequences  \n'
                                                                                           '<False>: The blosum encodings are left untouched')
    parser.add_argument('-num_samples', type=int, nargs='?', default=3, help='Number of samples from the posterior predictive. Only makes sense when using amortized inference with a guide function')
    parser.add_argument('-pretrained-model', type=str2None, nargs='?', default="None", help='Load the checkpoints (state_dict and optimizer) from a previous run \n'
                                                                                                '<None>: Trains model \n'
                                                                                                '<str path>: Loads pre-trained model from given path \n')

    args = parser.parse_args()
    if args.use_cuda:
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            parser.add_argument('--device',type=str,default="cuda" ,nargs='?', help='Device choice (cpu, cuda:0, cuda:1), linked to use_cuda')
        else:
            print("Cuda (gpu) not found falling back to cpu")
            torch.set_default_tensor_type(torch.DoubleTensor)
            parser.add_argument('--device',type=str,default="cpu" ,nargs='?', help='Device choice (cpu, cuda:0, cuda:1), linked to use_cuda')
            parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=False,help='True: Use GPU; False: Use CPU')
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        parser.add_argument('--device', type=str, default="cpu", nargs='?',help='Device choice (cpu, cuda:0, cuda:1), linked to use_cuda')
    if args.dataset_name in ["viral_dataset6","viral_dataset8"]:
        parser.add_argument('-num_classes', type=int, nargs='?', default=3,help='Number of prediction classes. The model performs a regression task and the binary classification is derived from the entropy value')
        parser.add_argument('-num_obs_classes', type=int, nargs='?', default=2,help='Number of prediction classes. The model performs a regression task and the binary classification is derived from the entropy value')

    else:
        parser.add_argument('-num_classes', type=int, nargs='?', default=2,
                            help='Number of data type classes (includes observed and unobserved). The model performs a regression task and the binary classification is derived from the entropy value')
        parser.add_argument('-num_obs_classes', type=int, nargs='?', default=2,
                            help='Number of observed classes (positives, negatives). The model performs a regression task and the binary classification is derived from the entropy value')

    args = parser.parse_args()
    #pyro.set_rng_seed(0)
    #torch.manual_seed(0)
    pyro.enable_validation(False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main()