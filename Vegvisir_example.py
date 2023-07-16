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
import vegvisir.plots as VegvisirPlots

if "CUDA_VISIBLE_DEVICES" in os.environ:
    pass
else:
    print("Cuda device has not been specified in your environment variables, setting it to cuda 0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Loading Vegvisir module from {}".format(vegvisir.__file__))
now = datetime.datetime.now()

def define_suffix(args):
    kmers = "_9mers" if args.filter_kmers else ""
    encoding = "_{}".format(args.encoding)
    if args.shuffle_sequence:
        if args.test:
            suffix = encoding + "_shuffled_TESTING{}".format(kmers)
        else:
            suffix = encoding + "_shuffled{}".format(kmers)
    elif args.random_sequences:
        if args.test:
            suffix = encoding + "_random_TESTING{}".format(kmers)
        else:
            suffix = encoding + "_random{}".format(kmers)
    elif args.num_mutations > 0:
        if args.test:
            suffix = encoding + "_{}_mutations_positions_{}_TESTING".format(args.num_mutations,
                                                                 args.idx_mutations if args.idx_mutations is not None else "random")
        else:
            suffix = encoding + "_{}_mutations_positions_{}".format(args.num_mutations,args.idx_mutations if args.idx_mutations is not None else "random")
    else:
        if args.test:
            suffix = encoding + "_TESTING{}".format(kmers)
        else:
            suffix = encoding + "{}".format(kmers)
    return suffix
def main():
    """Executes nnalignpy:
    1) Select the train/validation/test dataset
    2) Process the data and perform exploratory analysis
    2) Execute Vegvisir"""

    suffix = define_suffix(args)
    results_dir = "{}/PLOTS_Vegvisir_{}_{}_{}epochs_{}_{}{}".format(script_dir, args.dataset_name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs,args.learning_type,args.sequence_type,suffix)
    VegvisirUtils.folders(ntpath.basename(results_dir), script_dir)
    if args.k_folds > 1:
        for kfold in range(args.k_folds):
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Train_fold_{}".format(kfold)), script_dir)
            if args.test:
                VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Test_fold_{}".format(kfold)), script_dir)
                if args.validate:
                    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Valid_fold_{}".format(kfold)),
                                          script_dir)

            else:
                VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Valid_fold_{}".format(kfold)),script_dir)

    else:
        VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Train"), script_dir)
        VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Valid"), script_dir)
        VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Test"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Vegvisir_checkpoints"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Scripts"), script_dir)



    vegvisir_dataset = vegvisir.select_dataset(args.dataset_name, script_dir,args,results_dir, update=False)

    json.dump(args.__dict__, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)

    vegvisir.run(vegvisir_dataset,results_dir,args)


def analysis_models():
    """"""



    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_supervised,results_folder = "A_Stress_testing")

    # dict_results_supervised2 = {"supervised":{
    #                             r"viral-dataset-9-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/A1_Stress_testing/Supervised/PLOTS_Vegvisir_viral_dataset9_2023_07_10_23h20min56s048189ms_60epochs_supervised_Icore_onehot_TESTING",
    #                             r"viral-dataset-9-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/A1_Stress_testing/Supervised/PLOTS_Vegvisir_viral_dataset9_2023_07_11_15h36min10s584214ms_60epochs_supervised_Icore_blosum_TESTING"
    #                             },
    #                             "semisupervised":{
    #                             r"viral-dataset-8-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/A1_Stress_testing/Semisupervised/PLOTS_Vegvisir_viral_dataset8_2023_07_11_11h08min17s385658ms_60epochs_semisupervised_Icore_blosum_TESTING",
    #                             r"viral-dataset-10-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/A1_Stress_testing/Semisupervised/PLOTS_Vegvisir_viral_dataset10_2023_07_11_00h09min40s965374ms_60epochs_semisupervised_Icore_blosum_TESTING"
    #                             }}

    dict_results_all = {"supervised":{
                                r"viral-dataset-3-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/PLOTS_Vegvisir_viral_dataset3_2023_07_15_14h29min10s783758ms_1epochs_supervised_Icore_blosum_TESTING",
                                # r"viral-dataset-3-onehot-variable-length":"",
                                # r"viral-dataset-9-blosum-variable-length":"",
                                # r"viral-dataset-9-blosum-shuffled-variable-length":"",
                                # r"viral-dataset-9-blosum-random-variable-length":"",
                                # r"viral-dataset-9-onehot-variable-length":"",
                                # r"viral-dataset-9-onehot-shuffled-variable-length":"",
                                # r"viral-dataset-9-onehot-random-variable-length":"",
                                # r"viral-dataset-9-blosum-9mers":"",
                                # r"viral-dataset-9-blosum-shuffled-9mers":"",
                                # r"viral-dataset-9-blosum-random-9mers":"",
                                # r"viral-dataset-9-onehot-shuffled-9mers":"",
                                # r"viral-dataset-9-onehot-random-9mers":"",
                                },
                                "semisupervised":{
                                #r"viral-dataset-8-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/PLOTS_Vegvisir_viral_dataset8_2023_07_14_12h16min29s680283ms_60epochs_semisupervised_Icore_blosum_TESTING",
                                # r"viral-dataset-8-onehot-variable-length":"",
                                # r"viral-dataset-10-blosum-variable-length":"",
                                # r"viral-dataset-10-blosum-shuffled-variable-length":"",
                                # r"viral-dataset-10-blosum-random-variable-length":"",
                                # r"viral-dataset-10-blosum-random-9mers":"",
                                # r"viral-dataset-10-onehot-variable-length":"",
                                # r"viral-dataset-10-onehot-shuffled-variable-length":"",
                                # r"viral-dataset-10-onehot-random-variable-length": "",
                                # r"viral-dataset-10-onehot-9mers": "",
                                # r"viral-dataset-10-onehot-shuffled-9mers":"",
                                # r"viral-dataset-10-onehot-random-9mers":"",
                                }
    }




    #assert args.dataset_name in ["viral_dataset8","viral_dataset6"], "In order to analyse the semi supervised performance we need to set num_obs_classes correctly, please select viral_dataset8 or viral_dataset6 and leraning-type to semisupervised"


    VegvisirPlots.plot_kfold_comparisons2(args,script_dir,dict_results_all,kfolds=1,results_folder = "Benchmark")


    VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_all,kfolds=1,results_folder="Benchmark")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    parser.add_argument('-name','--dataset-name', type=str, nargs='?',
                        default="viral_dataset10",
                        help='Dataset project name, look at vegvisir.available_datasets(). The data should be always located at vegvisir/src/vegvisir/data \n'
                             'viral_dataset3 : Only sequences, partitioned into train,validation and (old) test.If args.test = True, then the (old) assigned test is used \n'
                             'viral_dataset4 : viral_dataset3 sequences + Features \n '
                             'viral_dataset5: Contains additional artificially generated negative data points in the (old) test dataset \n'
                             'viral_dataset6: Contains additional unobserved (negative and positive) data points for semi supervised learning. Train, validation and (old) test are mixed. If args.test = True, then 1 partition is selected as test \n'
                             'viral_dataset7: Same dataset as viral_dataset3, but the test dataset is mixed with the train and validation datasets \n'
                             'viral_dataset8: Same dataset as viral_dataset6 (containing unobserved datapoints), where the original test dataset is left out from the training (not mixed in).If args.test = True, then the (old) assigned test is used'
                             'viral_dataset9: Same as viral_dataset7 with a new test dataset (OLD test,train and validation are mixed). New test dataset available when using args.test=True'
                             'viral_dataset10: Same as viral_dataset6 (containing unobserved datapoints) with a new test dataset (OLD test,train and validation are mixed). New test available when using args.test=True')
    parser.add_argument('-subset_data', type=str, default="no",
                        help="Pick only the first <n> datapoints (epitopes) for testing the pipeline\n"
                             "<no>: Keep all \n"
                             "<insert_number>: Keep first <n> data points")
    parser.add_argument('--run-nnalign', type=bool, nargs='?', default=False, help='Executes NNAlign 2.1 as in https://services.healthtech.dtu.dk/service.php?NNAlign-2.1')
    parser.add_argument('-n', '--num-epochs', type=int, nargs='?', default=1, help='Number of epochs + 1  (number of times that the model is run through the entire dataset (all batches) ')
    parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=True, help='True: Use GPU; False: Use CPU')
    parser.add_argument('-encoding', type=str, nargs='?', default="blosum", help='<blosum> Use the matrix selected in args.subs_matrix to encode the sequences as blosum vectors'
                                                                                 '<onehot> One hot encoding of the sequences  ')

    #TODO: include more blosum matrix types?
    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str,
                        help='blosum matrix to create blosum embeddings, choose one from /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')

    parser.add_argument('-k-folds', type=int, nargs='?', default=2, help='Number of k-folds for k-fold cross validation.\n '
                                                                         'If set to 1 is a single run where 1 of the partitions is selected randomly'
                                                                         'as the validation')
    parser.add_argument('-batch-size', type=int, nargs='?', default=100, help='Batch size')
    parser.add_argument('-num-unobserved', type=int, nargs='?', default=5000, help='Use with datasets: <viral_dataset6> or <viral_dataset8>. \n'
                                                                                   'It establishes the number of unobserved(unlabelled) datapoints to use')

    parser.add_argument('-optimizer_name', type=str, nargs='?', default="Adam", help='Gradient optimizer name \n '
                                                                                     '<ClippedAdam>'
                                                                                     '<Adam>')
    parser.add_argument('-loss-func', type=str, nargs='?', default="elbo", help="Error loss function to be optimized, options are: \n"
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
    parser.add_argument('-validate', type=str2bool, nargs='?', default=True, help='Evaluate the model on the validation dataset')
    parser.add_argument('-test', type=str2bool, nargs='?', default=False, help='Evaluate the model on the external test dataset')
    parser.add_argument('-plot-all','--plot-all', type=str2bool, nargs='?', default=False, help='True: Plots all UMAPs and other computationally expensive plots. Do not use when args.k_folds > 1, it saturates the CPU & GPU memory'
                                                                                                'False: Only plots the computationally inexpensive ROC curves')
    parser.add_argument('-train', type=str2bool, nargs='?', default=True, help='<True> Run the model'
                                                                                '<False> Make models comparison ')


    parser.add_argument('-aa-types', type=int, nargs='?', default=20, help='Define the number of unique amino acid types. It determines the blosum matrix to be used. If the sequence contains gaps, the script will use 20 aa + 1 gap character ')
    parser.add_argument('-filter-kmers', type=str2bool, nargs='?', default=False, help="Filters the dataset to 9-mers only")

    parser.add_argument('-st','--sequence-type', type=str, nargs='?', default="Icore_non_anchor", help='Define the type of peptide sequence to use:\n'
                                                                                'Icore: Full peptide '
                                                                                'Icore_non_anchor: Peptide without the anchoring points marked by NetMHCPan 4.1')
    parser.add_argument('-p','--seq-padding', type=str, nargs='?', default="ends", help='Controls how the sequences are padded to the length of the longest sequence \n'
                                                                                    '<ends>: The sequences are padded at the end \n'
                                                                                    '<borders>: The sequences are padded at the beginning and the end. Random choice when the pad is an even number \n'
                                                                                    '<replicated_borders>: Padds by replicating the borders of the sequence \n'
                                                                                    '<random>: random insertion of 0 along the sequence \n')

    parser.add_argument('-shuffle','--shuffle_sequence', type=str2bool, nargs='?', default=False, help='Shuffling the sequence prior to padding for model stress-testing')
    parser.add_argument('-random','--random_sequences', type=str2bool, nargs='?', default=False, help='Create completely random peptide sequences for model stress-testing')
    parser.add_argument('-mutations','--num_mutations', type=int, nargs='?', default=0, help='Mutate the original sequences n times for model stress-testing')
    parser.add_argument('-idx-mutations','--idx_mutations', type=str, nargs='?', default=None, help='Positions where to perform the mutations for model stress-testing. Indicated as string of format [2,5,3], Set to None otherwise')

    parser.add_argument('-z-dim','--z-dim', type=int, nargs='?', default=30, help='Latent space dimension')
    parser.add_argument('-beta-scale', type=int, nargs='?', default=1, help='Scaling the KL (p(z) | p(z \mid x)) of the variational autoencoder')
    parser.add_argument('-hidden-dim', type=int, nargs='?', default=40, help='Dimensions of fully connected networks')
    parser.add_argument('-embedding-dim', type=int, nargs='?', default=40, help='Embedding dimensions, use with self-attention')
    parser.add_argument('-save-all', type=str2bool, nargs='?', default=False, help='<True> Saves every matrix output from the model'
                                                                                   '<False> Only saves a selection of model outputs necessary for benchmarking')

    parser.add_argument('-lt','--learning-type', type=str, nargs='?', default="semisupervised", help='<supervised_no_decoder> simpler model architecture with an encoder and a classifier'
                                                                                                 '<unsupervised> Unsupervised learning. No classification is performed \n'
                                                                                                 '<semisupervised> Semi-supervised model/learning. The likelihood of the class (p(c | z)) is only computed and maximized using the most confident scores. \n '
                                                                                                            'The non confident data points are inferred by the guide \n'
                                                                                                 '<supervised> Supervised model. All target observations are used to compute the likelihood of the class given the latent representation')

    parser.add_argument('-glitch','--glitch', type=str2bool, nargs='?', default=True, help='NOT USED'
                                                                                           '<True>: Applies a random noise distortion (via rotations) to the encoded vector within the conserved positions of the sequences  \n'
                                                                                           '<False>: The blosum encodings are left untouched')
    parser.add_argument('-num-samples','-num_samples', type=int, nargs='?', default=3, help='Number of samples from the posterior predictive. Only makes sense when using amortized inference with a guide function')
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
    if args.dataset_name in ["viral_dataset6","viral_dataset8","viral_dataset10"]:
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
    if args.train:
        main()
    else:
        analysis_models()

