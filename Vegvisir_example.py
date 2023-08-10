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
from argparse import Namespace

if "CUDA_VISIBLE_DEVICES" in os.environ:
    pass
else:
    print("Cuda device has not been specified in your environment variables, setting it to cuda device 0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Loading Vegvisir module from {}".format(vegvisir.__file__))
now = datetime.datetime.now()

def define_suffix(args):
    kmers = "_{}mers".format("9" if args.sequence_type == "Icore" else "8") if args.filter_kmers else ""
    #kmers_name = "_{}mers".format("9" if args.sequence_type == "Icore" else "8") if args.filter_kmers else "variable-length"
    encoding = "_{}".format(args.encoding)
    num_unobserved = "_{}_unobserved".format(args.num_unobserved) if args.learning_type == "semisupervised" else ""
    if args.shuffle_sequence:
        if args.test:
            suffix =  "_shuffled_TESTING{}".format(kmers)
        else:
            suffix ="_shuffled{}".format(kmers)
    elif args.random_sequences:
        if args.test:
            suffix =  "_random_TESTING{}".format(kmers)
        else:
            suffix = "_random{}".format(kmers)
    elif args.num_mutations > 0:
        if args.test:
            suffix = "_{}_mutations_positions_{}_TESTING".format(args.num_mutations,
                                                                 args.idx_mutations if args.idx_mutations is not None else "random")
        else:
            suffix = "_{}_mutations_positions_{}".format(args.num_mutations,args.idx_mutations if args.idx_mutations is not None else "random")
    else:
        if args.test:
            suffix = "_TESTING{}".format(kmers)
        else:
            suffix = "{}".format(kmers)
    #name = args.dataset_name + "-" + encoding + "-" + kmers_name
    return encoding + suffix + num_unobserved
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
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Train_fold_{}".format(kfold)), script_dir) #TODO: 2 folders for train
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
    """Analyses the results of all possible model combinations (stress testing)"""



    dict_results_likelihood = {"supervised(Icore)":
                            {"vd9-10":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_14h37min35s894838ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_10",
                            "vd9-20":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_16h14min44s785096ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_20",
                            "vd9-30":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_17h48min26s866987ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_30",
                            "vd9-40":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_19h21min17s085055ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_40",
                            "vd3-30": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_03_21h44min10s036982ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_30",
                            "vd3-40": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_03_23h04min07s833125ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_40",
                            "vd3-50": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_04_00h25min53s635200ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_50",
                            "vd3-60":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_04_01h47min14s179667ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_60"
                             }}




    dict_results_predefined_partitions_100 = {"Icore":{
                                                    "random-blosum-variable-length":"",
                                                    "random-blosum-9mers":"",
                                                    "shuffled-blosum-variable-length":"",
                                                    "shuffled-blosum-9mers":"",
                                                    "raw-blosum-variable-length":"",
                                                    "raw-blosum-9mers":"",
                                                     "raw-onehot-variable-length":"",
                                                     "raw-onehot-9mers":""
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "",
                                                     "random-blosum-8mers": "",
                                                     "shuffled-blosum-variable-length": "",
                                                     "shuffled-blosum-8mers": "",
                                                     "raw-blosum-variable-length": "",
                                                     "raw-blosum-8mers": "",
                                                     "raw-onehot-variable-length":"",
                                                     "raw-onehot-8mers":""
                                         }}

    dict_results_random_stratified_partitions_100 = {"Icore":{
                                                    "random-blosum-variable-length":"",
                                                    "random-blosum-9mers":"",
                                                    "shuffled-blosum-variable-length":"",
                                                    "shuffled-blosum-9mers":"",
                                                    "raw-blosum-variable-length":"",
                                                    "raw-blosum-9mers":"",
                                                     "raw-onehot-variable-length":"",
                                                     "raw-onehot-9mers":""
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "",
                                                     "random-blosum-8mers": "",
                                                     "shuffled-blosum-variable-length": "",
                                                     "shuffled-blosum-8mers": "",
                                                     "raw-blosum-variable-length": "",
                                                     "raw-blosum-8mers": "",
                                                     "raw-onehot-variable-length":"",
                                                     "raw-onehot-8mers":""
                                         }}




    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_likelihood,kfolds=5,results_folder = "Benchmark/Plots",title="likelihood_tuning",overwrite=False)




    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_100,kfolds=5,results_folder = "Benchmark/Plots",title="predefined_partitions_likelihood_100",overwrite=False)
    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_random_stratified_partitions_100,kfolds=5,results_folder = "Benchmark/Plots",title="random_stratified_partitions_likelihood_40",overwrite=True)

    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_random_stratified_partitions_40,kfolds=5,results_folder="Benchmark/Plots",subtitle="random_stratified_partitions_likelihood_40",overwrite_correlations=False,overwrite_all=True)
    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_predefined_partitions_100,kfolds=5,results_folder="Benchmark/Plots",subtitle="predefined_partitions_likelihood_100",overwrite_correlations=False,overwrite_all=True)
    exit()

    dict_results_benchmark= {
        "viral-dataset9-likelihood-40":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_19h21min17s085055ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_40"
    }


    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,folder="Benchmark")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    parser.add_argument('-name','--dataset-name', type=str, nargs='?',
                        default="viral_dataset9",
                        help='Dataset project name, look at vegvisir.available_datasets(). The data should be always located at vegvisir/src/vegvisir/data \n'
                             'custom_dataset: Perform training or prediction (by setting the args.pretrained_model to the folder path with the model checkpoints). Define also train_path, test_path'
                             'viral_dataset3 : Only sequences, partitioned into train,validation and (old) test.If args.test = True, then the (old) assigned test is used \n'
                             'viral_dataset4 : viral_dataset3 sequences + Features \n '
                             'viral_dataset5: Contains additional artificially generated negative data points in the (old) test dataset \n'
                             'viral_dataset6: Contains additional unobserved data points for semi supervised learning. Train, validation and (old) test are mixed. If args.test = True, then 1 partition is selected as test \n'
                             'viral_dataset7: Same dataset as viral_dataset3, but the test dataset is mixed with the train and validation datasets \n'
                             'viral_dataset8: Same dataset as viral_dataset6 (containing unobserved datapoints), where the original test dataset is left out from the training (not mixed in).If args.test = True, then the (old) assigned test is used \n'
                             'viral_dataset9: Same as viral_dataset7 with a new test dataset (OLD test,train and validation are mixed). New test dataset available when using args.test=True \n'
                             'viral_dataset10: Same as viral_dataset6 (containing unobserved datapoints) with a new test dataset (OLD test,train and validation are mixed). New test available when using args.test=True \n'
                             'viral_dataset11: Similar to viral_dataset6 (containing unobserved datapoints), but the (old) test is incorporated as an unobserved sequence as well. (old) test available when using args.test=True \n'
                             'viral_dataset12: Training only on the unobserved data points with randomly assigned labels. MHC binders without binary targets'
                             )


    parser.add_argument('-subset-data', type=str, default="no",
                        help="Pick only the first <n> datapoints (epitopes) for testing the pipeline\n"
                             "<no>: Keep all \n"
                             "<insert_number>: Keep first <n> data points")
    parser.add_argument('--run-nnalign', type=bool, nargs='?', default=False, help='Executes NNAlign 2.1 as in https://services.healthtech.dtu.dk/service.php?NNAlign-2.1')
    parser.add_argument('-n', '--num-epochs', type=int, nargs='?', default=20, help='Number of epochs + 1  (number of times that the model is run through the entire dataset (all batches) ')
    parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=True, help='True: Use GPU; False: Use CPU')
    parser.add_argument('-encoding', type=str, nargs='?', default="blosum", help='<blosum> Use the matrix selected in args.subs_matrix to encode the sequences as blosum vectors'
                                                                                 '<onehot> One hot encoding of the sequences  ')

    #TODO: include more blosum matrix types?
    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str,
                        help='blosum matrix to create blosum embeddings, choose one from /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')

    parser.add_argument('-k-folds', type=int, nargs='?', default=1, help='Number of k-folds for k-fold cross validation.\n '
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

    parser.add_argument('-train', type=str2bool, nargs='?', default=True ,help='<True> Run the model \n <False> Make models comparison ')
    parser.add_argument('-validate', type=str2bool, nargs='?', default=True, help='Evaluate the model on the validation dataset')
    parser.add_argument('-test', type=str2bool, nargs='?', default=True, help='Evaluate the model on the external test dataset')

    parser.add_argument('-train-path', type=str2None, nargs='?', default="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/custom_dataset/unobserved_grouped_alleles_train.tsv",help="Path to training dataset. Use only for training. ")
    parser.add_argument('-test-path', type=str2None, nargs='?', default= "", help='Path to sequences to predict')
    parser.add_argument('-predefined-partitions', type=str2bool, nargs='?', default= True, help='<True> Divides the dataset into train, validation and test according to pre-specified partitions (in the sequences file, use a column named partitions)'
                                                                                                '<False> Performs a random stratified train, validation and test split')


    parser.add_argument('-plot-all','--plot-all', type=str2bool, nargs='?', default=True, help='True: Plots all UMAPs and other computationally expensive plots. Do not use when args.k_folds > 1, it saturates the CPU & GPU memory'
                                                                                                'False: Only plots the computationally inexpensive ROC curves')

    parser.add_argument('-aa-types', type=int, nargs='?', default=20, help='Define the number of unique amino acid types. It determines the blosum matrix to be used. If the sequence contains gaps, the script will use 20 aa + 1 gap character ')
    parser.add_argument('-filter-kmers', type=str2bool, nargs='?', default=False, help="Filters the dataset to 9-mers only")

    parser.add_argument('-st','--sequence-type', type=str, nargs='?', default="Icore", help='Define the type of peptide sequence to use:\n'
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
    parser.add_argument('-likelihood-scale', type=int, nargs='?', default=100, help='Scaling the log p( class | Z) of the variational autoencoder (cold posterior)'
                                                                                   '100: Assign likelihood to batch size')
    parser.add_argument('-hidden-dim', type=int, nargs='?', default=40, help='Dimensions of fully connected networks')
    parser.add_argument('-embedding-dim', type=int, nargs='?', default=40, help='Embedding dimensions, use with self-attention')
    parser.add_argument('-save-all', type=str2bool, nargs='?', default=False, help='<True> Saves every matrix output from the model'
                                                                                   '<False> Only saves a selection of model outputs necessary for benchmarking')

    parser.add_argument('-lt','--learning-type', type=str, nargs='?', default="supervised", help='<supervised_no_decoder> simpler model architecture with an encoder and a classifier'
                                                                                                 '<unsupervised> Unsupervised learning. No classification is performed \n'
                                                                                                 '<semisupervised> Semi-supervised model/learning. The likelihood of the class (p(c | z)) is only computed and maximized using the most confident scores. \n '
                                                                                                            'The non confident data points are inferred by the guide \n'
                                                                                                 '<supervised> Supervised model. All target observations are used to compute the likelihood of the class given the latent representation')

    parser.add_argument('-glitch','--glitch', type=str2bool, nargs='?', default=False, help='NOT USED at the moment, does not seem necessary. Only works with blosum encodings'
                                                                                           '<True>: Applies a random noise distortion (via rotations) to the encoded vector within the conserved positions of the sequences (mainly anchor points)  \n'
                                                                                           '<False>: The blosum encodings are left untouched')
    parser.add_argument('-num-samples','-num_samples', type=int, nargs='?', default=3, help='Number of samples from the posterior predictive. Only makes sense when using amortized inference with a guide function')

    #path = /home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_01_15h28min54s913529ms_60epochs_supervised_Icore_blosum_TESTING
    parser.add_argument('-pretrained-model', type=str2None, nargs='?', default="None",help='Load the checkpoints (state_dict and optimizer) from a previous run \n'
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
    if args.dataset_name in ["viral_dataset6","viral_dataset8","viral_dataset10","viral_dataset11"]:
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
    if args.pretrained_model is not None:
        args_dict = json.load(open("{}/commandline_args.txt".format(args.pretrained_model)))
        args_dict["pretrained_model"] = args.pretrained_model
        if args_dict["dataset_name"] == args.dataset_name:
            print("Overriding your current args (from argparser) to load the ones in {}".format(args.pretrained_model))
            args = Namespace(**args_dict)
    if args.train:
        main()
    else:
        analysis_models()


