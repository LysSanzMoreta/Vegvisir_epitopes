#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""
import warnings

import numpy as np
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
import Vegvisir_analysis as VegvisirAnalysis

if "CUDA_VISIBLE_DEVICES" in os.environ:
    device = "cuda:{}".format(os.environ['CUDA_VISIBLE_DEVICES'])
else:
    print("Cuda device has not been specified in your environment variables, setting it to cuda device 0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda:0"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20000" #Not useful to prevent memory crashes :(
print("Loading Vegvisir module from {}".format(vegvisir.__file__))
now = datetime.datetime.now()

def define_suffix(args):
    kmers = "_{}mers".format("9" if args.sequence_type == "Icore" else "7") if args.filter_kmers else ""
    #kmers_name = "_{}mers".format("9" if args.sequence_type == "Icore" else "8") if args.filter_kmers else "variable-length"
    if args.hpo:
        encoding = "hpo_encoding"
    else:
        encoding = "_{}".format( json.load(open(args.config_dict,"r+"))["general_config"]["encoding"] if args.config_dict is not None and not isinstance(args.config_dict,dict) else args.num_epochs)
    num_unobserved = "_{}_unobserved".format(args.num_unobserved) if args.learning_type == "semisupervised" else ""
    if args.shuffle_sequence:
        if args.test:
            suffix =  "_shuffled_TESTING{}".format(kmers)
        else:
            suffix ="_shuffled{}".format(kmers)
    elif args.shuffle_labels:
        if args.test:
            suffix =  "_shuffled_labels_TESTING{}".format(kmers)
        else:
            suffix ="_shuffled_labels{}".format(kmers)
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

def main(args):
    """Executes Vegvisir:
    1) Select the train/validation/test dataset
    2) Process the data and perform exploratory analysis
    2) Execute Vegvisir"""

    suffix = define_suffix(args)
    if args.hpo:
        train_config = "HPO"
    else:

        train_config = "{}epochs".format( json.load(open(args.config_dict,"r+"))["general_config"]["num_epochs"] if args.config_dict is not None and not isinstance(args.config_dict,dict) else args.num_epochs)

    results_dir = "{}/PLOTS_Vegvisir_{}_{}_{}_{}_{}{}".format(script_dir, args.dataset_name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),train_config,args.learning_type,args.sequence_type,suffix)
    basepath = ntpath.basename(results_dir)
    VegvisirUtils.folders(basepath, script_dir)
    if args.k_folds > 1:
        for kfold in range(args.k_folds):
            VegvisirUtils.folders("{}/{}".format(basepath, "Train_fold_{}".format(kfold)), script_dir) #TODO: 2 folders for train
            if args.test:
                VegvisirUtils.folders("{}/{}".format(basepath, "Test_fold_{}".format(kfold)), script_dir)
                if args.validate:
                    VegvisirUtils.folders("{}/{}".format(basepath, "Valid_fold_{}".format(kfold)),script_dir)
            else:
                VegvisirUtils.folders("{}/{}".format(basepath, "Valid_fold_{}".format(kfold)),script_dir)
        if args.generate:
            VegvisirUtils.folders("{}/{}".format(basepath, "Generated"), script_dir)
        if args.immunomodulate:
            VegvisirUtils.folders("{}/{}".format(basepath, "Immunomodulated"), script_dir)
    else:
        VegvisirUtils.folders("{}/{}".format(basepath,"Train"), script_dir)
        VegvisirUtils.folders("{}/{}".format(basepath,"Valid"), script_dir)
        VegvisirUtils.folders("{}/{}".format(basepath,"Test"), script_dir)
        if args.generate:
            VegvisirUtils.folders("{}/{}".format(basepath, "Generated"), script_dir)
        if args.immunomodulate:
            VegvisirUtils.folders("{}/{}".format(basepath, "Immunomodulated"), script_dir)
    VegvisirUtils.folders("{}/{}".format(basepath,"Vegvisir_checkpoints"), script_dir)
    VegvisirUtils.folders("{}/{}".format(basepath,"Scripts"), script_dir)

    vegvisir_dataset,args = vegvisir.select_dataset(args.dataset_name, script_dir,args,results_dir, update=False)

    json.dump(args.__dict__, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)

    vegvisir.run(vegvisir_dataset,results_dir,args)


def flex_add_argument(f):
    '''Make the add_argument accept (and ignore) the gooey option.'''

    def f_decorated(*args, **kwargs):
        kwargs.pop('gooey_options', None)
        return f(*args, **kwargs)

    return f_decorated


def parser_args(parser,device,script_dir):


    if isinstance(parser,argparse.ArgumentParser):
        # force to ignore the gooey arguments
        parser.add_argument = flex_add_argument(parser.add_argument)

    parser.add_argument('-name', '--dataset-name', type=str, nargs='?',
                        # default="custom_dataset_random",
                        # default="custom_dataset_random_icore_non_anchor",
                        default="viral_dataset15",
                        help='Dataset project name, look at vegvisir.available_datasets(). In case the folder is not automatically created, you can manually place it at vegvisir/src/vegvisir/data \n'
                             'custom_dataset: Perform training or prediction (if predicting only,set the args.pretrained_model to the folder path with the model checkpoints) on a given dataset. Remember to define also train_path, test_path'
                             'viral_dataset3 : Supervised learning. Only sequences, partitioned into train,validation and (old) test.If args.test = True, then the (old) assigned test is used \n'
                             'viral_dataset4 : Supervised learning. viral_dataset3 sequences + Peptide Features \n '
                             'viral_dataset5: Supervised learning. Contains additional artificially generated negative data points in the (old) test dataset \n'
                             'viral_dataset6: Semisupervised learning. Contains additional unobserved data points for semi supervised learning. Train, validation and (old) test are mixed. If args.test = True, then 1 partition is selected as test \n'
                             'viral_dataset7: Supervised learning. Same dataset as viral_dataset3, but the (old) test dataset is mixed with the train and validation datasets \n'
                             'viral_dataset8: Semisupervised learning. Same dataset as viral_dataset6 (containing unobserved datapoints), where the original test dataset is left out from the training (not mixed in).If args.test = True, then the (old) assigned test is used \n'
                             'viral_dataset9: Supervised learning. Uses the OLD test,train and validation are mixed into the train. Same as viral_dataset7 with a new test dataset. New test dataset available when using args.test=True \n'
                             'viral_dataset10: Semisupervised learning. Same as viral_dataset6 (containing unobserved datapoints) with a new test dataset (OLD test,train and validation are mixed). New test available when using args.test=True \n'
                             'viral_dataset11: Semisupervised learning. Similar to viral_dataset6 (containing unobserved datapoints), but the (old) test is incorporated as an unobserved sequence as well. (old) test available when using args.test=True \n'
                             'viral_dataset12: Prediction only .Training only on the unobserved data points with randomly assigned labels. MHC binders without binary targets'
                             'viral_dataset13: Supervised training. Same train dataset as viral_dataset9. The test incluye includes discarded peptides without information about the number o patients tested'
                             'viral_dataset14: Supervised training. Peptide sequences restricted to binders from alleles HLA-A2402, HLA-A2301 and HLA-2407 '
                             'viral_dataset15: Supervised training. DATASET used in the ARTICLE. Supervised training. Same data points as in viral_dataset9 with different partitioning, everything mixed up proportionally without data leakage.'
                             'viral_dataset16: Supervised training. Same as viral_dataset15 restricted to binders to alleles HLA-A2402, HLA-A2301 and HLA-2407'
                             'viral_dataset17: Semisupervised training. Semisupervised equivalent to viral dataset 15 (with added unobserved data points per partition) ',
                        gooey_options = {"label_color": "#ff33d2"}
                        )
    # Highlight: Dataset configurations: Use with the default datasets (not necessary with the custom dataset, unless you want to do some stress testing)
    parser.add_argument('-predefined-partitions', type=str2bool, nargs='?', default=True,
                        help='<True> Divides the dataset into train, validation and test according to pre-specified partitions (to activate it, in the input file use a COLUMN named partitions)\n'
                             '<False> Performs a random stratified train, validation and test split')
    parser.add_argument('-num-unobserved', type=int, nargs='?', default=5000,
                        help='Use with datasets for semi supervised training: i.e <viral_dataset6> or <viral_dataset8>. \n'
                             'It establishes the number of unobserved(unlabelled) datapoints to incorporate, from the 50000 available')
    parser.add_argument('-aa-types', type=int, nargs='?', default=20,
                        help='Define the number of unique amino acid types. It determines the blosum matrix to be used. Gaps will be represented as # \n'
                             '<20>: If the sequence contains gaps, the script will use 20 aa + 1 gap character \n'
                             '<24>: Allows for rare amino acids \n')
    parser.add_argument('-filter-kmers', type=str2bool, nargs='?', default=False,
                        help="Filters the dataset to contain 9-mers only")
    parser.add_argument('-st', '--sequence-type', type=str, nargs='?', default="Icore",
                        help='Define the type of peptide sequence to use:\n'
                             'Icore: Full peptide (use this one)'
                             'Icore_non_anchor: Peptide without the anchoring points marked by NetMHCPan 4.1')
    parser.add_argument('-p', '--seq-padding', type=str, nargs='?', default="ends",
                        help='Controls how the sequences are padded to fit the length of the longest sequence \n'
                             '<ends>: The sequences are padded at the end (ARTICLE SETUP) \n'
                             '<borders>: The sequences are padded at the beginning and the end. Random choice when the pad is an even number \n'
                             '<replicated_borders>: Padds by replicating the borders of the sequence \n'
                             '<random>: random insertion of zeroes(gaps) along the sequence \n')
    # Highlight: Pytorch efficiency parameters
    parser.add_argument('-prc', '--precision', type=str, nargs='?', default="64",
                        help='Define the type of peptide sequence to use:\n'
                             '32: Float 32, lower precision, faster run, potentially slower convergence (requires more epochs) \n'
                             '64: Float 64, higher precision, slower run, potentially faster convergence (requires less epochs)',gooey_options= {"label_color":"#ff33d2"}) #,gooey_options= {"label_color":"#ff33d2"})


    # Highlight: Model stress testing

    parser.add_argument('-shuffle', '--shuffle_sequence', type=str2bool, nargs='?', default=False,
                        help='Stress-testing technique. Shuffling the original sequence aminoacid order  randomly ie AVINM -> shuffle -> NIAMA')
    parser.add_argument('-shuffle-labels', '--shuffle_labels', type=str2bool, nargs='?', default=False,
                        help='Stress-testing technique. Shuffling the labels across the dataset (preserves the classes proportions however it breaks the correct sequence-class association). ')
    parser.add_argument('-random', '--random_sequences', type=str2bool, nargs='?', default=False,
                        help='Stress-testing technique. Create completely random peptide sequences \n. '
                             'Transforms each of the given sequence into a random sequence of aminoacids. The resulting random dataset maintains the same length and classes proportions as the original one. ')
    parser.add_argument('-mutations', '--num_mutations', type=int, nargs='?', default=0,
                        help='Stress-testing technique (not used in the article). Per sequence it mutates the original sequences n times')
    parser.add_argument('-idx-mutations', '--idx_mutations', type=str, nargs='?', default=None,
                        help='Stress-testing technique (not used in the article). Positions where to perform the mutations. Indicated as string of format [2,5,3], Set to None otherwise')

    # Highlight: Model hyperparameters, those marked with HPO* are overridden by the dictionary given to args.config_dict unless it is set to None. The given args.config_dict contains the optimized parameters, do not change unless new data is used to train the model.
    parser.add_argument('-n', '--num-epochs', type=int, nargs='?', default=1,
                        help='HPO* . Number of epochs + 1  (number of times that the model is run through the entire dataset (all batches) ')
    parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=True, help='True: Use GPU; False: Use CPU',gooey_options= {"label_color":"#ff33d2"})
    parser.add_argument('-encoding', type=str, nargs='?', default="blosum", help='HPO* . Sequence encoding type'
                                                                                 '<blosum> Use the matrix selected in args.subs_matrix to encode the sequences as blosum vectors'
                                                                                 '<onehot> One hot encoding of the sequences  ')

    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str,
                        help='Blosum matrix used to to create blosum embeddings, choose one from python/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')

    parser.add_argument('-k-folds', type=int, nargs='?', default=1,
                        help='Number of k-folds for k-fold cross validation (value between 1 and 5).\n '
                             'If set to 1 is a single run where 1 of the partitions is selected randomly as the validation partition, and the test is partition 6'
                             'If set to >1 it will perform n kfold validation on n partitions  and the test is partition 6')
    parser.add_argument('-batch-size', type=int, nargs='?', default=170, help='HPO*. Batch size')
    parser.add_argument('-optimizer_name', type=str, nargs='?', default="Adam",
                        help='Gradient optimizer name. Adam with clip gradients is implemented in 2 modes \n '
                             '<ClippedAdam>'
                             '<Adam> : Adam + args.clip_gradients=True')
    parser.add_argument('-clip-gradients', type=bool, nargs='?', default=True,
                        help='Computes the 2D Euclidean norm of the gradient to normalize the gradient by that value and \n '
                             ' prevent exploding gradients (small gradients that lead to abscence of training) ')
    parser.add_argument('-hidden-dim', '--hidden-dim', type=int, nargs='?', default=20,
                        help="HPO*. Global parameter that controls the network's dimensionalities")
    parser.add_argument('-z-dim', '--z-dim', type=int, nargs='?', default=16,
                        help='HPO*. Latent space dimensionality')
    parser.add_argument('-likelihood-scale', type=int, nargs='?', default=40,
                        help='HPO* .Scaling the log p( class | Z) of the variational autoencoder (cold posterior)')

    parser.add_argument('-lt', '--learning-type', type=str, nargs='?', default="supervised",
                        help='<supervised_no_decoder> simpler model architecture with only an encoder and a classifier'
                             '<unsupervised> Unsupervised learning. No classification is performed \n'
                             '<semisupervised> Semi-supervised model/learning. The likelihood of the class (p(c | z)) is only computed and maximized using the most confident scores. \n '
                             'The non confident data points are inferred by the guide \n'
                             '<supervised> Supervised model. All target observations are used to compute the likelihood of the class given the latent representation')

    parser.add_argument('-num-samples', '-num_samples', type=int, nargs='?', default=3,
                        help='HPO* Number of samples from the posterior predictive, set to minimum 30, unless performing some debugging. Only makes sense when using amortized inference with a guide function')

    parser.add_argument('-hpo', type=str2bool, nargs='?', default=False,
                        help='<True> Performs Hyperparameter optimization with Ray Tune. Overwrites the Train/Valid/Test folders with the last result.\n'
                             ' Extract the best HPO results using VegvisirUtils.build_hpo_config_dict(folder_name) ')

    best_config = {0: "{}/BEST_hyperparameter_dict_blosum_supervised_vd15_z16.p".format(script_dir), #TODO: Make sure it is included in the standalone application, move to data/hpo ?
                   1: "{}/BEST_hyperparameter_dict_blosum_semisupervised.p".format(script_dir), #TODO: Make sure it is included in the standalone application
                   2: None} #None makes use of the hyperparameter values given to argparse
    parser.add_argument('-config-dict', nargs='?', default=best_config[0], type=str2None,
                        help='Path to the HPO optimized hyperparameter dict. Overrules the previous hyperparameters marked as HPO*.\n'
                             'Set to None to use the values in the parser.',
                        gooey_options= {"label_color":"#ff33d2"})

    # Highlight: Evaluation modes
    parser.add_argument('-train', type=str2bool, nargs='?', default=True, help='<True> Run the model over the training data '
                                                                               '\n <False> Makes benchmarking plots (Functions migrated to Vegvisir_analysis) or loads previously trained model, if pargs.pretrained_model is not None')
    parser.add_argument('-validate', type=str2bool, nargs='?', default=False,
                        help='Evaluate the model on the validation dataset. Only needed for model design')
    parser.add_argument('-test', type=str2bool, nargs='?', default=True,
                        help='Evaluate the model on the external test dataset')

    # Highlight: Generating new sequences from a trained model
    parser.add_argument('-generate', type=str2bool, nargs='?', default=False,
                        help='<True> Generate new neo-epitopes labelled and with a confidence score based on the training dataset. Please use args.validate False \n '
                             '<False> Do nothing')
    parser.add_argument('-num-synthetic-peptides', type=int, nargs='?', default=3,
                        help='Generate n new neo-epitopes labelled and with a confidence score. IMPORTANT: The total number of generated peptides is \n'
                             'equal to args.num_synthetic_peptides*args.num_samples*args.num_generate_loops')
    parser.add_argument('-num-generate-loops', type=int, nargs='?', default=1,
                        help='Number of times to repeat the sampling loop')
    parser.add_argument('-generate-sampling-type', type=str, nargs='?', default="conditional", help='<conditional> \n'
                                                                                                    '<independent>')
    parser.add_argument('-generate-argmax', type=str2bool, nargs='?', default=False, help='True \n False')

    # Highlight: immunomodulating (supression or enhancing of immunogencity) a sequence
    immunomodulate_path = {0: "{}/immunomodulate_sequences.txt".format(script_dir),
                           1: None}
    parser.add_argument('-immunomodulate', type=str2bool, nargs='?', default=False,
                        help='<True> Predict latent representation for the given sequences and generate new neo-epitopes labelled and with a confidence score \n'
                             ' based only on the input sequences via args.immunomodulate_path.Please use with args.validate False \n'
                             '<False> Do nothing')
    parser.add_argument('-num-immunomodulate-peptides', type=int, nargs='?', default=10,
                        help='Number of generated peptides generated from the sequence to immunomodulate/attempt to change their immunogenic class.\n'
                             ' The total number of generated peptides is equal to args.num_immunomodulates_peptides*args.num_samples')
    parser.add_argument('-immunomodulate-path', type=str2None, nargs='?', default=immunomodulate_path[0],
                        help='Path to text file containing sequences to change their immunogenicity')

    #/home/dragon/drive/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/variable_length_Icore_sequences_viral_dataset15_TRAIN.tsv
    # Highlight: Re-training the model/Using your own sequences
    unobserved_sequences = {0: f"{script_dir}/vegvisir/src/vegvisir/data/custom_dataset/unobserved_grouped_alleles_train.tsv", #TODO: Move to /data/immunomodulation and upload to drive
                            1: None,
                            2: f"{script_dir}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/random_variable_length_Icore_sequences_viral_dataset9.tsv",
                            3: f"{script_dir}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/variable_length_Icore_sequences_viral_dataset9_TRAIN.tsv",
                            4: f"{script_dir}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore_non_anchor/variable_length_Icore_non_anchor_sequences_viral_dataset9_TRAIN.tsv",
                            5: f"{script_dir}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore_non_anchor/random_variable_length_Icore_non_anchor_sequences_viral_dataset9.tsv",
                            6: f"{script_dir}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/variable_length_Icore_sequences_viral_dataset15_TRAIN.tsv",
                            7: f"{script_dir}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/random_variable_length_Icore_sequences_viral_dataset15.tsv"}
    parser.add_argument('-train-path', type=str2None, nargs='?', default=unobserved_sequences[6],
                        help="Path to an external training dataset. It only activates if args.dataset_name = custom_dataset. ")
    parser.add_argument('-test-path', type=str2None, nargs='?', default=unobserved_sequences[7],
                        help='Path to (test) sequences to predict/classify')

    # Highlight: Output saving settings
    parser.add_argument('-save-all', type=str2bool, nargs='?', default=False,
                        help='<True> Saves every matrix output from the model. Not recommended'
                             '<False> Only saves a selection of model outputs necessary for benchmarking')
    parser.add_argument('-plot-all', '--plot-all', type=str2bool, nargs='?', default=False,
                        help='<True>: Plots all UMAPs and other computationally expensive plots. Do not use when args.k_folds > 1,\n'
                             ' it saturates the CPU and GPU memory\n'
                             '<False>: Only plots the computationally inexpensive ROC curves')

    # Highlight: Use this one if you do not want to train the model, just predict, generate or immunomodulate
    pretrained_model = {
        0: "PLOTS_Vegvisir_viral_dataset9_2023_12_23_22h34min52s755194ms_100epochs_supervised_Icore_blosum_TESTING",
        1: None,
        2: "PLOTS_Vegvisir_viral_dataset9_2023_12_26_18h37min00s675744ms_60epochs_supervised_Icore_blosum_TESTING_z34",
        3: "PLOTS_Vegvisir_viral_dataset9_2023_12_26_18h38min16s083132ms_60epochs_supervised_Icore_blosum_TESTING_z4",
        4: "PLOTS_Vegvisir_viral_dataset9_2023_12_26_19h11min19s422780ms_60epochs_supervised_Icore_60_TESTING_z30",
        5: "PLOTS_Vegvisir_viral_dataset9_2024_01_08_03h27min20s749801ms_60epochs_supervised_Icore_60_random_TESTING",
        6: "PLOTS_Vegvisir_viral_dataset14_2024_01_09_15h22min37s940180ms_60epochs_supervised_Icore_blosum_TESTING",
        7: "PLOTS_Vegvisir_viral_dataset9_2024_01_11_15h29min17s494812ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
        8: "PLOTS_Vegvisir_viral_dataset15_2024_01_24_16h35min18s664516ms_60epochs_supervised_Icore_blosum_TESTING_z16"
        }

    parser.add_argument('-pretrained-model', type=str2None, nargs='?', default="{}".format(pretrained_model[1]),
                        help='Load the checkpoints (state_dict and optimizer) from a previous run \n'
                             '<None>: Trains model from scratch \n'
                             '<str path>: Loads pre-trained model from given path \n')

    # Highlight: DO NOT CHANGE ANYTHING ELSE DOWN HERE
    args = parser.parse_args()
    dtype_dict = VegvisirUtils.return_dtype_dict() #keep here for the GUI
    torch.set_default_dtype(dtype_dict[args.precision][0])
    if args.use_cuda:
        if torch.cuda.is_available():
            print("Using cuda")
            # device = "cuda"
            torch.set_default_device(device)  # use the device selected above
            args.__dict__["device"] = device
        else:
            print("Cuda (gpu) not found falling back to cpu. Depending on availability, please make sure to use cuda:0 (CUDA_VISIBLE_DEVICES=0) or cuda:1 (CUDA_VISIBLE_DEVICES=1)")
            device = "cpu"
            torch.set_default_device(device)
            args.__dict__["device"] = device
            args.__dict__["use_cuda"] = False
    else:
        device = "cpu"
        torch.set_default_device(device)
        args.__dict__["device"] = device

    if args.dataset_name in ["viral_dataset6", "viral_dataset8", "viral_dataset10", "viral_dataset11",
                             "viral_dataset17"]:
        args.__dict__["num_classes"] = 3  # Number of observed classes
        args.__dict__["num_obs_classes"] = 2  # Number of predicted classes
    else:
        args.__dict__["num_classes"] = 2  # Number of predicted classes
        args.__dict__["num_obs_classes"] = 2  # Number of observed classes
    # pyro.set_rng_seed(0)
    # torch.manual_seed(0)
    pyro.enable_validation(False)

    if args.pretrained_model is not None and args.train:
        args_dict = json.load(open("{}/commandline_args.txt".format(args.pretrained_model)))
        args_dict["pretrained_model"] = args.pretrained_model
        args_dict["num_epochs"] = 0
        args_dict["plot_all"] = args.plot_all
        if args_dict["dataset_name"] == args.dataset_name:
            warnings.warn(
                "You selected using the same dataset as in the pretrained model. Overriding your current args (from argparser) to load the ones in {}".format(
                    args.pretrained_model))
            args = Namespace(**args_dict)
        else:
            args_dict["dataset_name"] = args.dataset_name
            # args_dict["learning_type"] = args.learning_type
            args_dict["sequence_type"] = args.sequence_type
            args_dict["num_epochs"] = 0
            args_dict["train"] = args.train
            args_dict["validate"] = args.validate
            args_dict["test"] = args.test
            args_dict["generate"] = args.generate
            args_dict["immunomodulate"] = args.immunomodulate
            args_dict["train_path"] = args.train_path
            args_dict["test_path"] = args.test_path
            warnings.warn(
                "Overriding some of your current args except <dataset_name>,<sequence_type>,<num_obs_classes>,<num_classes>,<generate>,<iimunomodulate> ...")
            args = Namespace(**args_dict)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    args = parser_args(parser,device,script_dir)

    # VegvisirUtils.build_hpo_config_dict(hpo_folder="/home/dragon/drive/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset17_2024_10_16_22h39min35s293251ms_HPO_semisupervised_Icorehpo_encoding_5000_unobserved/HyperparamOptimization_results.tsv",
    #                                     name="semisupervised")


    if args.train and args.pretrained_model is None:
        main(args)
    else:
        VegvisirAnalysis.analysis_models(args)

