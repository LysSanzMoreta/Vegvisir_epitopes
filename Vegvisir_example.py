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

    dict_results_all = {"supervised(Icore)":{
                                r"viral-dataset-3-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_15_14h31min30s605832ms_60epochs_supervised_Icore_blosum_TESTING",
                                r"viral-dataset-3-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_15_15h53min19s522115ms_60epochs_supervised_Icore_onehot_TESTING",
                                r"viral-dataset-3-blosum-shuffled-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_21_15h02min41s885374ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                r"viral-dataset-3-onehot-shuffled-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_21_16h35min28s374245ms_60epochs_supervised_Icore_onehot_shuffled_TESTING",
                                r"viral-dataset-3-blosum-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_28_05h01min58s428968ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                r"viral-dataset-3-onehot-random-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_28_06h15min47s287899ms_60epochs_supervised_Icore_onehot_random_TESTING",
                                r"viral-dataset-3-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_21_11h15min37s877362ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                r"viral-dataset-3-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_21_12h01min17s340676ms_60epochs_supervised_Icore_onehot_TESTING_9mers",
                                r"viral-dataset-3-blosum-9mers-shuffled":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_28_07h21min38s605815ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                r"viral-dataset-3-onehot-9mers-shuffled":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_28_08h11min36s002925ms_60epochs_supervised_Icore_onehot_shuffled_TESTING_9mers",
                                r"viral-dataset-3-blosum-9mers-random":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_28_09h10min31s003786ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                r"viral-dataset-3-onehot-9mers-random":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset3_2023_07_28_09h58min27s397141ms_60epochs_supervised_Icore_onehot_random_TESTING_9mers",
                                r"viral-dataset-9-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_15_17h12min48s008216ms_60epochs_supervised_Icore_blosum_TESTING",
                                r"viral-dataset-9-blosum-shuffled-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_15_18h49min47s511026ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                r"viral-dataset-9-blosum-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_15_20h25min51s797758ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                r"viral-dataset-9-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_15_22h03min01s768393ms_60epochs_supervised_Icore_onehot_TESTING",
                                r"viral-dataset-9-onehot-shuffled-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_15_23h39min36s436270ms_60epochs_supervised_Icore_onehot_shuffled_TESTING",
                                r"viral-dataset-9-onehot-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_01h16min34s296949ms_60epochs_supervised_Icore_onehot_random_TESTING",
                                r"viral-dataset-9-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_02h52min45s452031ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                r"viral-dataset-9-blosum-shuffled-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_03h59min18s659832ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                r"viral-dataset-9-blosum-random-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_05h07min22s965725ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                r"viral-dataset-9-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_06h14min14s242596ms_60epochs_supervised_Icore_onehot_TESTING_9mers",
                                r"viral-dataset-9-onehot-shuffled-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_07h20min21s295545ms_60epochs_supervised_Icore_onehot_shuffled_TESTING_9mers",
                                r"viral-dataset-9-onehot-random-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset9_2023_07_16_08h27min59s620085ms_60epochs_supervised_Icore_onehot_random_TESTING_9mers",
                                r"viral-dataset-12-blosum-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset12_2023_07_31_10h51min08s664848ms_60epochs_supervised_Icore_blosum",
                                r"viral-dataset-12-onehot-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset12_2023_07_31_11h12min47s101833ms_60epochs_supervised_Icore_onehot",
                                r"viral-dataset-12-blosum-random-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset12_2023_07_31_11h49min37s800692ms_60epochs_supervised_Icore_blosum_9mers",
                                r"viral-dataset-12-onehot-random-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/PLOTS_Vegvisir_viral_dataset12_2023_07_31_11h57min30s948968ms_60epochs_supervised_Icore_onehot_9mers"
                                },
                           "supervised(Icore_non_anchor)":{
                                r"viral-dataset-3-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_15h49min18s185301ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                r"viral-dataset-3-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_17h06min04s335752ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                r"viral-dataset-3-blosum-shuffled-variable-lenght":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_18h21min33s366706ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                r"viral-dataset-3-onehot-shuffled-variable-lenght":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_19h35min48s247364ms_60epochs_supervised_Icore_non_anchor_onehot_shuffled_TESTING",
                                r"viral-dataset-3-blosum-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_20h53min11s784925ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                r"viral-dataset-3-onehot-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_22h09min05s527462ms_60epochs_supervised_Icore_non_anchor_onehot_random_TESTING",
                                r"viral-dataset-3-blosum-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_23h23min26s345013ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                r"viral-dataset-3-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_24_23h50min36s886072ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers",
                                r"viral-dataset-3-blosum-shuffled-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_25_00h17min09s042941ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                r"viral-dataset-3-onehot-shuffled-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_25_00h44min02s537273ms_60epochs_supervised_Icore_non_anchor_onehot_shuffled_TESTING_8mers",
                                r"viral-dataset-3-blosum-random-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_25_01h11min13s239115ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                r"viral-dataset-3-onehot-random-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset3_2023_07_25_01h37min20s146147ms_60epochs_supervised_Icore_non_anchor_onehot_random_TESTING_8mers",
                                r"viral-dataset-9-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_26_22h23min34s704348ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                r"viral-dataset-9-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_26_23h59min52s031839ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                r"viral-dataset-9-blosum-shuffled-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_01h35min33s191421ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                r"viral-dataset-9-onehot-shuffled-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_03h08min34s398130ms_60epochs_supervised_Icore_non_anchor_onehot_shuffled_TESTING",
                                #r"viral-dataset-9-blosum-random-variable-length":"weird",
                                r"viral-dataset-9-onehot-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_06h18min41s252650ms_60epochs_supervised_Icore_non_anchor_onehot_random_TESTING",
                                r"viral-dataset-9-blosum-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_07h51min21s590446ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                r"viral-dataset-9-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_08h22min01s223478ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers",
                                r"viral-dataset-9-blosum-shuffled-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_08h49min57s552823ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                r"viral-dataset-9-onehot-shuffled-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_09h21min36s968218ms_60epochs_supervised_Icore_non_anchor_onehot_shuffled_TESTING_8mers",
                                r"viral-dataset-9-blosum-random-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_27_09h53min55s317100ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                r"viral-dataset-9-onehot-random-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_07_28_04h25min40s585455ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers",

                               },
                                "semisupervised(Icore)":{
                                r"viral-dataset-8-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset8_2023_07_15_14h31min33s611098ms_60epochs_semisupervised_Icore_blosum_TESTING",
                                r"viral-dataset-8-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset8_2023_07_15_16h44min33s943638ms_60epochs_semisupervised_Icore_onehot_TESTING",
                                r"viral-dataset-8-blosum-9mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset8_2023_07_20_23h19min58s119739ms_60epochs_semisupervised_Icore_blosum_TESTING_9mers_5000_unobserved",
                                r"viral-dataset-8-onehot-9mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset8_2023_07_20_22h11min26s891845ms_60epochs_semisupervised_Icore_onehot_TESTING_9mers_5000_unobserved",
                                r"viral-dataset-10-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_15_18h55min27s403782ms_60epochs_semisupervised_Icore_blosum_TESTING",
                                r"viral-dataset-10-blosum-shuffled-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_15_21h21min55s833093ms_60epochs_semisupervised_Icore_blosum_shuffled_TESTING",
                                r"viral-dataset-10-blosum-random-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_15_23h48min38s564252ms_60epochs_semisupervised_Icore_blosum_random_TESTING",
                                r"viral-dataset-10-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_02h15min46s300531ms_60epochs_semisupervised_Icore_onehot_TESTING",
                                r"viral-dataset-10-onehot-shuffled-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_04h43min26s032420ms_60epochs_semisupervised_Icore_onehot_shuffled_TESTING",
                                r"viral-dataset-10-onehot-random-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_07h09min56s516985ms_60epochs_semisupervised_Icore_onehot_random_TESTING",
                                r"viral-dataset-10-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_09h36min19s776058ms_60epochs_semisupervised_Icore_blosum_TESTING_9mers",
                                r"viral-dataset-10-blosum-shuffled-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_10h53min29s452869ms_60epochs_semisupervised_Icore_blosum_shuffled_TESTING_9mers",
                                r"viral-dataset-10-blosum-random-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_12h05min59s638721ms_60epochs_semisupervised_Icore_blosum_random_TESTING_9mers",
                                r"viral-dataset-10-onehot-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_13h18min27s084168ms_60epochs_semisupervised_Icore_onehot_TESTING_9mers",
                                r"viral-dataset-10-onehot-shuffled-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_14h26min13s379085ms_60epochs_semisupervised_Icore_onehot_shuffled_TESTING_9mers",
                                r"viral-dataset-10-onehot-random-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset10_2023_07_16_15h45min22s417997ms_60epochs_semisupervised_Icore_onehot_random_TESTING_9mers",
                                r"viral-dataset-11-blosum-variable-length-5000-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_20_01h18min23s590600ms_60epochs_semisupervised_Icore_blosum_TESTING",
                                r"viral-dataset-11-onehot-variable-length-5000-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_20_03h48min14s678550ms_60epochs_semisupervised_Icore_onehot_TESTING",
                                r"viral-dataset-11-blosum-variable-length-0-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_20_13h18min06s764600ms_60epochs_semisupervised_Icore_blosum_5kfold_0unobserved",
                                r"viral-dataset-11-onehot-variable-length-0-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_20_14h59min21s130148ms_60epochs_semisupervised_Icore_onehot_TESTING_0_unobserved",
                                r"viral-dataset-11-blosum-9mers-0-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_21_04h12min18s186798ms_60epochs_semisupervised_Icore_blosum_TESTING_9mers_0_unobserved",
                                r"viral-dataset-11-onehot-9mers-0-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_21_03h19min10s604693ms_60epochs_semisupervised_Icore_onehot_TESTING_9mers_0_unobserved",
                                r"viral-dataset-11-blosum-9mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_21_06h20min57s241971ms_60epochs_semisupervised_Icore_blosum_TESTING_9mers_5000_unobserved",
                                r"viral-dataset-11-onehot-9mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore/PLOTS_Vegvisir_viral_dataset11_2023_07_21_05h09min53s227910ms_60epochs_semisupervised_Icore_onehot_TESTING_9mers_5000_unobserved",
                                },
                                "semisupervised(Icore_non_anchor)": {
                                    r"viral-dataset-8-blosum-variable-length-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_24_15h53min40s103983ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_5000_unobserved",
                                    r"viral-dataset-8-onehot-variable-length-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_24_18h00min33s334285ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_5000_unobserved",
                                    r"viral-dataset-8-blosum-shuffled-variable-length-5000-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_24_21h32min39s633060ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_TESTING_5000_unobserved",
                                    r"viral-dataset-8-onehot-shuffled-variable-length-5000-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_24_23h39min11s927221ms_60epochs_semisupervised_Icore_non_anchor_onehot_shuffled_TESTING_5000_unobserved",
                                    r"viral-dataset-8-blosum-8mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_24_20h08min17s969033ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-8-onehot-8mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_24_20h50min27s161827ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-8-blosum-shuffled-8mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_25_01h42min00s789533ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-8-onehot-shuffled-8mers-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset8_2023_07_25_02h24min43s633407ms_60epochs_semisupervised_Icore_non_anchor_onehot_shuffled_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-10-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_26_22h32min39s635034ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_5000_unobserved",
                                    r"viral-dataset-10-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_05h52min58s223671ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_5000_unobserved",
                                    r"viral-dataset-10-blosum-shuffled-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_00h58min13s034305ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_TESTING_5000_unobserved",
                                    r"viral-dataset-10-onehot-shuffled-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_28_00h02min01s954443ms_60epochs_semisupervised_Icore_non_anchor_onehot_shuffled_TESTING_5000_unobserved",
                                    r"viral-dataset-10-blosum-random-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_03h25min10s615651ms_60epochs_semisupervised_Icore_non_anchor_blosum_random_TESTING_5000_unobserved",
                                    r"viral-dataset-10-onehot-random-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_28_02h12min33s579051ms_60epochs_semisupervised_Icore_non_anchor_onehot_random_TESTING_5000_unobserved",
                                    r"viral-dataset-10-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_10h48min58s530091ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-10-onehot-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_12h44min58s851618ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-10-blosum-shuffled-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_11h28min01s487237ms_60epochs_semisupervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-10-onehot-shuffled-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_08h21min10s537715ms_60epochs_semisupervised_Icore_non_anchor_onehot_shuffled_TESTING_5000_unobserved",
                                    r"viral-dataset-10-blosum-random-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_12h07min21s483121ms_60epochs_semisupervised_Icore_non_anchor_blosum_random_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-10-onehot-random-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset10_2023_07_27_14h00min31s072893ms_60epochs_semisupervised_Icore_non_anchor_onehot_random_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-11-blosum-variable-length-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_13h16min32s959401ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_5000_unobserved",
                                    r"viral-dataset-11-onehot-variable-length-5000-unobserved": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_15h38min37s432591ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_5000_unobserved",
                                    r"viral-dataset-11-blosum-variable-length-0-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_11h22min15s221766ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_0_unobserved",
                                    r"viral-dataset-11-onehot-variable-length-0-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_26_10h30min56s050759ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_0_unobserved",
                                    r"viral-dataset-11-blosum-8mers-5000-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_19h15min30s709035ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-11-onehot-8mers-5000-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_20h03min49s582137ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_8mers_5000_unobserved",
                                    r"viral-dataset-11-blosum-8mers-0-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_18h01min57s747648ms_60epochs_semisupervised_Icore_non_anchor_blosum_TESTING_8mers_0_unobserved",
                                    r"viral-dataset-11-onehot-8mers-0-unobserved":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Semisupervised/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset11_2023_07_25_18h39min30s140533ms_60epochs_semisupervised_Icore_non_anchor_onehot_TESTING_8mers_0_unobserved",
                                }
    }




    # dict_results_all = {"supervised(Icore)":
    #                         {"vd9-10":"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_03_14h37min35s894838ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_10",
    #                         "vd9-20":"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_03_16h14min44s785096ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_20",
    #                         "vd9-30":"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_03_17h48min26s866987ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_30",
    #                         "vd9-40":"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_03_19h21min17s085055ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_40",
    #                         "vd3-30": "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset3_2023_08_03_21h44min10s036982ms_60epochs_supervised_Icore_blosum_TESTING",
    #                         "vd3-40": "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset3_2023_08_03_23h04min07s833125ms_60epochs_supervised_Icore_blosum_TESTING",
    #                         "vd3-50": "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset3_2023_08_04_00h25min53s635200ms_60epochs_supervised_Icore_blosum_TESTING",
    #                         "vd3-60":"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset3_2023_08_04_01h47min14s179667ms_60epochs_supervised_Icore_blosum_TESTING"
    #                          }
    #
    #                     }




    VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_all,kfolds=5,results_folder = "Benchmark")


    #VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_all,kfolds=5,results_folder="Benchmark")

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
    parser.add_argument('-n', '--num-epochs', type=int, nargs='?', default=60, help='Number of epochs + 1  (number of times that the model is run through the entire dataset (all batches) ')
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
    parser.add_argument('-likelihood-scale', type=int, nargs='?', default=100, help='Scaling the log p( class | Z) of the variational autoencoder (cold posterior)')
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
    parser.add_argument('-pretrained-model', type=str2None, nargs='?', default="None",
                                                                                                help='Load the checkpoints (state_dict and optimizer) from a previous run \n'
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


