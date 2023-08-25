#!/usr/bin/env python3
"""
=======================
2023: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""
import warnings

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
    if args.hpo:
        train_config = "HPO"
    else:
        train_config = "{}epochs".format(args.num_epochs)
    results_dir = "{}/PLOTS_Vegvisir_{}_{}_{}_{}_{}{}".format(script_dir, args.dataset_name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),train_config,args.learning_type,args.sequence_type,suffix)
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



    #Highlight: Likelihood 100
    dict_results_predefined_partitions_100 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_00h21min39s228581ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_02h02min31s779860ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_03h09min53s385938ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_04h47min05s598852ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_05h53min07s553393ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_07h30min17s042073ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_08h35min44s182897ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_10h14min30s578730ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_11h21min04s589546ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_13h00min11s055366ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_13h32min18s725061ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_15h11min03s424849ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_15h45min21s460981ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_17h22min59s537148ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_19h11min15s401671ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_20h49min36s271884ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}




    dict_results_random_stratified_partitions_100 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_00h21min41s081591ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_01h58min21s438713ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_03h03min37s477755ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_04h37min19s658399ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_05h41min19s365333ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_07h13min19s842305ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_08h17min38s989963ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_09h49min01s283379ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_10h53min57s031443ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_12h27min47s743657ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_13h01min00s990819ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_14h34min45s419777ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_15h07min24s308813ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_16h37min00s237374ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_19h11min45s466440ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_20h46min22s152822ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}



    #Highlight: Likelihood 80
    dict_results_predefined_partitions_80 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_16h22min50s455264ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_18h00min16s910930ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_18h59min48s556463ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_20h35min30s186381ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    #"raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_13_22h08min17s858639ms_60epochs_supervised_Icore_blosum_TESTING_lk80",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_21h41min01s940264ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_23h20min57s783599ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_00h27min33s253377ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_02h04min06s395090ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_03h11min14s891078ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_04h47min43s218957ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_05h19min21s010814ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_06h56min53s143680ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_07h30min54s481085ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_09h06min53s142223ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_09h40min41s329685ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_11h15min29s130588ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}

    dict_results_random_stratified_partitions_80 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_01h06min29s238603ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_02h40min26s491870ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_03h45min58s786810ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_05h17min55s585799ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_06h21min58s024047ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_07h58min09s427052ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_09h01min57s241008ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_10h31min10s087815ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_11h37min19s196100ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_13h01min31s378988ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_13h35min14s337920ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_15h07min22s959488ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_15h37min52s131820ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_17h15min35s592585ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_17h46min54s694050ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_19h11min20s045530ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}


    #Highlight: Likelihood 60:

    dict_results_predefined_partitions_60 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_23h23min22s794823ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_01h10min32s340159ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_02h23min08s887505ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_03h46min01s482802ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_04h43min52s657889ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_06h06min06s120049ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_07h15min00s364188ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_08h42min55s171815ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_15h34min52s686599ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_17h55min59s277441ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_18h36min54s466004ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_22h24min27s815811ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_23h00min36s001099ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_17_00h34min23s937002ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_17_01h10min24s412028ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_17_02h38min57s577222ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}

    dict_results_random_stratified_partitions_60 = {"Icore":{
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

    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_60,kfolds=5,results_folder = "Benchmark/Plots",title="predefined_partitions_likelihood_60",overwrite=False)

    # VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_100,kfolds=5,results_folder = "Benchmark/Plots",title="predefined_partitions_likelihood_100",overwrite=False)
    # VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_random_stratified_partitions_100,kfolds=5,results_folder = "Benchmark/Plots",title="random_stratified_partitions_likelihood_100",overwrite=False)
    #
    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_random_stratified_partitions_100,kfolds=5,results_folder="Benchmark/Plots",subtitle="random_stratified_partitions_likelihood_100",overwrite_correlations=False,overwrite_all=False)
    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_predefined_partitions_100,kfolds=5,results_folder="Benchmark/Plots",subtitle="predefined_partitions_likelihood_100",overwrite_correlations=False,overwrite_all=False)


    # VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_80,kfolds=5,results_folder = "Benchmark/Plots",title="predefined_partitions_likelihood_80",overwrite=False)
    # VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_random_stratified_partitions_80,kfolds=5,results_folder = "Benchmark/Plots",title="random_stratified_partitions_likelihood_80",overwrite=False)
    #
    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_predefined_partitions_80,kfolds=5,results_folder="Benchmark/Plots",subtitle="predefined_partitions_likelihood_80",overwrite_correlations=False,overwrite_all=False)
    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_random_stratified_partitions_80,kfolds=5,results_folder="Benchmark/Plots",subtitle="random_stratified_partitions_likelihood_80",overwrite_correlations=False,overwrite_all=False)
    #



    dict_results_benchmark= { "Icore" :{
        "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_04h43min52s657889ms_60epochs_supervised_Icore_blosum_TESTING"
    }
    }


    #VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,folder="Benchmark/Plots")


    VegvisirPlots.plot_model_stressing_comparison(dict_results_predefined_partitions_60,script_dir,folder="Benchmark/Plots")


    exit()


def hierarchical_clustering():

    vegvisir_folder = "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_21h41min01s940264ms_60epochs_supervised_Icore_blosum_TESTING"

    embedded_epitopes = "{}/vegvisir/src/vegvisir/data/viral_dataset9/similarities/Icore/All/diff_allele/diff_len/neighbours1/all/EMBEDDED_epitopes.tsv".format(script_dir)

    VegvisirPlots.plot_hierarchical_clustering(vegvisir_folder, embedded_epitopes,folder="/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Plots")

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
                             'viral_dataset7: Supervised learning. Same dataset as viral_dataset3, but the test dataset is mixed with the train and validation datasets \n'
                             'viral_dataset8: Semisupervised learning. Same dataset as viral_dataset6 (containing unobserved datapoints), where the original test dataset is left out from the training (not mixed in).If args.test = True, then the (old) assigned test is used \n'
                             'viral_dataset9: Supervised learning. Same as viral_dataset7 with a new test dataset (OLD test,train and validation are mixed). New test dataset available when using args.test=True \n'
                             'viral_dataset10: Semisupervised learning. Same as viral_dataset6 (containing unobserved datapoints) with a new test dataset (OLD test,train and validation are mixed). New test available when using args.test=True \n'
                             'viral_dataset11: Semisupervised learning.Similar to viral_dataset6 (containing unobserved datapoints), but the (old) test is incorporated as an unobserved sequence as well. (old) test available when using args.test=True \n'
                             'viral_dataset12: Prediction.Training only on the unobserved data points with randomly assigned labels. MHC binders without binary targets'
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
    parser.add_argument('-num-unobserved', type=int, nargs='?', default=5000, help='Use with datasets for semi supervised training: <viral_dataset6> or <viral_dataset8>. \n'
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

    parser.add_argument('-train', type=str2bool, nargs='?', default=True ,help='<True> Run the model \n <False> Make models comparison or load previous model if pargs.pretrained_model is not None ')
    parser.add_argument('-validate', type=str2bool, nargs='?', default=True, help='Evaluate the model on the validation dataset')
    parser.add_argument('-test', type=str2bool, nargs='?', default=True, help='Evaluate the model on the external test dataset')
    parser.add_argument('-hpo', type=str2bool, nargs='?', default=False, help='Hyperparameter optimization with Ray Tune')
    parser.add_argument('-generate', type=str2bool, nargs='?', default=True, help='<True> Generate new neo-epitopes labelled and with a confidence score'
                                                                                   '<False> Do nothing')
    best_config = {0:"/home/lys/Dropbox/PostDoc/vegvisir/BEST_hyperparameter_dict.p",1:None}
    parser.add_argument('-config-dict', nargs='?', default=best_config[1],type=str2None, help='Path to optimized hyperparameter dict')
    unobserved_sequences = {0:"/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/custom_dataset/unobserved_grouped_alleles_train.tsv",1:None}
    parser.add_argument('-train-path', type=str2None, nargs='?', default=unobserved_sequences[1],help="Path to training dataset. Use only for training. ")
    parser.add_argument('-test-path', type=str2None, nargs='?', default= "", help='Path to sequences to predict')
    parser.add_argument('-predefined-partitions', type=str2bool, nargs='?', default= True, help='<True> Divides the dataset into train, validation and test according to pre-specified partitions (in the sequences file, use a column named partitions)'
                                                                                                '<False> Performs a random stratified train, validation and test split')


    parser.add_argument('-plot-all','--plot-all', type=str2bool, nargs='?', default=False, help='True: Plots all UMAPs and other computationally expensive plots. Do not use when args.k_folds > 1, it saturates the CPU & GPU memory'
                                                                                                'False: Only plots the computationally inexpensive ROC curves')

    parser.add_argument('-aa-types', type=int, nargs='?', default=20, help='Define the number of unique amino acid types. It determines the blosum matrix to be used. \n'
                                                                           ' If the sequence contains gaps, the script will use 20 aa + 1 gap character. The script automatically corrects issues ')
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
    parser.add_argument('-likelihood-scale', type=int, nargs='?', default=50, help='Scaling the log p( class | Z) of the variational autoencoder (cold posterior)'
                                                                                   '100: Assign likelihood to batch size')
    parser.add_argument('-hidden-dim', type=int, nargs='?', default=40, help='Dimensions of fully connected networks')
    parser.add_argument('-embedding-dim', type=int, nargs='?', default=40, help='Embedding dimensions, use with self-attention. NOT USED---> DELETE SOOn')
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

    pretrained_model = {0:"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_14_23h17min39s118041ms_60epochs_supervised_Icore_blosum_TESTING_pretty_plots",1:None}
    parser.add_argument('-pretrained-model', type=str2None, nargs='?', default="{}".format(pretrained_model[1]),help='Load the checkpoints (state_dict and optimizer) from a previous run \n'
                                                                                                '<None>: Trains model from scratch \n'
                                                                                                '<str path>: Loads pre-trained model from given path \n')

    args = parser.parse_args()

    if args.use_cuda:
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            #cuda_device = "cuda:{}".format(os.environ["CUDA_VISIBLE_DEVICES"]) if args.hpo else "cuda"
            cuda_device = "cuda"
            parser.add_argument('--device',type=str,default="{}".format(cuda_device) ,nargs='?', help='Device choice (cpu, cuda:0, cuda:1, ...), behaviour linked to use_cuda')
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
        args_dict["num_epochs"] = 0
        args_dict["plot_all"] = args.plot_all
        if args_dict["dataset_name"] == args.dataset_name:
            warnings.warn("You selected using the same dataset as in the pretrained model. Overriding your current args (from argparser) to load the ones in {}".format(args.pretrained_model))
            args = Namespace(**args_dict)
        else:
            args_dict["dataset_name"] = args.dataset_name
            args_dict["learning_type"] = args.learning_type
            args_dict["sequence_type"] = args.sequence_type
            args_dict["num_epochs"] = 0
            warnings.warn("Overriding your current args except <learning_type>,<dataset_name>,<sequence_type>")
            args = Namespace(**args_dict)
    if args.train:
        main()
    else:
        analysis_models()
        #hierarchical_clustering()


