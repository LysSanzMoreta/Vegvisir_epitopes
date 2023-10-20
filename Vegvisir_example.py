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
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20000" #Not useful to prevent memory crashes :(
print("Loading Vegvisir module from {}".format(vegvisir.__file__))
now = datetime.datetime.now()

def define_suffix(args):
    kmers = "_{}mers".format("9" if args.sequence_type == "Icore" else "8") if args.filter_kmers else ""
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
    VegvisirUtils.folders(ntpath.basename(results_dir), script_dir)
    if args.k_folds > 1:
        for kfold in range(args.k_folds):
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Train_fold_{}".format(kfold)), script_dir) #TODO: 2 folders for train
            if args.test:
                VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Test_fold_{}".format(kfold)), script_dir)
                if args.validate:
                    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Valid_fold_{}".format(kfold)),script_dir)
            else:
                VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Valid_fold_{}".format(kfold)),script_dir)
        if args.generate:
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Generated"), script_dir)
        if args.immunomodulate:
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Immunomodulated"), script_dir)
    else:
        VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Train"), script_dir)
        VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Valid"), script_dir)
        VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Test"), script_dir)
        if args.generate:
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Generated"), script_dir)
        if args.immunomodulate:
            VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir), "Immunomodulated"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Vegvisir_checkpoints"), script_dir)
    VegvisirUtils.folders("{}/{}".format(ntpath.basename(results_dir),"Scripts"), script_dir)

    vegvisir_dataset = vegvisir.select_dataset(args.dataset_name, script_dir,args,results_dir, update=False)

    json.dump(args.__dict__, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)

    vegvisir.run(vegvisir_dataset,results_dir,args)


def analysis_models():
    """Analyses the results of all possible model combinations (stress testing)"""

    #Highlight: Likelihood HPO


    dict_results_predefined_partitions_onehot = {"Icore":{
                                                    "random-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_29_22h43min33s935024ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_00h02min34s769003ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_00h56min37s868085ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_02h13min16s897762ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_03h07min36s214823ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_04h26min47s161308ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_30_05h20min45s177235ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-onehot-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_30_06h37min30s804204ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_30_07h05min42s608732ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-onehot-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_30_08h20min28s072463ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_30_08h49min11s108414ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_30_10h04min22s042493ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers"
                                         }}

    dict_results_random_stratified_partitions_onehot = {"Icore":{
                                                        "random-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_19h24min13s555928ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                        "random-onehot-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_20h34min04s389783ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                        "shuffled-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_21h26min34s973459ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                        "shuffled-onehot-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_22h34min59s255207ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                        "raw-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_23h21min22s981405ms_60epochs_supervised_Icore_blosum_TESTING",
                                                        "raw-onehot-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_31_00h33min58s021227ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_31_01h20min23s474294ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-onehot-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_31_02h31min58s722045ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-onehot-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_31_02h57min21s149063ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-onehot-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_31_04h07min10s887834ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_31_04h30min30s086587ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_31_05h37min53s060750ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers"
                                         }}

    dict_results_predefined_partitions_blosum = {"Icore": {
        "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_14h40min14s079250ms_60epochs_supervised_Icore_blosum_random_TESTING",
        "random-blosum-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_17h15min50s152080ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
        "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_19h03min53s442087ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
        "shuffled-blosum-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_21h42min48s429149ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
        "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_23h31min43s312787ms_60epochs_supervised_Icore_blosum_TESTING",
        "raw-blosum-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_02h11min08s246199ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
    },
        "Icore_non_anchor": {
            "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_04h00min58s336351ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
            "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_06h38min18s627771ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
            "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_07h35min12s832669ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
            "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_10h15min43s438283ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
            "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_11h10min46s716817ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
            "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_10_01_13h49min37s945624ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers"
        }}

    dict_results_random_stratified_partitions_blosum = {"Icore": {
        "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_01_22h13min17s282581ms_60epochs_supervised_Icore_blosum_random_TESTING",
        "random-blosum-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_00h38min57s749396ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
        "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_02h22min01s193852ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
        "shuffled-blosum-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_04h45min19s516444ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
        "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_06h29min45s990464ms_60epochs_supervised_Icore_blosum_TESTING",
        "raw-blosum-9mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_08h54min51s012247ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
    },
        "Icore_non_anchor": {
            "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_10h38min29s855945ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
            "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_13h00min22s455965ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
            "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_13h50min42s039778ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
            "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_16h15min53s223912ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
            "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_17h06min15s507265ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
            "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Random_stratified/PLOTS_Vegvisir_viral_dataset9_2023_10_02_19h31min38s918485ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers"
        }}



    #Highlight: K-fold comparisons

    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_predefined_partitions_blosum,kfolds=5,results_folder = "Benchmark/Plots",title="predefined_partitions_HPO_blosum",overwrite=True)
    #VegvisirPlots.plot_kfold_comparisons(args,script_dir,dict_results_random_stratified_partitions_blosum,kfolds=5,results_folder = "Benchmark/Plots",title="random_stratified_partitions_HPO_blosum",overwrite=False)

    # VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_predefined_partitions_blosum,kfolds=5,results_folder="Benchmark/Plots",subtitle="predefined_partitions_HPO_blosum",overwrite_correlations=False,overwrite_all=False)
    #VegvisirPlots.plot_kfold_latent_correlations(args,script_dir,dict_results_random_stratified_partitions_blosum,kfolds=5,results_folder="Benchmark/Plots",subtitle="random_stratified_partitions_HPO_blosum",overwrite_correlations=False,overwrite_all=False)


    dict_results_benchmark= { "Icore" :{
        #"raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_03h07min36s214823ms_60epochs_supervised_Icore_blosum_TESTING",
        "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_23h31min43s312787ms_60epochs_supervised_Icore_blosum_TESTING",
    }}

    #Highlight: Benchmarking
    VegvisirPlots.plot_benchmarking_results(dict_results_benchmark,script_dir,keyname="raw-blosum-variable-length",folder="Benchmark/Plots",title="HPO_blosum_NEW")


    #Highlight: Model stress comparison
    #VegvisirPlots.plot_model_stressing_comparison(dict_results_predefined_partitions_blosum,script_dir,folder="Benchmark/Plots",encoding="blosum",title="HPO_blosum")
    #VegvisirPlots.plot_model_stressing_comparison(dict_results_predefined_partitions_onehot,script_dir,folder="Benchmark/Plots",encoding="onehot",title="HPO_onehot")

    exit()


def hierarchical_clustering():

    vegvisir_folder_original = "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_21h41min01s940264ms_60epochs_supervised_Icore_blosum_TESTING"
    vegvisir_folder_HPO_onehot = "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_onehot/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_30_03h07min36s214823ms_60epochs_supervised_Icore_blosum_TESTING"
    vegvisir_folder_HPO_blosum = "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/HPO_blosum/Predefined_partitions/PLOTS_Vegvisir_viral_dataset9_2023_09_30_23h31min43s312787ms_60epochs_supervised_Icore_blosum_TESTING"
    vegvisir_folder_HPO_blosum_z8 = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_10_18_11h02min32s539316ms_60epochs_supervised_Icore_blosum_TESTING"
    embedded_epitopes = "{}/vegvisir/src/vegvisir/data/viral_dataset9/similarities/Icore/All/diff_allele/diff_len/neighbours1/all/EMBEDDED_epitopes.tsv".format(script_dir)
    vegvisir_folder_HPO_blosum_z34 = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_10_18_15h30min25s524718ms_60epochs_supervised_Icore_blosum_TESTING"
    VegvisirPlots.plot_hierarchical_clustering(vegvisir_folder_HPO_blosum, embedded_epitopes,folder="/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Plots",title="HPO_blosum_z8")
    #VegvisirPlots.plot_hierarchical_clustering(vegvisir_folder, embedded_epitopes,folder="/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Plots",title="blosum")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    parser.add_argument('-name','--dataset-name', type=str, nargs='?',
                        default="viral_dataset9",
                        help='Dataset project name, look at vegvisir.available_datasets(). The data should be always located at vegvisir/src/vegvisir/data \n'
                             'custom_dataset: Perform training or prediction (by setting the args.pretrained_model to the folder path with the model checkpoints). Remember to define also train_path, test_path'
                             'viral_dataset3 : Supervised learning. Only sequences, partitioned into train,validation and (old) test.If args.test = True, then the (old) assigned test is used \n'
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
    #Highlight: Dataset configurations: Use with the default datasets (not custom ones)
    parser.add_argument('-predefined-partitions', type=str2bool, nargs='?', default= True, help='<True> Divides the dataset into train, validation and test according to pre-specified partitions (in the sequences file, use a column named partitions)'
                                                                                                '<False> Performs a random stratified train, validation and test split')
    parser.add_argument('-num-unobserved', type=int, nargs='?', default=5000, help='Use with datasets for semi supervised training: <viral_dataset6> or <viral_dataset8>. \n'
                                                                                   'It establishes the number of unobserved(unlabelled) datapoints to use')
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



    parser.add_argument('-subset-data', type=str, default="no",
                        help="Pick only the first <n> datapoints (epitopes) for testing the pipeline\n"
                             "<no>: Keep all \n"
                             "<insert_number>: Keep first <n> data points") #TODO: Remove
    parser.add_argument('--run-nnalign', type=bool, nargs='?', default=False, help='Executes NNAlign 2.1 as in https://services.healthtech.dtu.dk/service.php?NNAlign-2.1') #TODO: Remove

    #Highlight: Model hyperparameters, do not change unless you re-train the model
    parser.add_argument('-n', '--num-epochs', type=int, nargs='?', default=2, help='Number of epochs + 1  (number of times that the model is run through the entire dataset (all batches) ')
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
    parser.add_argument('-optimizer_name', type=str, nargs='?', default="Adam", help='Gradient optimizer name \n '
                                                                                     '<ClippedAdam>'
                                                                                     '<Adam>')
    parser.add_argument('-loss-func', type=str, nargs='?', default="elbo", help="Error loss function to be optimized, options are: \n"
                                                                                         "<bcelogits>: Binary Cross Entropy with logits (no activation in last layer) \n "
                                                                                         "<bceprobs>: Binary Cross Entropy with probabilities (sigmoid activation)\n"
                                                                                         "<weighted_bce>: Weighted Binary Cross Entropy \n"
                                                                                         "<ae_loss>: Uses a reconstruction and a classification error loss \n"
                                                                                         "<softloss>: Label smoothing + Taylorsoftmax \n "
                                                                                         "<elbo>: Evidence Lower Bound objective used for SVI") #TODO: Remove
    parser.add_argument('-clip-gradients', type=bool, nargs='?', default=True, help='Computes the 2D Euclidean norm of the gradient to normalize the gradient by that value and \n '
                                                                                    ' prevent exploding gradients (small gradients that lead to abscence of training) ')

    parser.add_argument('-guide', type=str, nargs='?', default="custom", help='<custom>: See guides.py \n'
                                                                              '<autodelta> : Automatic guide for amortized inference '
                                                                              'in Pyro see pyro.autoguides. Does not work with mini-batching,'
                                                                              ' (perhaps subsampling in the plate)') #TODO: Remove

    parser.add_argument('-z-dim','--z-dim', type=int, nargs='?', default=5, help='Latent space dimension')
    parser.add_argument('-likelihood-scale', type=int, nargs='?', default=50, help='Scaling the log p( class | Z) of the variational autoencoder (cold posterior)')
    parser.add_argument('-hidden-dim', type=int, nargs='?', default=40, help='Dimensions of fully connected networks')
    parser.add_argument('-embedding-dim', type=int, nargs='?', default=40, help='Embedding dimensions, use with self-attention. NOT USED---> DELETE SOOn')
    parser.add_argument('-lt','--learning-type', type=str, nargs='?', default="supervised", help='<supervised_no_decoder> simpler model architecture with an encoder and a classifier'
                                                                                                 '<unsupervised> Unsupervised learning. No classification is performed \n'
                                                                                                 '<semisupervised> Semi-supervised model/learning. The likelihood of the class (p(c | z)) is only computed and maximized using the most confident scores. \n '
                                                                                                            'The non confident data points are inferred by the guide \n'
                                                                                                 '<supervised> Supervised model. All target observations are used to compute the likelihood of the class given the latent representation')

    parser.add_argument('-glitch','--glitch', type=str2bool, nargs='?', default=False, help='NOT USED at the moment, does not seem necessary. Only works with blosum encodings'
                                                                                           '<True>: Applies a random noise distortion (via rotations) to the encoded vector within the conserved positions of the sequences (mainly anchor points)  \n'
                                                                                           '<False>: The blosum encodings are left untouched')
    parser.add_argument('-num-samples','-num_samples', type=int, nargs='?', default=3, help='Number of samples from the posterior predictive. Only makes sense when using amortized inference with a guide function')


    parser.add_argument('-hpo', type=str2bool, nargs='?', default=False,help='Hyperparameter optimization with Ray Tune')
    best_config = {0: "{}/BEST_hyperparameter_dict_onehot.p".format(script_dir),
                   1: "{}/BEST_hyperparameter_dict_blosum.p".format(script_dir),
                   2: None}
    parser.add_argument('-config-dict', nargs='?', default=best_config[1], type=str2None,help='Path to optimized hyperparameter dict')

    #Highlight: Evaluation modes
    parser.add_argument('-train', type=str2bool, nargs='?', default=True,help='<True> Run the model \n <False> Make models comparison or load previous model if pargs.pretrained_model is not None ')
    parser.add_argument('-validate', type=str2bool, nargs='?', default=False, help='Evaluate the model on the validation dataset. Only needed for model design')
    parser.add_argument('-test', type=str2bool, nargs='?', default=True, help='Evaluate the model on the external test dataset')

    #Highlight: Generating new sequences from pre-trained model
    parser.add_argument('-generate', type=str2bool, nargs='?', default=False, help='<True> Generate new neo-epitopes labelled and with a confidence score based on the training dataset. Please use args.validate False''<False> Do nothing')
    parser.add_argument('-num-synthetic-peptides', type=int, nargs='?', default=30, help='<True> Generate new neo-epitopes labelled and with a confidence score. Please use args.validate False''<False> Do nothing')
    parser.add_argument('-generate-num-samples', type=int, nargs='?', default=50, help='If args.generate == True, then per generated sequence, produce n samples to calculate a class probability')
    parser.add_argument('-generate-argmax', type=str2bool, nargs='?', default=False, help='If args.generate == True, ... DO NOT USE')

    #Highlight: immunomodulating a sequence
    immunomodulate_path = {0:"{}/immunomodulate_sequences.txt".format(script_dir),
                           1:None}
    parser.add_argument('-immunomodulate', type=str2bool, nargs='?', default=True, help='<True> Predict latent representation for the given sequences and generate new neo-epitopes labelled and with a confidence score based only on the input sequences via args.immunomodulate_path. Please use args.validate False''<False> Do nothing')
    parser.add_argument('-immunomodulate-path', type=str2None, nargs='?', default= immunomodulate_path[0], help='Path to text file containing sequences to change their immunogenicity')



    #Highlight: Re-training the model
    unobserved_sequences = {0:"/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/custom_dataset/unobserved_grouped_alleles_train.tsv",1:None}
    parser.add_argument('-train-path', type=str2None, nargs='?', default=unobserved_sequences[0],help="Path to training dataset. Use only for training. ")
    parser.add_argument('-test-path', type=str2None, nargs='?', default= "", help='Path to sequences to predict')



    #Highlight: Output saving settings
    parser.add_argument('-save-all', type=str2bool, nargs='?', default=False, help='<True> Saves every matrix output from the model'
                                                                                   '<False> Only saves a selection of model outputs necessary for benchmarking')
    parser.add_argument('-plot-all','--plot-all', type=str2bool, nargs='?', default=True, help='True: Plots all UMAPs and other computationally expensive plots. Do not use when args.k_folds > 1,\n'
                                                                                               ' it saturates the CPU & GPU memory'
                                                                                                'False: Only plots the computationally inexpensive ROC curves')

    #Highlight: Use this one if you do not want to train the model, just predict, generate or immunomodulate
    #pretrained_model = {0:"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_14_23h17min39s118041ms_60epochs_supervised_Icore_blosum_TESTING_pretty_plots",1:None}
    pretrained_model = {0:"/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_09_12_11h07min45s598775ms_60epochs_supervised_Icore_blosum_TESTING",
                        1:None}

    parser.add_argument('-pretrained-model', type=str2None, nargs='?', default="{}".format(pretrained_model[1]),help='Load the checkpoints (state_dict and optimizer) from a previous run \n'
                                                                                                '<None>: Trains model from scratch \n'
                                                                                                '<str path>: Loads pre-trained model from given path \n')

    #Highlight: DO NOT CHANGE ANYTHING ELSE DOWN HERE
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
            warnings.warn("Overriding your current args except <learning_type>,<dataset_name>,<sequence_type>,<num_obs_classes>,<num_classes>")
            args = Namespace(**args_dict)
    if args.train:
        main()
    else:
        analysis_models()
        #hierarchical_clustering()


