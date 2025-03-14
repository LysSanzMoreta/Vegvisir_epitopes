#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""
import gc
import json
import operator
import pickle
import subprocess
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Union
import dill
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms

import  matplotlib
#matplotlib.rc('text', usetex=True)
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import umap # numpy < 1.23
import vegvisir.utils as VegvisirUtils
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score, recall_score,precision_score
from sklearn.cluster import MiniBatchKMeans,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score
from joblib import Parallel, delayed
import multiprocessing
import os
from scipy import stats
#import vegvisir.similarities as VegvisirSimilarities
from collections import namedtuple
import dataframe_image as dfi
import statsmodels.api as sm
import logomaker
import dromi

MAX_WORKERs = ( multiprocessing. cpu_count() - 1 )
plt.style.use('ggplot')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

colors_dict = {0: "green", 1: "red",2:"navajowhite"}
colors_cluster_dict = {0: "seagreen", 1: "crimson",2:"gold",3:"mediumslateblue"}
colors_dict_labels = {0: "mediumaquamarine", 1: "orangered",2:"navajowhite"}
colors_dict_labels_cmap = {key:sns.light_palette("{}".format(val), as_cmap=True)for key,val in colors_dict_labels.items()}

labels_dict = {0:"negative",1:"positive",2:"unobserved"}
colors_list_aa = ["black", "plum", "lime", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange",
               "yellow", "green",
               "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal",
               "goldenrod", "chocolate", "cornflowerblue", "pink", "darkgrey", "indianred",
               "mediumspringgreen"]

PlotSettings = namedtuple("PlotSettings",["values","colormap_unique","colors_feature","unique_values"])

def plot_generated_labels_histogram(dataframe:pd.DataFrame,results_dir:str):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(11, 10))

    plt.hist(dataframe["Positive_score"],color=colors_dict[1],bins=10)
    plt.hist(dataframe["Negative_score"],color=colors_dict[0],bins=10)
    plt.title("Target distribution score among generated sequences")

    plt.savefig("{}/Histogram_target_probabilities".format(results_dir), dpi=500)
    plt.clf()
    plt.close(fig)

def plot_data_information(data:np.ndarray, filters_dict:dict, storage_folder:str, args:namedtuple, name_suffix:str):
    """"""
    #ndata = data.shape[0]
    fig, ax = plt.subplots(nrows=3,ncols=4, figsize=(11, 10))
    num_bins = 50

    ############LABELS #############
    unique,counts = np.unique(data["target"].to_numpy(),return_counts=True)

    patches_list = []
    position = 0
    position_labels = []
    for label,count in zip(unique,counts):
        position_labels.append(position)
        bar = ax[0][0].bar(position,count, label=label, color=colors_dict[label], width=0.1, edgecolor='white')
        patches_list.append(bar.patches[0])
        position += 0.5
    ax[0][0].set_xlabel('Target/Label (0: Non-binder, \n 1: Binder)')
    ax[0][0].set_title('Histogram of targets/labels \n',fontsize=10)
    ax[0][0].xaxis.set_ticks(position_labels)
    ax[0][0].set_xticklabels(range(args.num_classes))
    # Annotate the bars.
    n_data = sum(counts)
    for bar in patches_list:
        ax[0][0].annotate(
            "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data), 2),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 8),
            textcoords='offset points')

    ############LABELS CORRECTED #############
    #freq, bins, patches = ax[0][1].hist(data["target_corrected"].to_numpy(), bins=args.num_classes, density=True, edgecolor='white')

    unique,counts = np.unique(data["target_corrected"].to_numpy(),return_counts=True)
    patches_list = []
    position = 0
    position_labels = []
    for label,count in zip(unique,counts):
        position_labels.append(position)
        bar = ax[0][1].bar(position,count, label=label, color=colors_dict[label], width=0.1, edgecolor='white')
        patches_list.append(bar.patches[0])
        position += 0.5

    ax[0][1].set_xlabel('Target/Label (0: Non-binder, \n 1: Binder)')
    ax[0][1].set_title('Histogram of re-assigned \n targets/labels',fontsize=10)
    ax[0][1].xaxis.set_ticks(position_labels)
    ax[0][1].set_xticklabels(range(args.num_classes))
    # Annotate the bars.
    n_data = sum(counts)

    for bar in patches_list:
        ax[0][1].annotate(
            "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data), 2),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 8),
            textcoords='offset points')
    ####### Immunodominance scores ###################
    ax[1][0].hist(data["immunodominance_score_scaled"].to_numpy(), num_bins, density=True)
    ax[1][0].set_xlabel('Minmax scaled \n immunodominance score \n (N_+ / Total Nsubjects)',fontsize=10)
    ax[1][0].set_title('Histogram of \n immunodominance scores',fontsize=10)
    ######## Sequence length distribution ####################
    ######Train#############################3
    data_lens_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_unobserved = data.loc[(data["training"] == True) & (data["target_corrected"] == 2.), filters_dict["filter_kmers"][2]].str.len()

    dict_counts = {0: data_lens_train_negative.value_counts(), 1: data_lens_train_positive.value_counts(),2:data_lens_train_unobserved.value_counts()}
    all_lens = []
    for key,vals in dict_counts.items():
        all_lens += list(vals.keys())
    unique_sorted = sorted(set(all_lens))[::-1]
    position = 0
    positions = []
    for seq_len in unique_sorted:
        positions.append(position + 0.2)
        for label,val_counts in dict_counts.items():
            if seq_len in val_counts.keys():
                ax[2][0].bar(position, val_counts[seq_len], label=seq_len, color=colors_dict[label], width=0.1, edgecolor='white')
                position += 0.2

    ax[2][0].xaxis.set_ticks(positions)
    ax[2][0].set_xticklabels(unique_sorted)
    ax[2][0].set_title("Sequence length distribution of \n  the Train-valid dataset",fontsize=10)
    ###### Test #####################
    data_lens_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_unobserved = data.loc[(data["training"] == False) & (data["target_corrected"] == 2.), filters_dict["filter_kmers"][2]].str.len()

    dict_counts = {0: data_lens_test_negative.value_counts(), 1: data_lens_test_positive.value_counts(),2:data_lens_test_unobserved.value_counts()}
    all_lens = []
    for key,vals in dict_counts.items():
        all_lens += list(vals.keys())
    unique_sorted = sorted(set(all_lens))[::-1]
    position = 0
    positions = []
    for seq_len in unique_sorted:
        positions.append(position + 0.2)
        for label,val_counts in dict_counts.items():
            if seq_len in val_counts.keys():
                ax[2][1].bar(position, val_counts[seq_len], label=seq_len, color=colors_dict[label], width=0.1, edgecolor='white')
                position += 0.2
    ax[2][1].xaxis.set_ticks(positions)
    ax[2][1].set_xticklabels(unique_sorted)
    ax[2][1].set_title("Sequence length distribution of \n  the Test dataset",fontsize=10)
    ############TEST PROPORTIONS #############
    data_partitions = data[["partition", "training", "target_corrected"]]
    test_counts = data_partitions[data_partitions["training"] == False].value_counts("target_corrected")  # returns a dict
    test_counts = dict(sorted(test_counts.items(), key=operator.itemgetter(1), reverse=True))  # sort dict by values
    if test_counts:
        bar_list= []
        for label,counts in test_counts.items():
            bar = ax[1][1].bar(0, counts, label=labels_dict[label], color=colors_dict[label], width=0.1, edgecolor='white')
            bar = bar.patches[0]
            bar_list.append(bar)
        n_data_test = sum([val for val in test_counts.values()])
        for bar in bar_list:
            ax[1][1].annotate("{}({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data_test), 2),
                              (bar.get_x() + bar.get_width() / 2,
                               bar.get_height()), ha='center', va='center',
                              size=12, xytext=(0, 8),
                              textcoords='offset points')
        ax[1][1].xaxis.set_ticks([0])
        ax[1][1].set_xticklabels(["Test proportions"], fontsize=10)
        ax[1][1].set_title('Test dataset \n +/- proportions', fontsize=10)
    #################################################################
    ################ TRAIN PARTITION PROPORTIONS###################################

    train_set = data_partitions[data_partitions["training"] == True]

    partitions_groups = [train_set.groupby('partition').get_group(x) for x in train_set.groupby('partition').groups]
    i = 0
    partitions_names = []
    for group in partitions_groups:
        name = group["partition"].iloc[0]
        group_counts = group.value_counts("target_corrected")  # returns a dict
        group_counts = dict( sorted(group_counts.items(), key=operator.itemgetter(1),reverse=True)) #sort dict by value counts
        if group_counts:
            bar_list = []
            for label,count in group_counts.items():
                bar = ax[0][2].bar(i, count, label=labels_dict[int(label)], color=colors_dict[int(label)], width=0.1, edgecolor='white')
                bar = bar.patches[0]
                bar_list.append(bar)

            i += 0.4
            n_data_partition = sum([val for key, val in group_counts.items()])
            for bar in bar_list:
                ax[0][2].annotate(
                    "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data_partition), 2),
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height()), ha='center', va='center',
                    size=8, xytext=(0, 8),
                    textcoords='offset points')

        partitions_names.append(name)
    ax[0][2].xaxis.set_ticks([0, 0.4, 0.8, 1.2, 1.6])
    ax[0][2].set_xticklabels(["Part. {}".format(int(i)) for i in partitions_names])
    ax[0][2].set_title('TrainEval dataset \n +/- proportions per partition',fontsize=10)
    ################### ALLELES PROPORTIONS PER CLASS ##############################################
    if not filters_dict["group_alleles"][0]:
        #Train
        data_alleles_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), "allele"].value_counts().to_dict()
        data_alleles_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), "allele"].value_counts().to_dict()

        dict_counts = {0: data_alleles_train_negative, 1: data_alleles_train_positive}
        longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
        position = 0
        positions = []
        for val_i, count_i in dict_counts[longest].items():
            ax[1][2].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.4)
            if val_i in dict_counts[shortest].keys():
                count_j = dict_counts[shortest][val_i]
                ax[1][2].bar(position + 0.45, count_j, label=shortest, color=colors_dict[shortest], width=0.4)
            positions.append(position + 0.1)
            position += 1.1
        ax[1][2].xaxis.set_ticks(positions)
        ax[1][2].set_xticklabels(dict_counts[longest].keys(),rotation=90,fontsize=2)
        ax[1][2].xaxis.set_tick_params(width=0.1)
        ax[1][2].set_title("Allele distribution of \n  the Train-valid dataset")
        #Test
        data_alleles_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), "allele"].value_counts().to_dict()
        data_alleles_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), "allele"].value_counts().to_dict()
        dict_counts = {0: data_alleles_test_negative, 1: data_alleles_test_positive}
        longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
        position = 0
        positions = []
        for val_i, count_i in dict_counts[longest].items():
            ax[2][2].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.4)
            if val_i in dict_counts[shortest].keys():
                count_j = dict_counts[shortest][val_i]
                ax[2][2].bar(position + 0.45, count_j, label=shortest, color=colors_dict[shortest], width=0.4)
            positions.append(position + 0.1)
            position += 1.1
        ax[2][2].xaxis.set_ticks(positions)
        ax[2][2].set_xticklabels(dict_counts[longest].keys(),rotation=90,fontsize=2)
        ax[2][2].xaxis.set_tick_params(width=0.1)
        ax[2][2].set_title("Allele distribution of \n  the Test dataset",fontsize=10)


    ax[0][3].axis("off")
    ax[1][3].axis("off")
    #ax[2][2].axis("off")
    ax[1][2].axis("off")
    ax[2][3].axis("off")

    legends = [mpatches.Patch(color=color, label='Class {}'.format(label)) for label, color in colors_dict.items()]
    fig.legend(handles=legends, prop={'size': 12}, loc='center right', bbox_to_anchor=(0.9, 0.5))
    fig.tight_layout(pad=0.1)
    plt.savefig("{}/{}/Viruses_histograms_{}".format(storage_folder, args.dataset_name, name_suffix), dpi=500)
    plt.clf()
    plt.close(fig)

def plot_data_information_reduced1(data:pd.DataFrame, filters_dict:dict, storage_folder:str, args:namedtuple, name_suffix:str):
    """"""
    ndata = data.shape[0]
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15, 10))
    fontsize = 20
    num_bins = 50

    ############LABELS #############
    unique,counts = np.unique(data["target"].to_numpy(),return_counts=True)

    patches_list = []
    position = 0
    position_labels = []
    for label,count in zip(unique,counts):
        position_labels.append(position)
        bar = ax[0][0].bar(position,count, label=label, color=colors_dict[label], width=0.1, edgecolor='white')
        patches_list.append(bar.patches[0])
        position += 0.5 if args.num_classes == 3 else 0.25
    ax[0][0].margins(y=0.1)
    ax[0][0].set_xlabel('Binary target class')
    ax[0][0].set_title('Binary targets \n counts',fontsize=fontsize)
    ax[0][0].xaxis.set_ticks(position_labels)
    ax[0][0].set_xticklabels(range(args.num_classes))
    # Annotate the bars.
    n_data = sum(counts)
    for bar in patches_list:
        ax[0][0].annotate(
            "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data), 2),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 8),
            textcoords='offset points',weight='bold')

    ############LABELS CORRECTED #############
    #freq, bins, patches = ax[0][1].hist(data["target_corrected"].to_numpy(), bins=args.num_classes, density=True, edgecolor='white')

    unique,counts = np.unique(data["target_corrected"].to_numpy(),return_counts=True)
    patches_list = []
    position = 0
    position_labels = []
    for label,count in zip(unique,counts):
        position_labels.append(position)
        bar = ax[0][1].bar(position,count, label=label, color=colors_dict[label], width=0.1, edgecolor='white')
        patches_list.append(bar.patches[0])
        position += 0.5 if args.num_classes == 3 else 0.25

    ax[0][1].margins(y=0.1)
    ax[0][1].set_xlabel('Binary target class')
    ax[0][1].set_title('Corrected binary \n targets counts',fontsize=fontsize)
    ax[0][1].xaxis.set_ticks(position_labels)
    ax[0][1].set_xticklabels(range(args.num_classes))
    # Annotate the bars.
    n_data = sum(counts)

    for bar in patches_list:
        ax[0][1].annotate(
            "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data), 2),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
            size=int(fontsize/2) + 1, xytext=(0, 8),
            textcoords='offset points',weight='bold')
    ####### Immunodominance scores ###################
    ax[0][3].hist(data["immunodominance_score_scaled"].to_numpy(), num_bins, density=True)
    ax[0][3].set_xlabel('Minmax scaled \n immunoprevalence score \n (N + / Total)',fontsize=fontsize)
    ax[0][3].set_title('Histogram of \n immunoprevalence \n scores',fontsize=fontsize)
    ############TEST PROPORTIONS #############
    data_partitions = data[["partition", "training", "target_corrected"]]
    test_counts = data_partitions[data_partitions["training"] == False].value_counts("target_corrected")  # returns a dict
    test_counts = dict(sorted(test_counts.items(), key=operator.itemgetter(1), reverse=True))  # sort dict by values
    if test_counts:
        bar_list= []
        for label,counts in test_counts.items():
            bar = ax[1][0].bar(0, counts, label=labels_dict[label], color=colors_dict[label], width=0.1, edgecolor='white')
            bar = bar.patches[0]
            bar_list.append(bar)
        n_data_test = sum([val for val in test_counts.values()])
        for bar in bar_list:
            ax[1][0].annotate("{}({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data_test), 2),
                              (bar.get_x() + bar.get_width() / 2,
                               bar.get_height()), ha='center', va='center',
                              size=int(fontsize/2) + 1, xytext=(0, 8),
                              textcoords='offset points',weight='bold')
        ax[1][0].margins(y=0.1)
        ax[1][0].xaxis.set_ticks([0])
        ax[1][0].set_xticklabels(["Test proportions"], fontsize=fontsize)
        ax[1][0].set_title('Test dataset. \n Class proportions', fontsize=fontsize)
    ################ TRAIN PARTITION PROPORTIONS###################################

    train_set = data_partitions[data_partitions["training"] == True]

    partitions_groups = [train_set.groupby('partition').get_group(x) for x in train_set.groupby('partition').groups]
    i = 0
    partitions_names = []
    for group in partitions_groups:
        name = group["partition"].iloc[0]
        group_counts = group.value_counts("target_corrected")  # returns a dict
        group_counts = dict( sorted(group_counts.items(), key=operator.itemgetter(1),reverse=True)) #sort dict by value counts
        if group_counts:
            bar_list = []
            for label,count in group_counts.items():
                bar = ax[0][2].bar(i, count, label=labels_dict[int(label)], color=colors_dict[int(label)], width=0.1, edgecolor='white')
                bar = bar.patches[0]
                bar_list.append(bar)

            i += 0.4
            n_data_partition = sum([val for key, val in group_counts.items()])
            for bar in bar_list:
                ax[0][2].annotate(
                    "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data_partition), 2),
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height()), ha='center', va='center',
                    size=int(fontsize/2) + 1, xytext=(0, 8),
                    textcoords='offset points',weight='bold')

        partitions_names.append(name)

    ax[0][2].margins(y=0.1)
    ax[0][2].xaxis.set_ticks([0, 0.4, 0.8, 1.2, 1.6])
    ax[0][2].set_xticklabels(["Part. {}".format(int(i)) for i in partitions_names])
    ax[0][2].set_title('Train-validation dataset. \n Class proportions \n per partition',fontsize=fontsize)
    ######## Sequence length distributions ####################
    ######Train#############################3
    data_lens_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_unobserved = data.loc[(data["training"] == True) & (data["target_corrected"] == 2.), filters_dict["filter_kmers"][2]].str.len()

    dict_counts = {0: data_lens_train_negative.value_counts(), 1: data_lens_train_positive.value_counts(),2:data_lens_train_unobserved.value_counts()}
    all_lens = []
    for key,vals in dict_counts.items():
        all_lens += list(vals.keys())
    unique_sorted = sorted(set(all_lens))[::-1]
    position = 0
    positions = []
    for seq_len in unique_sorted:
        positions.append(position + 0.2)
        for label,val_counts in dict_counts.items():
            if seq_len in val_counts.keys():
                ax[1][1].bar(position, val_counts[seq_len], label=seq_len, color=colors_dict[label], width=0.1, edgecolor='white')
                position += 0.2

    ax[1][1].xaxis.set_ticks(positions)
    ax[1][1].set_xticklabels(unique_sorted)
    ax[1][1].set_title("Sequence length \n distribution \n of the Train-validation dataset",fontsize=fontsize)

    ###### Test #####################
    data_lens_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_unobserved = data.loc[(data["training"] == False) & (data["target_corrected"] == 2.), filters_dict["filter_kmers"][2]].str.len()

    dict_counts = {0: data_lens_test_negative.value_counts(), 1: data_lens_test_positive.value_counts(),2:data_lens_test_unobserved.value_counts()}
    all_lens = []
    for key,vals in dict_counts.items():
        all_lens += list(vals.keys())
    unique_sorted = sorted(set(all_lens))[::-1]
    position = 0
    positions = []
    for seq_len in unique_sorted:
        positions.append(position + 0.2)
        for label,val_counts in dict_counts.items():
            if seq_len in val_counts.keys():
                ax[1][2].bar(position, val_counts[seq_len], label=seq_len, color=colors_dict[label], width=0.1, edgecolor='white')
                position += 0.2
    ax[1][2].xaxis.set_ticks(positions)
    ax[1][2].set_xticklabels(unique_sorted)
    ax[1][2].set_title("Sequence length \n distribution \n of the Test dataset",fontsize=fontsize)

    #################################################################

    # ################### ALLELES PROPORTIONS PER CLASS ##############################################
    # if not filters_dict["group_alleles"][0]:
    #     #Train
    #     data_alleles_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), "allele"].value_counts().to_dict()
    #     data_alleles_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), "allele"].value_counts().to_dict()
    #
    #     dict_counts = {0: data_alleles_train_negative, 1: data_alleles_train_positive}
    #     longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
    #     position = 0
    #     positions = []
    #     for val_i, count_i in dict_counts[longest].items():
    #         ax[1][2].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.4)
    #         if val_i in dict_counts[shortest].keys():
    #             count_j = dict_counts[shortest][val_i]
    #             ax[1][2].bar(position + 0.45, count_j, label=shortest, color=colors_dict[shortest], width=0.4)
    #         positions.append(position + 0.1)
    #         position += 1.1
    #     ax[1][2].xaxis.set_ticks(positions)
    #     ax[1][2].set_xticklabels(dict_counts[longest].keys(),rotation=90,fontsize=2)
    #     ax[1][2].xaxis.set_tick_params(width=0.1)
    #     ax[1][2].set_title("Allele distribution of \n  the Train-valid dataset")
    #     #Test
    #     data_alleles_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), "allele"].value_counts().to_dict()
    #     data_alleles_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), "allele"].value_counts().to_dict()
    #     dict_counts = {0: data_alleles_test_negative, 1: data_alleles_test_positive}
    #     longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
    #     position = 0
    #     positions = []
    #     for val_i, count_i in dict_counts[longest].items():
    #         ax[2][2].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.4)
    #         if val_i in dict_counts[shortest].keys():
    #             count_j = dict_counts[shortest][val_i]
    #             ax[2][2].bar(position + 0.45, count_j, label=shortest, color=colors_dict[shortest], width=0.4)
    #         positions.append(position + 0.1)
    #         position += 1.1
    #     ax[2][2].xaxis.set_ticks(positions)
    #     ax[2][2].set_xticklabels(dict_counts[longest].keys(),rotation=90,fontsize=2)
    #     ax[2][2].xaxis.set_tick_params(width=0.1)
    #     ax[2][2].set_title("Allele distribution of \n  the Test dataset",fontsize=10)


    #ax[0][3].axis("off")
    #ax[1][3].axis("off")
    ax[1][3].axis("off")


    legends = [mpatches.Patch(color=color, label='Class {}'.format(label))  for label, color in colors_dict.items() if label < args.num_classes]
    fig.legend(handles=legends, prop={'size': fontsize}, loc='center right', bbox_to_anchor=(0.85, 0.2))
    fig.tight_layout(pad=0.1)
    fig.suptitle("Dataset distributions",fontsize=fontsize)
    plt.subplots_adjust(right=0.9,top=0.85,hspace=0.4,wspace=0.4)
    plt.savefig("{}/{}/Viruses_histograms_{}".format(storage_folder, args.dataset_name, name_suffix), dpi=700)
    plt.clf()
    plt.close(fig)

def plot_data_information_reduced2(data:pd.DataFrame, filters_dict:dict, storage_folder:str, args:namedtuple, name_suffix:str):
    """"""
    ndata = data.shape[0]
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15, 10))
    fontsize = 20
    num_bins = 50

    ############LABELS #############
    unique,counts = np.unique(data["target"].to_numpy(),return_counts=True)

    patches_list = []
    position = 0
    position_labels = []
    for label,count in zip(unique,counts):
        position_labels.append(position)
        bar = ax[0][0].bar(position,count, label=label, color=colors_dict[label], width=0.1, edgecolor='white')
        patches_list.append(bar.patches[0])
        position += 0.5 if args.num_classes == 3 else 0.25
    ax[0][0].margins(y=0.1)
    ax[0][0].set_xlabel('Binary target class')
    ax[0][0].set_title('Binary targets \n counts',fontsize=fontsize)
    ax[0][0].xaxis.set_ticks(position_labels)
    ax[0][0].set_xticklabels(range(args.num_classes))
    # Annotate the bars.
    n_data = sum(counts)
    for bar in patches_list:
        ax[0][0].annotate(
            "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data), 2),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
            size=10, xytext=(0, 8),
            textcoords='offset points',weight='bold')

    ############LABELS CORRECTED #############
    #freq, bins, patches = ax[0][1].hist(data["target_corrected"].to_numpy(), bins=args.num_classes, density=True, edgecolor='white')

    unique,counts = np.unique(data["target_corrected"].to_numpy(),return_counts=True)
    patches_list = []
    position = 0
    position_labels = []
    for label,count in zip(unique,counts):
        position_labels.append(position)
        bar = ax[0][1].bar(position,count, label=label, color=colors_dict[label], width=0.1, edgecolor='white')
        patches_list.append(bar.patches[0])
        position += 0.5 if args.num_classes == 3 else 0.25

    ax[0][1].margins(y=0.1)
    ax[0][1].set_xlabel('Binary target class')
    ax[0][1].set_title('Corrected binary \n targets counts',fontsize=fontsize)
    ax[0][1].xaxis.set_ticks(position_labels)
    ax[0][1].set_xticklabels(range(args.num_classes))
    # Annotate the bars.
    n_data = sum(counts)

    for bar in patches_list:
        ax[0][1].annotate(
            "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data), 2),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height()), ha='center', va='center',
            size=int(fontsize/2) + 1, xytext=(0, 8),
            textcoords='offset points',weight='bold')
    ####### Immunodominance scores ###################
    # ax[0][3].hist(data["immunodominance_score_scaled"].to_numpy(), num_bins, density=True)
    # ax[0][3].set_xlabel('Minmax scaled \n immunoprevalence score \n (N + / Total)',fontsize=fontsize)
    # ax[0][3].set_title('Histogram of \n immunoprevalence \n scores',fontsize=fontsize)
    ax[0][3].axis("off")
    ############TEST PROPORTIONS #############
    data_partitions = data[["partition", "training", "target_corrected"]]
    test_counts = data_partitions[data_partitions["training"] == False].value_counts("target_corrected")  # returns a dict
    test_counts = dict(sorted(test_counts.items(), key=operator.itemgetter(1), reverse=True))  # sort dict by values
    if test_counts:
        bar_list= []
        for label,counts in test_counts.items():
            bar = ax[1][0].bar(0, counts, label=labels_dict[label], color=colors_dict[label], width=0.1, edgecolor='white')
            bar = bar.patches[0]
            bar_list.append(bar)
        n_data_test = sum([val for val in test_counts.values()])
        for bar in bar_list:
            ax[1][0].annotate("{}({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data_test), 2),
                              (bar.get_x() + bar.get_width() / 2,
                               bar.get_height()), ha='center', va='center',
                              size=int(fontsize/2) + 1, xytext=(0, 8),
                              textcoords='offset points',weight='bold')
        ax[1][0].margins(y=0.1)
        ax[1][0].xaxis.set_ticks([0])
        ax[1][0].set_xticklabels(["Test proportions"], fontsize=fontsize)
        ax[1][0].set_title('Test dataset. \n Class proportions', fontsize=fontsize)
    ################ TRAIN PARTITION PROPORTIONS###################################

    train_set = data_partitions[data_partitions["training"] == True]

    partitions_groups = [train_set.groupby('partition').get_group(x) for x in train_set.groupby('partition').groups]
    i = 0
    partitions_names = []
    for group in partitions_groups:
        name = group["partition"].iloc[0]
        group_counts = group.value_counts("target_corrected")  # returns a dict
        group_counts = dict( sorted(group_counts.items(), key=operator.itemgetter(1),reverse=True)) #sort dict by value counts
        if group_counts:
            bar_list = []
            for label,count in group_counts.items():
                bar = ax[0][2].bar(i, count, label=labels_dict[int(label)], color=colors_dict[int(label)], width=0.1, edgecolor='white')
                bar = bar.patches[0]
                bar_list.append(bar)

            i += 0.4
            n_data_partition = sum([val for key, val in group_counts.items()])
            for bar in bar_list:
                ax[0][2].annotate(
                    "{}\n({}%)".format(bar.get_height(), np.round((bar.get_height() * 100) / n_data_partition), 2),
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height()), ha='center', va='center',
                    size=int(fontsize/2) + 1, xytext=(0, 8),
                    textcoords='offset points',weight='bold')

        partitions_names.append(name)

    ax[0][2].margins(y=0.1)
    ax[0][2].xaxis.set_ticks([0, 0.4, 0.8, 1.2, 1.6])
    ax[0][2].set_xticklabels(["Part. {}".format(int(i)) for i in partitions_names])
    ax[0][2].set_title('Train-validation dataset. \n Class proportions \n per partition',fontsize=fontsize)
    ######## Sequence length distributions ####################
    ######Train#############################3
    data_lens_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_unobserved = data.loc[(data["training"] == True) & (data["target_corrected"] == 2.), filters_dict["filter_kmers"][2]].str.len()

    dict_counts = {0: data_lens_train_negative.value_counts(), 1: data_lens_train_positive.value_counts(),2:data_lens_train_unobserved.value_counts()}
    all_lens = []
    for key,vals in dict_counts.items():
        all_lens += list(vals.keys())
    unique_sorted = sorted(set(all_lens))[::-1]
    position = 0
    positions = []
    for seq_len in unique_sorted:
        positions.append(position + 0.2)
        for label,val_counts in dict_counts.items():
            if seq_len in val_counts.keys():
                ax[1][1].bar(position, val_counts[seq_len], label=seq_len, color=colors_dict[label], width=0.1, edgecolor='white')
                position += 0.2

    ax[1][1].xaxis.set_ticks(positions)
    ax[1][1].set_xticklabels(unique_sorted)
    ax[1][1].set_title("Sequence length \n distribution \n of the Train-validation dataset",fontsize=fontsize)

    ###### Test #####################
    data_lens_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_unobserved = data.loc[(data["training"] == False) & (data["target_corrected"] == 2.), filters_dict["filter_kmers"][2]].str.len()

    dict_counts = {0: data_lens_test_negative.value_counts(), 1: data_lens_test_positive.value_counts(),2:data_lens_test_unobserved.value_counts()}
    all_lens = []
    for key,vals in dict_counts.items():
        all_lens += list(vals.keys())
    unique_sorted = sorted(set(all_lens))[::-1]
    position = 0
    positions = []
    for seq_len in unique_sorted:
        positions.append(position + 0.2)
        for label,val_counts in dict_counts.items():
            if seq_len in val_counts.keys():
                ax[1][2].bar(position, val_counts[seq_len], label=seq_len, color=colors_dict[label], width=0.1, edgecolor='white')
                position += 0.2
    ax[1][2].xaxis.set_ticks(positions)
    ax[1][2].set_xticklabels(unique_sorted)
    ax[1][2].set_title("Sequence length \n distribution \n of the Test dataset",fontsize=fontsize)

    #################################################################

    # ################### ALLELES PROPORTIONS PER CLASS ##############################################
    # if not filters_dict["group_alleles"][0]:
    #     #Train
    #     data_alleles_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), "allele"].value_counts().to_dict()
    #     data_alleles_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), "allele"].value_counts().to_dict()
    #
    #     dict_counts = {0: data_alleles_train_negative, 1: data_alleles_train_positive}
    #     longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
    #     position = 0
    #     positions = []
    #     for val_i, count_i in dict_counts[longest].items():
    #         ax[1][2].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.4)
    #         if val_i in dict_counts[shortest].keys():
    #             count_j = dict_counts[shortest][val_i]
    #             ax[1][2].bar(position + 0.45, count_j, label=shortest, color=colors_dict[shortest], width=0.4)
    #         positions.append(position + 0.1)
    #         position += 1.1
    #     ax[1][2].xaxis.set_ticks(positions)
    #     ax[1][2].set_xticklabels(dict_counts[longest].keys(),rotation=90,fontsize=2)
    #     ax[1][2].xaxis.set_tick_params(width=0.1)
    #     ax[1][2].set_title("Allele distribution of \n  the Train-valid dataset")
    #     #Test
    #     data_alleles_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), "allele"].value_counts().to_dict()
    #     data_alleles_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), "allele"].value_counts().to_dict()
    #     dict_counts = {0: data_alleles_test_negative, 1: data_alleles_test_positive}
    #     longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
    #     position = 0
    #     positions = []
    #     for val_i, count_i in dict_counts[longest].items():
    #         ax[2][2].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.4)
    #         if val_i in dict_counts[shortest].keys():
    #             count_j = dict_counts[shortest][val_i]
    #             ax[2][2].bar(position + 0.45, count_j, label=shortest, color=colors_dict[shortest], width=0.4)
    #         positions.append(position + 0.1)
    #         position += 1.1
    #     ax[2][2].xaxis.set_ticks(positions)
    #     ax[2][2].set_xticklabels(dict_counts[longest].keys(),rotation=90,fontsize=2)
    #     ax[2][2].xaxis.set_tick_params(width=0.1)
    #     ax[2][2].set_title("Allele distribution of \n  the Test dataset",fontsize=10)


    #ax[0][3].axis("off")
    #ax[1][3].axis("off")
    ax[1][3].axis("off")


    legends = [mpatches.Patch(color=color, label='Class {}'.format(label))  for label, color in colors_dict.items() if label < args.num_classes]
    fig.legend(handles=legends, prop={'size': fontsize}, loc='center right', bbox_to_anchor=(0.85, 0.5))
    fig.tight_layout(pad=0.1)
    fig.suptitle("Dataset distributions",fontsize=fontsize,x=0.35)
    plt.subplots_adjust(right=0.9,top=0.85,hspace=0.4,wspace=0.4)
    plt.savefig("{}/{}/Viruses_histograms_{}.jpg".format(storage_folder, args.dataset_name, name_suffix), dpi=700)
    plt.clf()
    plt.close(fig)

def plot_features_histogram(data, features_names, results_dir, name_suffix):
    """Plots the histogram densities featues of all data points
    Notes: If feeling fancy try: https://stackoverflow.com/questions/38830250/how-to-fill-matplotlib-bars-with-a-gradient"""
    max_cols = 4
    num_bins = 50
    nplots = [int(len(features_names)) if int(len(features_names)) % 2 == 0 else int(len(features_names)) + 1][0]
    ncols = [int(nplots / 2) if int(nplots / 2) <= max_cols else max_cols][0]
    nrows = [int(nplots / ncols) if nplots / ncols % 2 == 0 else int(nplots / ncols) + 1][0]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,squeeze=False)  # check this: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
    axs = axs.ravel()
    total_ax = len(axs)
    if isinstance(data,pd.DataFrame):
        i = 0
        data_negative = data.loc[data["target_corrected"] == 0]
        data_positive = data.loc[data["target_corrected"] == 1]
        for idx,feature_name in enumerate(features_names):
            freq, bins, patches = axs[idx].hist(data_negative[feature_name].to_numpy(), bins=num_bins, density=True,label="Negative",color=colors_dict[0],alpha=0.5)
            freq, bins, patches = axs[idx].hist(data_positive[feature_name].to_numpy(), bins=num_bins, density=True,label="Positive",color = colors_dict[1],alpha=0.5)
            axs[idx].set_xlabel('')
            axs[idx].set_title('{} \n'.format(feature_name),fontsize=6)
            axs[idx].tick_params(axis='both', which='both', labelsize=5)
            axs[idx].yaxis.offsetText.set_fontsize(5) #referred as scientific notation
            i = idx
        for j in range(i +1 ,total_ax):
            axs[j].axis("off")

        fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
        fig.suptitle("Histogram Features")
        negative_patch = mpatches.Patch(color=colors_dict[0], label='Class 0')
        positive_patch = mpatches.Patch(color=colors_dict[1], label='Class 1')
        plt.legend(handles=[negative_patch, positive_patch], prop={'size': 8}, loc='center',bbox_to_anchor=(-1.6, -0.2), ncol=2)
        plt.savefig("{}/Viruses_features_histograms_{}".format(results_dir, name_suffix), dpi=300)
        plt.clf()
        plt.close()

    elif isinstance(data,torch.Tensor) or isinstance(data,np.ndarray):
        i = 0
        seq_max_len = data.shape[2] - len(features_names)
        data_negative = data[data[:,0,0,0] == 0]
        data_negative_features = data_negative[:,1, seq_max_len:, 0]
        data_positive = data[data[:,0,0,0] == 1]
        data_positive_features = data_positive[:,1, seq_max_len:, 0]
        for idx, feature_name in enumerate(features_names):
            freq, bins, patches = axs[idx].hist(data_negative_features[:,idx].numpy(), bins=num_bins, density=True,label="Negative",color=colors_dict[0],alpha=0.5)
            freq, bins, patches = axs[idx].hist(data_positive_features[:,idx].numpy(), bins=num_bins, density=True,label="Positive",color = colors_dict[1],alpha=0.5)
            axs[idx].set_xlabel('')
            axs[idx].set_title('{} \n'.format(feature_name), fontsize=6)
            axs[idx].tick_params(axis='both', which='both', labelsize=5)
            axs[idx].yaxis.offsetText.set_fontsize(5)  # referred as scientific notation
            i = idx
        for j in range(i + 1, total_ax):
            axs[j].axis("off")

        fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
        fig.suptitle("Histogram Features (preprocessed)")
        negative_patch = mpatches.Patch(color=colors_dict[0], label='Class 0')
        positive_patch = mpatches.Patch(color=colors_dict[1], label='Class 1')
        plt.legend(handles=[negative_patch, positive_patch], prop={'size': 8}, loc='center right',bbox_to_anchor=(-1.6, -0.2), ncol=2)
        plt.savefig("{}/Viruses_features_histograms_{}".format(results_dir, name_suffix),
                    dpi=300)
        plt.clf()
        plt.close(fig)

    else:
        print("Data type not implemented, not plotting features histograms")

def plot_data_umap(data_array_blosum_norm,seq_max_len,max_len,script_dir,dataset_name):
    """Plotting the projections of the data"""
    if seq_max_len == max_len:
        data_sequences_norm = data_array_blosum_norm[:,1]
    else:
        data_sequences_norm = data_array_blosum_norm[:,1,:seq_max_len]
        data_features = data_array_blosum_norm[:,1,seq_max_len:]

    reducer = umap.UMAP()
    true_labels = data_array_blosum_norm[:,0,0]
    colors_true = np.vectorize(colors_dict_labels.get)(true_labels)

    confidence_scores = data_array_blosum_norm[:,0,5]
    confidence_scores_unique = np.unique(confidence_scores).tolist()
    colormap_confidence = matplotlib.cm.get_cmap('plasma_r', len(confidence_scores_unique))
    colors_dict = dict(zip(confidence_scores_unique, colormap_confidence.colors))
    colors_confidence = np.vectorize(colors_dict.get, signature='()->(n)')(confidence_scores)

    immunodominance_scores = data_array_blosum_norm[:,0,4]
    immunodominance_scores_unique = np.unique(immunodominance_scores).tolist()
    colormap_immunodominance = matplotlib.cm.get_cmap('plasma_r', len(immunodominance_scores_unique))
    colors_dict = dict(zip(immunodominance_scores_unique, colormap_immunodominance.colors))
    colors_immunodominance = np.vectorize(colors_dict.get, signature='()->(n)')(immunodominance_scores)

    umap_proj = reducer.fit_transform(data_sequences_norm)
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2,ncols=3, figsize=(17, 12), gridspec_kw={'width_ratios': [4.5, 4.5,1], 'height_ratios': [4, 4]})

    fig.suptitle('UMAP projections of blosum norms', fontsize=20)
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, alpha=1, s=30)
    ax1.set_title("True labels", fontsize=20)
    ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_confidence, alpha=1, s=30)
    ax2.set_title("Confidence scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence),ax=ax2) #cax= fig.add_axes([0.9, 0.5, 0.01, 0.09])
    ax4.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_immunodominance, alpha=1, s=30)
    ax4.set_title("Immunoprevalence (aggregated +) scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_immunodominance),ax=ax4) #cax= fig.add_axes([0.9, 0.5, 0.01, 0.09])
    ax3.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
    plt.legend(handles=[negative_patch, positive_patch], prop={'size': 20}, loc='center right',
               bbox_to_anchor=(0.6, 0.5), ncol=1)
    plt.savefig("{}/{}/umap_data_norm".format(script_dir,dataset_name))
    plt.clf()
    plt.close(fig)


    if seq_max_len != max_len:
        umap_proj = reducer.fit_transform(data_features)
        fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(17, 12),
                                                               gridspec_kw={'width_ratios': [4.5, 4.5,1],
                                                                            'height_ratios': [4, 4]})
        fig.suptitle('UMAP projections of data features', fontsize=20)
        ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, alpha=1, s=30)
        ax1.set_title("True labels", fontsize=20)
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_confidence, alpha=1, s=30)
        ax2.set_title("Confidence scores", fontsize=20)
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence), ax=ax2)  # cax= fig.add_axes([0.9, 0.5, 0.01, 0.09])
        ax4.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_immunodominance, alpha=1, s=30)
        ax4.set_title("Immunodominance scores", fontsize=20)
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_immunodominance), ax=ax4)  # cax= fig.add_axes([0.9, 0.5, 0.01, 0.09])
        ax3.axis("off")
        ax5.axis("off")
        ax6.axis("off")
        negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
        positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
        plt.legend(handles=[negative_patch, positive_patch], prop={'size': 20}, loc='center right',
                   bbox_to_anchor=(0.7, 0.5), ncol=1)
        plt.savefig("{}/{}/umap_data_features".format(script_dir, dataset_name))
        plt.clf()
        plt.close(fig)

def plot_aa_frequencies(data_array:Union[np.ndarray,pd.DataFrame],aa_types,aa_dict,max_len,storage_folder,args,analysis_mode,mode):
    """Creates a bar plot per position in the sequence to show the amino acid frequencies"""

    aa_groups_colors_dict,aa_groups_dict,groups_names_colors_dict,aa_by_groups_dict = VegvisirUtils.aminoacids_groups(aa_dict)
    reverse_aa_dict = {val:key for key,val in aa_dict.items()}
    frequencies_per_position = VegvisirUtils.calculate_aa_frequencies(data_array,aa_types)
    aa_patches = [mpatches.Patch(color=colors_list_aa[i], label='{}'.format(aa)) for aa,i in aa_dict.items()]
    aa_groups_patches = [mpatches.Patch(color=color, label='{}'.format(group)) for group,color in groups_names_colors_dict.items()]
    aa_colors_dict = {i: colors_list_aa[i] for aa, i in aa_dict.items()}
    aa_int = np.array(list(aa_dict.values()))
    fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(12, 6),gridspec_kw={'width_ratios': [4.5, 4.5,1]})  # check this: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots

    x_pos = 0
    x_pos_labels = []
    for p_idx, frequencies_position in enumerate(frequencies_per_position):
        x_pos_labels.append(x_pos)
        sorted_idx = np.argsort(frequencies_position)[::-1]
        frequencies_position_sorted = frequencies_position[sorted_idx]
        sorted_aa = aa_int[sorted_idx]
        group_frequencies_inverted_dict = dict.fromkeys(list(groups_names_colors_dict.values()))
        #group_frequencies_inverted_dict = defaultdict(float)
        for freq,aa in zip(frequencies_position_sorted,sorted_aa):
            group_color_aa = aa_groups_colors_dict[aa]
            #print("aa idx {}, aa name {},freq {},group color {}".format(aa,reverse_aa_dict[aa],freq,group_color_aa))
            if group_frequencies_inverted_dict[group_color_aa] is not None:
                group_frequencies_inverted_dict[group_color_aa] += freq
            else:
                group_frequencies_inverted_dict[group_color_aa] = freq
        group_frequencies_inverted_dict = {k: v for k, v in group_frequencies_inverted_dict.items() if v is not None}
        assert np.sum(frequencies_position_sorted) == sum(list(group_frequencies_inverted_dict.values()))

        prev_group_freq = 0
        for group_color,group_frequency in sorted(group_frequencies_inverted_dict.items(), key=lambda x: x[1],reverse=True):
            if group_frequency != 0:
                if group_frequency == prev_group_freq: # avoids overlapping frequencies with equal values
                    x_pos += 0.3
                axs[1].bar(x_pos, group_frequency,color=group_color,width = 0.2)
                prev_group_freq = group_frequency
        prev_aa_freq = 0
        for aa_idx, aa_frequency in zip(sorted_aa,frequencies_position_sorted):
            if aa_frequency != 0:
                if aa_frequency == prev_aa_freq: # avoids overlapping frequencies with equal values
                    x_pos += 0.3
                axs[0].bar(x_pos, aa_frequency,color=aa_colors_dict[aa_idx],width = 0.2)
                prev_aa_freq = aa_frequency
        x_pos += 1

    axs[0].set_xticks(x_pos_labels)
    axs[0].set_xticklabels(labels=list(range(max_len)))
    axs[1].set_xticks(x_pos_labels)
    axs[1].set_xticklabels(labels=list(range(max_len)))
    axs[2].axis("off")
    legend1 = plt.legend(handles=aa_patches, prop={'size': 8}, loc='center right',
                         bbox_to_anchor=(1.4, 0.5), ncol=1)
    plt.legend(handles=aa_groups_patches, prop={'size': 8}, loc='center right',
               bbox_to_anchor=(0.7, 0.45), ncol=1)
    plt.gca().add_artist(legend1)
    plt.suptitle("Amino acid/Group counts per position ()".format(mode))
    plt.savefig("{}/{}/{}/Aminoacids_frequency_counts_{}".format(storage_folder, args.dataset_name,analysis_mode,mode), dpi=500)
    plt.clf()
    plt.close(fig)

def plot_heatmap(array:np.ndarray, title,file_name):
    fig = plt.figure(figsize=(20, 20))
    sns.heatmap(array, cmap='RdYlGn_r',yticklabels=False,xticklabels=False)
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()
    plt.close(fig)

def plot_logos(sequences_list:list,results_dir:str,filename:str=""):
    """
    Notes:
    -Logomaker docs: https://github.com/jbkinney/logomaker/blob/master/logomaker/examples/logos_from_datafiles.ipynb
    -https://stackoverflow.com/questions/42615527/sequence-logos-in-matplotlib-aligning-xticks
    :return:
    """
    print("Generating logos plots based on amino acid frequencies")

    filepath = "{}/SEQUENCES_{}".format(results_dir,filename)
    with open(filepath,"w+") as f:
        sequences_unpadded =  list(map(lambda seq: seq.replace("#","-"), sequences_list))
        f.write("\n".join(sequences_unpadded))

    command = 'Rscript'
    script_path = os.path.abspath(os.path.dirname(__file__))
    path2script = "{}/logos.R".format(script_path)
    inputpath = filepath
    outputpath = "{}/LOGOS_GGSEQLOGO_{}.jpg".format(results_dir,filename)
    print("Building logos plots using ggseqlogo")
    subprocess.check_call([command, path2script, inputpath, outputpath])

    sequences_list = list(map(lambda seq: seq.replace("\n",""), sequences_list))
    ww_counts_df = logomaker.alignment_to_matrix(sequences=sequences_list, to_type='counts', characters_to_ignore='.-X\n')

    logomaker.Logo(ww_counts_df,color_scheme="chemistry")
    plt.savefig("{}/Logos{}.png".format(results_dir,filename))

def plot_umap1(array:np.ndarray,labels:list,storage_folder:str,args:namedtuple,title_name:str,file_name:str):
    from matplotlib.colors import ListedColormap
    print("Plotting UMAP ---")
    unique = np.unique(labels)
    if len(unique) == 2:
        labels = np.array(labels).squeeze(-1)
        colors_dict = {0:"red",1:"green"}
        color_labels = np.vectorize(colors_dict.get)(labels)
    else:
        colormap = matplotlib.cm.get_cmap('plasma_r', len(unique))
        colors_dict = dict(zip(unique,colormap.colors))
        color_labels = np.vectorize(colors_dict.get, signature='()->(n)')(labels)

    reducer = umap.UMAP()
    v_pr = reducer.fit_transform(array)
    fig,ax = plt.subplots(figsize=(18, 18))
    plt.scatter(v_pr[:, 0], v_pr[:, 1],c = color_labels ,alpha=0.7, s=50)
    if len(unique) == 2:
        patches = [mpatches.Patch(color=color, label='Class {}'.format(label)) for label,color in colors_dict.items() ]
        fig.legend(handles=patches, prop={'size': 20},loc= 'center right',bbox_to_anchor=(1,0.5))
    else:
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap),ax=ax) #cax= fig.add_axes([0.9, 0.5, 0.01, 0.09])
    plt.title("UMAP dimensionality reduction of {}".format(title_name),fontsize=20)
    plt.savefig("{}/{}/similarities_old/{}.png".format(storage_folder,args.dataset_name,file_name))
    plt.clf()
    plt.close(fig)

def plot_loss(train_loss:list,valid_loss:list,epochs_list:list,fold:str,results_dir:str):
    """Plots the model's error loss
    :param list train_loss: list of accumulated error losses during training
    :param list valid_loss: list of accumulated error losses during validation
    :param list epochs_list: list of epochs
    :param str results_dict: path to results directory
    """
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    epochs_idx = np.array(epochs_list)
    train_loss = train_loss[epochs_idx.astype(int)] #select the same epochs as the vaidation

    if np.isnan(train_loss).any():
        print("Error loss contains nan")
        pass
    else:
        fig = plt.figure()
        plt.plot(epochs_idx,train_loss, color="dodgerblue",label="train")
        if valid_loss is not None:
            plt.plot(epochs_idx,valid_loss, color="darkorange", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("-ELBO")
        #plt.yscale('log')
        plt.title("Error loss (Train/valid)")
        plt.legend()
        plt.savefig("{}/error_loss_{}fold.png".format(results_dir,fold))
        plt.close(fig)
        plt.clf()

def plot_accuracy(train_accuracies: Union[dict,list],valid_accuracies: Union[dict,list],epochs_list:list,mode:str,results_dir:str):
    """Plots the model's accuracies, both for target label and for sequence reconstruction loss
    :param list train_elbo: list of accumulated error losses during training
    :param str results_dict: path to results directory
    """
    if isinstance(train_accuracies,dict):
        epochs_idx = np.array(epochs_list)
        train_accuracies_mean = np.array(train_accuracies["mean"])[epochs_idx.astype(int)]
        valid_accuracies_mean = np.array(valid_accuracies["mean"])
        train_accuracies_std = np.array(train_accuracies["std"])[epochs_idx.astype(int)]
        valid_accuracies_std = np.array(valid_accuracies["std"])
        epochs_idx = np.array(epochs_list)
        fig = plt.figure()
        plt.plot(epochs_idx, train_accuracies_mean, color="deepskyblue", label="train")
        plt.fill_between(epochs_idx,train_accuracies_mean-train_accuracies_std,train_accuracies_mean+train_accuracies_std,color="cyan",alpha=0.2)
        if valid_accuracies is not None:
            plt.plot(epochs_idx, valid_accuracies_mean, color="salmon", label="validation")
            plt.fill_between(epochs_idx, valid_accuracies_mean - valid_accuracies_std,
                             valid_accuracies_mean + valid_accuracies_std, color="salmon", alpha=0.2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (average number of correct \n  amino acids across all sequences)")
        plt.title("Sequence Reconstruction Accuracy (Train/valid)")
        plt.legend()
        plt.savefig("{}/reconstruction_accuracies_{}fold.png".format(results_dir, mode))
        plt.clf()
        plt.close(fig)


    else:
        train_accuracies = np.array(train_accuracies)
        valid_accuracies = np.array(valid_accuracies)
        epochs_idx = np.array(epochs_list)
        train_accuracies = train_accuracies[epochs_idx.astype(int)] #select the same epochs as the vaidation
        fig = plt.figure()
        plt.plot(epochs_idx,train_accuracies, color="deepskyblue",label="train")
        if valid_accuracies is not None:
            plt.plot(epochs_idx,valid_accuracies, color="salmon", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (number of correct predictions)")
        plt.title("Accuracy (Train/valid)")
        plt.legend()
        plt.savefig("{}/accuracies_{}fold.png".format(results_dir,mode))
        plt.clf()
        plt.close(fig)

def plot_classification_score(train_auc:list,valid_auc:list,epochs_list:list,fold: Union[int,float],results_dir:str,method:str):
    """Plots the AUC/AUK scores while training
    :param list train_auc: list of accumulated AUC during training
    :param list valid_auc: list of accumulated AUC during validation
    :param str results_dict: path to results directory
    :param str method: AUC or AUK
    """
    epochs_idx = np.array(epochs_list)
    fig = plt.figure()
    plt.plot(epochs_idx,train_auc, color="deepskyblue",label="train")
    if valid_auc is not None:
        plt.plot(epochs_idx,valid_auc, color="salmon", label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("{}".format(method))
    plt.title("{} (Train/valid)".format(method))
    plt.legend()
    plt.savefig("{}/{}_{}fold.png".format(results_dir,method,fold))
    plt.clf()
    plt.close(fig)

def plot_latent_vector(latent_space:np.ndarray,predictions_dict:dict,fold:Union[int,float],results_dir:str,method:str):

    print("Plotting latent vector...")
    latent_vectors = latent_space[:,6:]
    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:,0])
    colors_predicted = np.vectorize(colors_dict_labels.get)(predictions_dict["class_logits_predictions_samples_argmax_mode"])
    #Highlight: Confidence scores colors
    confidence_scores = latent_space[:,4]
    confidence_scores_unique = np.unique(confidence_scores).tolist()
    colormap_confidence = matplotlib.cm.get_cmap('plasma_r', len(confidence_scores_unique))
    colors_dict = dict(zip(confidence_scores_unique, colormap_confidence.colors))
    colors_confidence = np.vectorize(colors_dict.get, signature='()->(n)')(confidence_scores)

    fig, [[ax1, ax2, ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]] = plt.subplots(3, 3,figsize=(17, 12),gridspec_kw={'width_ratios': [4.5,4.5,1],'height_ratios': [4,4,2]})
    fig.suptitle('UMAP projections',fontsize=20)
    for i in range(latent_vectors.shape[0]):
        ax1.plot(latent_vectors[i], color=colors_true[i], alpha=1,linewidth=4)
        ax1.set_title("True labels",fontsize=20)
        ax2.plot(latent_vectors[i], color=colors_predicted[i], alpha=1)
        ax2.set_title("Predicted labels (samples mode)",fontsize=20)
        ax4.plot(latent_vectors[i], color=colors_confidence[i], alpha=1)
        ax4.set_title("Confidence scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence),ax=ax4)
    ax3.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")
    ax9.axis("off")

    negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
    fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
    plt.legend(handles=[negative_patch,positive_patch], prop={'size': 20},loc= 'center right',bbox_to_anchor=(1,0.5),ncol=1)
    plt.savefig("{}/{}/zvector_fold{}".format(results_dir,method,fold))
    plt.clf()
    plt.close(fig)
    del latent_vectors,confidence_scores
    gc.collect()

def plot_clusters_features_distributions(dataset_info:namedtuple,cluster_assignments:np.ndarray,n_clusters:Union[int,float],predictions_dict:dict,sample_mode:str,results_dir:str,method:str,vector_name:str):
    """
    Notes:
        - https://www.researchgate.net/publication/15556561_Global_Fold_Determination_from_a_Small_Number_of_Distance_Restraints/figures?lo=1
        - Radius table: https://www.researchgate.net/publication/15556561_Global_Fold_Determination_from_a_Small_Number_of_Distance_Restraints/figures?lo=1
        - Peptide properties: https://cran.r-project.org/web/packages/Peptides/Peptides.pdf
        -Peptide properties:
                http://biotools.nubic.northwestern.edu/proteincalc.html
                https://web.nmsu.edu/~talipovm/lib/exe/fetch.php?media=world:pasted:table08.pdf
        -Peptides and lenghts: https://academic.oup.com/bioinformatics/article/22/22/2761/197569
        -Aminoacid side chain bulkiness:
                    Zimmerman scale: https://pubmed.ncbi.nlm.nih.gov/5700434/
                    https://search.r-project.org/CRAN/refmans/alakazam/html/bulk.html


    """

    custom_features_dicts = VegvisirUtils.build_features_dicts(dataset_info)
    #Highlight: Load features dictionaries
    aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]
    gravy_dict = custom_features_dicts["gravy_dict"]
    volume_dict = custom_features_dicts["volume_dict"]
    radius_dict = custom_features_dicts["radius_dict"]
    side_chain_pka_dict = custom_features_dicts["side_chain_pka_dict"]
    isoelectric_dict = custom_features_dicts["isoelectric_dict"]
    bulkiness_dict = custom_features_dicts["bulkiness_dict"]

    data_int = predictions_dict["data_int_{}".format(sample_mode)]
    sequences = data_int[:,1:].squeeze(1)
    if dataset_info.corrected_aa_types == 20:
        sequences_mask=np.zeros_like(sequences).astype(bool)
    else:
        sequences_mask = np.array((sequences == 0))

    sequences_lens = np.sum(~sequences_mask,axis=1)
    sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(sequences)
    #aminoacid_frequencies = VegvisirUtils.calculate_aa_frequencies(sequences_raw,dataset_info.corrected_aa_types)
    sequences_list = sequences_raw.tolist()

    #aminoacid_frequencies_dict = VegvisirUtils.CalculatePeptideFeatures(dataset_info.seq_max_len,sequences_list,dataset_info.storage_folder).calculate_aminoacid_frequencies()

    # gravy_scores = np.vectorize(gravy_dict.get)(sequences)
    # gravy_scores = np.ma.masked_array(gravy_scores, mask=sequences_mask, fill_value=0)
    # gravy_scores = np.ma.mean(gravy_scores, axis=1)
    
    bulkiness_scores = np.vectorize(bulkiness_dict.get)(sequences)
    bulkiness_scores = np.ma.masked_array(bulkiness_scores, mask=sequences_mask, fill_value=0)
    bulkiness_scores = np.ma.sum(bulkiness_scores, axis=1)
    
    volume_scores = np.vectorize(volume_dict.get)(sequences)
    volume_scores = np.ma.masked_array(volume_scores, mask=sequences_mask, fill_value=0)
    volume_scores = np.ma.sum(volume_scores, axis=1) #TODO: Mean? or sum?
    
    radius_scores = np.vectorize(radius_dict.get)(sequences)
    radius_scores = np.ma.masked_array(radius_scores, mask=sequences_mask, fill_value=0)
    radius_scores = np.ma.sum(radius_scores, axis=1)

    side_chain_pka_scores = np.vectorize(side_chain_pka_dict.get)(sequences)
    side_chain_pka_scores = np.ma.masked_array(side_chain_pka_scores, mask=sequences_mask, fill_value=0)
    side_chain_pka_scores = np.ma.mean(side_chain_pka_scores, axis=1) #Highlight: before I was doing just the sum

    isoelectric_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_isoelectric(seq), sequences_list)))
    aromaticity_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_aromaticity(seq), sequences_list)))
    gravy_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_gravy(seq), sequences_list)))
    molecular_weight_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_molecular_weight(seq), sequences_list)))
    extintion_coefficient_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_extintioncoefficient(seq)[0], sequences_list)))

    fig, [[ax0,ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8,ax9]] = plt.subplots(2, 5,figsize=(26,22))

    clusters_info = defaultdict(lambda : defaultdict(lambda : defaultdict()))
    i = 0
    labels = []

    all_gravy = []
    all_volumes = []
    all_radius = []
    all_isoelectric = []
    all_aromaticity = []
    all_side_chain_pka = []
    all_dict_lens = []
    all_molecular_weights = []
    all_bulkiness = []
    all_extintion_coefficients = []
    all_colors = []
    all_dict_true_labels = []
    label_locations = []
    for cluster in range(n_clusters):
        idx_cluster = (cluster_assignments[..., None] == cluster).any(-1)
        data_int_cluster = data_int[idx_cluster]
        idx_observed = (data_int_cluster[:, 0, 0][..., None] != 2).any(-1)
        for mode,idx in zip(["observed","unobserved"],[idx_observed,np.invert(idx_observed)]):
            sequences_cluster = data_int_cluster[idx][:, 1:].squeeze(1)
            true_labels_cluster =  data_int_cluster[idx][:, 0,0]

            if sequences_cluster.size != 0:
                sequences_raw_cluster = np.vectorize(aminoacids_dict_reversed.get)(sequences_cluster)
                sequences_cluster_list = sequences_raw_cluster.tolist()
                if dataset_info.corrected_aa_types == 20:
                    sequences_mask = np.zeros_like(sequences_cluster).astype(bool)
                else:
                    sequences_mask = np.array((sequences_cluster == 0))
                sequences_len = np.sum(~sequences_mask,axis=1)
                # gravy = np.vectorize(gravy_dict.get)(sequences_cluster)
                # gravy = np.ma.masked_array(gravy,mask=sequences_mask,fill_value=0)
                # gravy = np.ma.sum(gravy,axis=1)
                gravy = np.array(list(map(lambda seq: VegvisirUtils.calculate_gravy(seq), sequences_cluster_list)))
                aromaticity = np.array(list(map(lambda seq: VegvisirUtils.calculate_aromaticity(seq), sequences_cluster_list)))

                volumes = np.vectorize(volume_dict.get)(sequences_cluster)
                volumes = np.ma.masked_array(volumes,mask=sequences_mask,fill_value=0)
                volumes = np.ma.sum(volumes,axis=1)

                bulkiness = np.vectorize(bulkiness_dict.get)(sequences_cluster)
                bulkiness = np.ma.masked_array(bulkiness, mask=sequences_mask, fill_value=0)
                bulkiness = np.ma.sum(bulkiness, axis=1)
                
                radius = np.vectorize(radius_dict.get)(sequences_cluster)
                radius = np.ma.masked_array(radius,mask=sequences_mask,fill_value=0)
                radius = np.ma.sum(radius,axis=1)
                
                side_chain_pka = np.vectorize(side_chain_pka_dict.get)(sequences_cluster)
                side_chain_pka = np.ma.masked_array(side_chain_pka,mask=sequences_mask,fill_value=0)
                side_chain_pka = np.ma.mean(side_chain_pka,axis=1) #Highlight: before I was doing just the sum

                # isoelectric = np.vectorize(isoelectric_dict.get)(sequences_cluster)
                # isoelectric = np.ma.masked_array(isoelectric,mask=sequences_mask,fill_value=0)
                # isoelectric = np.ma.sum(isoelectric,axis=1)
                isoelectric = np.array(list(map(lambda seq: VegvisirUtils.calculate_isoelectric(seq), sequences_cluster_list)))
                molecular_weight = np.array(list(map(lambda seq: VegvisirUtils.calculate_molecular_weight(seq), sequences_cluster_list)))
                extintion_coefficient = np.array(list(map(lambda seq: VegvisirUtils.calculate_extintioncoefficient(seq)[0], sequences_cluster_list)))


                clusters_info["Cluster_{}".format(cluster)][mode]["gravy"] = gravy.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["volumes"] = volumes.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["radius"] = radius.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["side_chain_pka"] = side_chain_pka.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["isoelectric"] = isoelectric.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["aromaticity"] = aromaticity.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["lengths"] = sequences_lens.mean() #TODO: Change to mode
                clusters_info["Cluster_{}".format(cluster)][mode]["molecular_weights"] = molecular_weight.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["bulkiness_scores"] = bulkiness.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["extintion_coefficient"] = extintion_coefficient.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["true_labels"] = true_labels_cluster.mean()


                all_side_chain_pka.append(side_chain_pka)
                all_isoelectric.append(isoelectric)
                all_gravy.append(gravy)
                all_volumes.append(volumes)
                all_radius.append(radius)
                all_aromaticity.append(aromaticity)
                all_molecular_weights.append(molecular_weight)
                all_bulkiness.append(bulkiness)
                all_extintion_coefficients.append(extintion_coefficient)
                all_colors.append(colors_cluster_dict[cluster])

                unique_len,counts_len = np.unique(sequences_len,return_counts=True)
                len_dict = dict(zip(unique_len,counts_len))
                len_dict =  dict(sorted(len_dict.items(), key=operator.itemgetter(1), reverse=True))
                all_dict_lens.append(len_dict)

                unique_true_labels,counts_true_labels = np.unique(true_labels_cluster,return_counts=True)
                true_labels_dict = dict(zip(unique_true_labels,counts_true_labels))
                true_labels_dict =  dict(sorted(true_labels_dict.items(), key=operator.itemgetter(1), reverse=True))
                all_dict_true_labels.append(true_labels_dict)

                labels.append("Cluster {}, \n {}".format(cluster,mode))
                label_locations.append(i + 0.2)
                i+=0.4

    boxplot0 = ax0.boxplot( all_gravy,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster]
    boxplot1 = ax1.boxplot( all_volumes,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]

    boxplot2 = ax2.boxplot( all_radius,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]
    boxplot3 = ax3.boxplot( all_isoelectric,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]

    boxplot4 = ax4.boxplot( all_side_chain_pka,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]
    boxplot5 = ax5.boxplot( all_aromaticity,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]
    boxplot6 = ax6.boxplot( all_molecular_weights,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]
    
    boxplot7 = ax7.boxplot( all_bulkiness,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]

    i = 0
    j = 0
    k= 0
    hist_labels_lens = []
    hist_labels_locations_lens = []
    hist_labels_true = []
    hist_labels_locations_true = []
    for color,label,cluster_dict_lens,cluster_dict_true in zip(all_colors,labels,all_dict_lens,all_dict_true_labels):
        for seq_len,count in cluster_dict_lens.items():
            hist_labels_locations_lens.append(i+j)
            ax8.bar(i+j,count,label=seq_len, color=color, width=0.3, edgecolor='white')
            i += 0.4
            hist_labels_lens.append(seq_len)
        for true_label,count in cluster_dict_true.items():
            hist_labels_locations_true.append(i+k)
            ax9.bar(i+k,count,label=true_label, color=color, width=0.3, edgecolor='white')
            k += 0.4
            hist_labels_true.append(true_label)
        j = i + 0.1
        k = i + 0.1

        
    # fill with colors
    #colors = ['pink', 'lightblue', 'lightgreen',"gold"]
    for bplot in (boxplot0, boxplot1,boxplot2,boxplot3,boxplot4,boxplot5,boxplot6,boxplot7):
        for patch, color in zip(bplot['boxes'], all_colors):
            patch.set_facecolor(color)

    ax0.set_title("Gravy",fontsize=20)
    ax1.set_title("Volumes",fontsize=20)
    ax2.set_title("Radius",fontsize=20)
    ax3.set_title("Isoelectric",fontsize=20)
    ax4.set_title("Side chain PKA",fontsize=20)
    ax5.set_title("Aromaticity",fontsize=20)
    ax6.set_title("Molecular weights",fontsize=20)
    ax7.set_title("Bulkiness",fontsize=20)
    ax8.set_title("Sequences lenghts",fontsize=20)


    ax0.set_xticklabels(rotation=90,labels=labels)
    ax1.set_xticklabels(rotation=90,labels=labels)
    ax2.set_xticklabels(rotation=90,labels=labels)
    ax3.set_xticklabels(rotation=90,labels=labels)
    ax4.set_xticklabels(rotation=90,labels=labels)
    ax5.set_xticklabels(rotation=90,labels=labels)
    ax6.set_xticklabels(rotation=90,labels=labels)
    ax7.set_xticklabels(rotation=90,labels=labels)
    ax8.set_xticks(hist_labels_locations_lens)
    ax8.set_xticklabels(rotation=90,labels=hist_labels_lens)
    ax9.set_xticks(hist_labels_locations_true)
    ax9.set_xticklabels(rotation=90,labels=hist_labels_true)


    #ax1.set_xticks(label_locations)
    #ax1.set_xticklabels(labels=labels1,rotation=45,fontsize=8)
    #plt.legend(handles=[negative_patch,positive_patch], prop={'size': 20},loc= 'center right',bbox_to_anchor=(1,0.5),ncol=1)
    plt.savefig("{}/{}/clusters_features_{}_{}".format(results_dir,method,vector_name,sample_mode))
    plt.clf()
    plt.close(fig)
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    features_dicts = VegvisirUtils.CalculatePeptideFeatures(dataset_info.seq_max_len,sequences_list,storage_folder,return_aa_freqs=True).features_summary()

    features_dict = {"gravy_scores":gravy_scores,
                "isoelectric_scores":isoelectric_scores,
                "volume_scores":volume_scores,
                "radius_scores":radius_scores,
                "side_chain_pka_scores":side_chain_pka_scores,
                "aromaticity_scores":aromaticity_scores,
                "molecular_weights":molecular_weight_scores,
                "bulkiness_scores":bulkiness_scores,
                "sequences_lens":sequences_lens,
                "extintion_coefficients":extintion_coefficient_scores,
                "clusters_info":clusters_info,
                "tryptophan":features_dicts["Tryptophan"]}
    
    return features_dict,sequences_raw

def define_colormap(feature:np.ndarray,cmap_name:str):
    """

    :param feature:
    :param cmap_name:
    :return:
    """

    unique_values = np.unique(feature)
    feature, unique_values = VegvisirUtils.replace_nan(feature, unique_values)
    colormap_unique = matplotlib.cm.get_cmap(cmap_name, len(unique_values.tolist()))

    colors_dict = dict(zip(unique_values, colormap_unique.colors))
    colors_feature = np.vectorize(colors_dict.get, signature='()->(n)')(feature)

    return PlotSettings(values=feature,
                        colormap_unique=colormap_unique,
                        colors_feature=colors_feature,
                        unique_values=unique_values)

def plot_preprocessing(umap_proj:np.ndarray,dataset_info:namedtuple,predictions_dict:dict,sample_mode:str,results_dir:str,method:str,vector_name:str="latent_space_z",n_clusters:Union[int,float]=4):
    """

    :param umap_proj:
    :param dataset_info:
    :param predictions_dict:
    :param sample_mode:
    :param results_dir:
    :param method:
    :param vector_name:
    :param n_clusters:
    :return:
    """
    print("Preparing UMAP plots of {}...".format(vector_name))

    cluster_assignments = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=100, max_iter=10,
                                          reassignment_ratio=0, n_init="auto").fit_predict(umap_proj)
    colors_clusters = np.vectorize(colors_cluster_dict.get)(cluster_assignments)

    # gravy_scores,volume_scores,radius_scores,side_chain_pka_scores,isoelectric_scores,aromaticity_scores,sequences_lens,clusters_info=plot_clusters_features_distributions(dataset_info,cluster_assignments,n_clusters,predictions_dict,sample_mode,results_dir,method)
    features_dict,sequences_raw = plot_clusters_features_distributions(dataset_info, cluster_assignments, n_clusters,
                                                         predictions_dict, sample_mode, results_dir, method,
                                                         vector_name)

    sequences_lens_settings = define_colormap(features_dict["sequences_lens"],cmap_name="viridis")
    gravy_scores_settings = define_colormap(features_dict["gravy_scores"],cmap_name="viridis")
    volume_scores_settings = define_colormap(features_dict["volume_scores"],cmap_name="viridis")
    radius_scores_settings = define_colormap(features_dict["radius_scores"],cmap_name="viridis")
    side_chain_pka_settings = define_colormap(features_dict["side_chain_pka_scores"],cmap_name="magma")
    isoelectric_scores_settings = define_colormap(features_dict["isoelectric_scores"],cmap_name="magma")
    aromaticity_scores_settings = define_colormap(features_dict["aromaticity_scores"],cmap_name="cividis")
    molecular_weights_settings = define_colormap(features_dict["molecular_weights"],cmap_name="cividis")
    bulkiness_scores_settings = define_colormap(features_dict["bulkiness_scores"],cmap_name="magma")
    extintion_coefficients_settings = define_colormap(features_dict["extintion_coefficients"],cmap_name="magma")
    tryptophan_coefficients_settings = define_colormap(features_dict["tryptophan"],cmap_name="magma")


    return {"features_dict":features_dict,
            "sequences_raw":sequences_raw,
            "cluster_assignments":cluster_assignments,
            "colors_clusters":colors_clusters,
            "sequence_lens_settings":sequences_lens_settings,
            "gravy_scores_settings":gravy_scores_settings,
            "volume_scores_settings":volume_scores_settings,
            "radius_scores_settings":radius_scores_settings,
            "side_chain_pka_settings":side_chain_pka_settings,
            "isoelectric_scores_settings":isoelectric_scores_settings,
            "aromaticity_scores_settings":aromaticity_scores_settings,
            "molecular_weights_settings":molecular_weights_settings,
            "bulkiness_scores_settings":bulkiness_scores_settings,
            "extintion_coefficients_settings":extintion_coefficients_settings,
            "tryptophan_coefficients_settings":tryptophan_coefficients_settings
            }

def plot_scatter(umap_proj:np.ndarray,dataset_info:namedtuple,latent_space:np.ndarray,predictions_dict:dict,sample_mode:str,results_dir:str,method:str,settings:dict,vector_name:str="latent_space_z",n_clusters:Union[int,float]=4):
    """

    :param umap_proj:
    :param dataset_info:
    :param latent_space:
    :param predictions_dict:
    :param sample_mode:
    :param results_dir:
    :param method:
    :param settings:
    :param vector_name:
    :param n_clusters:
    :return:
    """
    print("Plotting scatter UMAP of {}...".format(vector_name))

    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:, 0])
    if sample_mode== "single_sample":
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_dict["class_binary_predictions_single_sample"])
    else:
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_dict["class_binary_predictions_samples_mode"])
        #colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_dict["class_binary_predictions_samples_logits_average_argmax"])

    # Highlight: Confidence scores colors
    confidence_scores = latent_space[:, 4]
    confidence_scores_unique = np.unique(confidence_scores).tolist()
    colormap_confidence = matplotlib.cm.get_cmap('plasma_r', len(confidence_scores_unique))
    colors_dict = dict(zip(confidence_scores_unique, colormap_confidence.colors))

    colors_confidence = np.vectorize(colors_dict.get, signature='()->(n)')(confidence_scores)
    # Highlight: Immunodominance scores colors
    immunodominance_scores = latent_space[:, 3]
    immunodominance_scores_unique = np.unique(immunodominance_scores)
    immunodominance_scores, immunodominance_scores_unique = VegvisirUtils.replace_nan(immunodominance_scores,
                                                                                      immunodominance_scores_unique)
    colormap_immunodominance = matplotlib.cm.get_cmap('plasma_r', len(immunodominance_scores_unique.tolist()))
    colors_dict = dict(zip(immunodominance_scores_unique, colormap_immunodominance.colors))
    colors_immunodominance = np.vectorize(colors_dict.get, signature='()->(n)')(immunodominance_scores)
    # Highlight: Frequency scores per class: https://stackoverflow.com/questions/65927253/linearsegmentedcolormap-to-list
    frequency_class0_unique = np.unique(predictions_dict["class_binary_predictions_samples_frequencies"][:, 0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn',len(frequency_class0_unique))  # This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array = np.array([colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_predictions_samples_frequencies"][:, 0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_predictions_samples_frequencies"][:, 1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array = np.array([colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(predictions_dict["class_binary_predictions_samples_frequencies"][:, 1])
    alpha = 0.7
    size = 5
    fig, [[ax1, ax2, ax3, ax4],
          [ax5, ax6, ax7, ax8],
          [ax9, ax10, ax11, ax12],
          [ax13, ax14, ax15, ax16],
          [ax17, ax18, ax19, ax20],
          [ax21,ax22,ax23,ax24]] = plt.subplots(6, 4, figsize=(20, 15),
                                                   gridspec_kw={'width_ratios': [4.5, 4.5, 4.5, 1],
                                                                'height_ratios': [4, 4, 4, 4, 4,4]})
    fig.suptitle('UMAP projections', fontsize=20)
    #sns.kdeplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax1, cmap="Blues", n_levels=30, fill=True, thresh=0.05,alpha=0.5)  # cmap='Blues'
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=latent_space[:, 0], alpha=alpha, s=size)

    ax1.set_title("Binary targets", fontsize=20)
    if sample_mode == "single_sample":
        #sns.kdeplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax2, cmap="Blues", n_levels=30, fill=True,thresh=0.05, alpha=0.5)  # cmap='Blues'
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted_binary, alpha=alpha, s=size)
        ax2.set_title("Predicted binary targets \n (single sample)", fontsize=20)
    else:
        #sns.kdeplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax2, cmap="Blues", n_levels=30, fill=True,thresh=0.05, alpha=0.5)  # cmap='Blues'
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted_binary, alpha=alpha, s=size)
        ax2.set_title("Predicted binary targets \n (samples mode)", fontsize=20)

    ax3.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_confidence, alpha=alpha, s=size)
    ax3.set_title("Confidence scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence), ax=ax3)
    ax5.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_frequency_class0, alpha=alpha, s=size)
    ax5.set_title("Probability class 0 \n (frequency argmax)", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap_frequency_class0), ax=ax5)
    ax6.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_frequency_class1, alpha=alpha, s=size)
    ax6.set_title("Probability class 1 \n (frequency argmax)", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_frequency_class1), ax=ax6)
    ax7.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_immunodominance, alpha=alpha, s=size)
    ax7.set_title("Immunodominance scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_immunodominance), ax=ax7)

    ax9.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["colors_clusters"], alpha=alpha, s=size)
    ax9.set_title("Coloured by Kmeans cluster")

    ax10.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["gravy_scores_settings"].colors_feature, alpha=alpha, s=size)
    ax10.set_title("Coloured by Gravy")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["gravy_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["gravy_scores_settings"].unique_values),
                                                      vmax=np.max(settings["gravy_scores_settings"].unique_values))), ax=ax10)

    ax11.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["volume_scores_settings"].colors_feature, alpha=alpha, s=size)
    ax11.set_title("Coloured by Peptide volume")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["volume_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["volume_scores_settings"].unique_values),
                                                     vmax=np.max(settings["volume_scores_settings"].unique_values))),
                 ax=ax11)

    ax13.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["side_chain_pka_settings"].colors_feature, alpha=alpha, s=size)
    ax13.set_title("Coloured by Side chain pka")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["side_chain_pka_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["side_chain_pka_settings"].unique_values),
                                                      vmax=np.max(settings["side_chain_pka_settings"].unique_values))), ax=ax13)

    ax14.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["isoelectric_scores_settings"].colors_feature, alpha=alpha, s=size)
    ax14.set_title("Coloured by Isoelectric point")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["isoelectric_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["isoelectric_scores_settings"].unique_values),
                                                      vmax=np.max(settings["isoelectric_scores_settings"].unique_values))), ax=ax14)

    ax15.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["aromaticity_scores_settings"].colors_feature, alpha=alpha, s=size)
    ax15.set_title("Coloured by aromaticity")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["isoelectric_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["isoelectric_scores_settings"].unique_values),
                                                      vmax=np.max(settings["isoelectric_scores_settings"].unique_values))), ax=ax15)

    ax17.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["sequence_lens_settings"].colors_feature, alpha=alpha, s=size)
    ax17.set_title("Coloured by sequence len")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["sequence_lens_settings"].colormap_unique, norm=Normalize(vmin=np.min(settings["sequence_lens_settings"].unique_values),
                                                                         vmax=np.max(settings["sequence_lens_settings"].unique_values))),ax=ax17)

    ax18.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["molecular_weights_settings"].colors_feature, alpha=alpha, s=size)
    ax18.set_title("Coloured by molecular weight")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["molecular_weights_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["molecular_weights_settings"].unique_values),
                                                      vmax=np.max(settings["molecular_weights_settings"].unique_values))), ax=ax18)

    ax19.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["bulkiness_scores_settings"].colors_feature, alpha=alpha, s=size)
    ax19.set_title("Coloured by bulkiness")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["bulkiness_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["bulkiness_scores_settings"].unique_values),
                                                      vmax=np.max(settings["bulkiness_scores_settings"].unique_values))), ax=ax19)

    ax21.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["extintion_coefficients_settings"].colors_feature, alpha=alpha, s=size)
    ax21.set_title("Coloured by extintion_coefficient")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["extintion_coefficients_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["extintion_coefficients_settings"].unique_values),
                                                      vmax=np.max(settings["extintion_coefficients_settings"].unique_values))), ax=ax21)


    ax4.axis("off")
    ax8.axis("off")
    ax12.axis("off")
    ax16.axis("off")
    ax20.axis("off")
    ax22.axis("off")
    ax23.axis("off")
    ax24.axis("off")


    fig.suptitle("UMAP of {}".format(vector_name))

    negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
    fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
    plt.legend(handles=[negative_patch, positive_patch], prop={'size': 20}, loc='center right',
               bbox_to_anchor=(1.5, 2.5), ncol=1)
    plt.savefig("{}/{}/umap_SCATTER_{}_{}".format(results_dir, method, vector_name, sample_mode),dpi=500)
    plt.clf()
    plt.close(fig)

    #gc.collect()

def colorbar(mappable:matplotlib.colorbar):
    """Places a figure color bar without squeezing the plot"""
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def plot_scatter_reduced(umap_proj:np.ndarray,args:namedtuple,dataset_info:namedtuple,latent_space:np.ndarray,predictions_dict:dict,sample_mode:str,results_dir:str,method:str,settings:dict,vector_name:str="latent_space_z",n_clusters:Union[int,float]=4):
    print("Plotting (reduced) scatter UMAP of {}...".format(vector_name))
    storage_folder = os.path.dirname(os.path.abspath(__file__)) + "/data"

    title_dict = {"latent_space_z": "Latent representation (z)",
                  "encoder_final_hidden_state":"Encoder Hf",
                  "decoder_final_hidden_state": "Decoder Hf"}
    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:, 0])
    if sample_mode == "single_sample":
        predictions_binary = predictions_dict["class_binary_predictions_single_sample"]
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_binary)
    else:
        predictions_binary = predictions_dict["class_binary_predictions_samples_mode"]
        #predictions_binary = predictions_dict["class_binary_predictions_samples_logits_average_argmax"]
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_binary)

    #epitopes_df = VegvisirUtils.save_results_table(predictions_dict, latent_space, args, dataset_info, results_dir,method=method, merge_netmhc=True, save_df=False)

    dataframe = pd.DataFrame({"UMAP_x":umap_proj[:,0],
                              "UMAP_y": umap_proj[:, 1],
                              "Binary targets":latent_space[:, 0],
                              "predictions_binary":predictions_binary,
                              "alleles":latent_space[:,5],
                              "Immunoprevalence":latent_space[:, 3],
                              "frequency_0":predictions_dict["class_binary_predictions_samples_frequencies"][:, 0],
                              "frequency_1":predictions_dict["class_binary_predictions_samples_frequencies"][:, 1],
                              "Sequence_lens":settings["sequence_lens_settings"].values,
                             })



    # Highlight: Immunodominance scores colors
    immunodominance_scores = latent_space[:, 3]
    immunodominance_scores_unique = np.unique(immunodominance_scores)
    immunodominance_scores, immunodominance_scores_unique = VegvisirUtils.replace_nan(immunodominance_scores,
                                                                                      immunodominance_scores_unique)
    colormap_immunodominance = matplotlib.cm.get_cmap('plasma_r', len(immunodominance_scores_unique.tolist()))
    colors_dict = dict(zip(immunodominance_scores_unique, colormap_immunodominance.colors))
    colors_immunodominance = np.vectorize(colors_dict.get, signature='()->(n)')(immunodominance_scores)
    # Highlight: Frequency scores per class: https://stackoverflow.com/questions/65927253/linearsegmentedcolormap-to-list
    frequency_class0_unique = np.unique(predictions_dict["class_binary_predictions_samples_frequencies"][:, 0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn',len(frequency_class0_unique))  # This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array = np.array([colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(predictions_dict["class_binary_predictions_samples_frequencies"][:, 0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_predictions_samples_frequencies"][:, 1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array = np.array([colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(predictions_dict["class_binary_predictions_samples_frequencies"][:, 1])

    plot_types = {0:"immunoprevalence",1:"alleles",2:"sequence_lengths",3:""}
    plot_type = plot_types[1]
    alpha = 0.5
    size = 4
    fontsize= 18
    plot_kde=False

    # Highlight: Scatter and density plot
    colors_dict_labels_updated = colors_dict_labels.copy()
    hue_order = [0,1,2]
    if args.num_obs_classes == args.num_classes:
        del colors_dict_labels_updated[2]
        hue_order = [0,1]
    colors_palette = list(colors_dict_labels_updated.values())
    g0 = sns.jointplot(data=dataframe, x="UMAP_x", y="UMAP_y", hue="Binary targets", alpha=alpha, s=8,
                        palette=colors_palette,
                        hue_order=hue_order
                        )
    g0.ax_joint.legend(fancybox=True, framealpha=0.5)
    g0.ax_joint.axis("off")
    g0.ax_marg_x.axis("off")
    g0.ax_marg_y.axis("off")

    #Highlight: Predictions scatter plot
    g1 = sns.FacetGrid(dataframe,hue="predictions_binary", subplot_kws={"fc": "white"}, margin_titles=True,
    palette = colors_palette,
    hue_order = [0, 1]
    )
    g1.set(yticks=[])
    g1.set(xticks=[])
    g1_axes = g1.axes.flatten()
    g1_axes[0].set_title("Predicted targets",fontsize=fontsize)
    g1.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
    g1_axes[0].set_xlabel("")
    g1_axes[0].set_ylabel("")

    if plot_type == "immunoprevalence":
        #Highlight: Immunodominace scatter plot


        g2 = sns.FacetGrid(dataframe, hue="Immunoprevalence", subplot_kws={"fc": "white"},
                           palette=colormap_immunodominance.colors,
                           hue_order=immunodominance_scores_unique)
        g2.set(yticks=[])
        g2.set(xticks=[])
        g2_axes = g2.axes.flatten()
        g2_axes[0].set_title("Immunoprevalence",fontsize=fontsize)
        g2.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
        g2_axes[0].set_xlabel("")
        g2_axes[0].set_ylabel("")

    elif plot_type == "alleles":
        #read in the groups of alleles
        alleles_cluster_colors = pd.read_csv(open('{}/common_files/allele_colors2.tsv'.format(storage_folder)),sep="\t")
        alleles_cluster_colors.allele = alleles_cluster_colors.allele.str.replace(":","")
        max_cluster = str(len(pd.unique(alleles_cluster_colors.cluster1)) + 1)
        alleles_cluster_grouped = alleles_cluster_colors.groupby('cluster1', as_index=False)[["cluster1","allele","color1"]].agg(lambda x: list(x)[0])
        alleles_colors = alleles_cluster_grouped.color1


        alleles_labels_dict = dict(zip(alleles_cluster_grouped.cluster1.tolist(),alleles_cluster_grouped["allele"].tolist()))
        alleles_labels_dict[int(max_cluster)] = "Others"
        #alleles_cluster_colors = alleles_cluster_colors.groupby('cluster1', as_index=False)[["allele"]].agg(lambda x: list(x))
        alleles_cluster_colors_dict = dict(zip( alleles_cluster_colors.allele,alleles_cluster_colors.cluster1))
        alleles_encoded_dict = json.load(open('{}/common_files/alleles_dict.txt'.format(storage_folder),"r+"))
        alleles_encoded_dict_reversed = {val:key for key,val in alleles_encoded_dict.items()}
        dataframe["alleles"] = dataframe["alleles"].astype(int).astype(str) #if this does not work, is because your alleles contain nan values
        dataframe = dataframe.replace({"alleles":alleles_encoded_dict})
        #dataframe = dataframe.replace({"alleles":alleles_cluster_colors_dict}) #returns the original value if not found in dict
        dataframe["alleles"] = dataframe["alleles"].map(alleles_cluster_colors_dict) #returns nan if not found in dict
        dataframe["alleles"] = dataframe["alleles"].fillna(max_cluster).astype(int)
        # Highlight: Alleles colors
        alleles_settings = define_colormap(dataframe["alleles"].tolist(), "tab20")

        alleles_custom_cmap = LinearSegmentedColormap.from_list(f"alleles_cmap_{vector_name}_{sample_mode}_{method}_list", alleles_colors)

        #matplotlib.cm.register("alleles_cmap", alleles_custom_cmap,override_builtin=True) #deprecated
        print(f"registering: alleles_cmap_{vector_name}_{sample_mode}_{method}")
        matplotlib.colormaps.register(name = f"alleles_cmap_{vector_name}_{sample_mode}_{method}",cmap = alleles_custom_cmap)
        #alleles_palette = sns.color_palette("alleles_cmap", n_colors=len(alleles_colors), desat=0)
        alleles_palette = sns.color_palette(f"alleles_cmap_{vector_name}_{sample_mode}_{method}", n_colors=len(alleles_colors), desat=0)
        alleles_palette_hue = pd.unique(alleles_cluster_grouped["allele"].map(alleles_encoded_dict_reversed)) #TODO: There is a nan allele, so that it follows the same order as the allele_colors.tsv--> needs to be mapped to 0,1,--

        if plot_kde:
            g2 = sns.jointplot(data=dataframe,
                               x="UMAP_x", y="UMAP_y",
                               hue="alleles",
                               alpha=alpha, s=8,
                               kind="scatter",
                               palette=alleles_settings.colormap_unique.colors,
                               hue_order=alleles_settings.unique_values,
                               legend=False
                               )
            # g2.ax_joint.legend(fancybox=True, framealpha=0.5)
            g2.ax_joint.axis("off")
            g2.ax_marg_x.axis("off")
            g2.ax_marg_y.axis("off")
        else:
            g2 = sns.FacetGrid(dataframe, hue="alleles", subplot_kws={"fc": "white"},
                               palette=alleles_settings.colormap_unique.colors,
                               #palette=alleles_palette,
                               hue_order=alleles_settings.unique_values,
                               #hue_order=alleles_palette_hue,
                               )
            g2.set(yticks=[])
            g2.set(xticks=[])
            g2_axes = g2.axes.flatten()
            g2_axes[0].set_title("Alleles", fontsize=fontsize)
            g2.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
            g2_axes[0].set_xlabel("")
            g2_axes[0].set_ylabel("")



    elif plot_type == "seq_lens":
        #Highlight: Sequences lens plots

        g2 = sns.FacetGrid(dataframe, hue="Sequence_lens", subplot_kws={"fc": "white"},
                           palette=settings["sequence_lens_settings"].colormap_unique.colors,
                           hue_order=settings["sequence_lens_settings"].unique_values)
        g2.set(yticks=[])
        g2.set(xticks=[])
        g2_axes = g2.axes.flatten()
        g2_axes[0].set_title("Sequence lengths", fontsize=fontsize)
        g2.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
        g2_axes[0].set_xlabel("")
        g2_axes[0].set_ylabel("")
    else: pass

    #Highlight: Frequencies plots

    g3 = sns.FacetGrid(dataframe, hue="frequency_0", subplot_kws={"fc": "white"},
                       palette=colormap_frequency_class0_array,
                       hue_order=frequency_class0_unique
                       )
    g3.set(yticks=[])
    g3.set(xticks=[])
    g3_axes = g3.axes.flatten()
    g3_axes[0].set_title("Posterior predictive (class 0)",fontsize=fontsize)
    g3.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
    g3_axes[0].set_xlabel("")
    g3_axes[0].set_ylabel("")

    g4 = sns.FacetGrid(dataframe, hue="frequency_1", subplot_kws={"fc": "white"},
                       palette=colormap_frequency_class1_array,
                       hue_order=frequency_class1_unique
                       )
    g4.set(yticks=[])
    g4.set(xticks=[])
    g4_axes = g4.axes.flatten()
    g4_axes[0].set_title("Posterior predictive (class 1)",fontsize=fontsize)
    g4.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
    g4_axes[0].set_xlabel("")
    g4_axes[0].set_ylabel("")

    fig = plt.figure(figsize=(17, 8))
    gs = gridspec.GridSpec(2, 6, width_ratios=[2, 1, 0.1, 0.07, 1, 0.1])

    mg0 = VegvisirUtils.SeabornFig2Grid(g0, fig, gs[0:2, 0])
    mg1 = VegvisirUtils.SeabornFig2Grid(g1, fig, gs[0, 1])
    mg2 = VegvisirUtils.SeabornFig2Grid(g2, fig, gs[0, 4])
    mg3 = VegvisirUtils.SeabornFig2Grid(g3, fig, gs[1, 1])
    mg4 = VegvisirUtils.SeabornFig2Grid(g4, fig, gs[1, 4])

    gs.update(top=0.9)
    #gs.update(right=0.4)
    # Following: https://www.sc.eso.org/~bdias/pycoffee/codes/20160407/gridspec_demo.html
    cbax2 = plt.subplot(gs[0, 5])  # Place it where it should be.
    cbax3 = plt.subplot(gs[1, 2])  # Place it where it should be.
    cbax4 = plt.subplot(gs[1, 5])  # Place it where it should be.

    if plot_type == "immunoprevalence":
        cb2 = Colorbar(ax=cbax2, mappable=plt.cm.ScalarMappable(cmap=colormap_immunodominance))
    elif plot_type == "sequence_lens":
        unique_lens = settings["sequence_lens_settings"].unique_values
        cb2 = Colorbar(ax=cbax2, mappable=plt.cm.ScalarMappable(norm=BoundaryNorm(unique_lens, len(unique_lens)),cmap=settings["sequence_lens_settings"].colormap_unique),ticks=unique_lens, boundaries=unique_lens)
        cb2.set_ticks(unique_lens)
    elif plot_type == "alleles":
        unique_alleles = alleles_settings.unique_values
        alleles_labels = [alleles_labels_dict[group] for group in unique_alleles]
        unique_alleles = unique_alleles.tolist() + [max(unique_alleles) + 1]
        colormap_unique = matplotlib.cm.get_cmap("tab20", len(unique_alleles))
        #colormap_unique = matplotlib.cm.get_cmap("alleles_cmap", len(unique_alleles))
        cb2 = Colorbar(ax=cbax2,
                       mappable=plt.cm.ScalarMappable(norm=BoundaryNorm(unique_alleles, len(unique_alleles)), cmap=colormap_unique),
                       boundaries=unique_alleles)

        #ticks_locations = np.arange(0.5, len(unique_alleles), 1)
        #ticks_locations = cb2.get_ticks()
        # ticks_locations = np.array(unique_alleles) + 0.5
        # ticks_locations = ticks_locations
        #cb2.set_ticks(np.array(ticks_locations) - 0.5 )
        cb2.set_ticklabels(alleles_labels + [""])
        cb2.ax.tick_params(size=0)  # set size of the ticks to 0

    cb3 = Colorbar(ax=cbax3, mappable=plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap_frequency_class0))
    cb4 = Colorbar(ax=cbax4, mappable=plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap_frequency_class1))

    fig.suptitle("UMAP latent space (z) projections",fontsize=fontsize)

    plt.savefig("{}/{}/umap_SCATTER_reduced_{}_{}".format(results_dir, method, vector_name, sample_mode))
    plt.clf()
    plt.close(fig)

    #gc.collect()

def plot_scatter_quantiles(umap_proj:np.ndarray,dataset_info:namedtuple,latent_space:np.ndarray,predictions_dict:dict,sample_mode:str,results_dir:str,method:str,settings:dict,vector_name:str="latent_space_z",n_clusters:Union[int,float]=4):
    print("Plotting scatter (quantiles) UMAP of {}...".format(vector_name))

    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:, 0])
    if sample_mode == "single_sample":
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(
            predictions_dict["class_binary_predictions_single_sample"])
    else:
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(
            predictions_dict["class_binary_predictions_samples_mode"])
    features_dict = settings["features_dict"]
    # Highlight: Confidence scores colors
    confidence_scores = latent_space[:, 4]
    confidence_scores_unique = np.unique(confidence_scores).tolist()
    colormap_confidence = matplotlib.cm.get_cmap('plasma_r', len(confidence_scores_unique))
    colors_dict = dict(zip(confidence_scores_unique, colormap_confidence.colors))
    colors_confidence = np.vectorize(colors_dict.get, signature='()->(n)')(confidence_scores)
    # Highlight: Immunodominance scores colors
    immunodominance_scores = latent_space[:, 3]
    immunodominance_scores_unique = np.unique(immunodominance_scores)
    immunodominance_scores, immunodominance_scores_unique = VegvisirUtils.replace_nan(immunodominance_scores,
                                                                                      immunodominance_scores_unique)
    colormap_immunodominance = matplotlib.cm.get_cmap('plasma_r', len(immunodominance_scores_unique.tolist()))
    colors_dict = dict(zip(immunodominance_scores_unique, colormap_immunodominance.colors))
    colors_immunodominance = np.vectorize(colors_dict.get, signature='()->(n)')(immunodominance_scores)
    # Highlight: Frequency scores per class: https://stackoverflow.com/questions/65927253/linearsegmentedcolormap-to-list
    frequency_class0_unique = np.unique(predictions_dict["class_binary_predictions_samples_frequencies"][:, 0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn',
                                                       len(frequency_class0_unique))  # This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array = np.array(
        [colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_predictions_samples_frequencies"][:, 0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_predictions_samples_frequencies"][:, 1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array = np.array(
        [colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_predictions_samples_frequencies"][:, 1])
    alpha = 0.7
    size = 5
    fig, [[ax1, ax2, ax3, ax4],
          [ax5, ax6, ax7, ax8],
          [ax9, ax10, ax11, ax12],
          [ax13, ax14, ax15, ax16],
          [ax17, ax18, ax19, ax20],
          [ax21, ax22, ax23, ax24]] = plt.subplots(6, 4, figsize=(20, 15),
                                                   gridspec_kw={'width_ratios': [4.5, 4.5, 4.5, 1],
                                                                'height_ratios': [4, 4, 4, 4, 4, 4]})
    fig.suptitle('UMAP projections', fontsize=20)
    #sns.kdeplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax1, cmap="Blues", n_levels=30, fill=True, thresh=0.05,alpha=0.5)  # cmap='Blues'
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=latent_space[:, 2], alpha=alpha, s=size)

    ax1.set_title("True labels", fontsize=20)
    if sample_mode == "single_sample":
        #sns.kdeplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax2, cmap="Blues", n_levels=30, fill=True,thresh=0.05, alpha=0.5)  # cmap='Blues'
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted_binary, alpha=alpha, s=size)
        ax2.set_title("Predicted labels \n (single sample)", fontsize=20)
    else:
        #sns.kdeplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax2, cmap="Blues", n_levels=30, fill=True,thresh=0.05, alpha=0.5)  # cmap='Blues'
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted_binary, alpha=alpha, s=size)
        ax2.set_title("Predicted binary labels \n (samples mode)", fontsize=20)

    ax3.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_confidence, alpha=alpha, s=size)
    ax3.set_title("Confidence scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence), ax=ax3)
    ax5.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_frequency_class0, alpha=alpha, s=size)
    ax5.set_title("Probability class 0 \n (frequency argmax)", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap_frequency_class0), ax=ax5)
    ax6.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_frequency_class1, alpha=alpha, s=size)
    ax6.set_title("Probability class 1 \n (frequency argmax)", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_frequency_class1), ax=ax6)
    ax7.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_immunodominance, alpha=alpha, s=size)
    ax7.set_title("Immunodominance scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_immunodominance), ax=ax7)

    ax9.scatter(umap_proj[:, 0], umap_proj[:, 1], c=settings["colors_clusters"], alpha=alpha, s=size)
    ax9.set_title("Coloured by Kmeans cluster")
    quantiles = [0.1,0.9]
    quantiles_gravy = np.quantile(features_dict["gravy_scores"],quantiles)

    quantile_gravy_idx = (features_dict["gravy_scores"][...,None] < quantiles_gravy[0]).any(-1), (features_dict["gravy_scores"][...,None] > quantiles_gravy[1]).any(-1)
    gravy_colors = settings["gravy_scores_settings"].colors_feature
    gravy_colors = gravy_colors[quantile_gravy_idx[0] | quantile_gravy_idx[1]]
    gravy_umap = umap_proj[quantile_gravy_idx[0] | quantile_gravy_idx[1]]

    ax10.scatter(gravy_umap[:,0],gravy_umap[:,1], c=gravy_colors, alpha=alpha, s=size)
    ax10.set_title("Coloured by Gravy")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["gravy_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["gravy_scores_settings"].unique_values),
                                                      vmax=np.max(settings["gravy_scores_settings"].unique_values))), ax=ax10)


    
    quantiles_volume = np.quantile(features_dict["volume_scores"],quantiles)
    quantile_volume_idx = (features_dict["volume_scores"][...,None] < quantiles_volume[0]).any(-1), (features_dict["volume_scores"][...,None] > quantiles_volume[1]).any(-1)
    volume_colors = settings["volume_scores_settings"].colors_feature
    volume_colors = volume_colors[quantile_volume_idx[0] | quantile_volume_idx[1]]
    volume_umap = umap_proj[quantile_volume_idx[0] | quantile_volume_idx[1]]
    ax11.scatter(volume_umap[:, 0], volume_umap[:, 1], c=volume_colors, alpha=alpha, s=size)
    ax11.set_title("Coloured by Peptide volume")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["volume_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["volume_scores_settings"].unique_values),
                                                     vmax=np.max(settings["volume_scores_settings"].unique_values))),ax=ax11)
    
    
    
    
    quantiles_side_chain_pka = np.quantile(features_dict["side_chain_pka_scores"],quantiles)
    quantile_side_chain_pka_idx = (features_dict["side_chain_pka_scores"][...,None] < quantiles_side_chain_pka[0]).any(-1), (features_dict["side_chain_pka_scores"][...,None] > quantiles_side_chain_pka[1]).any(-1)
    side_chain_pka_colors = settings["side_chain_pka_settings"].colors_feature
    side_chain_pka_colors = side_chain_pka_colors[quantile_side_chain_pka_idx[0] | quantile_side_chain_pka_idx[1]]
    side_chain_pka_umap = umap_proj[quantile_side_chain_pka_idx[0] | quantile_side_chain_pka_idx[1]]

    ax13.scatter(side_chain_pka_umap[:, 0], side_chain_pka_umap[:, 1], c=side_chain_pka_colors, alpha=alpha, s=size)
    ax13.set_title("Coloured by Side chain pka")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["side_chain_pka_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["side_chain_pka_settings"].unique_values),
                                                      vmax=np.max(settings["side_chain_pka_settings"].unique_values))), ax=ax13)

    quantiles_isoelectric = np.quantile(features_dict["isoelectric_scores"], quantiles)
    quantile_isoelectric_idx = (features_dict["isoelectric_scores"][..., None] < quantiles_isoelectric[0]).any(-1), (
                features_dict["isoelectric_scores"][..., None] > quantiles_isoelectric[1]).any(-1)
    isoelectric_colors = settings["isoelectric_scores_settings"].colors_feature
    isoelectric_colors = isoelectric_colors[quantile_isoelectric_idx[0] | quantile_isoelectric_idx[1]]
    isoelectric_umap = umap_proj[quantile_isoelectric_idx[0] | quantile_isoelectric_idx[1]]

    ax14.scatter(isoelectric_umap[: , 0], isoelectric_umap[: , 1], c=isoelectric_colors, alpha=alpha, s=size)
    ax14.set_title("Coloured by Isoelectric point")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["isoelectric_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["isoelectric_scores_settings"].unique_values),
                                                      vmax=np.max(settings["isoelectric_scores_settings"].unique_values))), ax=ax14)
    
    quantiles_aromaticity = np.quantile(features_dict["aromaticity_scores"],quantiles)
    quantile_aromaticity_idx = (features_dict["aromaticity_scores"][...,None] < quantiles_aromaticity[0]).any(-1), (features_dict["aromaticity_scores"][...,None] > quantiles_aromaticity[1]).any(-1)
    aromaticity_colors = settings["aromaticity_scores_settings"].colors_feature
    aromaticity_colors = aromaticity_colors[quantile_aromaticity_idx[0] | quantile_aromaticity_idx[1]]
    aromaticity_umap = umap_proj[quantile_aromaticity_idx[0] | quantile_aromaticity_idx[1]]

    ax15.scatter(aromaticity_umap[:, 0], aromaticity_umap[:, 1], c=aromaticity_colors, alpha=alpha, s=size)
    ax15.set_title("Coloured by aromaticity")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["isoelectric_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["isoelectric_scores_settings"].unique_values),
                                                      vmax=np.max(settings["isoelectric_scores_settings"].unique_values))), ax=ax15)

    quantiles_sequence_lens = np.quantile(features_dict["sequences_lens"], quantiles)
    quantile_sequence_lens_idx = (features_dict["sequences_lens"][..., None] < quantiles_sequence_lens[0]).any(-1), (
                features_dict["sequences_lens"][..., None] > quantiles_sequence_lens[1]).any(-1)
    sequence_lens_colors = settings["sequence_lens_settings"].colors_feature
    sequence_lens_colors = sequence_lens_colors[quantile_sequence_lens_idx[0] | quantile_sequence_lens_idx[1]]
    sequence_lens_umap = umap_proj[quantile_sequence_lens_idx[0] | quantile_sequence_lens_idx[1]]

    ax17.scatter(sequence_lens_umap[:, 0], sequence_lens_umap[:, 1], c=sequence_lens_colors, alpha=alpha, s=size)
    ax17.set_title("Coloured by sequence len")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["sequence_lens_settings"].colormap_unique, norm=Normalize(vmin=np.min(settings["sequence_lens_settings"].unique_values),
                                                                         vmax=np.max(settings["sequence_lens_settings"].unique_values))),ax=ax17)
    
    quantiles_molecular_weights = np.quantile(features_dict["molecular_weights"],quantiles)
    quantile_molecular_weights_idx = (features_dict["molecular_weights"][...,None] < quantiles_molecular_weights[0]).any(-1), (features_dict["molecular_weights"][...,None] > quantiles_molecular_weights[1]).any(-1)
    molecular_weights_colors = settings["molecular_weights_settings"].colors_feature
    molecular_weights_colors = molecular_weights_colors[quantile_molecular_weights_idx[0] | quantile_molecular_weights_idx[1]]
    molecular_weights_umap = umap_proj[quantile_molecular_weights_idx[0] | quantile_molecular_weights_idx[1]]

    ax18.scatter(molecular_weights_umap[:, 0], molecular_weights_umap[:, 1], c=molecular_weights_colors, alpha=alpha, s=size)
    ax18.set_title("Coloured by molecular weight")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["molecular_weights_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["molecular_weights_settings"].unique_values),
                                                      vmax=np.max(settings["molecular_weights_settings"].unique_values))), ax=ax18)

    quantiles_bulkiness = np.quantile(features_dict["bulkiness_scores"], quantiles)
    quantile_bulkiness_idx = (features_dict["bulkiness_scores"][..., None] < quantiles_bulkiness[0]).any(-1), (
                features_dict["bulkiness_scores"][..., None] > quantiles_bulkiness[1]).any(-1)
    bulkiness_colors = settings["bulkiness_scores_settings"].colors_feature
    bulkiness_colors = bulkiness_colors[quantile_bulkiness_idx[0] | quantile_bulkiness_idx[1]]
    bulkiness_umap = umap_proj[quantile_bulkiness_idx[0] | quantile_bulkiness_idx[1]]

    ax19.scatter(bulkiness_umap[:, 0], bulkiness_umap[:, 1], c=bulkiness_colors, alpha=alpha, s=size)
    ax19.set_title("Coloured by bulkiness")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["bulkiness_scores_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["bulkiness_scores_settings"].unique_values),
                                                      vmax=np.max(settings["bulkiness_scores_settings"].unique_values))), ax=ax19)
    
    
    
    quantiles_extintion_coefficients = np.quantile(features_dict["extintion_coefficients"], quantiles)
    quantile_extintion_coefficients_idx = (features_dict["extintion_coefficients"][..., None] < quantiles_extintion_coefficients[0]).any(-1), (
                features_dict["extintion_coefficients"][..., None] > quantiles_extintion_coefficients[1]).any(-1)
    extintion_coefficients_colors = settings["extintion_coefficients_settings"].colors_feature
    extintion_coefficients_colors = extintion_coefficients_colors[quantile_extintion_coefficients_idx[0] | quantile_extintion_coefficients_idx[1]]
    extintion_coefficients_umap = umap_proj[quantile_extintion_coefficients_idx[0] | quantile_extintion_coefficients_idx[1]]
    ax21.scatter(extintion_coefficients_umap[:, 0], extintion_coefficients_umap[:, 1], c=extintion_coefficients_colors, alpha=alpha, s=size)
    ax21.set_title("Coloured by extintion_coefficient")
    fig.colorbar(plt.cm.ScalarMappable(cmap=settings["extintion_coefficients_settings"].colormap_unique,
                                       norm=Normalize(vmin=np.min(settings["extintion_coefficients_settings"].unique_values),
                                                      vmax=np.max(settings["extintion_coefficients_settings"].unique_values))), ax=ax21)



    ax4.axis("off")
    ax8.axis("off")
    ax12.axis("off")
    ax16.axis("off")
    ax20.axis("off")
    ax22.axis("off")
    ax23.axis("off")
    ax24.axis("off")
    fig.suptitle("UMAP of {}".format(vector_name))

    negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
    fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
    plt.legend(handles=[negative_patch, positive_patch], prop={'size': 20}, loc='center right',
               bbox_to_anchor=(1.5, 2.5), ncol=1)
    plt.savefig("{}/{}/umap_SCATTER_quantiles_{}_{}".format(results_dir, method, vector_name, sample_mode),dpi=500)
    plt.clf()
    plt.close(fig)

def plot_latent_correlations(umap_proj:np.ndarray,dataset_info:namedtuple,latent_space:np.ndarray,predictions_dict:dict,sample_mode:str,results_dir:str,method:str,vector_name:str):
    """
    :param umap_proj:
    :param dataset_info:
    :param latent_space:
    :param predictions_dict:
    :param sample_mode:
    :param results_dir:
    :param method:
    :param vector_name:
    :return:

    -Notes: https://medium.com/@outside2SDs/an-overview-of-correlation-measures-between-categorical-and-continuous-variables-4c7f85610365
    """
    print("Latent space distances")
    fig, [ax1,ax2,ax3]= plt.subplots(1, 3, figsize=(15, 12),dpi=100,gridspec_kw={'width_ratios': [4.5,4.5,1]})

    immunodominance_scores = latent_space[:,3]
    #TODO: Filter by high immunodominance?
    true_labels = latent_space[:,0]
    class_0_idx = (true_labels[...,None] == 0).any(-1)
    class_1_idx = (true_labels[...,None] == 1).any(-1)

    cosine_dist = 1- dromi.cosine_similarity(umap_proj[:,None], umap_proj[:,None], correlation_matrix=False, parallel=False).squeeze(-1).squeeze(-1)
    euclidean = np.linalg.norm(umap_proj[:,None]-umap_proj[None,:],axis=-1)

    def compute(distance,ax):
        average_sim_class_0_class_0 = np.mean(distance[class_0_idx][:,class_0_idx],axis=1)
        average_sim_class_0_class_1 = np.mean(distance[class_0_idx][:,class_1_idx],axis=1)
        average_sim_class_1_class_1 = np.mean(distance[class_1_idx][:,class_1_idx],axis=1)
        average_sim_class_1_class_0 = np.mean(distance[class_1_idx][:,class_0_idx],axis=1)

        all_distances = [average_sim_class_0_class_0,average_sim_class_0_class_1,average_sim_class_1_class_1,average_sim_class_1_class_0]
        all_labels = ["average_sim_class_0_class_0","average_sim_class_0_class_1","average_sim_class_1_class_1","average_sim_class_1_class_0"]
        boxplot = ax.boxplot(all_distances,
                               vert=True,  # vertical box alignment
                               patch_artist=True,  # fill with color
                               labels=all_labels
                               )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]
        ax.set_xticklabels(rotation=90,labels=all_labels)

        return boxplot

    boxplot1 = compute(cosine_dist,ax1)
    boxplot2 = compute(euclidean,ax2)

    all_colors = [colors_dict[0], "plum", colors_dict[1], "bisque"]
    for boxplot in (boxplot1,boxplot2):
        for patch, color in zip(boxplot['boxes'], all_colors):
            patch.set_facecolor(color)
    plt.savefig("{}/{}/Latent_quantitative_analysis_{}_{}".format(results_dir, method, vector_name, sample_mode),dpi=500)
    plt.clf()
    plt.close(fig)

def plot_latent_correlations_1d(umap_proj_1d:np.ndarray,args:namedtuple,settings:dict,dataset_info:namedtuple,latent_space:np.ndarray,sample_mode:str,results_dir:str,method:str,vector_name:str,plot_scatter_correlations:bool=False,calculate_covariance:bool=True,plot_covariance:bool=True,filter_correlations:bool=False):
    """
    :param umap_proj:
    :param dataset_info:
    :param latent_space:
    :param predictions_dict:
    :param sample_mode:
    :param results_dir:
    :param method:
    :param vector_name:
    :return:

    -Notes: https://medium.com/@outside2SDs/an-overview-of-correlation-measures-between-categorical-and-continuous-variables-4c7f85610365
    """
    print("Latent space correlations")

    features_dict = settings["features_dict"]
    features_dict.pop('clusters_info', None)
    #features_dict["immunodominance_scores"] = latent_space[:,3]
    true_labels = latent_space[:,0]
    features_dict["binary_targets"] = true_labels
    # #Highlight: Bring the pre-calculate peptide features back. PRESERVING THE ORDER OF THE SEQUENCES!
    compute_all_feats = False

    if (not args.shuffle_sequence) and (not args.random_sequences) and (not args.num_mutations != 0) and (args.sequence_type == "Icore"):
        if (args.num_classes == args.num_obs_classes):
            try:
                sequences_raw = settings["sequences_raw"]  # the sequences are following the order from the data loader
                sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), sequences_raw))
                sequences_raw = pd.DataFrame({"Icore": sequences_raw})
                all_feats = pd.read_csv("{}/common_files/dataset_all_features_with_test.tsv".format(dataset_info.storage_folder),sep="\t")
                peptide_feats_cols = all_feats.columns[(all_feats.columns.str.contains("Icore")) | (all_feats.columns.str.contains(pat = 'pep_'))]
                peptide_feats = all_feats[peptide_feats_cols]
                sequences_feats = VegvisirUtils.merge_in_left_order(sequences_raw, peptide_feats, "Icore")

                sequences_feats = sequences_feats.groupby('Icore', as_index=False, sort=False)[peptide_feats_cols[peptide_feats_cols != "Icore"]].agg(lambda x: sum(list(x)) / len(list(x))) #sort Falsse to not mess up the order in which the sequences come out from the model
                sequences_feats = sequences_feats[~sequences_feats[peptide_feats_cols[1]].isna()]
                sequences_feats = sequences_feats.to_dict(orient="list")
                sequences_feats.pop('Icore', None)

                #Highlight: Merge both features dict if all the features have been found for all the sequences
                if len(sequences_feats[peptide_feats_cols[1]]) == umap_proj_1d.shape[0]:
                    if not compute_all_feats:
                        print("Filtering features")
                        sequences_feats = {key: sequences_feats[key] for key in ["pep_secstruc_turn", "pep_secstruc_helix", "pep_secstruc_sheet"]}
                    features_dict = {**features_dict, **sequences_feats}
                else:
                    print("Could NOT find feature-information about all the sequences")

            except:
                print("Not all of the sequences are in the pre-computed features or some other error came up, skipping")
                pass

    #Highlight: The variable is named pearson but some are calculated via point biserial
    pearson_correlations = list(map(lambda feat1,feat2: VegvisirUtils.calculate_correlations(feat1, feat2),[umap_proj_1d]*len(features_dict.keys()),list(features_dict.values())))
    pearson_correlations = list(zip(*pearson_correlations))
    pearson_coefficients = np.array(pearson_correlations[0])
    pearson_coefficients = np.round(pearson_coefficients,2)
    pearson_pvalues = np.array(pearson_correlations[1])
    pearson_pvalues = np.round(pearson_pvalues,3)
    if filter_correlations:
        pearson_coef_idx = np.argwhere((pearson_coefficients >= 0.1) | (pearson_coefficients <-0.1))
        pearson_coefficients=pearson_coefficients[pearson_coef_idx]

    #spearman_correlations = list(map(lambda feat1,feat2: scipy.stats.stats.spearmanr(feat1,feat2),[umap_proj_1d]*len(features_dict.keys()),list(features_dict.values())))
    spearman_correlations = list(map(lambda feat1,feat2: VegvisirUtils.calculate_correlations(feat1,feat2,method="spearman"),[umap_proj_1d]*len(features_dict.keys()),list(features_dict.values())))
    spearman_correlations = list(zip(*spearman_correlations))

    spearman_coefficients = np.array(spearman_correlations[0])
    spearman_coefficients = np.round(spearman_coefficients,2)
    spearman_pvalues = np.array(spearman_correlations[1])
    spearman_pvalues = np.round(spearman_pvalues,3)
    #spearman_corr_idx = np.argwhere((spearman_correlations >= 0.1) | (spearman_correlations <-0.1))
    if filter_correlations:
        spearman_coefficients = spearman_coefficients[pearson_coef_idx]

    features_dict_new = defaultdict()
    for idx,(key,val) in enumerate(features_dict.items()):
        if filter_correlations:
          if idx in pearson_coef_idx:
            features_dict_new[key] = val
        else:
            features_dict_new[key] = val


    if features_dict_new:
        if plot_scatter_correlations:
            ncols = int(len(features_dict_new.keys())/2)
            nrows = [int(len(features_dict_new.keys())/ncols) + 1 if len(features_dict_new.keys())/ncols % 2 != 0 else int(len(features_dict_new.keys())/ncols)][0]
            fig, axs= plt.subplots(nrows, ncols, figsize=(40, 30),dpi=500)
            axs = axs.ravel()
            i= 0
            for feature,pearson_coef,pval,spearman_coef,color in zip(features_dict_new.keys(),pearson_coefficients,pearson_pvalues,spearman_coefficients,colors_list_aa):
                axs[i].scatter(umap_proj_1d,features_dict_new[feature],color=color,s=10)
                axs[i].set_title("{}: \n Pearson: {} \n Spearman: {}".format(feature,pearson_coef.item(),spearman_coef.item()),fontsize=22)
                axs[i].tick_params(axis='both', which='major', labelsize=15,width=0.1)
                axs[i].tick_params(axis='both', which='minor', labelsize=15,width=0.1)
                i += 1

            plt.margins(0.5)
            plt.subplots_adjust(bottom=0.15,hspace=0.25)
            plt.savefig("{}/{}/Latent_features_correlations_{}_{}".format(results_dir, method, vector_name, sample_mode),dpi=500)
            plt.clf()
            plt.close(fig)
    else:
        print("No relevant latent space-features correlations found")
    if calculate_covariance:
        features_names = ["UMAP-1D"] + list(features_dict.keys())
        features_matrix = np.array(list(features_dict.values()))
        features_matrix = np.vstack([umap_proj_1d[None, :], features_matrix])
        features_covariance = np.cov(features_matrix)
        features_correlations = np.corrcoef(features_matrix)
        if plot_covariance:
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 20))
            # norm = plt.Normalize(0, 1)
            norm = None
            cmap = sns.color_palette("rocket_r", as_cmap=True)
            cbar = True

            sns.heatmap(features_correlations, ax=ax1, cbar=cbar, cmap=cmap, norm=norm, annot=True,annot_kws={"fontsize": "small"}, fmt=".1f")
            ax1.set_xticks(np.arange(len(features_names)), labels=features_names, rotation=45)
            ax1.spines['left'].set_visible(False)
            # ax1.yaxis.set_ticklabels([])
            ax1.set_yticks(np.arange(len(features_names)) + 0.5, labels=features_names, rotation=360)
            ax1.set_title("Covariance matrix features")

            fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.2)
            plt.margins(0.6)
            fig.suptitle("Features covariance")
            #plt.savefig("{}/{}/HEATMAP_features_UMAP_1D_covariance_{}_{}.png".format(results_dir, method, vector_name,sample_mode), dpi=600)
            plt.savefig("{}/{}/HEATMAP_features_UMAP_1D_correlations_{}_{}.png".format(results_dir, method, vector_name,sample_mode), dpi=600)
            plt.clf()
            plt.close(fig)

        return {"features_covariance":features_covariance,
                "features_names":features_names,
                "pearson_coefficients":pearson_coefficients,
                "pearson_pvalues":pearson_pvalues,
                "spearman_coefficients":spearman_coefficients,
                "spearman_pvalues":spearman_pvalues
                }

def plot_latent_space(args:namedtuple,dataset_info:namedtuple,latent_space,predictions_dict,sample_mode,results_dir,method,vector_name="latent_space_z",n_clusters=4,plot_correlations=True):
    """
    -Notes on UMAP: https://www.arxiv-vanity.com/papers/2009.12981/
    """
    if latent_space.size > 0 or not None:
        reducer = umap.UMAP()
        print("Started UMAP projections")
        umap_proj = reducer.fit_transform(latent_space[:, 6:])
        print("Finished UMAP projections")
        settings =plot_preprocessing(umap_proj, dataset_info, predictions_dict, sample_mode, results_dir, method,
                           vector_name="latent_space_z", n_clusters=4)

        plot_scatter(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name=vector_name,n_clusters=n_clusters)
        plot_scatter_reduced(umap_proj,args,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name=vector_name,n_clusters=n_clusters)
        if vector_name == "latent_space_z":
            plot_scatter_quantiles(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name=vector_name,n_clusters=n_clusters)
        if plot_correlations and vector_name == "latent_space_z":
            reducer = umap.UMAP(n_components=1)
            umap_proj_1d = reducer.fit_transform(latent_space[:, 6:]).squeeze(-1)
            plot_latent_correlations_1d(umap_proj_1d,args, settings,dataset_info, latent_space, sample_mode, results_dir, method,
                                     vector_name)
        del umap_proj
        gc.collect()
    else:
        print("Latent space size : {}".format(latent_space.shape))
        print("No latent representations could be sampled")

def plot_gradients(gradient_norms,results_dir,mode):
    print("Plotting gradients")
    #fig = plt.figure(figsize=(13, 6), dpi=100).set_facecolor('white')
    fig, (ax1,ax2)= plt.subplots(1, 2, figsize=(15, 12),dpi=100,gridspec_kw={'width_ratios': [3.7, 1]})
    clrs = plt.get_cmap("nipy_spectral", len(gradient_norms.keys()))
    for idx,(name_i, grad_norms) in enumerate(gradient_norms.items()):
        ax1.plot(grad_norms, label=name_i,c=clrs(idx))
    plt.xlabel('epochs')
    plt.ylabel('gradient norm')
    plt.yscale('log')
    ax2.axis('off')
    plt.legend(loc= 'center right',bbox_to_anchor=(1.5,0.5),fontsize=11, borderaxespad=0.)
    fig.suptitle('Gradient norms of model parameters')

    plt.savefig("{}/gradients_{}".format(results_dir,mode))
    plt.clf()
    plt.close(fig)

def plot_ROC_curve(fpr,tpr,roc_auc,auk_score,results_dir,fold,method):
    fig = plt.figure()
    plt.title('Receiver Operating Characteristic',fontdict={"size":20})
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f \n '
                                  'AUK = %0.2f' % (roc_auc,auk_score))
    plt.legend(loc='lower right',prop={'size': 15})
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.savefig("{}/ROC_curve_fold{}_{}".format(results_dir,fold,method))
    plt.clf()
    plt.close(fig)

def plot_blosum_cosine(blosum_array,storage_folder,args):
    """

    :param blosum_array:
    :param storage_folder:
    :param args:
    """
    fig,_=plt.subplots(figsize=(10, 10))
    blosum_cosine = VegvisirUtils.cosine_similarity(blosum_array[1:, 1:], blosum_array[1:, 1:])
    aa_dict = VegvisirUtils.aminoacid_names_dict(21, zero_characters=["#"])
    aa_list = [key for key, val in aa_dict.items() if val in list(blosum_array[:, 0])]
    blosum_cosine_df = pd.DataFrame(blosum_cosine, columns=aa_list, index=aa_list)
    sns.heatmap(blosum_cosine_df.to_numpy(),
                xticklabels=blosum_cosine_df.columns.values,
                yticklabels=blosum_cosine_df.columns.values, annot=True, annot_kws={"size": 8}, fmt=".2f")
    plt.title("Amino acids blosum vector cosine similarity", fontsize=10)
    plt.savefig('{}/{}/blosum_cosine.png'.format(storage_folder, args.dataset_name), dpi=500)
    plt.clf()
    plt.close(fig)

def plot_feature_importance(feature_dict:dict,max_len:int,features_names:list,results_dir:str) -> None:
    """
    :rtype: object
    :param feature_dict:
    :param max_len:
    :param features_names:
    :param results_dir:

    """
    colors_list = ["plum", "lime", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow",
                   "green",
                   "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal",
                   "goldenrod",
                   "black", "chocolate", "cornflowerblue", "pink", "darkgrey", "indianred", "mediumspringgreen",
                   "cadetblue", "sienna",
                   "crimson", "deepbluesky", "wheat", "silver"]
    # plot
    ncols = int(len(feature_dict.keys())/2)
    nrows = [int(len(feature_dict.keys())/ncols) + 1 if len(feature_dict.keys())/ncols % 2 != 0 else int(len(feature_dict.keys())/ncols)][0]
    if len(feature_dict["Fold_0"]) == max_len:
        labels = ["Pos.{}".format(pos) for pos in list(range(max_len))]
    else:
        labels = ["Pos.{}".format(pos) for pos in list(range(max_len))] + features_names
    colors_dict = dict(zip(labels,colors_list))
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols+1 ,squeeze=False) #check this: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
    axs = axs.ravel()
    for i,(fold,features) in enumerate(feature_dict.items()): #ax = fig.add_subplot(2,2,a+1)
        axs[i].bar(range(len(features)),features,color=colors_dict.values())
        #ax[i].set_xticks(np.arange(len(labels)),labels,rotation=45,fontsize=8)
        axs[i].set_title("{}".format(fold))
    axs[-1].set_axis_off()
    patches = [mpatches.Patch(color='{}'.format(val), label='{}'.format(key)) for key,val in colors_dict.items()]
    fig.legend(handles=patches, prop={'size': 8},loc= 'center right',bbox_to_anchor=(1.0,0.3),ncols=2)
    fig.tight_layout(pad=3.0,w_pad=1.5, h_pad=2.0)
    fig.suptitle("Feature importance")
    plt.savefig("{}/feature_importance_xgboost".format(results_dir))
    plt.close(fig)

def plot_mutual_information(full_data,full_labels,feature_names,results_dir):
    """
    :param full_data: (N,n_feats)
    :param full_labels:
    :param features_names:
    :param results_dir:
    """
    colors_list = ["plum", "lime", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow",
                   "green",
                   "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal",
                   "goldenrod",
                   "black", "chocolate", "cornflowerblue", "pink", "darkgrey", "indianred", "mediumspringgreen",
                   "cadetblue", "sienna",
                   "crimson", "deepbluesky", "wheat", "silver"]
    colors_dict = dict(zip(feature_names,colors_list))

    mutual_information = mutual_info_regression(full_data[:,1], full_labels, discrete_features=False, n_neighbors=3, copy=True,random_state=None)
    fig = plt.figure()
    plt.bar(range(len(feature_names)), feature_names,color=colors_dict.values())
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, fontsize=8)
    plt.title("Mutual Information feature importance")
    patches = [mpatches.Patch(color='{}'.format(val), label='{}'.format(key)) for key,val in colors_dict.items()]
    # pos = fig.get_position()
    # fig.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    plt.legend(handles=patches, prop={'size': 10},loc= 'center right',bbox_to_anchor=(1.37,0.5))
    plt.savefig("{}/mi_feature_importance".format(results_dir),dpi=500)
    plt.clf()
    plt.close(fig)

def plot_confusion_matrix(confusion_matrix,performance_metrics,results_dir,fold,method):
    """Plot confusion matrix
    :param pandas dataframe confusion_matrix
    :param dict performance_metrics"""
    confusion_matrix_array = confusion_matrix.to_numpy()
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,7),gridspec_kw={'width_ratios': [3, 1]})
    ax[0].imshow(confusion_matrix_array,cmap='Pastel1_r')
    for i in range(confusion_matrix_array.shape[0]):
        for j in range(confusion_matrix_array.shape[1]):
              ax[0].text(j, i, "{:.2f}".format(confusion_matrix_array[i, j]), ha="center", va="center")
    #[[true_negatives,false_positives],[false_negatives,true_positives]]
    ax[1].axis("off")
    ax[0].set_xticks([0,1],confusion_matrix.columns)
    ax[0].set_yticks([0,1],confusion_matrix.index)
    fig.suptitle("Confusion matrix")
    patches = [mpatches.Circle((0.5, 0.5),radius = 0.25,color=colors_dict[0], label='{}:{}'.format(key,np.round(val,2))) for key,val in performance_metrics.items()]
    ax[0].legend(handles=patches, prop={'size': 10}, loc='right',bbox_to_anchor=(1.5, 0.5), ncol=1)
    plt.savefig("{}/confusion_matrix_fold{}_{}.png".format(results_dir,fold,method),dpi=100)
    plt.clf()
    plt.close(fig)

def micro_auc(args:namedtuple,onehot_labels,y_prob,idx):
    """Calculates the AUC for a multi-class problem"""

    micro_roc_auc_ovr = roc_auc_score(
        onehot_labels[idx],
        y_prob[idx],
        multi_class="ovr",
        average="micro",
    )
    fprs = dict()
    tprs = dict()
    roc_aucs = dict()
    for i in range(2):
          fprs[i], tprs[i], _ = roc_curve(onehot_labels[:, i], y_prob[:, i])
          roc_aucs[i] = auc(fprs[i], tprs[i])
    return [micro_roc_auc_ovr,fprs,tprs,roc_aucs]

def plot_precision_recall_curve(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,key_name,stats_name,idx,idx_name,save_plot=True):
    """Following https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate."""
    onehot_targets = onehot_labels[idx]
    target_scores = predictions_dict[stats_name][idx]
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    try:
        for i in range(args.num_obs_classes):
            precision[i], recall[i], _ = precision_recall_curve(onehot_targets[:, i], target_scores[:, i])
            average_precision[i] = average_precision_score(onehot_targets[:, i], target_scores[:, i])

            precision_recall_joint = np.concatenate([precision[i][None, :], recall[i][None, :]], axis=0)
            max_precision_recall_combo = precision_recall_joint[:, precision_recall_joint.sum(axis=0).argmax()]
            precision["max_{}".format(i)] = max_precision_recall_combo[0]
            recall["max_{}".format(i)] = max_precision_recall_combo[1]


        if save_plot:
            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                onehot_targets.ravel(), target_scores.ravel()
            )
            average_precision["micro"] = average_precision_score(onehot_targets, target_scores, average="micro")
            average_precision["weighted"] = average_precision_score(onehot_targets, target_scores, average="weighted")
            fig = plt.figure()
            plt.plot(recall["micro"],precision["micro"], label="Average Precision (AP): {}".format(average_precision["weighted"]))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Precision', fontsize=20)
            plt.xlabel('Recall', fontsize=20)
            plt.legend(loc='lower right', prop={'size': 15})
            plt.title("Average Precision curves")
            plt.savefig("{}/{}/PrecisionRecall_curves_fold{}_{}".format(results_dir, mode, fold, "{}_{}".format(key_name, idx_name)))
            plt.clf()
            plt.close(fig)
    except:
        print("Could not calculate AUC, only one class found")
    return {"precision":precision,"recall":recall,"average_precision":average_precision}

def plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,key_name,stats_name,idx,idx_name,save=True):
    """

    :param labels:
    :param onehot_labels:
    :param predictions_dict:
    :param args:
    :param results_dir:
    :param mode:
    :param fold:
    :param key_name:
    :param str stats_name: also known as prob_mode
    :param idx:
    :param idx_name:
    :param save:
    :return:
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pvals = dict()
    ppv_mod = dict()
    labels = labels[idx]
    onehot_targets = onehot_labels[idx]
    target_scores = predictions_dict[stats_name][idx]
    #TODO??:
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
    # https://www.statalist.org/forums/forum/general-stata-discussion/general/1319291-p-value-for-auc-after-roctab-command
    # https://stats.stackexchange.com/questions/75050/in-r-how-to-compute-the-p-value-for-area-under-roc
    try:
        fig,[ax1,ax2] = plt.subplots(1,2,figsize=(9, 6),gridspec_kw={'width_ratios': [4.5,1]})
        # ROC AUC per class
        for i in range(args.num_obs_classes):
            fpr[i], tpr[i], _ = roc_curve(onehot_targets[:, i], target_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc["auc01_class_{}".format(i)] = roc_auc_score(onehot_targets[:, i], target_scores[:, i],average="weighted",max_fpr=0.1)
            ax1.plot(fpr[i], tpr[i], label='Class {} AUC : {}'.format(i, round(roc_auc[i],2)), c=colors_dict_labels[i])
        # Micro ROC AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(onehot_targets.ravel(), target_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #ax1.plot(fpr["micro"], tpr["micro"], label="micro-average AUC : {}".format(round(roc_auc["micro"],3)),linestyle="-.", color="magenta")
        # Macro ROC AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_obs_classes)]))
        fpr["macro"] = all_fpr
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(args.num_obs_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) #linear interpolation of data points
        tpr["macro"] = mean_tpr / args.num_obs_classes
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        #ax1.plot(fpr["macro"], tpr["macro"], label="macro-average AUC : {}".format(round(roc_auc["macro"],3)),linestyle="-.", color="blue")
        ax1.plot([0, 1], [0, 1], 'r--')
        fig.legend( bbox_to_anchor=(0.71, 0.35), prop={'size': 15}) #loc='lower right'
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        plt.margins(1)
        ax1.set_ylabel('True Positive Rate', fontsize=20)
        ax1.set_xlabel('False Positive Rate', fontsize=20)
        fig.suptitle("ROC curves",fontsize=20)
        ax2.axis("off")
        if save:
            plt.savefig("{}/{}/ROC_curves_fold{}_{}.png".format(results_dir, mode, fold, "{}_{}".format(key_name, idx_name)))

        plt.clf()
        plt.close(fig)
    except:
        print("Could not calculate AUC, only one class found. (or another error came up)")
        roc_auc[0] = np.nan
        roc_auc[1] = np.nan
        roc_auc["auc01_class_0"] = np.nan
        roc_auc["auc01_class_1"] = np.nan
    try:
        for i in range(args.num_obs_classes):
            lrm = sm.Logit(onehot_targets[:, i], target_scores[:, i]).fit(disp=0)
            pvals[i] = lrm.pvalues.item()
    except:
        print("Regression failed")
        pvals[0] = np.nan
        pvals[1] = np.nan

    for i in range(args.num_obs_classes):
        ppv_mod[i] = calculate_ppv_modified(onehot_targets[:, i], target_scores[:, i])

    return fpr,tpr,roc_auc,pvals,ppv_mod

def plot_classification_metrics(args,predictions_dict,fold,results_dir,mode="Train",per_sample=False):
    """
    Notes:
        -http://www.med.mcgill.ca/epidemiology/hanley/software/Hanley_McNeil_Radiology_82.pdf
        -https://jorgetendeiro.github.io/SHARE-UMCG-14-Nov-2019/Part2
        -Avoid AUC: https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2007.00358.x
        - "Can Micro-Average ROC AUC score be larger than Class ROC AUC scores
        - https://arxiv.org/pdf/2107.13171.pdf
        - Find optimal treshold: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        - Interpretation: https://glassboxmedicine.com/2020/07/14/the-complete-guide-to-auc-and-average-precision-simulations-and-visualizations/#:~:text=the%20PR%20curve.-,Average%20precision%20indicates%20whether%20your%20model%20can%20correctly%20identify%20all,to%201.0%20(perfect%20model).
    :param predictions_dict: {"mode": tensor of (N,), "frequencies": tensor of (N, num_classes)}
    :param labels:
    :param fold:
    :param results_dir:
    :param mode:
    :return:
    """
    evaluation_modes = ["samples","single_sample"] if predictions_dict["true_single_sample"] is not None else ["samples"]
    probability_modes = ["class_probs_predictions_samples_average","class_probs_predictions_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_probs_predictions_samples_average"]
    #binary_modes = ["class_binary_predictions_samples_mode","class_binary_predictions_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_mode"]
    binary_modes = ["class_binary_predictions_samples_logits_average_argmax","class_binary_predictions_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_logits_average_argmax"]

    metrics_summary_dict = defaultdict(lambda:defaultdict(lambda : defaultdict()))
    for sample_mode,prob_mode,binary_mode in zip(evaluation_modes,probability_modes,binary_modes):
        labels = predictions_dict["true_{}".format(sample_mode)]
        onehot_labels = predictions_dict["true_onehot_{}".format(sample_mode)]
        confidence_scores = predictions_dict["confidence_scores_{}".format(sample_mode)]
        idx_all = np.ones_like(labels).astype(bool)
        if args.num_classes > args.num_obs_classes:
            idx_all = (labels[..., None] != 2).any(-1)  # Highlight: unlabelled data has been assigned labelled 2, we give high confidence to the labelled data (for now)
        idx_highconfidence = (confidence_scores[..., None] > 0.7).any(-1)

        for idx,idx_name in zip([idx_all,idx_highconfidence],["ALL","HIGH_CONFIDENCE"]):
            print("---------------- {} data points ----------------\n ".format(idx_name))
            print("---------------- {} data points ----------------\n ".format(idx_name),file=open("{}/AUC_out.txt".format(results_dir), "a"))

            if predictions_dict[prob_mode] is not None:
                #fpr, tpr, threshold = roc_curve(y_true=onehot_labels[idx], y_score=predictions_dict[stats_name][idx])
                try:
                    micro_roc_auc_ovr = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ovr",
                        average="micro",
                    )
                except:
                    micro_roc_auc_ovr = None
                try:
                    micro_roc_auc_ovo = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ov0",
                        average="micro",
                    )
                except:
                    micro_roc_auc_ovo = None

                try:
                    macro_roc_auc_ovr = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ovr",
                        average="macro",
                    )
                except:
                    macro_roc_auc_ovr = None
                try:
                    macro_roc_auc_ovo = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ovo",
                        average="macro",
                    )
                except:
                    macro_roc_auc_ovo = None
                try:
                    weighted_roc_auc_ovr = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ovr",
                        average="weighted",
                    )
                except:
                    weighted_roc_auc_ovr = None
                try:
                    weighted_roc_auc_ovo = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ovo",
                        average="weighted",
                    )
                except:
                    weighted_roc_auc_ovo = None
                if predictions_dict[binary_mode] is not None:
                    try:
                        auk_score_binary = VegvisirUtils.AUK(predictions_dict[binary_mode][idx],
                                                             labels[idx]).calculate_auk()
                    except:
                        auk_score_binary = None
                else:
                    auk_score_binary = None

                fpr,tpr,roc_auc,pvals,ppv_mod = plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name)
                ap_dict = plot_precision_recall_curve(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name)

                print("---------------- {} ----------------\n".format(prob_mode))
                print("---------------- {} ----------------\n ".format(prob_mode),file=open("{}/AUC_out.txt".format(results_dir), "a"))
                scores_dict = {"micro_roc_auc_ovr":micro_roc_auc_ovr,
                               "micro_roc_auc_ovo": micro_roc_auc_ovo,
                               "macro_roc_auc_ovr": macro_roc_auc_ovr,
                               "macro_roc_auc_ovo": macro_roc_auc_ovo,
                               "weighted_roc_auc_ovr": weighted_roc_auc_ovr,
                               "weighted_roc_auc_ovo": weighted_roc_auc_ovo,
                               "auk_score_binary":auk_score_binary}

                #json.dump(scores_dict, open("{}/AUC_out_{}_fold_{}.txt".format(results_dir,mode,fold), "a"), indent=2)


                metrics_summary_dict[sample_mode][idx_name]["fpr"] = fpr
                metrics_summary_dict[sample_mode][idx_name]["tpr"] = tpr
                metrics_summary_dict[sample_mode][idx_name]["roc_auc"] = roc_auc
                metrics_summary_dict[sample_mode][idx_name]["pvals"] = pvals
                metrics_summary_dict[sample_mode][idx_name]["precision"] = ap_dict["precision"]
                metrics_summary_dict[sample_mode][idx_name]["recall"] = ap_dict["recall"]
                metrics_summary_dict[sample_mode][idx_name]["average_precision"] = ap_dict["average_precision"]

            #for key_name_2,stats_name_2 in zip(["samples_mode","single_sample"],["class_binary_predictions_samples_mode","class_binary_predictions_single_sample"]):
            if predictions_dict[binary_mode] is not None:
                targets = labels[idx]
                scores = predictions_dict[binary_mode][idx]
                try:
                    #TODO: Change to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
                    tn, fp, fn, tp = confusion_matrix(y_true=targets, y_pred=scores).ravel()
                    confusion_matrix_df = pd.DataFrame([[tp, fp],
                                                        [fn, tn]],
                                                    index=["Positive\n(Pred)", "Negative\n(Pred)"],
                                                    columns=["Positive\n(True)", "Negative\n(True)"])
                    recall = tp/(tp + fn)
                    precision = tp/(tp + fp)
                    f1score = 2*tp/(2*tp + fp + fn)
                    tnr = tn/(tn + fp)
                    mcc_custom = (tp*tn - fp*fn)/np.sqrt([(tp + tp)*(tp + fn)*(tn + fp)*(tn + fn)])[0]
                    mcc = matthews_corrcoef(targets,scores)
                    accuracy = 100*((scores == targets).sum()/targets.shape[0])
                    performance_metrics = {"recall/tpr":recall,"precision/ppv":precision,"accuracy":accuracy,"f1score":f1score,"tnr":tnr,"samples\naverage\naccuracy":predictions_dict["samples_average_accuracy"],
                                           "Matthew CC":mcc, "AUK":auk_score_binary}
                    plot_confusion_matrix(confusion_matrix_df,performance_metrics,"{}/{}".format(results_dir,mode),fold,"{}_{}".format(sample_mode,idx_name))
                except:
                    print("Only one class found, not plotting confusion matrix")

            if per_sample:
                #Calculate metrics for every individual samples
                samples_results = Parallel(n_jobs=MAX_WORKERs)(delayed(micro_auc)(args,onehot_labels, sample, idx) for sample in np.transpose(predictions_dict["class_probs_predictions_samples"],(1,0,2)))
                average_micro_auc = 0
                fig, [ax1, ax2] = plt.subplots(1, 2,figsize=(17, 12),gridspec_kw={'width_ratios': [6, 2]})
                for i in range(args.num_samples):
                    micro_roc_auc_ovr, fprs, tprs, roc_aucs = samples_results[i]
                    average_micro_auc += micro_roc_auc_ovr
                    for j in range(args.num_classes):
                        ax1.plot(fprs[j], tprs[j], label='AUC_{}: {} MicroAUC: {}'.format(i, roc_aucs[j],micro_roc_auc_ovr), c=colors_dict[j])
                ax1.plot([0, 1], [0, 1], 'r--')
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])
                ax1.set_ylabel('True Positive Rate', fontsize=20)
                ax1.set_xlabel('False Positive Rate', fontsize=20)
                ax2.axis("off")
                ax1.legend(loc='lower right', prop={'size': 6},bbox_to_anchor=(1.5, 0.))
                fig.suptitle("ROC curve. AUC_micro_ovr_average: {}".format(average_micro_auc/args.num_samples),fontsize=12)
                plt.savefig("{}/{}/ROC_curves_PER_SAMPLE_{}".format(results_dir, mode, "{}".format(idx_name)))
                plt.clf()
                plt.close(fig)
                del samples_results
                gc.collect()

    return metrics_summary_dict

def plot_classification_metrics_per_species(dataset_info,args,predictions_dict,fold,results_dir,mode="Train",per_sample=False):
    """
    Notes:
        -http://www.med.mcgill.ca/epidemiology/hanley/software/Hanley_McNeil_Radiology_82.pdf
        -https://jorgetendeiro.github.io/SHARE-UMCG-14-Nov-2019/Part2
        -Avoid AUC: https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2007.00358.x
        - "Can Micro-Average ROC AUC score be larger than Class ROC AUC scores
        - https://arxiv.org/pdf/2107.13171.pdf
        - Find optimal treshold: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        - Interpretation: https://glassboxmedicine.com/2020/07/14/the-complete-guide-to-auc-and-average-precision-simulations-and-visualizations/#:~:text=the%20PR%20curve.-,Average%20precision%20indicates%20whether%20your%20model%20can%20correctly%20identify%20all,to%201.0%20(perfect%20model).
    :param predictions_dict: {"mode": tensor of (N,), "frequencies": tensor of (N, num_classes)}
    :param labels:
    :param fold:
    :param results_dir:
    :param mode:
    :return:
    """
    species_dict = pickle.load(open('{}/{}/org_name_dict.pkl'.format(dataset_info.storage_folder,args.dataset_name),"rb"))

    evaluation_modes = ["samples","single_sample"] if predictions_dict["true_single_sample"] is not None else ["samples"]
    probability_modes = ["class_probs_predictions_samples_average","class_probs_prediction_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_probs_predictions_samples_average"]
    binary_modes = ["class_binary_predictions_samples_mode","class_binary_predictions_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_mode"]
    #binary_modes = ["class_binary_predictions_samples_logits_average_argmax","class_binary_predictions_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_logits_average_argmax"]

    #for sample_mode,prob_mode,binary_mode in zip(["samples","single_sample"],["class_probs_predictions_samples_average","class_probs_prediction_single_sample"],["class_binary_predictions_samples_mode","class_binary_predictions_single_sample"]):
    for sample_mode,prob_mode,binary_mode in zip(evaluation_modes,probability_modes,binary_modes):
        data_int = predictions_dict["data_int_{}".format(sample_mode)]
        org_name = data_int[:,0,6]
        unique_org_name,counts = np.unique(org_name,return_counts=True)
        labels = predictions_dict["true_{}".format(sample_mode)]
        onehot_labels = predictions_dict["true_onehot_{}".format(sample_mode)]
        confidence_scores = predictions_dict["confidence_scores_{}".format(sample_mode)]
        idx_all = np.ones_like(labels).astype(bool)
        if args.num_classes > args.num_obs_classes:
            idx_all = (labels[..., None] != 2).any(-1)  # Highlight: unlabelled data has been assigned labelled 2, we give high confidence to the labelled data (for now)
        #idx_highconfidence = (confidence_scores[..., None] > 0.7).any(-1)
        fig = plt.figure()
        for species in unique_org_name:
            idx = (org_name[...,None] == species).any(-1)
            idx *= idx_all #TODO: Check
            idx_name = str(int(species))
            species_labels = labels[idx]
            if species_labels.shape[0] > 100:
                print("Number data points: \n  class 0 : {} \n class 1:{}".format(species_labels.shape[0]-species_labels.sum(),species_labels.sum()))
                print("---------------- species number {} ----------------\n ".format(species))
                print("---------------- species number {} ----------------\n ".format(species),file=open("{}/AUC_out.txt".format(results_dir), "a"))

                if predictions_dict[prob_mode] is not None:
                    #fpr, tpr, threshold = roc_curve(y_true=onehot_labels[idx], y_score=predictions_dict[stats_name][idx])
                    micro_roc_auc_ovr = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ovr",
                        average="micro",
                    )
                    micro_roc_auc_ovo = roc_auc_score(
                        onehot_labels[idx],
                        predictions_dict[prob_mode][idx],
                        multi_class="ov0",
                        average="micro",
                    )
                    try:
                        macro_roc_auc_ovr = roc_auc_score(
                            onehot_labels[idx],
                            predictions_dict[prob_mode][idx],
                            multi_class="ovr",
                            average="macro",
                        )
                    except:
                        macro_roc_auc_ovr = None
                    try:
                        macro_roc_auc_ovo = roc_auc_score(
                            onehot_labels[idx],
                            predictions_dict[prob_mode][idx],
                            multi_class="ovo",
                            average="macro",
                        )
                    except:
                        macro_roc_auc_ovo = None
                    try:
                        weighted_roc_auc_ovr = roc_auc_score(
                            onehot_labels[idx],
                            predictions_dict[prob_mode][idx],
                            multi_class="ovr",
                            average="weighted",
                        )
                    except:
                        weighted_roc_auc_ovr = None
                    try:
                        weighted_roc_auc_ovo = roc_auc_score(
                            onehot_labels[idx],
                            predictions_dict[prob_mode][idx],
                            multi_class="ovo",
                            average="weighted",
                        )
                    except:
                        weighted_roc_auc_ovo = None
                    if predictions_dict[binary_mode] is not None:
                        try:
                            auk_score_binary = VegvisirUtils.AUK(predictions_dict[binary_mode][idx],
                                                                 labels[idx]).calculate_auk()
                        except:
                            auk_score_binary = None
                    else:
                        auk_score_binary = None


                    fpr,tpr,roc_auc,pvals,ppv_mod = plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name,save=False)
                    for i in range(args.num_obs_classes):
                        plt.plot(fpr[i], tpr[i], label='({})AUC_{}: {}'.format(species_dict[species],i, roc_auc[i]), c=colors_dict[i])

                    plot_precision_recall_curve(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name)

                    print("---------------- {} ----------------\n".format(prob_mode))
                    print("---------------- {} ----------------\n ".format(prob_mode),file=open("{}/AUC_out.txt".format(results_dir), "a"))
                    scores_dict = {"micro_roc_auc_ovr":micro_roc_auc_ovr,
                                   "micro_roc_auc_ovo": micro_roc_auc_ovo,
                                   "macro_roc_auc_ovr": macro_roc_auc_ovr,
                                   "macro_roc_auc_ovo": macro_roc_auc_ovo,
                                   "weighted_roc_auc_ovr": weighted_roc_auc_ovr,
                                   "weighted_roc_auc_ovo": weighted_roc_auc_ovo,
                                   "auk_score_binary":auk_score_binary}

                    json.dump(scores_dict, open("{}/AUC_out.txt".format(results_dir), "a"), indent=2)

                if predictions_dict[binary_mode] is not None:
                    targets = labels[idx]
                    scores = predictions_dict[binary_mode][idx]
                    try: #if len(np.unique(targets)) >= 2:
                        #TODO: Change to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
                        tn, fp, fn, tp = confusion_matrix(y_true=targets, y_pred=scores).ravel()
                        confusion_matrix_df = pd.DataFrame([[tp, fp],
                                                            [fn, tn]],
                                                        index=["Positive\n(Pred)", "Negative\n(Pred)"],
                                                        columns=["Positive\n(True)", "Negative\n(True)"])
                        recall = tp/(tp + fn)
                        precision = tp/(tp + fp)
                        f1score = 2*tp/(2*tp + fp + fn)
                        tnr = tn/(tn + fp)
                        mcc_custom = (tp*tn - fp*fn)/np.sqrt([(tp + tp)*(tp + fn)*(tn + fp)*(tn + fn)])[0]
                        mcc = matthews_corrcoef(targets,scores)
                        accuracy = 100*((scores == targets).sum()/targets.shape[0])
                        performance_metrics = {"recall/tpr":recall,"precision/ppv":precision,"accuracy":accuracy,"f1score":f1score,"tnr":tnr,"samples\naverage\naccuracy":predictions_dict["samples_average_accuracy"],
                                               "Matthew CC":mcc, "AUK":auk_score_binary}
                        plot_confusion_matrix(confusion_matrix_df,performance_metrics,"{}/{}".format(results_dir,mode),fold,"{}_{}".format(sample_mode,idx_name))
                    except:
                        print("Only one class found, not plotting confusion matrix")

                if per_sample:
                    #Calculate metrics for every individual samples
                    samples_results = Parallel(n_jobs=MAX_WORKERs)(delayed(micro_auc)(args,onehot_labels, sample, idx) for sample in np.transpose(predictions_dict["class_probs_predictions_samples"],(1,0,2)))
                    average_micro_auc = 0
                    fig, [ax1, ax2] = plt.subplots(1, 2,figsize=(17, 12),gridspec_kw={'width_ratios': [6, 2]})
                    for i in range(args.num_samples):
                        micro_roc_auc_ovr, fprs, tprs, roc_aucs = samples_results[i]
                        average_micro_auc += micro_roc_auc_ovr
                        for j in range(args.num_classes):
                            ax1.plot(fprs[j], tprs[j], label='AUC_{}: {} MicroAUC: {}'.format(i, roc_aucs[j],micro_roc_auc_ovr), c=colors_dict[j])
                    ax1.plot([0, 1], [0, 1], 'r--')
                    ax1.set_xlim([0, 1])
                    ax1.set_ylim([0, 1])
                    ax1.set_ylabel('True Positive Rate', fontsize=20)
                    ax1.set_xlabel('False Positive Rate', fontsize=20)
                    ax2.axis("off")
                    fig.legend(bbox_to_anchor=(0.71, 0.35), prop={'size': 15})  # loc='lower right'
                    fig.suptitle("ROC curve. AUC_micro_ovr_average: {}".format(average_micro_auc/args.num_samples),fontsize=12)
                    plt.savefig("{}/{}/ROC_curves_PER_SAMPLE_{}".format(results_dir, mode, "{}".format(idx_name)))
                    plt.clf()
                    plt.close(fig)
                    del samples_results
                    gc.collect()
        plt.title("ROC curves per species")
        plt.savefig("{}/{}/ROC_curves_per_species_fold{}_{}.png".format(results_dir, mode, fold, sample_mode))
        plt.clf()
        plt.close(fig)

def plot_attention_weights(summary_dict,dataset_info,results_dir,method="Train"):
    """

    :param summary_dict:
    :param results_dir:
    :param method:
    :return:
    Notes:
        https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    """

    if dataset_info.corrected_aa_types == 20:
        aminoacids_dict = VegvisirUtils.aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=[])
    else:
        aminoacids_dict = VegvisirUtils.aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=["#"])
    aa_groups_colors_dict, aa_groups_dict, groups_names_colors_dict,aa_by_groups_dict = VegvisirUtils.aminoacids_groups(aminoacids_dict)
    aa_groups_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("aa_cm", list(aa_groups_colors_dict.values()))

    colors_list = ["black","plum", "lime", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow","green",
                   "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal",
                   "goldenrod", "chocolate", "cornflowerblue", "pink", "darkgrey", "indianred",
                   "mediumspringgreen"]
    aa_colors_dict = {i:colors_list[i] for aa,i in aminoacids_dict.items()}
    aa_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("aa_cm", list(aa_colors_dict.values()))
    aa_patches = [mpatches.Patch(color=colors_list[i], label='{}'.format(aa)) for aa,i in aminoacids_dict.items()]
    aa_groups_patches = [mpatches.Patch(color=color, label='{}'.format(group)) for group,color in groups_names_colors_dict.items()]

    evaluation_modes = ["single_sample","samples"] if summary_dict["true_single_sample"] is not None else ["samples"]


    for sample_mode in evaluation_modes:
        data_int = summary_dict["data_int_{}".format(sample_mode)]
        confidence_scores = summary_dict["confidence_scores_{}".format(sample_mode)]
        idx_all = np.ones_like(confidence_scores).astype(bool)
        idx_highconfidence = (confidence_scores[..., None] > 0.7).any(-1)
        for data_points,idx in zip(["all","high_confidence"],[idx_all,idx_highconfidence]):
            true_labels = summary_dict["true_{}".format(sample_mode)][idx]
            positives_idx = (true_labels == 1)
            for class_type,idx_class in zip(["positives","negatives"],[positives_idx,~positives_idx]):
                attention_weights = summary_dict["attention_weights_{}".format(sample_mode)][idx][idx_class]
                aminoacids = data_int[idx][idx_class]
                if sample_mode == "single_sample":
                    attention_weights = attention_weights[:,:,0]
                else:
                    attention_weights = attention_weights[:,:,:,0].mean(axis=1)
                #try:

                fig,[[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(nrows=2,ncols=3,figsize=(9, 6),gridspec_kw={'width_ratios': [4.5, 4.5,1],'height_ratios': [4.5, 4.5]})
                print("{}/{}".format(sample_mode,data_points))
                if np.prod(attention_weights.shape) == 0.:
                    print("No data points, no attention plots")
                    pass
                else:
                    #Highlight: Attention weights
                    sns.heatmap(attention_weights,ax=ax1)
                    ax1.set_xticks(np.arange(attention_weights.shape[1]) + 0.5,labels=["{}".format(i) for i in range(attention_weights.shape[1])])
                    ax1.spines['left'].set_visible(False)
                    ax1.yaxis.set_ticklabels([])
                    ax1.set_title("Attention by weight")
                    #Highlight: Aminoacids coloured by name
                    if np.rint(attention_weights).max() != 1:
                        aminoacids_masked = (aminoacids[:,1])*attention_weights.astype(bool).astype(int)
                    else:
                        aminoacids_masked = (aminoacids[:,1])*np.rint(attention_weights)
                    sns.heatmap(aminoacids_masked,ax=ax2,cbar=False,cmap=aa_colormap)
                    ax2.set_xticks(np.arange(attention_weights.shape[1] +1 ) + 0.5,labels=["{}".format(i) for i in range(attention_weights.shape[1] + 1)])
                    ax2.spines['left'].set_visible(False)
                    ax2.yaxis.set_ticklabels([])
                    ax2.set_title("Attention by amino acid type")
                    #Highlight: Aminoacids coloured by functional group (i.e positive, negative ...)
                    sns.heatmap(aminoacids_masked,ax=ax4,cbar=False,cmap=aa_groups_colormap)
                    ax4.set_xticks(np.arange(attention_weights.shape[1] +1) + 0.5,labels=["{}".format(i) for i in range(attention_weights.shape[1] + 1)])
                    ax4.spines['left'].set_visible(False)
                    ax4.yaxis.set_ticklabels([])
                    ax4.set_title("Attention by amino acid group")

                    ax3.axis("off")
                    ax5.axis("off")
                    ax6.axis("off")

                    legend1 = plt.legend(handles=aa_patches, prop={'size': 8}, loc='center right',
                               bbox_to_anchor=(0.9, 0.7), ncol=1)
                    plt.legend(handles=aa_groups_patches, prop={'size': 8}, loc='center right',
                               bbox_to_anchor=(0.1, 0.5), ncol=1)
                    plt.gca().add_artist(legend1)

                    fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
                    fig.suptitle("Attention weights: {}, {}, {}".format(sample_mode,data_points,class_type))
                    plt.savefig("{}/{}/Attention_plots_{}_{}_{}.png".format(results_dir,method,sample_mode,data_points,class_type))
                    plt.clf()
                    plt.close(fig)
    del data_int,attention_weights,confidence_scores,aminoacids,aminoacids_masked
    gc.collect()

def plot_hidden_dimensions(summary_dict, dataset_info, results_dir,args, method="Train"):
    """"""
    print("Analyzing hidden dimensions ...")
    if dataset_info.corrected_aa_types == 20:
        aminoacids_dict = VegvisirUtils.aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=[])
    else:
        aminoacids_dict = VegvisirUtils.aminoacid_names_dict(dataset_info.corrected_aa_types,zero_characters=["#"])
    aa_groups_colors_dict, aa_groups_dict, groups_names_colors_dict,aa_by_groups_dict = VegvisirUtils.aminoacids_groups(aminoacids_dict)
    if dataset_info.corrected_aa_types == 20: #TODO: Review
        aa_groups_colors_dict[0] = "black"  #otherwise it colors everyhing with the color of R, and the weights have been adjusted to 0
    aa_groups_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("aa_cm", list(aa_groups_colors_dict.values()))

    colors_list = ["black","plum", "lime", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow","green",
                   "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal",
                   "goldenrod", "chocolate", "cornflowerblue", "pink", "darkgrey", "indianred",
                   "mediumspringgreen"]
    aa_colors_dict = {i:colors_list[i] for aa,i in aminoacids_dict.items()}
    if dataset_info.corrected_aa_types == 20:  #TODO: Review
        aa_colors_dict[0] = "black"

    aa_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("aa_cm", list(aa_colors_dict.values()))
    aa_patches = [mpatches.Patch(color=colors_list[i], label='{}'.format(aa)) for aa,i in aminoacids_dict.items()]
    aa_groups_patches = [mpatches.Patch(color=color, label='{}'.format(group)) for group,color in groups_names_colors_dict.items()]
    max_len = dataset_info.seq_max_len
    diag_idx_maxlen = np.diag_indices(max_len)

    evaluation_modes = ["single_sample","samples"] if summary_dict["true_single_sample"] is not None else ["samples"]

    for sample_mode in evaluation_modes:
        data_int = summary_dict["data_int_{}".format(sample_mode)]
        data_mask = summary_dict["data_mask_{}".format(sample_mode)]
        data_mask_seq = data_mask[:, 1:,:,0].squeeze(1)
        #true_labels = summary_dict["true_{}".format(sample_mode)]
        confidence_scores = summary_dict["confidence_scores_{}".format(sample_mode)]
        idx_all = np.ones_like(confidence_scores).astype(bool)

        idx_highconfidence = (confidence_scores[..., None] > 0.7).any(-1)
        encoder_final_hidden_state = summary_dict["encoder_final_hidden_state_{}".format(sample_mode)]
        decoder_final_hidden_state = summary_dict["decoder_final_hidden_state_{}".format(sample_mode)]
        #print("Plotting Hidden dimensions latent space")
        #plot_latent_space(args,dataset_info,encoder_final_hidden_state, summary_dict, sample_mode, results_dir, method, vector_name="encoder_final_hidden_state")
        #plot_latent_space(args,dataset_info,decoder_final_hidden_state, summary_dict, sample_mode, results_dir, method, vector_name="decoder_final_hidden_state")

        for data_points,idx_conf in zip(["all","high_confidence"],[idx_all,idx_highconfidence]):
            true_labels = summary_dict["true_{}".format(sample_mode)][idx_conf]
            positives_idx = (true_labels == 1)
            for class_type,idx_class in zip(["positives","negatives"],[positives_idx,~positives_idx]):
                #warnings.warn("Change the line of code below!")
                #idx_class = np.ones_like(idx_class).astype(bool)
                encoder_hidden_states = summary_dict["encoder_hidden_states_{}".format(sample_mode)][idx_conf][idx_class]
                decoder_hidden_states = summary_dict["decoder_hidden_states_{}".format(sample_mode)][idx_conf][idx_class] #TODO: Review the values
                if encoder_hidden_states.size != 0:
                    #Highlight: Compute the cosine similarity measure (distance = 1 - similarity) among the hidden states of the sequence
                    if sample_mode == "single_sample":
                        # for seq,seq_mask in zip(encoder_hidden_states,data_mask_seq):
                        #     VegvisirUtils.information_shift(seq,seq_mask,diag_idx_maxlen,dataset_info.seq_max_len)
                        # Highlight: Encoder
                        encoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs,backend='loky')(
                            delayed(VegvisirUtils.information_shift)(seq,seq_mask,diag_idx_maxlen,dataset_info.seq_max_len) for seq,seq_mask in
                            zip(encoder_hidden_states,data_mask_seq))
                        encoder_information_shift_weights = np.concatenate(encoder_information_shift_weights,axis=0)
                        #Highlight: Decoder
                        decoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs,backend='loky')(
                            delayed(VegvisirUtils.information_shift)(seq,seq_mask,diag_idx_maxlen,dataset_info.seq_max_len) for seq,seq_mask in
                            zip(decoder_hidden_states,data_mask_seq))
                        decoder_information_shift_weights = np.concatenate(decoder_information_shift_weights,axis=0)
                    else:
                        encoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs,backend='loky')(
                            delayed(VegvisirUtils.information_shift_samples)(encoder_hidden_states[:, sample_idx],
                                                                            data_mask_seq, diag_idx_maxlen,dataset_info.seq_max_len) for sample_idx in range(args.num_samples))

                        decoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs,backend='loky')(
                            delayed(VegvisirUtils.information_shift_samples)(decoder_hidden_states[:, sample_idx],
                                                                            data_mask_seq, diag_idx_maxlen,
                                                                            dataset_info.seq_max_len) for sample_idx in range(args.num_samples))

                        encoder_information_shift_weights = np.concatenate(encoder_information_shift_weights,axis=1) #N,samples, L
                        encoder_information_shift_weights = encoder_information_shift_weights.mean(axis=1)
                        decoder_information_shift_weights = np.concatenate(decoder_information_shift_weights,axis=1) #N,samples, L
                        decoder_information_shift_weights = decoder_information_shift_weights.mean(axis=1)
                    aminoacids = data_int[idx_conf][idx_class]
                    for nn_name,weights in zip(["Encoder","Decoder"],[encoder_information_shift_weights,decoder_information_shift_weights]):
                        #Highlight: Start figure
                        fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(9, 6),
                                                                               gridspec_kw={'width_ratios': [4.5, 4.5, 1],
                                                                                            'height_ratios': [4.5, 4.5]})
                        print("{}/{}/{}/{}".format(sample_mode, data_points,class_type,nn_name))
                        if np.prod(weights.shape) == 0.:
                            print("No data points, no information shift - weights plots")
                            pass
                        else:
                            # Highlight: Attention weights
                            divider = make_axes_locatable(ax1)
                            #cbar_ax1 = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
                            cbar_ax1 = divider.append_axes("right", size="5%", pad=0.05)

                            fig.add_axes(cbar_ax1)
                            sns.heatmap(weights,
                                        ax=ax1,
                                        cbar_ax=cbar_ax1,
                                        annot=False,
                                        #square=True,
                                        cbar_kws={ "orientation": "vertical" })
                            #cb2 = Colorbar(ax=cbax2, mappable=plt.cm.ScalarMappable(cmap=colormap_immunodominance))

                            ax1.set_xticks(np.arange(weights.shape[1]) + 0.5,
                                           labels=["{}".format(i) for i in range(weights.shape[1])])
                            ax1.spines['left'].set_visible(False)
                            ax1.yaxis.set_ticklabels([])
                            ax1.set_title("Information shift by weight")
                            # Highlight: Aminoacids coloured by name

                            if np.rint(weights).max() != 1:
                                weights_adjusted = np.array((weights > weights.mean()))
                                aminoacids_masked = (aminoacids[:, 1]) * weights_adjusted.astype(int)
                            else:
                                aminoacids_masked = (aminoacids[:, 1]) * np.rint(weights)

                            sns.heatmap(aminoacids_masked, ax=ax2, cbar=False, cmap=aa_colormap)
                            ax2.set_xticks(np.arange(weights.shape[1]) + 0.5,labels=["{}".format(i) for i in range(weights.shape[1])])
                            ax2.spines['left'].set_visible(False)
                            ax2.yaxis.set_ticklabels([])
                            ax2.set_title("Information shift by \n amino acid type")
                            # Highlight: Aminoacids coloured by functional group (i.e positive, negative ...)
                            sns.heatmap(aminoacids_masked, ax=ax4, cbar=False, cmap=aa_groups_colormap)
                            ax4.set_xticks(np.arange(max_len) + 0.5,
                                           labels=["{}".format(i) for i in range(max_len)])
                            ax4.spines['left'].set_visible(False)
                            ax4.yaxis.set_ticklabels([])
                            ax4.set_title("Information shift by \n amino acid group")

                            ax3.axis("off")
                            ax5.axis("off")
                            ax6.axis("off")

                            # legend1 = plt.legend(handles=aa_patches, prop={'size': 8}, loc='best',
                            #                      bbox_to_anchor=(2.1, -7.4), ncol=2)
                            # plt.legend(handles=aa_groups_patches, prop={'size': 8}, loc='best',
                            #            bbox_to_anchor=(1.6, -7.4), ncol=1)
                            # plt.gca().add_artist(legend1)

                            legend1 = plt.legend(handles=aa_patches, prop={'size': 10}, loc='best',
                                                 bbox_to_anchor=(29, -0.3), ncol=2)
                            plt.legend(handles=aa_groups_patches, prop={'size': 10}, loc='best',
                                       bbox_to_anchor=(14, -0.4), ncol=1)
                            plt.gca().add_artist(legend1)

                            #fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
                            plt.subplots_adjust(left=0.1,hspace=0.39,wspace=0.3)
                            #fig.suptitle("{}. Information shift weights: {}, {}, {}".format(nn_name,sample_mode, data_points, class_type))
                            fig.suptitle("{}.Information shift weights".format(nn_name))
                            plt.savefig("{}/{}/{}_information_shift_plots_{}_{}_{}.png".format(results_dir, method, nn_name,sample_mode, data_points, class_type))
                            plt.clf()
                            plt.close(fig)
                            del aminoacids_masked


                else:
                    print("Not data points available for {}/{}/{}/{}".format(sample_mode, data_points,class_type,nn_name))

    del aminoacids,encoder_information_shift_weights,decoder_information_shift_weights,encoder_hidden_states,decoder_hidden_states
    del data_mask,data_int,data_mask_seq,idx_all,idx_highconfidence
    gc.collect()

def plot_logits_entropies(train_dict,valid_dict,epochs_list, mode,results_dir):
    """Plot positional entropies calculated from logits"""
    train_entropies = np.array(train_dict["entropies"]) #[num_epochs, num_positions]
    valid_entropies = np.array(valid_dict["entropies"])

    epochs_idx = np.array(epochs_list)
    train_entropies = train_entropies[epochs_idx.astype(int)]  # select the same epochs as the validation
    colormap_train = matplotlib.cm.get_cmap('Blues', train_entropies.shape[1])
    colormap_valid = matplotlib.cm.get_cmap('Oranges', train_entropies.shape[1])

    fig, [ax1, ax2] = plt.subplots(1, 2,figsize=(17, 12),gridspec_kw={'width_ratios': [4.5,1]})

    for position in range(train_entropies.shape[1]):
        ax1.plot(epochs_idx, train_entropies[:,position], color=colormap_train(position), label="train_{}".format(position))
    if valid_entropies is not None:
        for position in range(valid_entropies.shape[1]):
            ax1.plot(epochs_idx, valid_entropies[:,position], color=colormap_valid(position), label="validation_{}".format(position))

    ax2.axis("off")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Entropy (positional)",fontsize=15)
    ax1.set_title("Entropy (Train/valid)",fontsize=15)
    ax1.legend(prop={'size': 15},loc= 'center right',bbox_to_anchor=(1.3,0.5),ncol=1)
    plt.savefig("{}/entropies_{}fold.png".format(results_dir, mode))
    plt.clf()
    plt.close(fig)

def plot_volumetrics(volumetrics_dict,seq_max_len,labels,storage_folder,args,subfolders,tag=""):
    fig, [[ax1, ax2, ax3],[ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(9, 6),gridspec_kw={'width_ratios': [4.5,4.5,1],'height_ratios': [4.5, 4.5]})
    if volumetrics_dict["volume"] is not None:
        #norm = plt.Normalize(0, 1)
        norm= None
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        cbar = True

        sns.heatmap(volumetrics_dict["volume"], ax=ax1, cbar=cbar, cmap=cmap,norm=norm)
        ax1.set_xticks(np.arange(1,seq_max_len) + 0.5,labels=["{}".format(i) for i in range(1,seq_max_len)])
        ax1.spines['left'].set_visible(False)
        ax1.yaxis.set_ticklabels([])
        if labels is not None:
            pearson = np.corrcoef(np.array(labels),volumetrics_dict["volume"].sum(axis=1))
            pointbiserial = stats.pointbiserialr(np.array(labels),volumetrics_dict["volume"].sum(axis=1))
            spearman = stats.spearmanr(a= volumetrics_dict["volume"].sum(axis=1), b=np.array(labels),axis=1)
            ax1.set_title("Volume.\n Pointbiserial : {} \n Spearman: {} \n Pearson: {} ".format(round(pointbiserial.correlation,2),round(spearman.correlation,2),round(pearson[0][1],2)))
        else:
            ax1.set_title("Volume")


        sns.heatmap(volumetrics_dict["molecular_weights"], ax=ax2, cbar=cbar, cmap=cmap,norm=norm)
        ax2.set_xticks(np.arange(1,seq_max_len) + 0.5,labels=["{}".format(i) for i in range(1,seq_max_len)])
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.set_ticklabels([])
        if labels is not None:
            pearson = np.corrcoef(np.array(labels),volumetrics_dict["molecular_weights"].sum(axis=1))
            pointbiserial = stats.pointbiserialr(np.array(labels), volumetrics_dict["molecular_weights"].sum(axis=1))
            spearman = stats.spearmanr(a=volumetrics_dict["molecular_weights"].sum(axis=1), b=np.array(labels), axis=1)
            ax2.set_title("Molecular weights.\n Pointbiserial : {}\n Spearman: {} \n Pearson: {}".format(round(pointbiserial.correlation,2),round(spearman.correlation,2),round(pearson[0][1],2)))
        else:
            ax2.set_title("Molecular weights")

        sns.heatmap(volumetrics_dict["radius"], ax=ax4, cbar=cbar, cmap=cmap,norm=norm)
        ax4.set_xticks(np.arange(1,seq_max_len) + 0.5,labels=["{}".format(i) for i in range(1,seq_max_len)])
        ax4.spines['left'].set_visible(False)
        ax4.yaxis.set_ticklabels([])
        if labels is not None:
            pearson = np.corrcoef(np.array(labels),volumetrics_dict["radius"].sum(axis=1))
            pointbiserial = stats.pointbiserialr(np.array(labels), volumetrics_dict["radius"].sum(axis=1))
            spearman = stats.spearmanr(a=volumetrics_dict["radius"].sum(axis=1), b=np.array(labels), axis=1)
            ax4.set_title("Radius.\n Pointbiserial : {}\n Spearman: {} \n Pearson: {}".format(round(pointbiserial.correlation,2),round(spearman.correlation,2),round(pearson[0][1],2)))
        else:
            ax4.set_title("Radius")


        sns.heatmap(volumetrics_dict["bulkiness"], ax=ax5, cbar=cbar, cmap=cmap,norm=norm)
        ax5.set_xticks(np.arange(1,seq_max_len) + 0.5,labels=["{}".format(i) for i in range(1,seq_max_len)])
        ax5.spines['left'].set_visible(False)
        ax5.yaxis.set_ticklabels([])
        if labels is not None:
            pearson = np.corrcoef(np.array(labels),volumetrics_dict["bulkiness"].sum(axis=1))
            pointbiserial = stats.pointbiserialr(np.array(labels), volumetrics_dict["bulkiness"].sum(axis=1))
            spearman = stats.spearmanr(a=volumetrics_dict["bulkiness"].sum(axis=1), b=np.array(labels), axis=1)
            ax5.set_title("Bulkiness. \n Pointbiserial : {}\n Spearman: {} \n Pearson: {}".format(round(pointbiserial.correlation,2),round(spearman.correlation,2),round(pearson[0][1],2)))
        else:
            ax5.set_title("Bulkiness")

        ax3.axis("off")
        ax6.axis("off")


        fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.2)

        fig.suptitle("Volumetrics")

        plt.savefig("{}/{}/similarities/{}/HEATMAP_volumetrics{}.png".format(storage_folder,args.dataset_name,subfolders,tag))
        plt.clf()
        plt.close(fig)

def plot_features_covariance(sequences_raw,features_dict,seq_max_len,labels,storage_folder,args,subfolders,tag="",use_precomputed_features=True):
    """
    :param labels: immunodominance scores or  binary targets
    """

    fontsize=25
    label_names = {"_immunodominance_scores":"Immunodominance",
                   "_binary_labels":"Binary targets"}
    #Highlight: Load the precomputed (with other softwares) sequence features
    if (not args.shuffle_sequence) and (not args.random_sequences) and (not args.num_mutations != 0) and (args.sequence_type == "Icore"):
        if (args.num_classes == args.num_obs_classes) and use_precomputed_features:
            sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), sequences_raw))
            sequences_raw = pd.DataFrame({"Icore": sequences_raw})
            all_feats = pd.read_csv("{}/common_files/dataset_all_features_with_test.tsv".format(storage_folder),sep="\t")
            peptide_feats_cols = all_feats.columns[(all_feats.columns.str.contains("Icore")) | (all_feats.columns.str.contains(pat = 'pep_'))]
            peptide_feats = all_feats[peptide_feats_cols]
            sequences_feats = VegvisirUtils.merge_in_left_order(sequences_raw, peptide_feats, "Icore")
            sequences_feats = sequences_feats.groupby('Icore', as_index=False, sort=False)[peptide_feats_cols[peptide_feats_cols != "Icore"]].agg(lambda x: sum(list(x)) / len(list(x))) #sort Falsse to not mess up the order in which the sequences come out from the model
            sequences_feats = sequences_feats[~sequences_feats[peptide_feats_cols[1]].isna()]

            sequences_feats = sequences_feats.to_dict(orient="list")
            sequences_feats.pop('Icore', None)

            #Highlight: Merge both features dict
            if sequences_feats[peptide_feats_cols[1]]:
                features_dict = {**features_dict, **sequences_feats}



    features_dict1 = {key: features_dict[key] for i, key in enumerate(features_dict) if i < 10}
    features_dict2 = {key: features_dict[key] for i, key in enumerate(features_dict) if i >= 10 and i < 30}

    def calculate_spearman(features_dict, targets, method="spearman"):
        spearman_correlations = list(map(lambda feat1, feat2: VegvisirUtils.calculate_correlations(feat1, feat2, method=method),[targets] * len(features_dict.keys()), list(features_dict.values())))
        spearman_correlations = list(zip(*spearman_correlations))
        spearman_coefficients = np.array(spearman_correlations[0])
        spearman_coefficients = np.round(spearman_coefficients, 2)
        spearman_pvalues = np.array(spearman_correlations[1])
        spearman_pvalues = np.round(spearman_pvalues, 3)
        return spearman_correlations, spearman_coefficients, spearman_pvalues
    
    if tag == "_immunodominance_scores" and features_dict["volume"] is not None:

        spearman_correlations, spearman_coefficients, spearman_pvalues = calculate_spearman(features_dict,labels, "spearman")
        spearman_correlations1, spearman_coefficients1, spearman_pvalues1 = calculate_spearman(features_dict1,labels, "spearman")
        spearman_correlations2, spearman_coefficients2, spearman_pvalues2 = calculate_spearman(features_dict2,labels, "spearman")

    else:
        if features_dict["volume"] is not None:
            # spearman_correlations = list(map(lambda feat1,feat2: VegvisirUtils.calculate_correlations(feat2, feat1),[labels]*len(features_dict.keys()),list(features_dict.values())))
            # spearman_correlations = list(zip(*spearman_correlations))
            # spearman_coefficients = np.array(spearman_correlations[0])
            # spearman_coefficients = np.round(spearman_coefficients,2)
            # spearman_pvalues = np.array(spearman_correlations[1])
            # spearman_pvalues = np.round(spearman_pvalues,3)

            spearman_correlations, spearman_coefficients, spearman_pvalues = calculate_spearman(features_dict, labels,"")
            spearman_correlations1, spearman_coefficients1, spearman_pvalues1 = calculate_spearman(features_dict1, labels,"")
            spearman_correlations2, spearman_coefficients2, spearman_pvalues2 = calculate_spearman(features_dict2, labels,"")

    if features_dict["volume"] is not None:
        with plt.style.context('classic'):
            fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(25, 20), gridspec_kw={'width_ratios': [4.5, 1]})


            features_names = list(features_dict.keys()) + [tag.replace("_","")]
            features_matrix = np.array(list(features_dict.values()))
            features_matrix = np.vstack([features_matrix,labels[None,:]])
            features_covariance = np.cov(features_matrix)

            #norm = plt.Normalize(0, 1)
            norm= None
            cmap = sns.color_palette("rocket_r", as_cmap=True)
            cbar = True

            sns.heatmap(features_covariance, ax=ax1, cbar=cbar, cmap=cmap,norm=norm,annot=True,annot_kws={"fontsize":14},fmt=".2f")
            ax1.set_xticks(np.arange(len(features_names)) ,labels=features_names,rotation=45,fontsize=fontsize,weight='bold')
            ax1.spines['left'].set_visible(False)
            #ax1.yaxis.set_ticklabels([])
            ax1.set_yticks(np.arange(len(features_names)) + 0.5,labels=features_names,rotation=360,fontsize=fontsize,weight='bold')
            ymax=len(features_names)-1
            xpos=0
            ax1.add_patch(matplotlib.patches.Rectangle((ymax, xpos), 1, len(features_names), fill=False, edgecolor='green', lw=3))
            ax1.set_title("Covariance matrix features ({})".format(subfolders.replace("/",",")),fontsize=fontsize,weight='bold')

            ax2.axis("off")
            fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.2)
            #fig.suptitle("Features covariance")
            plt.savefig("{}/{}/similarities/{}/HEATMAP_features_covariance{}.png".format(storage_folder,args.dataset_name,subfolders,tag))
            plt.clf()
            plt.close(fig)

        #Highlight: Plot correlations
        with plt.style.context('classic'):
            fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(18, 11),sharey="col") #gridspec_kw={'width_ratios': [4.5, 0.5]}


            n_feats = len(features_dict1.keys())
            index = np.arange(n_feats)
            positive_idx = np.array(spearman_coefficients1 >= 0) #divide the coefficients in negative and positive to plot them separately
            right_arr = np.zeros(n_feats)
            left_arr = np.zeros(n_feats)
            right_arr[positive_idx] = spearman_coefficients1[positive_idx]
            left_arr[~positive_idx] = spearman_coefficients1[~positive_idx]

            ax1.barh(index,left_arr, align="center",color="mediumorchid",zorder=1) #zorder indicates the plotting order, supposedly
            ax1.barh(index,right_arr, align="center",color="seagreen",zorder=2)

            position_labels = list(range(0,n_feats))
            ax1.axvline(0)
            aa_dict = VegvisirUtils.aa_dict_1letter_full()
            def clean_labels(label):
                if label == "extintion_coefficient_cystines":
                    label = "extintion coefficient \n (cystines)"
                elif label == "extintion_coefficient_cysteines":
                    label = "extintion coefficient \n (cysteines)"
                elif label in aa_dict.keys():
                    label = aa_dict[label]
                else:
                    label = label.replace("_"," ")
                return label
            labels_names = list(map(lambda label: clean_labels(label), list(features_dict1.keys())))
            ax1.yaxis.set_ticks(position_labels)
            ax1.set_yticklabels(labels_names,fontsize=fontsize,rotation=0,weight='bold')
            ax1.tick_params(axis="x",labelsize=fontsize-8,top=False)
            #ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
            ax1.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                left=False,
                right=False)  # labels along the bottom edge are off
            plt.subplots_adjust(left=0.28)
            #ax1.margins(y=0.15)
            ax1.spines[['right', 'top','left']].set_visible(False)
            ##########################################################################
            n_feats = len(features_dict2.keys())
            index = np.arange(n_feats)
            positive_idx = np.array(spearman_coefficients2 >= 0)  # divide the coefficients in negative and positive to plot them separately
            right_arr = np.zeros(n_feats)
            left_arr = np.zeros(n_feats)
            right_arr[positive_idx] = spearman_coefficients2[positive_idx]
            left_arr[~positive_idx] = spearman_coefficients2[~positive_idx]

            ax2.barh(index, left_arr, align="center", color="mediumorchid",zorder=1)  # zorder indicates the plotting order, supposedly
            ax2.barh(index, right_arr, align="center", color="seagreen", zorder=2)

            position_labels = list(range(0, n_feats))
            ax2.axvline(0)

            labels_names = list(map(lambda label: clean_labels(label), list(features_dict2.keys())))
            ax2.yaxis.set_ticks(position_labels)
            ax2.set_yticklabels(labels_names, fontsize=fontsize, rotation=0, weight='bold')
            ax2.tick_params(axis="x", labelsize=fontsize-8,top=False)
            # ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
            ax2.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                left=False,
                right=False)  # labels along the bottom edge are off
            plt.subplots_adjust(left=0.25,wspace=0.6)
            # ax2.margins(y=0.15)
            ax2.spines[['right', 'top', 'left']].set_visible(False)
            
            

            fig.suptitle("Correlation coefficients: Features vs {}".format(label_names[tag]),fontsize=fontsize + 8,weight='bold')
            plt.savefig("{}/{}/similarities/{}/HISTOGRAM_features_correlations{}.png".format(storage_folder,args.dataset_name,subfolders,tag),dpi=700)

def calculate_species_roc_auc_helper(summary_dict,args,script_dir,idx_all,fold,prob_mode,sample_mode,mode="train_species",compute_species_auc=False):
    
    if compute_species_auc:
        data_int = summary_dict["data_int_{}".format(sample_mode)]
        org_name = data_int[:, 0, 6]
        unique_org_name, counts = np.unique(org_name, return_counts=True)
        labels = summary_dict["true_{}".format(sample_mode)]
        onehot_labels = summary_dict["true_onehot_{}".format(sample_mode)]
        # confidence_scores = summary_dict["confidence_scores_{}".format(sample_mode)]
        # idx_all = np.ones_like(labels).astype(bool)
        # if args.num_classes > args.num_obs_classes:
        #     idx_all = (labels[..., None] != 2).any(
        #         -1)  # Highlight: unlabelled data has been assigned labelled 2, we give high confidence to the labelled data (for now)
        # idx_highconfidence = (confidence_scores[..., None] > 0.7).any(-1)
        species_auc_class_0 = []
        species_auc_class_1 = []

        species_pval_class_0 = []
        species_pval_class_1 = []

        for species in unique_org_name.tolist():
            idx = (org_name[..., None] == species).any(-1)
            idx *= idx_all  # TODO: Check
            idx_name = str(int(species))
            species_labels = labels[idx]

            #onehot_species_labels = onehot_labels[idx]

            if species_labels.shape[0] > 100:
                idx_name = "all"
                sample_mode = "samples"

                fpr, tpr, roc_auc,pvals,ppv_mod = plot_ROC_curves(labels, onehot_labels, summary_dict, args, script_dir, mode, fold,
                                                    sample_mode,
                                                    prob_mode, idx, idx_name, save=False) #Highlight: We input labels and not species_labels because it is indexed inside the function again
                species_auc_class_0.append(roc_auc[0])
                species_auc_class_1.append(roc_auc[1])
                species_pval_class_0.append(pvals[0])
                species_pval_class_1.append(pvals[1])


        species_auc_class_0 = np.array(species_auc_class_0)
        species_auc_class_0 = species_auc_class_0[~np.isnan(species_auc_class_0)]
        species_auc_class_1 = np.array(species_auc_class_1)
        species_auc_class_1 = species_auc_class_1[~np.isnan(species_auc_class_1)]

        species_pval_class_0 = np.array(species_pval_class_0)
        species_pval_class_0 = species_pval_class_0[~np.isnan(species_pval_class_0)]
        species_pval_class_1 = np.array(species_pval_class_1)
        species_pval_class_1 = species_pval_class_1[~np.isnan(species_pval_class_1)]
        return np.mean(species_auc_class_0),np.mean(species_auc_class_1),np.mean(species_pval_class_0),np.mean(species_pval_class_1)
    else:
        return np.nan,np.nan,np.nan,np.nan

def plot_kfold_comparison_helper(metrics_keys,script_dir,folder,overwrite,kfolds):
    """"""
    metrics_results_train = dict.fromkeys(metrics_keys)
    metrics_results_valid = dict.fromkeys(metrics_keys)
    metrics_results_test = dict.fromkeys(metrics_keys)
    metrics_results_train_species = dict.fromkeys(metrics_keys)
    metrics_results_valid_species = dict.fromkeys(metrics_keys)
    metrics_results_test_species = dict.fromkeys(metrics_keys)
    if os.path.exists("{}/Vegvisir_checkpoints/roc_auc_train.p".format(folder)) and not overwrite:
        print("Loading pre-computed ROC-AUC values")
        metrics_results_train = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_train.p".format(folder), "rb"))

        train_folds_ap_class_0,train_folds_ap_class_1,train_folds_ppv,train_folds_ppv_mod_class_0,train_folds_ppv_mod_class_1,train_folds_fpr, train_folds_tpr, train_folds_roc_auc_class_0, train_folds_roc_auc_class_1,train_folds_auc01_class_0, train_folds_auc01_class_1, train_folds_pvals_class_0, train_folds_pvals_class_1 = (
            metrics_results_train["ap_class_0"],metrics_results_train["ap_class_1"],metrics_results_train["ppv"],metrics_results_train["ppv_mod_class_0"],metrics_results_train["ppv_mod_class_1"],metrics_results_train["fpr"], metrics_results_train["tpr"], metrics_results_train["roc_auc_class_0"], metrics_results_train["roc_auc_class_1"],metrics_results_train["auc01_class_0"], metrics_results_train["auc01_class_1"], metrics_results_train["pval_class_0"], metrics_results_train["pval_class_1"])

        train_folds_precision_class_0,train_folds_precision_class_1,train_folds_recall_class_0,train_folds_recall_class_1 = metrics_results_train["precision_class_0"],metrics_results_train["precision_class_1"],metrics_results_train["recall_class_0"],metrics_results_train["recall_class_1"]

       
       
        metrics_results_train_species = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_train_species.p".format(folder), "rb"))
        train_species_folds_ap_class_0,train_species_folds_ap_class_1,train_species_folds_ppv,train_species_folds_ppv_mod_class_0,train_species_folds_ppv_mod_class_1,train_species_folds_fpr, train_species_folds_tpr, train_species_folds_roc_auc_class_0, train_species_folds_roc_auc_class_1,train_species_folds_auc01_class_0, train_species_folds_auc01_class_1,train_species_folds_pvals_class_0, train_species_folds_pvals_class_1 = (
            metrics_results_train_species["ap_class_0"],metrics_results_train_species["ap_class_1"],metrics_results_train_species["ppv"],metrics_results_train_species["ppv_mod_class_0"],metrics_results_train_species["ppv_mod_class_1"],metrics_results_train_species["fpr"], metrics_results_train_species["tpr"], metrics_results_train_species["roc_auc_class_0"], metrics_results_train_species["roc_auc_class_1"],metrics_results_train_species["auc01_class_0"], metrics_results_train_species["auc01_class_1"], metrics_results_train_species["pval_class_0"], metrics_results_train_species["pval_class_1"])

        train_species_folds_precision_class_0,train_species_folds_precision_class_1,train_species_folds_recall_class_0,train_species_folds_recall_class_1 = metrics_results_train_species["precision_class_0"],metrics_results_train_species["precision_class_1"],metrics_results_train_species["recall_class_0"],metrics_results_train_species["recall_class_1"]

        
        
        if os.path.exists("{}/Vegvisir_checkpoints/roc_auc_valid.p".format(folder)):
            metrics_results_valid = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_valid.p".format(folder), "rb"))
            valid_folds_ap_class_0,valid_folds_ap_class_1,valid_folds_ppv,valid_folds_ppv_mod_class_0,valid_folds_ppv_mod_class_1,valid_folds_fpr, valid_folds_tpr, valid_folds_roc_auc_class_0, valid_folds_roc_auc_class_1,valid_folds_auc01_class_0, valid_folds_auc01_class_1, train_folds_pvals_class_0, train_folds_pvals_class_1 = (
                metrics_results_valid["ap_class_0"],metrics_results_valid["ap_class_1"],metrics_results_valid["ppv"],metrics_results_valid["ppv_mod_class_0"],metrics_results_valid["ppv_mod_class_1"],metrics_results_valid["fpr"], metrics_results_valid["tpr"], metrics_results_valid["roc_auc_class_0"],
                metrics_results_valid["roc_auc_class_1"],metrics_results_valid["auc01_class_0"],metrics_results_valid["auc01_class_1"], metrics_results_valid["pval_class_0"], metrics_results_valid["pval_class_1"])

            valid_folds_precision_class_0, valid_folds_precision_class_1, valid_folds_recall_class_0, valid_folds_recall_class_1 = \
            metrics_results_valid["precision_class_0"], metrics_results_valid["precision_class_1"], \
            metrics_results_valid["recall_class_0"], metrics_results_valid["recall_class_1"]



            metrics_results_valid_species = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_valid_species.p".format(folder), "rb"))
            valid_species_folds_ap_class_0,valid_species_folds_ap_class_1,valid_species_folds_ppv,valid_species_folds_ppv_mod_class_0,valid_species_folds_ppv_mod_class_1,valid_species_folds_fpr, valid_species_folds_tpr, valid_species_folds_roc_auc_class_0, valid_species_folds_roc_auc_class_1,valid_species_folds_auc01_class_0, valid_species_folds_auc01_class_1, valid_species_folds_pvals_class_0, valid_species_folds_pvals_class_1 =(
                metrics_results_valid_species["ap_class_0"],metrics_results_valid_species["ap_class_1"],metrics_results_valid_species["ppv"],metrics_results_valid_species["ppv_mod_class_0"],metrics_results_valid_species["ppv_mod_class_1"], metrics_results_valid_species["fpr"], metrics_results_valid_species["tpr"],metrics_results_valid_species["roc_auc_class_0"], metrics_results_valid_species["roc_auc_class_1"],metrics_results_valid_species["auc01_class_0"], metrics_results_valid_species["auc01_class_1"], metrics_results_valid_species["pval_class_0"], metrics_results_valid_species["pval_class_1"])

            valid_species_folds_precision_class_0, valid_species_folds_precision_class_1, valid_species_folds_recall_class_0, valid_species_folds_recall_class_1 = \
            metrics_results_valid_species["precision_class_0"], metrics_results_valid_species["precision_class_1"], \
            metrics_results_valid_species["recall_class_0"], metrics_results_valid_species["recall_class_1"]

        if os.path.exists("{}/Vegvisir_checkpoints/roc_auc_test.p".format(folder)):
            metrics_results_test = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_test.p".format(folder), "rb"))
            test_folds_ap_class_0,test_folds_ap_class_1,test_folds_ppv,test_folds_ppv_mod_class_0,test_folds_ppv_mod_class_1,test_folds_fpr, test_folds_tpr, test_folds_roc_auc_class_0, test_folds_roc_auc_class_1,test_folds_auc01_class_0, test_folds_auc01_class_1, test_folds_pvals_class_0, test_folds_pvals_class_1 = (
                metrics_results_test["ap_class_0"],metrics_results_test["ap_class_1"],metrics_results_test["ppv"],metrics_results_test["ppv_mod_class_0"],metrics_results_test["ppv_mod_class_1"],metrics_results_test["fpr"], metrics_results_test["tpr"], metrics_results_test["roc_auc_class_0"], metrics_results_test["roc_auc_class_1"],metrics_results_test["auc01_class_0"], metrics_results_test["auc01_class_1"], metrics_results_test["pval_class_0"], metrics_results_test["pval_class_1"])

            test_folds_precision_class_0, test_folds_precision_class_1, test_folds_recall_class_0, test_folds_recall_class_1 = \
                metrics_results_test["precision_class_0"], metrics_results_test["precision_class_1"], \
                    metrics_results_test["recall_class_0"], metrics_results_test["recall_class_1"]
            
            
            metrics_results_test_species = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_test_species.p".format(folder), "rb"))
            test_species_folds_ap_class_0,test_species_folds_ap_class_1,test_species_folds_ppv,test_species_folds_ppv_mod_class_0,test_species_folds_ppv_mod_class_1,test_species_folds_fpr, test_species_folds_tpr, test_species_folds_roc_auc_class_0, test_species_folds_roc_auc_class_1,test_species_folds_auc01_class_0, test_species_folds_auc01_class_1, test_species_folds_pvals_class_0, test_species_folds_pvals_class_1 = (
                metrics_results_test_species["ap_class_0"],metrics_results_test_species["ap_class_1"],metrics_results_test_species["ppv"],metrics_results_test_species["ppv_mod_class_0"],metrics_results_test_species["ppv_mod_class_1"],metrics_results_test_species["fpr"], metrics_results_test_species["tpr"], metrics_results_test_species["roc_auc_class_0"], metrics_results_test_species["roc_auc_class_1"],metrics_results_test_species["auc01_class_0"], metrics_results_test_species["auc01_class_1"], metrics_results_test_species["pval_class_0"], metrics_results_test_species["pval_class_1"])
            
            test_species_folds_precision_class_0, test_species_folds_precision_class_1, test_species_folds_recall_class_0, test_species_folds_recall_class_1 = \
            metrics_results_test_species["precision_class_0"], metrics_results_test_species["precision_class_1"], \
            metrics_results_test_species["recall_class_0"], metrics_results_test_species["recall_class_1"]
    
    
    else:
        print("calculating ROC-AUC values")
        train_folds_ap_class_0,train_folds_ap_class_1,train_folds_ppv,train_folds_ppv_mod_class_0,train_folds_ppv_mod_class_1,train_folds_fpr, train_folds_tpr, train_folds_roc_auc_class_0, train_folds_roc_auc_class_1, train_folds_auc01_class_0, train_folds_auc01_class_1, train_folds_pvals_class_0, train_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[],[],[],[],[]
        train_folds_precision_class_0,train_folds_precision_class_1,train_folds_recall_class_0,train_folds_recall_class_1 = [],[],[],[]
        
        valid_folds_ap_class_0,valid_folds_ap_class_1,valid_folds_ppv,valid_folds_ppv_mod_class_0,valid_folds_ppv_mod_class_1,valid_folds_fpr, valid_folds_tpr, valid_folds_roc_auc_class_0, valid_folds_roc_auc_class_1,valid_folds_auc01_class_0, valid_folds_auc01_class_1, valid_folds_pvals_class_0, valid_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[],[],[],[],[]
        valid_folds_precision_class_0,valid_folds_precision_class_1,valid_folds_recall_class_0,valid_folds_recall_class_1 = [],[],[],[]

        test_folds_ap_class_0,test_folds_ap_class_1,test_folds_ppv,test_folds_ppv_mod_class_0,test_folds_ppv_mod_class_1,test_folds_fpr, test_folds_tpr, test_folds_roc_auc_class_0, test_folds_roc_auc_class_1,test_folds_auc01_class_0, test_folds_auc01_class_1, test_folds_pvals_class_0, test_folds_pvals_class_1 =[], [], [], [], [], [], [],[],[],[],[],[],[]
        test_folds_precision_class_0,test_folds_precision_class_1,test_folds_recall_class_0,test_folds_recall_class_1 = [],[],[],[]

        train_species_folds_ap_class_0,train_species_folds_ap_class_1,train_species_folds_ppv,train_species_folds_ppv_mod_class_0,train_species_folds_ppv_mod_class_1,train_species_folds_fpr, train_species_folds_tpr, train_species_folds_roc_auc_class_0, train_species_folds_roc_auc_class_1,train_species_folds_auc01_class_0, train_species_folds_auc01_class_1, train_species_folds_pvals_class_0, train_species_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[],[],[],[],[]
        train_species_folds_precision_class_0,train_species_folds_precision_class_1,train_species_folds_recall_class_0,train_species_folds_recall_class_1 = [],[],[],[]

        valid_species_folds_ap_class_0,valid_species_folds_ap_class_1,valid_species_folds_ppv,valid_species_folds_ppv_mod_class_0,valid_species_folds_ppv_mod_class_1,valid_species_folds_fpr, valid_species_folds_tpr, valid_species_folds_roc_auc_class_0, valid_species_folds_roc_auc_class_1,valid_species_folds_auc01_class_0, valid_species_folds_auc01_class_1, valid_species_folds_pvals_class_0, valid_species_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[],[],[],[],[]
        valid_species_folds_precision_class_0,valid_species_folds_precision_class_1,valid_species_folds_recall_class_0,valid_species_folds_recall_class_1 = [],[],[],[]

        test_species_folds_ap_class_0,test_species_folds_ap_class_1,test_species_folds_ppv,test_species_folds_ppv_mod_class_0,test_species_folds_ppv_mod_class_1,test_species_folds_fpr, test_species_folds_tpr, test_species_folds_roc_auc_class_0, test_species_folds_roc_auc_class_1,test_species_folds_auc01_class_0, test_species_folds_auc01_class_1, test_species_folds_pvals_class_0, test_species_folds_pvals_class_1 = [], [], [],[], [], [], [],[],[],[],[],[],[]
        test_species_folds_precision_class_0,test_species_folds_precision_class_1,test_species_folds_recall_class_0,test_species_folds_recall_class_1 = [],[],[],[]

        for fold in range(kfolds):
            print("-------------FOLD {}--------------".format(fold))
            if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold)): #use the results of the entire training dataset
                train_load = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold))  #TODO: I changed train_out to train_load
                args = train_load["args"]
                train_out = train_load["summary_dict"]
            else:
                train_load = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_valid_fold_{}.p".format(folder, fold))  #use the results of the training dataset minus the validation
                args = train_load["args"]
                train_out = train_load["summary_dict"]
            valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(folder, fold))["summary_dict"]
            if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold)):
                test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold))["summary_dict"]
            else:
                test_out = None

            for mode, summary_dict in zip(["train", "valid", "test"], [train_out, valid_out, test_out]):

                if summary_dict is not None:
                    labels = summary_dict["true_samples"]
                    onehot_labels = summary_dict["true_onehot_samples"]
                    prob_mode = "class_probs_predictions_samples_average"
                    idx_all = np.ones_like(labels).astype(bool)
                    if args.num_classes > args.num_obs_classes:
                        idx_all = (labels[..., None] != 2).any(-1)  # Highlight: unlabelled data has been assigned labelled 2,we skip it
                    idx_name = "all"
                    sample_mode = "samples"

                    fpr, tpr, roc_auc, pvals, ppv_mod = plot_ROC_curves(labels, onehot_labels, summary_dict, args, script_dir,
                                                               mode,
                                                               fold, sample_mode,
                                                               prob_mode, idx_all, idx_name, save=False)

                    binary_predictions = np.argmax(summary_dict[prob_mode],axis=1)

                    if args.num_classes == 2: #supervised
                        tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=binary_predictions).ravel()
                        precision = tp / (tp + fp)
                    elif args.num_classes == 3: #semisupervised
                        cm = confusion_matrix(y_true=labels, y_pred=binary_predictions)
                        precisions = precision_score(y_true=labels, y_pred=binary_predictions, average=None)
                        precision = np.mean(precisions) #average precision

                    ap_dict = plot_precision_recall_curve(labels, onehot_labels, summary_dict, args, script_dir, mode, fold,"", prob_mode, idx_all, idx_name, save_plot=False)

                    if mode == "train":

                        train_folds_ap_class_0.append(ap_dict["average_precision"][0])
                        train_folds_ap_class_1.append(ap_dict["average_precision"][1])
                        train_folds_ppv.append(precision)
                        train_folds_recall_class_0.append(ap_dict["recall"]["max_0"])
                        train_folds_recall_class_1.append(ap_dict["recall"]["max_1"])
                        train_folds_precision_class_0.append(ap_dict["precision"]["max_0"])
                        train_folds_precision_class_1.append(ap_dict["precision"]["max_1"])
                        train_folds_ppv_mod_class_0.append(ppv_mod[0])
                        train_folds_ppv_mod_class_1.append(ppv_mod[1])
                        train_folds_fpr.append(fpr)
                        train_folds_tpr.append(tpr)
                        train_folds_roc_auc_class_0.append(roc_auc[0])
                        train_folds_roc_auc_class_1.append(roc_auc[1])
                        train_folds_auc01_class_0.append(roc_auc["auc01_class_0"])
                        train_folds_auc01_class_1.append(roc_auc["auc01_class_1"])
                        train_folds_pvals_class_0.append(pvals[0])
                        train_folds_pvals_class_1.append(pvals[1])
                        species_results = calculate_species_roc_auc_helper(summary_dict, args, script_dir, idx_all,fold, prob_mode, sample_mode,mode="{}_species".format(mode))

                        
                        train_species_folds_ap_class_0.append(np.nan)
                        train_species_folds_ap_class_1.append(np.nan)
                        train_species_folds_ppv.append(np.nan)
                        train_species_folds_recall_class_0.append(ap_dict["recall"]["max_0"])
                        train_species_folds_recall_class_1.append(ap_dict["recall"]["max_1"])
                        train_species_folds_precision_class_0.append(ap_dict["precision"]["max_0"])
                        train_species_folds_precision_class_1.append(ap_dict["precision"]["max_1"])
                        train_species_folds_ppv_mod_class_0.append(ppv_mod)
                        train_species_folds_ppv_mod_class_1.append(ppv_mod)
                        train_species_folds_fpr.append(np.nan)
                        train_species_folds_tpr.append(np.nan)
                        train_species_folds_roc_auc_class_0.append(species_results[0])
                        train_species_folds_roc_auc_class_1.append(species_results[1])
                        train_species_folds_auc01_class_0.append(np.nan)
                        train_species_folds_auc01_class_1.append(np.nan)
                        train_species_folds_pvals_class_0.append(species_results[2])
                        train_species_folds_pvals_class_1.append(species_results[3])

                    elif mode == "valid":
                        valid_folds_ap_class_0.append(ap_dict["average_precision"][0])
                        valid_folds_ap_class_1.append(ap_dict["average_precision"][1])
                        valid_folds_ppv.append(precision)
                        valid_folds_recall_class_0.append(ap_dict["recall"]["max_0"])
                        valid_folds_recall_class_1.append(ap_dict["recall"]["max_1"])
                        valid_folds_precision_class_0.append(ap_dict["precision"]["max_0"])
                        valid_folds_precision_class_1.append(ap_dict["precision"]["max_1"])
                        valid_folds_ppv_mod_class_0.append(ppv_mod[0])
                        valid_folds_ppv_mod_class_1.append(ppv_mod[1])
                        valid_folds_fpr.append(fpr)
                        valid_folds_tpr.append(tpr)
                        valid_folds_roc_auc_class_0.append(roc_auc[0])
                        valid_folds_roc_auc_class_1.append(roc_auc[1])
                        valid_folds_auc01_class_0.append(roc_auc["auc01_class_0"])
                        valid_folds_auc01_class_1.append(roc_auc["auc01_class_1"])
                        valid_folds_pvals_class_0.append(pvals[0])
                        valid_folds_pvals_class_1.append(pvals[1])
                        species_results = calculate_species_roc_auc_helper(summary_dict, args, script_dir, idx_all,
                                                                           fold, prob_mode, sample_mode,
                                                                           mode="{}_species".format(mode))
                        valid_species_folds_ap_class_0.append(np.nan)
                        valid_species_folds_ap_class_1.append(np.nan)
                        valid_species_folds_ppv.append(np.nan)
                        valid_species_folds_recall_class_0.append(np.nan)
                        valid_species_folds_recall_class_1.append(np.nan)
                        valid_species_folds_precision_class_0.append(np.nan)
                        valid_species_folds_precision_class_1.append(np.nan)

                        valid_species_folds_ppv_mod_class_0.append(np.nan)
                        valid_species_folds_ppv_mod_class_1.append(np.nan)
                        valid_species_folds_fpr.append(np.nan)
                        valid_species_folds_tpr.append(np.nan)
                        valid_species_folds_roc_auc_class_0.append(species_results[0])
                        valid_species_folds_roc_auc_class_1.append(species_results[1])
                        valid_species_folds_auc01_class_0.append(np.nan)
                        valid_species_folds_auc01_class_1.append(np.nan)
                        valid_species_folds_pvals_class_0.append(species_results[2])
                        valid_species_folds_pvals_class_1.append(species_results[3])

                    else:
                        test_folds_ap_class_0.append(ap_dict["average_precision"][0])
                        test_folds_ap_class_1.append(ap_dict["average_precision"][1])
                        test_folds_ppv.append(precision)
                        test_folds_recall_class_0.append(ap_dict["recall"]["max_0"])
                        test_folds_recall_class_1.append(ap_dict["recall"]["max_1"])
                        test_folds_precision_class_0.append(ap_dict["precision"]["max_0"])
                        test_folds_precision_class_1.append(ap_dict["precision"]["max_1"])
                        test_folds_ppv_mod_class_0.append(ppv_mod[0])
                        test_folds_ppv_mod_class_1.append(ppv_mod[1])
                        test_folds_fpr.append(fpr)
                        test_folds_tpr.append(tpr)
                        test_folds_roc_auc_class_0.append(roc_auc[0])
                        test_folds_roc_auc_class_1.append(roc_auc[1])
                        test_folds_auc01_class_0.append(roc_auc["auc01_class_0"])
                        test_folds_auc01_class_1.append(roc_auc["auc01_class_1"])
                        test_folds_pvals_class_0.append(pvals[0])
                        test_folds_pvals_class_1.append(pvals[1])
                        
                        test_species_folds_ap_class_0.append(np.nan)
                        test_species_folds_ap_class_1.append(np.nan)
                        test_species_folds_ppv.append(np.nan)
                        test_species_folds_recall_class_0.append(np.nan) 
                        test_species_folds_recall_class_1.append(np.nan)
                        test_species_folds_precision_class_0.append(np.nan)
                        test_species_folds_precision_class_1.append(np.nan)
                        test_species_folds_ppv_mod_class_0.append(np.nan)
                        test_species_folds_ppv_mod_class_1.append(np.nan)
                        test_species_folds_fpr.append(np.nan)
                        test_species_folds_tpr.append(np.nan)
                        test_species_folds_roc_auc_class_0.append(np.nan)
                        test_species_folds_roc_auc_class_1.append(np.nan)
                        test_species_folds_auc01_class_0.append(np.nan)
                        test_species_folds_auc01_class_1.append(np.nan)
                        test_species_folds_pvals_class_0.append(np.nan)
                        test_species_folds_pvals_class_1.append(np.nan)

                else:
                    if mode == "valid":
                        valid_folds_ap_class_0.append(np.nan)
                        valid_folds_ap_class_1.append(np.nan)
                        valid_folds_fpr.append(np.nan)
                        valid_folds_tpr.append(np.nan)
                        valid_folds_ppv.append(np.nan)
                        valid_folds_ppv_mod_class_0.append(np.nan)
                        valid_folds_ppv_mod_class_1.append(np.nan)
                        valid_folds_roc_auc_class_0.append(np.nan)
                        valid_folds_roc_auc_class_1.append(np.nan)
                        valid_folds_auc01_class_0.append(np.nan)
                        valid_folds_auc01_class_1.append(np.nan)
                        valid_folds_pvals_class_0.append(np.nan)
                        valid_folds_pvals_class_1.append(np.nan)
                        
                        valid_species_folds_ppv.append(np.nan)
                        valid_species_folds_fpr.append(np.nan)
                        valid_species_folds_tpr.append(np.nan)
                        valid_species_folds_roc_auc_class_0.append(np.nan)
                        valid_species_folds_roc_auc_class_1.append(np.nan)
                        valid_species_folds_pvals_class_0.append(np.nan)
                        valid_species_folds_pvals_class_1.append(np.nan)
                    else:
                        test_folds_ap_class_0.append(np.nan)
                        test_folds_ap_class_1.append(np.nan)
                        test_folds_fpr.append(np.nan)
                        test_folds_tpr.append(np.nan)
                        test_folds_ppv.append(np.nan)
                        test_folds_recall_class_0.append(np.nan)
                        test_folds_recall_class_1.append(np.nan)
                        test_folds_precision_class_0.append(np.nan)
                        test_folds_precision_class_1.append(np.nan)
                        test_folds_ppv_mod_class_0.append(np.nan)
                        test_folds_ppv_mod_class_1.append(np.nan)
                        test_folds_roc_auc_class_0.append(np.nan)
                        test_folds_roc_auc_class_1.append(np.nan)
                        test_folds_auc01_class_0.append(np.nan)
                        test_folds_auc01_class_1.append(np.nan)
                        test_folds_pvals_class_0.append(np.nan)
                        test_folds_pvals_class_1.append(np.nan)
                        
                        test_species_folds_ppv.append(np.nan)
                        test_species_folds_fpr.append(np.nan)
                        test_species_folds_tpr.append(np.nan)
                        test_species_folds_roc_auc_class_0.append(np.nan)
                        test_species_folds_roc_auc_class_1.append(np.nan)
                        test_species_folds_pvals_class_0.append(np.nan)
                        test_species_folds_pvals_class_1.append(np.nan)

        
        
        metrics_results_train["ap_class_0"] = train_folds_ap_class_0
        metrics_results_train["ap_class_1"] = train_folds_ap_class_1
        metrics_results_train["ppv"] = train_folds_ppv
        metrics_results_train["precision_class_0"] = train_folds_precision_class_0
        metrics_results_train["precision_class_1"] = train_folds_precision_class_1
        metrics_results_train["recall_class_0"] = train_folds_recall_class_0
        metrics_results_train["recall_class_1"] = train_folds_recall_class_1
        metrics_results_train["ppv_mod_class_0"] = train_folds_ppv_mod_class_0
        metrics_results_train["ppv_mod_class_1"] = train_folds_ppv_mod_class_1
        metrics_results_train["fpr"] = train_folds_fpr
        metrics_results_train["tpr"] = train_folds_tpr
        metrics_results_train["roc_auc_class_0"] = train_folds_roc_auc_class_0
        metrics_results_train["roc_auc_class_1"] = train_folds_roc_auc_class_1
        metrics_results_train["auc01_class_0"] = train_folds_auc01_class_0
        metrics_results_train["auc01_class_1"] = train_folds_auc01_class_1

        train_folds_pvals_class_0 = np.array(train_folds_pvals_class_0)
        metrics_results_train["pval_class_0"] = train_folds_pvals_class_0[~np.isnan(train_folds_pvals_class_0)]
        train_folds_pvals_class_1 = np.array(train_folds_pvals_class_1)
        metrics_results_train["pval_class_1"] = train_folds_pvals_class_1[~np.isnan(train_folds_pvals_class_1)]

        
        metrics_results_valid["ap_class_0"] = valid_folds_ap_class_0
        metrics_results_valid["ap_class_1"] = valid_folds_ap_class_1
        metrics_results_valid["ppv"] = valid_folds_ppv
        metrics_results_valid["precision_class_0"] = valid_folds_precision_class_0
        metrics_results_valid["precision_class_1"] = valid_folds_precision_class_1
        metrics_results_valid["recall_class_0"] = valid_folds_recall_class_0
        metrics_results_valid["recall_class_1"] = valid_folds_recall_class_1
        metrics_results_valid["ppv_mod_class_0"] = valid_folds_ppv_mod_class_0
        metrics_results_valid["ppv_mod_class_1"] = valid_folds_ppv_mod_class_1
        metrics_results_valid["fpr"] = valid_folds_fpr
        metrics_results_valid["tpr"] = valid_folds_tpr
        metrics_results_valid["roc_auc_class_0"] = valid_folds_roc_auc_class_0
        metrics_results_valid["roc_auc_class_1"] = valid_folds_roc_auc_class_1
        metrics_results_valid["auc01_class_0"] = valid_folds_auc01_class_0
        metrics_results_valid["auc01_class_1"] = valid_folds_auc01_class_1
        valid_folds_pvals_class_0 = np.array(valid_folds_pvals_class_0)
        metrics_results_valid["pval_class_0"] = valid_folds_pvals_class_0[~np.isnan(valid_folds_pvals_class_0)]
        valid_folds_pvals_class_1 = np.array(valid_folds_pvals_class_1)
        metrics_results_valid["pval_class_1"] = valid_folds_pvals_class_1[~np.isnan(valid_folds_pvals_class_1)]

        metrics_results_test["ap_class_0"] = test_folds_ap_class_0
        metrics_results_test["ap_class_1"] = test_folds_ap_class_1
        metrics_results_test["ppv"] = test_folds_ppv
        metrics_results_test["precision_class_0"] = test_folds_precision_class_0
        metrics_results_test["precision_class_1"] = test_folds_precision_class_1
        metrics_results_test["recall_class_0"] = test_folds_recall_class_0
        metrics_results_test["recall_class_1"] = test_folds_recall_class_1
        metrics_results_test["ppv_mod_class_0"] = test_folds_ppv_mod_class_0
        metrics_results_test["ppv_mod_class_1"] = test_folds_ppv_mod_class_1
        metrics_results_test["fpr"] = test_folds_fpr
        metrics_results_test["tpr"] = test_folds_tpr
        metrics_results_test["roc_auc_class_0"] = test_folds_roc_auc_class_0
        metrics_results_test["roc_auc_class_1"] = test_folds_roc_auc_class_1
        metrics_results_test["auc01_class_0"] = test_folds_auc01_class_0
        metrics_results_test["auc01_class_1"] = test_folds_auc01_class_1
        test_folds_pvals_class_0 = np.array(test_folds_pvals_class_0)
        metrics_results_test["pval_class_0"] = test_folds_pvals_class_0[~np.isnan(test_folds_pvals_class_0)]
        test_folds_pvals_class_1 = np.array(test_folds_pvals_class_1)
        metrics_results_test["pval_class_1"] = test_folds_pvals_class_1[~np.isnan(test_folds_pvals_class_1)]

        
        
        metrics_results_train_species["ap_class_0"] = train_species_folds_ap_class_0
        metrics_results_train_species["ap_class_1"] = train_species_folds_ap_class_1
        metrics_results_train_species["ppv"] = train_species_folds_ppv
        metrics_results_train_species["precision_class_0"] = train_species_folds_precision_class_0
        metrics_results_train_species["precision_class_1"] = train_species_folds_precision_class_1
        metrics_results_train_species["recall_class_0"] = train_species_folds_recall_class_0
        metrics_results_train_species["recall_class_1"] = train_species_folds_recall_class_1
        metrics_results_train_species["ppv_mod_class_0"] = train_species_folds_ppv_mod_class_0
        metrics_results_train_species["ppv_mod_class_1"] = train_species_folds_ppv_mod_class_1
        metrics_results_train_species["fpr"] = train_species_folds_fpr
        metrics_results_train_species["tpr"] = train_species_folds_tpr
        metrics_results_train_species["roc_auc_class_0"] = train_species_folds_roc_auc_class_0
        metrics_results_train_species["roc_auc_class_1"] = train_species_folds_roc_auc_class_1
        metrics_results_train_species["auc01_class_0"] = train_species_folds_auc01_class_0
        metrics_results_train_species["auc01_class_1"] = train_species_folds_auc01_class_1
        train_species_folds_pvals_class_0 = np.array(train_species_folds_pvals_class_0)
        metrics_results_train_species["pval_class_0"] = train_species_folds_pvals_class_0[~np.isnan(train_species_folds_pvals_class_0)]
        train_species_folds_pvals_class_1 = np.array(train_species_folds_pvals_class_1)
        metrics_results_train_species["pval_class_1"] = train_species_folds_pvals_class_1[~np.isnan(train_species_folds_pvals_class_1)]

        
        metrics_results_valid_species["ap_class_0"] = valid_species_folds_ap_class_0
        metrics_results_valid_species["ap_class_1"] = valid_species_folds_ap_class_1
        metrics_results_valid_species["ppv"] = valid_species_folds_ppv
        metrics_results_valid_species["precision_class_0"] = valid_species_folds_precision_class_0
        metrics_results_valid_species["precision_class_1"] = valid_species_folds_precision_class_1
        metrics_results_valid_species["recall_class_0"] = valid_species_folds_recall_class_0
        metrics_results_valid_species["recall_class_1"] = valid_species_folds_recall_class_1
        metrics_results_valid_species["ppv_mod_class_0"] = valid_species_folds_ppv_mod_class_0
        metrics_results_valid_species["ppv_mod_class_1"] = valid_species_folds_ppv_mod_class_1
        metrics_results_valid_species["fpr"] = valid_species_folds_fpr
        metrics_results_valid_species["tpr"] = valid_species_folds_tpr
        metrics_results_valid_species["roc_auc_class_0"] = valid_species_folds_roc_auc_class_0
        metrics_results_valid_species["roc_auc_class_1"] = valid_species_folds_roc_auc_class_1
        metrics_results_valid_species["auc01_class_0"] = valid_species_folds_auc01_class_0
        metrics_results_valid_species["auc01_class_1"] = valid_species_folds_auc01_class_1
        valid_species_folds_pvals_class_0 = np.array(valid_species_folds_pvals_class_0)
        metrics_results_valid_species["pval_class_0"] = valid_species_folds_pvals_class_0[~np.isnan(valid_species_folds_pvals_class_0)]
        valid_species_folds_pvals_class_1 = np.array(valid_species_folds_pvals_class_1)
        metrics_results_valid_species["pval_class_1"] = valid_species_folds_pvals_class_1[~np.isnan(valid_species_folds_pvals_class_1)]

        
        metrics_results_test_species["ap_class_0"] = test_species_folds_ap_class_0
        metrics_results_test_species["ap_class_1"] = test_species_folds_ap_class_1
        metrics_results_test_species["ppv"] = test_species_folds_ppv
        metrics_results_test_species["precision_class_0"] = test_species_folds_precision_class_0
        metrics_results_test_species["precision_class_1"] = test_species_folds_precision_class_1
        metrics_results_test_species["recall_class_0"] = test_species_folds_recall_class_0
        metrics_results_test_species["recall_class_1"] = test_species_folds_recall_class_1
        metrics_results_test_species["ppv_mod_class_0"] = test_species_folds_ppv_mod_class_0
        metrics_results_test_species["ppv_mod_class_1"] = test_species_folds_ppv_mod_class_1
        metrics_results_test_species["fpr"] = test_species_folds_fpr
        metrics_results_test_species["tpr"] = test_species_folds_tpr
        metrics_results_test_species["roc_auc_class_0"] = test_species_folds_roc_auc_class_0
        metrics_results_test_species["roc_auc_class_1"] = test_species_folds_roc_auc_class_1
        metrics_results_test_species["auc01_class_0"] = test_species_folds_auc01_class_0
        metrics_results_test_species["auc01_class_1"] = test_species_folds_auc01_class_1
        test_species_folds_pvals_class_0 = np.array(test_species_folds_pvals_class_0)
        metrics_results_test_species["pval_class_0"] = test_species_folds_pvals_class_0[~np.isnan(test_species_folds_pvals_class_0)]
        test_species_folds_pvals_class_1 = np.array(test_species_folds_pvals_class_1)
        metrics_results_test_species["pval_class_1"] = test_species_folds_pvals_class_1[~np.isnan(test_species_folds_pvals_class_1)]

        pickle.dump(metrics_results_train, open("{}/Vegvisir_checkpoints/roc_auc_train.p".format(folder), "wb"))
        pickle.dump(metrics_results_valid, open("{}/Vegvisir_checkpoints/roc_auc_valid.p".format(folder), "wb"))
        pickle.dump(metrics_results_test, open("{}/Vegvisir_checkpoints/roc_auc_test.p".format(folder), "wb"))

        pickle.dump(metrics_results_train_species,
                    open("{}/Vegvisir_checkpoints/roc_auc_train_species.p".format(folder), "wb"))
        pickle.dump(metrics_results_valid_species,
                    open("{}/Vegvisir_checkpoints/roc_auc_valid_species.p".format(folder), "wb"))
        pickle.dump(metrics_results_test_species,
                    open("{}/Vegvisir_checkpoints/roc_auc_test_species.p".format(folder), "wb"))


    return {"train":metrics_results_train,
        "valid":metrics_results_valid,
        "test":metrics_results_test,
        "train-species":metrics_results_train_species,
        "valid-species":metrics_results_valid_species,
        "test-species":metrics_results_test_species,
        "train-auc-0":train_folds_roc_auc_class_0,
        "train-auc-1":train_folds_roc_auc_class_1,
        "valid-auc-0":valid_folds_roc_auc_class_0,
        "valid-auc-1":valid_folds_roc_auc_class_1,
        "test-auc-0": test_folds_roc_auc_class_0,
        "test-auc-1": test_folds_roc_auc_class_1}

def plot_kfold_comparisons(args, script_dir, dict_results, kfolds=5, results_folder="Benchmark",title="",overwrite=False):
    """Compares average ROC AUC, Average Precision and Precision across runs"""
    metrics_auc_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    metrics_pvals_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    metrics_aps_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    metrics_auc_all_latex = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    metrics_pvals_all_latex = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    metrics_aps_all_latex = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    fig, [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]] = plt.subplots(nrows=2, ncols=4, figsize=(19, 6),
                                                                     gridspec_kw={'width_ratios': [5, 5, 5, 1],
                                                                                  'height_ratios': [6, 3.5]})

    i = 0
    names = []
    positions = []
    patches_list_train = []
    patches_list_valid = []
    patches_list_test = []
    metrics_keys = ["ap","ppv","fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1","pval_class_0","pval_class_1"]
    tuples_idx = []
    for learning_type,values in dict_results.items():
        for name, folder in values.items():
            print("-------------NAME {}--------------".format(name))

            tuples_idx.append((learning_type,name))
            metrics_results_dict = plot_kfold_comparison_helper(metrics_keys,script_dir, folder, overwrite, kfolds)

            metrics_results_train = metrics_results_dict["train"]
            metrics_results_train_species = metrics_results_dict["train-species"]
            metrics_results_valid = metrics_results_dict["valid"]
            metrics_results_valid_species = metrics_results_dict["valid-species"]
            metrics_results_test = metrics_results_dict["test"]
            metrics_results_test_species = metrics_results_dict["test-species"]
            train_folds_roc_auc_class_0 = metrics_results_dict["train-auc-0"]
            train_folds_roc_auc_class_1 = metrics_results_dict["train-auc-1"]
            valid_folds_roc_auc_class_0 = metrics_results_dict["valid-auc-0"]
            valid_folds_roc_auc_class_1 = metrics_results_dict["valid-auc-1"]
            test_folds_roc_auc_class_0 = metrics_results_dict["test-auc-0"]
            test_folds_roc_auc_class_1 = metrics_results_dict["test-auc-1"]

            
            #Highlight: Train
            metrics_auc_all_latex[learning_type][name]["train"]= str(
                np.round((np.mean(np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train["roc_auc_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_train["roc_auc_class_0"])) + np.std(np.array(metrics_results_train["roc_auc_class_1"])))/2, 2))

            metrics_pvals_all_latex[learning_type][name]["train"]= str(
                np.round((np.mean(np.array(metrics_results_train["pval_class_0"])) + np.mean(np.array(metrics_results_train["pval_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_train["pval_class_0"])) + np.std(np.array(metrics_results_train["pval_class_1"])))/2, 2))

            metrics_aps_all_latex[learning_type][name]["train"] = str(
                np.round((np.mean(np.array(metrics_results_train["ap_class_0"])) + np.mean(
                    np.array(metrics_results_train["ap_class_1"]))) / 2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_train["ap_class_0"])) + np.std(
                    np.array(metrics_results_train["ap_class_1"]))) / 2, 2))

            #Highlight: Validation

            metrics_auc_all_latex[learning_type][name]["valid"] = str(
                np.round((np.mean((np.array(metrics_results_valid["roc_auc_class_0"]))) + np.mean(np.array(metrics_results_valid["roc_auc_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_valid["roc_auc_class_0"])) + np.std(np.array(metrics_results_valid["roc_auc_class_1"])))/2, 2))

            metrics_pvals_all_latex[learning_type][name]["valid"] = str(
                np.round((np.mean((np.array(metrics_results_valid["pval_class_0"]))) + np.mean(
                    np.array(metrics_results_valid["pval_class_1"]))) / 2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_valid["pval_class_0"])) + np.std(
                    np.array(metrics_results_valid["pval_class_1"]))) / 2, 2))

            metrics_aps_all_latex[learning_type][name]["valid"] = str(
                np.round((np.mean((np.array(metrics_results_valid["ap_class_0"]))) + np.mean(
                    np.array(metrics_results_valid["ap_class_1"]))) / 2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_valid["ap_class_0"])) + np.std(
                    np.array(metrics_results_valid["ap_class_1"]))) / 2, 2))


            #Highlight: Test

            metrics_auc_all_latex[learning_type][name]["test"] = str(
                np.round((np.mean(np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test["roc_auc_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_test["roc_auc_class_0"])) + np.std(np.array(metrics_results_test["roc_auc_class_1"])))/2, 2)) if metrics_results_test["roc_auc_class_0"] is not None else metrics_results_test["roc_auc_class_0"]


            metrics_pvals_all_latex[learning_type][name]["test"] = str(
                np.round((np.mean(np.array(metrics_results_test["pval_class_0"])) + np.mean(np.array(metrics_results_test["pval_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_test["pval_class_0"])) + np.std(np.array(metrics_results_test["pval_class_1"])))/2, 2)) if metrics_results_test["pval_class_0"] is not None else metrics_results_test["pval_class_0"]

            metrics_aps_all_latex[learning_type][name]["test"] = str(
                np.round((np.mean(np.array(metrics_results_test["ap_class_0"])) + np.mean(
                    np.array(metrics_results_test["ap_class_1"]))) / 2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_test["ap_class_0"])) + np.std(
                    np.array(metrics_results_test["ap_class_1"]))) / 2, 2)) if metrics_results_test["ap_class_0"] is not None else metrics_results_test["ap_class_0"]
            
                                                                                     
            #Highlight: Train-species

            metrics_auc_all_latex[learning_type][name]["train-species"] = str(
                np.round((np.mean(np.array(metrics_results_train_species["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train_species["roc_auc_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_train_species["roc_auc_class_0"])) + np.std(np.array(metrics_results_train_species["roc_auc_class_1"])))/2, 2))

            metrics_pvals_all_latex[learning_type][name]["train-species"] = str(
                np.round((np.mean(np.array(metrics_results_train_species["pval_class_0"])) + np.mean(
                    np.array(metrics_results_train_species["pval_class_1"]))) / 2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_train_species["pval_class_0"])) + np.std(
                    np.array(metrics_results_train_species["pval_class_1"]))) / 2, 2))
            
            metrics_aps_all_latex[learning_type][name]["train-species"] = str(
                np.round((np.mean(np.array(metrics_results_train_species["ap_class_0"])) + np.mean(
                    np.array(metrics_results_train_species["ap_class_1"]))) / 2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_train_species["ap_class_0"])) + np.std(
                    np.array(metrics_results_train_species["ap_class_1"]))) / 2, 2))
            
            #Highlight: valid species

            metrics_auc_all_latex[learning_type][name]["valid-species"] = str(
                np.round((np.mean((np.array(metrics_results_valid_species["roc_auc_class_0"]))) + np.mean(np.array(metrics_results_valid_species["roc_auc_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_valid_species["roc_auc_class_0"])) + np.std(np.array(metrics_results_valid_species["roc_auc_class_1"])))/2, 2))

            metrics_pvals_all_latex[learning_type][name]["valid-species"] = str(
                np.round((np.mean((np.array(metrics_results_valid_species["pval_class_0"]))) + np.mean(np.array(metrics_results_valid_species["pval_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_valid_species["pval_class_0"])) + np.std(np.array(metrics_results_valid_species["pval_class_1"])))/2, 2))


            metrics_aps_all_latex[learning_type][name]["valid-species"] = str(
                np.round((np.mean((np.array(metrics_results_valid_species["ap_class_0"]))) + np.mean(np.array(metrics_results_valid_species["ap_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_valid_species["ap_class_0"])) + np.std(np.array(metrics_results_valid_species["ap_class_1"])))/2, 2))


            
            #Highlight: Test species
            
            metrics_auc_all_latex[learning_type][name]["test-species"] = str(
                np.round((np.mean(np.array(metrics_results_test_species["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test_species["roc_auc_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_test_species["roc_auc_class_0"])) + np.std(np.array(metrics_results_test_species["roc_auc_class_1"])))/2, 2)) if metrics_results_test_species["roc_auc_class_0"] is not None else metrics_results_test_species["roc_auc_class_0"]

            metrics_pvals_all_latex[learning_type][name]["test-species"] = str(
                np.round((np.mean(np.array(metrics_results_test_species["pval_class_0"])) + np.mean(np.array(metrics_results_test_species["pval_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_test_species["pval_class_0"])) + np.std(np.array(metrics_results_test_species["pval_class_1"])))/2, 2)) if metrics_results_test_species["pval_class_0"] is not None else metrics_results_test_species["pval_class_0"]


            metrics_aps_all_latex[learning_type][name]["test-species"] = str(
                np.round((np.mean(np.array(metrics_results_test_species["ap_class_0"])) + np.mean(np.array(metrics_results_test_species["ap_class_1"])))/2, 2)) + " $\pm$ " + str(
                np.round((np.std(np.array(metrics_results_test_species["ap_class_0"])) + np.std(np.array(metrics_results_test_species["ap_class_1"])))/2, 2)) if metrics_results_test_species["ap_class_0"] is not None else metrics_results_test_species["ap_class_0"]




            track_all = False
            metrics_auc_all[learning_type][name]["train"] = np.round((np.mean(np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train["roc_auc_class_1"])))/2, 2)
            metrics_pvals_all[learning_type][name]["train"] = np.round((np.mean(np.array(metrics_results_train["pval_class_0"])) + np.mean(np.array(metrics_results_train["pval_class_1"])))/2, 2)
            metrics_aps_all[learning_type][name]["train"] = np.round((np.mean(np.array(metrics_results_train["ap_class_0"])) + np.mean(np.array(metrics_results_train["ap_class_1"])))/2, 2)
            if track_all:
                metrics_auc_all[learning_type][name]["valid"] = np.round((np.mean((np.array(metrics_results_valid["roc_auc_class_0"]))) + np.mean(np.array(metrics_results_valid["roc_auc_class_1"])))/2, 2)
                metrics_pvals_all[learning_type][name]["valid"] = np.round((np.mean((np.array(metrics_results_valid["pval_class_0"]))) + np.mean(np.array(metrics_results_valid["pval_class_1"])))/2, 2)
                metrics_aps_all[learning_type][name]["valid"] = np.round((np.mean((np.array(metrics_results_valid["ap_class_0"]))) + np.mean(np.array(metrics_results_valid["ap_class_1"])))/2, 2)
            metrics_auc_all[learning_type][name]["test"] = np.round((np.mean(np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test["roc_auc_class_1"])))/2, 2) if metrics_results_test["roc_auc_class_0"] is not None else metrics_results_test["roc_auc_class_0"]
            metrics_pvals_all[learning_type][name]["test"] = np.round((np.mean(np.array(metrics_results_test["pval_class_0"])) + np.mean(np.array(metrics_results_test["pval_class_1"])))/2, 2) if metrics_results_test["pval_class_0"] is not None else metrics_results_test["pval_class_0"]
            metrics_aps_all[learning_type][name]["test"] = np.round((np.mean(np.array(metrics_results_test["ap_class_0"])) + np.mean(np.array(metrics_results_test["ap_class_1"])))/2, 2) if metrics_results_test["ap_class_0"] is not None else metrics_results_test["ap_class_0"]
            metrics_auc_all[learning_type][name]["train-species"] = np.round((np.mean(np.array(metrics_results_train_species["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train_species["roc_auc_class_1"]))) / 2, 2)
            metrics_pvals_all[learning_type][name]["train-species"] = np.round((np.mean(np.array(metrics_results_train_species["pval_class_0"])) + np.mean(np.array(metrics_results_train_species["pval_class_1"]))) / 2, 2)
            metrics_aps_all[learning_type][name]["train-species"] = np.round((np.mean(np.array(metrics_results_train_species["ap_class_0"])) + np.mean(np.array(metrics_results_train_species["ap_class_1"]))) / 2, 2)

            if track_all:
                metrics_auc_all[learning_type][name]["valid-species"] = np.round((np.mean((np.array(metrics_results_valid_species["roc_auc_class_0"]))) + np.mean(np.array(metrics_results_valid_species["roc_auc_class_1"])))/2, 2)
                metrics_pvals_all[learning_type][name]["valid-species"] = np.round((np.mean((np.array(metrics_results_valid_species["pval_class_0"]))) + np.mean(np.array(metrics_results_valid_species["pval_class_1"])))/2, 2)
                metrics_aps_all[learning_type][name]["valid-species"] = np.round((np.mean((np.array(metrics_results_valid_species["ap_class_0"]))) + np.mean(np.array(metrics_results_valid_species["ap_class_1"])))/2, 2)
                metrics_auc_all[learning_type][name]["test-species"] = np.round((np.mean(np.array(metrics_results_test_species["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test_species["roc_auc_class_1"])))/2, 2)if metrics_results_test_species["roc_auc_class_0"] is not None else metrics_results_test_species["roc_auc_class_0"]
                metrics_pvals_all[learning_type][name]["test-species"] = np.round((np.mean(np.array(metrics_results_test_species["pval_class_0"])) + np.mean(np.array(metrics_results_test_species["pval_class_1"])))/2, 2)if metrics_results_test_species["pval_class_0"] is not None else metrics_results_test_species["pval_class_0"]
                metrics_aps_all[learning_type][name]["test-species"] = np.round((np.mean(np.array(metrics_results_test_species["ap_class_0"])) + np.mean(np.array(metrics_results_test_species["ap_class_1"])))/2, 2)if metrics_results_test_species["ap_class_0"] is not None else metrics_results_test_species["ap_class_0"]

            bsize = 0.7
            bar1_0 = ax1.bar(i, np.mean(train_folds_roc_auc_class_0), yerr=np.std(train_folds_roc_auc_class_0), width=bsize,
                             color="steelblue")
            bar1_1 = ax1.bar(i + 1.2, np.mean(train_folds_roc_auc_class_1), yerr=np.std(train_folds_roc_auc_class_1),
                             width=bsize, color="steelblue")

            bar2_0 = ax2.bar(i, np.mean(valid_folds_roc_auc_class_0), yerr=np.std(valid_folds_roc_auc_class_0), width=bsize,
                             color="coral")
            bar2_1 = ax2.bar(i + 1.2, np.mean(valid_folds_roc_auc_class_1), yerr=np.std(valid_folds_roc_auc_class_1),
                             width=bsize, color="coral")

            bar3_0 = ax3.bar(i, np.mean(test_folds_roc_auc_class_0), yerr=np.std(test_folds_roc_auc_class_0), width=bsize,
                             color="red")
            bar3_1 = ax3.bar(i + 1.2, np.mean(test_folds_roc_auc_class_1), yerr=np.std(test_folds_roc_auc_class_1),
                             width=bsize, color="red")

            fsize = 6
            elevation = 0.02
            #Highlight: I am uncertain why it does not let me do this in a loop
            for bar in bar1_0.patches:
                ax1.annotate(format(bar.get_height(), '.2f'),
                             (bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + elevation), ha='center', va='center',
                             size=fsize, xytext=(0, 8),
                             textcoords='offset points')
            for bar in bar1_1.patches:
                ax1.annotate(format(bar.get_height(), '.2f'),
                             (bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + elevation), ha='center', va='center',
                             size=fsize, xytext=(0, 8),
                             textcoords='offset points')
            for bar in bar2_0.patches:
                ax2.annotate(format(bar.get_height(), '.2f'),
                             (bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + elevation), ha='center', va='center',
                             size=fsize, xytext=(0, 8),
                             textcoords='offset points')
            for bar in bar2_1.patches:
                ax2.annotate(format(bar.get_height(), '.2f'),
                             (bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + elevation), ha='center', va='center',
                             size=fsize, xytext=(0, 8),
                             textcoords='offset points')
            for bar in bar3_0.patches:
                ax3.annotate(format(bar.get_height(), '.2f'),
                             (bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + elevation), ha='center', va='center',
                             size=fsize, xytext=(0, 8),
                             textcoords='offset points')
            for bar in bar3_1.patches:
                ax3.annotate(format(bar.get_height(), '.2f'),
                             (bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + elevation), ha='center', va='center',
                             size=fsize, xytext=(0, 8),
                             textcoords='offset points')
            patches_list_train.append(bar1_0.patches[0])
            patches_list_train.append(bar1_1.patches[0])
            patches_list_valid.append(bar2_0.patches[0])
            patches_list_valid.append(bar2_1.patches[0])
            patches_list_test.append(bar3_0.patches[0])
            patches_list_test.append(bar3_1.patches[0])

            names.append("{}_class_0".format(name))
            names.append("{}_class_1".format(name))

            positions.append(i)
            positions.append(i + 1)

            i += 2.7


    ax1.set_xticks(positions, labels=names, rotation=90, fontsize=8)
    ax2.set_xticks(positions, labels=names, rotation=90, fontsize=8)
    ax3.set_xticks(positions, labels=names, rotation=90, fontsize=8)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)

    ax1.set_title("Train")
    ax2.set_title("Valid")
    ax3.set_title("Test")

    ax4.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")

    fig.suptitle("Model comparison (5-fold cross validation)")
    plt.savefig("{}/{}/methods_comparison_HISTOGRAM.png".format(script_dir, results_folder), dpi=600)

    def convert_dict(results_dict):
        df = pd.DataFrame.from_dict(results_dict, orient="index").stack().to_frame()
        # to break out the lists into columns
        df = pd.DataFrame(df[0].values.tolist(),index=df.index)
        #df = df.reindex(index=df.index.reindex(['s', 'r'], level=1)[0]) #preserving the original order because it tends to become unordered
        #df = df.stack().unstack(level=1)  # transposes
        new_index = pd.MultiIndex.from_tuples(tuples_idx)
        df = df.reindex(index=new_index) #guarantees the same order as the dictionary
        return df

    df_latex = convert_dict(metrics_auc_all_latex)
    df_latex.style.format(na_rep="-").to_latex('{}/{}/methods_comparison_LATEX_{}.tex'.format(script_dir, results_folder,title))

    #a = {"w": {"r": {"train": 1, "test": 0.7}, "s": {"train": 1, "test": 0.7}},"t": {"r": {"train": 0.8, "test": 0.6}, "s": {"train": 0.9, "test": 0.8}}}
    def process_dict(metrics_dict,title):
        metrics_dict = {key.replace(r"\textbf{", "").replace("}", ""): val for key, val in metrics_dict.items()}
        df = convert_dict(metrics_dict)
        #Highlight: https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex
        colors = {"supervised(Icore)": matplotlib.colors.to_rgba('lavender'),
                  "Icore": matplotlib.colors.to_rgba('lavender'),
                  "supervised(Icore_non_anchor)": matplotlib.colors.to_rgba('palegreen'),
                  "Icore_non_anchor": matplotlib.colors.to_rgba('palegreen'),
                  "semisupervised(Icore)":matplotlib.colors.to_rgba('paleturquoise'),
                  "semisupervised(Icore_non_anchor)":matplotlib.colors.to_rgba('plum')}
        c = {k: matplotlib.colors.rgb2hex(v) for k, v in colors.items()}
        idx = df.index.get_level_values(0)
        css = [{'selector': f'.row{i}.level0', 'props': [('background-color', c[v])]} for i, v in enumerate(idx)]
        df_styled = df.style.format(na_rep="-", escape="latex",precision=2).background_gradient(axis=None,cmap="YlOrBr").set_table_styles(css)  # TODO: Switch to escape="latex-math" for pandas 2.1
        dfi.export(df_styled, '{}/{}/metrics_comparison_{}.png'.format(script_dir, results_folder,title), max_cols=-1,max_rows=-1,dpi=600)


    process_dict(metrics_auc_all, "ROC_AUC_{}".format(title))
    process_dict(metrics_pvals_all, "Pvalues_AUC_{}".format(title))
    process_dict(metrics_aps_all, "Average_precision_{}".format(title))

def plot_latent_correlations_helper(train_out,valid_out,test_out,reducer,covariances_dict_train,covariances_dict_valid,covariances_dict_test,learning_type,name,args,script_dir,results_folder):
    for mode, summary_dict in zip(["train", "valid", "test"], [train_out, valid_out, test_out]):
        if summary_dict is not None:
            dataset_info = summary_dict["dataset_info"]
            latent_space = summary_dict["latent_space"]
            if "umap_1d" not in summary_dict.keys():
                print("Constructing UMAP-1d")
                umap_proj_1d = reducer.fit_transform(latent_space[:, 6:]).squeeze(-1)
                #warnings.warn("using random UMAP for debugging")
                #umap_proj_1d = np.random.rand(latent_space.shape[0]) #used to do fast debugging

            else:
                print("Umap found, not calculating")
                umap_proj_1d = summary_dict["umap_1d"]
            custom_features_dicts = VegvisirUtils.build_features_dicts(dataset_info)

            aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]
            gravy_dict = custom_features_dicts["gravy_dict"]
            volume_dict = custom_features_dicts["volume_dict"]
            radius_dict = custom_features_dicts["radius_dict"]
            side_chain_pka_dict = custom_features_dicts["side_chain_pka_dict"]
            isoelectric_dict = custom_features_dicts["isoelectric_dict"]
            bulkiness_dict = custom_features_dicts["bulkiness_dict"]
            data_int = summary_dict["summary_dict"]["data_int_samples"]
            sequences = data_int[:, 1:].squeeze(1)
            if dataset_info.corrected_aa_types == 20:
                sequences_mask = np.zeros_like(sequences).astype(bool)
            else:
                sequences_mask = np.array((sequences == 0))

            sequences_lens = np.sum(~sequences_mask, axis=1)
            sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(sequences)
            sequences_list = sequences_raw.tolist()
            bulkiness_scores = np.vectorize(bulkiness_dict.get)(sequences)
            bulkiness_scores = np.ma.masked_array(bulkiness_scores, mask=sequences_mask, fill_value=0)
            bulkiness_scores = np.ma.sum(bulkiness_scores, axis=1)

            volume_scores = np.vectorize(volume_dict.get)(sequences)
            volume_scores = np.ma.masked_array(volume_scores, mask=sequences_mask, fill_value=0)
            volume_scores = np.ma.sum(volume_scores, axis=1)  # TODO: Mean? or sum?

            radius_scores = np.vectorize(radius_dict.get)(sequences)
            radius_scores = np.ma.masked_array(radius_scores, mask=sequences_mask, fill_value=0)
            radius_scores = np.ma.sum(radius_scores, axis=1)

            side_chain_pka_scores = np.vectorize(side_chain_pka_dict.get)(sequences)
            side_chain_pka_scores = np.ma.masked_array(side_chain_pka_scores, mask=sequences_mask, fill_value=0)
            side_chain_pka_scores = np.ma.mean(side_chain_pka_scores,
                                               axis=1)  # Highlight: before I was doing just the sum

            isoelectric_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_isoelectric(seq), sequences_list)))
            aromaticity_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_aromaticity(seq), sequences_list)))
            gravy_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_gravy(seq), sequences_list)))
            molecular_weight_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_molecular_weight(seq), sequences_list)))
            extintion_coefficient_reduced_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_extintioncoefficient(seq)[0], sequences_list)))
            extintion_coefficient_cystines_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_extintioncoefficient(seq)[1], sequences_list)))

            storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
            features_dict = VegvisirUtils.CalculatePeptideFeatures(dataset_info.seq_max_len, sequences_list,
                                                                    storage_folder,
                                                                    return_aa_freqs=True).features_summary()

            settings = {"features_dict": {"gravy_scores": gravy_scores,
                                          "isoelectric_scores": isoelectric_scores,
                                          "volume_scores": volume_scores,
                                          "radius_scores": radius_scores,
                                          "side_chain_pka_scores": side_chain_pka_scores,
                                          "aromaticity_scores": aromaticity_scores,
                                          "molecular_weights": molecular_weight_scores,
                                          "bulkiness_scores": bulkiness_scores,
                                          "sequences_lens": sequences_lens,
                                          "extintion_coefficients_reduced": extintion_coefficient_reduced_scores,
                                          "extintion_coefficients_cystines": extintion_coefficient_cystines_scores,
                                          "tryptophan":features_dict["Tryptophan"]
                                          },
                        "sequences_raw": sequences_raw}

            correlations_dict = plot_latent_correlations_1d(umap_proj_1d,
                                                            args,
                                                            settings,
                                                            dataset_info,
                                                            latent_space,
                                                            sample_mode="samples",
                                                            results_dir="{}/{}".format(script_dir, results_folder),
                                                            method=mode,
                                                            vector_name="latent_space_z",
                                                            plot_scatter_correlations=False,
                                                            calculate_covariance=True,
                                                            plot_covariance=False,
                                                            filter_correlations=False)


            if mode == "train":
                covariances_dict_train[learning_type][name]["covariance"].append(np.abs(correlations_dict["features_covariance"])) #make positive because umap vector can be orientated anywhere
                covariances_dict_train[learning_type][name]["features_names"].append(correlations_dict["features_names"])
                covariances_dict_train[learning_type][name]["pearson_coefficients"].append(np.abs(correlations_dict["pearson_coefficients"]))
                covariances_dict_train[learning_type][name]["pearson_pvalues"].append(correlations_dict["pearson_pvalues"])
                covariances_dict_train[learning_type][name]["spearman_coefficients"].append(np.abs(correlations_dict["spearman_coefficients"]))
                covariances_dict_train[learning_type][name]["spearman_pvalues"].append(correlations_dict["spearman_pvalues"])
                covariances_dict_train[learning_type][name]["umap_1d"].append(umap_proj_1d)
            elif mode == "valid":
                covariances_dict_valid[learning_type][name]["covariance"].append(np.abs(correlations_dict["features_covariance"]))
                covariances_dict_valid[learning_type][name]["features_names"].append(correlations_dict["features_names"])
                covariances_dict_valid[learning_type][name]["pearson_coefficients"].append(np.abs(correlations_dict["pearson_coefficients"]))
                covariances_dict_valid[learning_type][name]["pearson_pvalues"].append(correlations_dict["pearson_pvalues"])
                covariances_dict_valid[learning_type][name]["spearman_coefficients"].append(np.abs(correlations_dict["spearman_coefficients"]))
                covariances_dict_valid[learning_type][name]["spearman_pvalues"].append(correlations_dict["spearman_pvalues"])
                covariances_dict_valid[learning_type][name]["umap_1d"].append(umap_proj_1d)
            elif mode == "test":
                covariances_dict_test[learning_type][name]["covariance"].append(np.abs(correlations_dict["features_covariance"]))
                covariances_dict_test[learning_type][name]["features_names"].append(correlations_dict["features_names"])
                covariances_dict_test[learning_type][name]["pearson_coefficients"].append(np.abs(correlations_dict["pearson_coefficients"]))
                covariances_dict_test[learning_type][name]["pearson_pvalues"].append(correlations_dict["pearson_pvalues"])
                covariances_dict_test[learning_type][name]["spearman_coefficients"].append(np.abs(correlations_dict["spearman_coefficients"]))
                covariances_dict_test[learning_type][name]["spearman_pvalues"].append(correlations_dict["spearman_pvalues"])
                covariances_dict_test[learning_type][name]["umap_1d"].append(umap_proj_1d)

    return covariances_dict_train,covariances_dict_valid,covariances_dict_test

def plot_kfold_latent_correlations(args,script_dir,dict_results,kfolds=5,results_folder="Benchmark",overwrite_correlations=False,overwrite_all=False,subtitle=""):
     """Computes the average UMAP1d vs peptide feature correlations across the n-folds"""
     new_feature_names = {"UMAP-1D":"UMAP-1D",
                          "gravy_scores":"Gravy (Hydropathy index)", #larger the number is, the more hydrophobic the amino acid
                          "isoelectric_scores":"Isoelectric",
                          "volume_scores":"Volume",
                          "radius_scores":"Radius",
                          "side_chain_pka_scores":"Side chain Pka",
                          "aromaticity_scores":"Aromaticity",
                          "molecular_weights":"Molecular weight",
                          "bulkiness_scores":"Bulkiness",
                          "sequences_lens":"Sequences  lengths",
                          "extintion_coefficients_reduced":"Extintion coeff (red)",
                          "extintion_coefficients_cystines":"Extintion coeff (ox)",
                          "immunodominance_scores":"Immunodominance",
                          "binary_targets":"Targets",
                          "pep_secstruc_turn":"Coil struct",
                          "pep_secstruc_helix":"Helix struct",
                          "pep_secstruc_sheet":"Sheet struct",
                          "tryptophan":"Tryptophan"}
     covariances_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     pearson_coefficients_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     pearson_pvalues_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     spearman_coefficients_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     spearman_coefficients_all_latex = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     spearman_pvalues_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))

     covariances_dict_train= defaultdict(lambda :defaultdict(lambda : defaultdict(lambda : [])))
     covariances_dict_valid= defaultdict(lambda :defaultdict(lambda : defaultdict(lambda : [])))
     covariances_dict_test= defaultdict(lambda :defaultdict(lambda : defaultdict(lambda : [])))

     reducer = umap.UMAP(n_components=1)
     tuples_idx = []
     plot_all = True
     for learning_type, values in dict_results.items():
         for name, folder in values.items():
             tuples_idx.append((learning_type,name,"train"))
             if plot_all:
                tuples_idx.append((learning_type,name,"valid"))
             tuples_idx.append((learning_type,name,"test"))
             print("{}".format(name))
             if os.path.exists("{}/Vegvisir_checkpoints/covariances_train.p".format(folder)) and not overwrite_correlations and not overwrite_all:
                print("Loading pre computed correlations ---------------------------------------")
                #Highlight: Dill is an *** , if you change the module's non mutable structures (i.e named tuples), it will complain
                covariances_dict_train = dill.load(open("{}/Vegvisir_checkpoints/covariances_train.p".format(folder), "rb"))
                covariances_dict_valid = dill.load(open("{}/Vegvisir_checkpoints/covariances_valid.p".format(folder), "rb"))
                if os.path.exists("{}/Vegvisir_checkpoints/covariances_test.p".format(folder)):
                    covariances_dict_test = dill.load(open("{}/Vegvisir_checkpoints/covariances_test.p".format(folder), "rb"))
             elif os.path.exists("{}/Vegvisir_checkpoints/covariances_train.p".format(folder)) and overwrite_correlations and not overwrite_all:
                 print("Loading pre computed UMAP, re-calculating correlations ---------------------------------------")
                 # Highlight: Dill is an *** , if you change the module's non mutable structures (i.e named tuples), it will complain
                 covariances_dict_train_precomputed = dill.load(open("{}/Vegvisir_checkpoints/covariances_train.p".format(folder), "rb"))
                 covariances_dict_valid_precomputed = dill.load(open("{}/Vegvisir_checkpoints/covariances_valid.p".format(folder), "rb"))
                 if os.path.exists("{}/Vegvisir_checkpoints/covariances_test.p".format(folder)):
                     covariances_dict_test_precomputed = dill.load(open("{}/Vegvisir_checkpoints/covariances_test.p".format(folder), "rb"))

                 for fold in range(kfolds):
                     print("Computing correlations & covariances")
                     print("FOLD: {} ------------------------".format(fold))
                     if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold)):
                        train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold))
                     else:
                        train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_valid_fold_{}.p".format(folder, fold))

                     #Highlight: Load the UMAPs from this fold (precomputed)

                     covariances_dict_train_fold = covariances_dict_train_precomputed[learning_type][name]
                     covariances_dict_train_fold = dict(zip(list(covariances_dict_train_fold.keys()),list(zip(*covariances_dict_train_fold.values()))[fold]))
                     train_out = {**train_out, **covariances_dict_train_fold}
                     valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(folder, fold))
                     covariances_dict_valid_fold = covariances_dict_valid_precomputed[learning_type][name]
                     covariances_dict_valid_fold = dict(zip(list(covariances_dict_valid_fold.keys()),list(zip(*covariances_dict_valid_fold.values()))[fold]))
                     valid_out = {**valid_out, **covariances_dict_valid_fold}
                     if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold)):
                         test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold))
                         covariances_dict_test_fold = covariances_dict_test_precomputed[learning_type][name]
                         covariances_dict_test_fold = dict(zip(list(covariances_dict_test_fold.keys()),
                                                                list(zip(*covariances_dict_test_fold.values()))[fold]))
                         test_out = {**test_out, **covariances_dict_test_fold}
                     else:
                         test_out = None

                     args = train_out["args"]
                     plot_latent_correlations_helper(
                         train_out, valid_out, test_out, reducer, covariances_dict_train, covariances_dict_valid,
                         covariances_dict_test, learning_type, name, args, script_dir, results_folder)
                     #print(len(covariances_dict_train[learning_type][name]["umap_1d"]))

                 dill.dump(covariances_dict_train,open("{}/Vegvisir_checkpoints/covariances_train.p".format(folder), "wb"))
                 dill.dump(covariances_dict_valid,open("{}/Vegvisir_checkpoints/covariances_valid.p".format(folder), "wb"))
                 dill.dump(covariances_dict_test,open("{}/Vegvisir_checkpoints/covariances_test.p".format(folder), "wb"))

             else:
                for fold in range(kfolds):
                     print("Files not existing or you set overwrite_all = True.Computing correlation covariances and UMAP")
                     if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold)):
                        train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold))
                     else:
                        train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_valid_fold_{}.p".format(folder, fold))

                     valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(folder, fold))
                     if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold)):
                         test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold))
                     else:
                         test_out = None

                     args = train_out["args"]

                     plot_latent_correlations_helper(train_out,valid_out,test_out,reducer,covariances_dict_train,covariances_dict_valid,covariances_dict_test,learning_type,name,args,script_dir,results_folder)



                dill.dump(covariances_dict_train,open("{}/Vegvisir_checkpoints/covariances_train.p".format(folder), "wb"))
                dill.dump(covariances_dict_valid,open("{}/Vegvisir_checkpoints/covariances_valid.p".format(folder), "wb"))
                dill.dump(covariances_dict_test,open("{}/Vegvisir_checkpoints/covariances_test.p".format(folder), "wb"))


             covariances_all[learning_type][name]["train"] = np.mean([covariance[0,1:] for covariance in covariances_dict_train[learning_type][name]["covariance"]],axis=0)
             if plot_all:
                covariances_all[learning_type][name]["valid"] = np.mean([covariance[0,1:] for covariance in covariances_dict_valid[learning_type][name]["covariance"]],axis=0)
             covariances_all[learning_type][name]["test"] = np.mean([covariance[0,1:] for covariance in covariances_dict_test[learning_type][name]["covariance"]],axis=0)


             pearson_pvalues_all[learning_type][name]["train"] = np.mean([pvalues for pvalues in covariances_dict_train[learning_type][name]["pearson_pvalues"]],axis=0)
             if plot_all:
                pearson_pvalues_all[learning_type][name]["valid"] = np.mean([pvalues for pvalues in covariances_dict_valid[learning_type][name]["pearson_pvalues"]],axis=0)
             pearson_pvalues_all[learning_type][name]["test"] = np.mean([pvalues for pvalues in covariances_dict_test[learning_type][name]["pearson_pvalues"]],axis=0)


             pearson_coefficients_all[learning_type][name]["train"] = np.mean([coefficients for coefficients in covariances_dict_train[learning_type][name]["pearson_coefficients"]],axis=0)
             if plot_all:
                pearson_coefficients_all[learning_type][name]["valid"] = np.mean([coefficients for coefficients in covariances_dict_valid[learning_type][name]["pearson_coefficients"]],axis=0)
             pearson_coefficients_all[learning_type][name]["test"] = np.mean([coefficients for coefficients in covariances_dict_test[learning_type][name]["pearson_coefficients"]],axis=0)

             spearman_pvalues_all[learning_type][name]["train"] = np.mean([pvalues for pvalues in covariances_dict_train[learning_type][name]["spearman_pvalues"]], axis=0)
             if plot_all:
                 spearman_pvalues_all[learning_type][name]["valid"] = np.mean([pvalues for pvalues in covariances_dict_valid[learning_type][name]["spearman_pvalues"]], axis=0)
             spearman_pvalues_all[learning_type][name]["test"] = np.mean([pvalues for pvalues in covariances_dict_test[learning_type][name]["spearman_pvalues"]], axis=0)

             spearman_coefficients_all[learning_type][name]["train"] = np.mean([coefficients for coefficients in covariances_dict_train[learning_type][name]["spearman_coefficients"]],axis=0)
             if plot_all:
                 spearman_coefficients_all[learning_type][name]["valid"] = np.mean([coefficients for coefficients in covariances_dict_valid[learning_type][name]["spearman_coefficients"]], axis=0)
             spearman_coefficients_all[learning_type][name]["test"] = np.mean([coefficients for coefficients in covariances_dict_test[learning_type][name]["spearman_coefficients"]],axis=0)

             if covariances_dict_train[learning_type][name]["features_names"]: #UMAP-1D exists
                 features_names = covariances_dict_train[learning_type][name]["features_names"][0][1:] # remove UMAP-1D
                 features_names = [new_feature_names[feat] if feat in new_feature_names.keys() else feat for feat in features_names]

             else:
                 features_names = covariances_dict_train[learning_type][name]["features_names"]

             n_feats = len(features_names)
             covariances_all[learning_type][name]["train"] = dict(zip(features_names,np.round(covariances_all[learning_type][name]["train"],2).tolist())) if features_names else dict(zip(features_names,[np.nan]*n_feats))
             if plot_all:
                covariances_all[learning_type][name]["valid"] = dict(zip(features_names,np.round(covariances_all[learning_type][name]["valid"],2).tolist())) if features_names else dict(zip(features_names,[np.nan]*n_feats))
             covariances_all[learning_type][name]["test"] = dict(zip(features_names,np.round(covariances_all[learning_type][name]["test"],2).tolist())) if features_names and type(covariances_all[learning_type][name]["test"]) != np.float64 else dict(zip(features_names,[np.nan]*n_feats))

             pearson_pvalues_all[learning_type][name]["train"] = dict(zip(features_names,np.round(pearson_pvalues_all[learning_type][name]["train"],3).tolist())) if features_names else dict(zip(features_names,[np.nan]*n_feats))
             if plot_all:
                pearson_pvalues_all[learning_type][name]["valid"] = dict(zip(features_names,np.round(pearson_pvalues_all[learning_type][name]["valid"],3).tolist())) if features_names else dict(zip(features_names,[np.nan]*n_feats))
             pearson_pvalues_all[learning_type][name]["test"] = dict(zip(features_names,np.round(pearson_pvalues_all[learning_type][name]["test"],3).tolist())) if features_names and type(pearson_pvalues_all[learning_type][name]["test"]) != np.float64 else dict(zip(features_names,[np.nan]*n_feats))


             pearson_coefficients_all[learning_type][name]["train"] = dict(zip(features_names,np.round(pearson_coefficients_all[learning_type][name]["train"],3).tolist())) if features_names else dict(zip(features_names,[np.nan]*n_feats))
             if plot_all:
                pearson_coefficients_all[learning_type][name]["valid"] = dict(zip(features_names,np.round(pearson_coefficients_all[learning_type][name]["valid"],3).tolist())) if features_names else dict(zip(features_names,[np.nan]*n_feats))
             pearson_coefficients_all[learning_type][name]["test"] = dict(zip(features_names,np.round(pearson_coefficients_all[learning_type][name]["test"],3).tolist())) if features_names and type(pearson_coefficients_all[learning_type][name]["test"]) != np.float64 else dict(zip(features_names,[np.nan]*n_feats))

             spearman_pvalues_all[learning_type][name]["train"] = dict(zip(features_names, np.round(
                 spearman_pvalues_all[learning_type][name]["train"], 3).tolist())) if features_names else dict(
                 zip(features_names, [np.nan] * n_feats))
             if plot_all:
                 spearman_pvalues_all[learning_type][name]["valid"] = dict(zip(features_names, np.round(
                     spearman_pvalues_all[learning_type][name]["valid"], 3).tolist())) if features_names else dict(
                     zip(features_names, [np.nan] * n_feats))
             spearman_pvalues_all[learning_type][name]["test"] = dict(zip(features_names, np.round(
                 spearman_pvalues_all[learning_type][name]["test"], 3).tolist())) if features_names and type(
                 spearman_pvalues_all[learning_type][name]["test"]) != np.float64 else dict(
                 zip(features_names, [np.nan] * n_feats))

             spearman_coefficients_all[learning_type][name]["train"] = dict(zip(features_names, np.round(
                 spearman_coefficients_all[learning_type][name]["train"], 3).tolist())) if features_names else dict(
                 zip(features_names, [np.nan] * n_feats))
             if plot_all:
                 spearman_coefficients_all[learning_type][name]["valid"] = dict(zip(features_names, np.round(
                     spearman_coefficients_all[learning_type][name]["valid"], 3).tolist())) if features_names else dict(
                     zip(features_names, [np.nan] * n_feats))
             spearman_coefficients_all[learning_type][name]["test"] = dict(zip(features_names, np.round(
                 spearman_coefficients_all[learning_type][name]["test"], 3).tolist())) if features_names and type(
                 spearman_coefficients_all[learning_type][name]["test"]) != np.float64 else dict(
                 zip(features_names, [np.nan] * n_feats))

             spearman_coefficients_all_latex[learning_type][name]["train"] = spearman_coefficients_all[learning_type][name]["train"]
             spearman_coefficients_all_latex[learning_type][name]["valid"] = spearman_coefficients_all[learning_type][name]["valid"]
             spearman_coefficients_all_latex[learning_type][name]["test"] = spearman_coefficients_all[learning_type][name]["test"]



     print("Finished loop, building dataframe")

     def process_dict(metric_dict,title="",subtitle=""):
         metric_dict = {key.replace(r"\textbf{", "").replace("}", ""): val for key, val in metric_dict.items()}

         df = pd.DataFrame.from_dict({(i, j,k): metric_dict[i][j][k] for i in metric_dict.keys() for j in metric_dict[i].keys() for k in metric_dict[i][j]},orient='index')

         # Highlight: https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex
         colors = {"supervised(Icore)": matplotlib.colors.to_rgba('lavender'),
                   "Icore": matplotlib.colors.to_rgba('lavender'),
                   "supervised(Icore_non_anchor)": matplotlib.colors.to_rgba('palegreen'),
                   "Icore_non_anchor": matplotlib.colors.to_rgba('palegreen'),
                   "semisupervised(Icore)": matplotlib.colors.to_rgba('paleturquoise'),
                   "semisupervised(Icore_non_anchor)": matplotlib.colors.to_rgba('plum')}
         c = {k: matplotlib.colors.rgb2hex(v) for k, v in colors.items()}
         idx = df.index.get_level_values(0)
         css = [{'selector': f'.row{i}.level0', 'props': [('background-color', c[v])]} for i, v in enumerate(idx)]
         new_index = pd.MultiIndex.from_tuples(tuples_idx)
         df = df.reindex(index=new_index)  # guarantees the same order as the dictionary
         df_styled = df.style.format(na_rep="-", escape="latex",precision=2).background_gradient(axis=None,cmap="YlOrBr").set_table_styles(css)  # TODO: Switch to escape="latex-math" when pandas 2.1 arrives

         dfi.export(df_styled, '{}/{}/{}_DATAFRAME_{}.png'.format(script_dir, results_folder,title,subtitle), max_cols=-1,max_rows=-1)

         return df

     def latex_with_lines(df, *args, **kwargs):
         kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels + ['r'] * df.shape[1] + [''])
         kwargs["na_rep"] = ""
         kwargs["float_format"] = "%.2f"
         res = df.to_latex(*args, **kwargs)
         res = res.replace("\multirow[t]{24}{*}{Icore}","\multirow[t]{24}{*}{\colorbox{blue!20}{\makebox[3cm]{Icore}}}")
         res = res.replace("\multirow[t]{24}{*}{Icore_non_anchor}","\multirow[t]{24}{*}{\colorbox{magenta!20}{\makebox[3cm]{Icore-non-anchor}}}")
         res = res.replace("\multirow[t]","\multirow")
         res = res.replace("variable","var")
         res = res.replace("\multirow{3}{*}{raw-blosum-variable-length}","\multirow{3}{*}{\textbf{raw-blosum-variable-length}}")
         res = res.replace("\multirow{3}{*}{raw-variable-length}","\multirow{3}{*}{\textbf{raw-variable-length}}")
         return res #.replace('\\\\\n', '\\\\ \\midrule\n')

     df_latex = process_dict(spearman_coefficients_all_latex)

     #df_latex.style.format(na_rep="0",precision=2).to_latex('{}/{}/Pearson_coefficients_LATEX_{}.tex'.format(script_dir, results_folder,subtitle),hrules=True)
     df_latex = latex_with_lines(df_latex)
     latex_file = open('{}/{}/Spearman_coefficients_LATEX_{}.tex'.format(script_dir, results_folder,subtitle),"w+")
     latex_file.write(df_latex)

     process_dict(covariances_all,"Latent_covariances",subtitle)
     process_dict(pearson_pvalues_all,"Latent_pearson_pvalues",subtitle)
     process_dict(pearson_coefficients_all,"Latent_pearson_coefficients",subtitle)
     process_dict(spearman_pvalues_all,"Latent_spearman_pvalues",subtitle)
     process_dict(spearman_coefficients_all,"Latent_spearman_coefficients",subtitle)

def calculate_auc(targets, predictions,overlap=None):
    """Calculates the ROC AUC and derivatives of the AUC"""
    #auc_dict = {"roc_auc":None,"auc01":None}
    try:
        if overlap is not None:
            idx_not_overlapped = np.invert(overlap.to_numpy()) #keep the non overlapping sequences results
        else:
            idx_not_overlapped = np.ones(len(targets)).astype(bool)  # keep all sequences
        targets = targets.to_numpy()[idx_not_overlapped]
        predictions = predictions.to_numpy()[idx_not_overlapped]
        predictions_nan = np.isnan(predictions) #some predictions are not made ...
        targets = targets[~predictions_nan]
        predictions = predictions[~predictions_nan]
        fpr, tpr, _ = roc_curve(y_true=targets,y_score=predictions)
        roc_auc = auc(fpr, tpr)
        auc01 = roc_auc_score(y_true=targets,y_score=predictions,max_fpr=0.1)
        # auc_dict["roc_auc"] = roc_auc
        # auc_dict["auc01"] = auc01
        return roc_auc,auc01
    except:
        # auc_dict["roc_auc"] = np.nan
        # auc_dict["auc01"] = np.nan
        return (np.nan,np.nan)

def calculate_ppv(targets, predictions,overlap=None):
    """Estimates the PPV (Precision) value"""
    if overlap is not None:
        idx_not_overlapped = np.invert(overlap)  # keep the non overlapping sequences results
    else:
        idx_not_overlapped = np.ones(len(targets)).astype(bool)  # keep all sequences
    targets = targets.to_numpy()[idx_not_overlapped]
    predictions = predictions.to_numpy()[idx_not_overlapped]
    try:
        try:
            binary_predictions = np.where(predictions >= 0.5, 1,0)  # for the rank this is useless, but I want to preserve the error
        except:
            binary_predictions = predictions
        binary_predictions_nan = pd.isnull(binary_predictions)
        targets = targets[~binary_predictions_nan]
        binary_predictions = binary_predictions[~binary_predictions_nan]
        tn, fp, fn, tp = confusion_matrix(y_true=targets, y_pred=binary_predictions).ravel()
        precision = tp / (tp + fp)
        return precision
    except:
        return np.nan

def calculate_ppv_modified(targets, predictions,overlap=None):
    """Estimates the a modified version of the PPV (Precision) value. It is referred as the proportion of true positives among
    the top N highest scoring predictions, where N is equal to the number of positive peptides in the given data set."""
    n_pos_data = int(np.sum(targets))
    if overlap is not None:
        idx_not_overlapped = np.invert(overlap.to_numpy())  # keep the non overlapping sequences results
    else:
        idx_not_overlapped = np.ones(len(targets)).astype(bool)  # keep all sequences
    if not isinstance(targets,np.ndarray):
        targets = targets.to_numpy()[idx_not_overlapped]
    if not isinstance(predictions,np.ndarray):
        predictions = predictions.to_numpy()[idx_not_overlapped]
    try:
        idx_descending = np.argsort(predictions)[::-1]
        targets = targets[idx_descending]
        predictions = predictions[idx_descending]
        targets_topscoring = targets[:n_pos_data]
        predictions_topscoring = predictions[:n_pos_data]
        ppv_mod = np.sum(targets_topscoring)/n_pos_data
        return ppv_mod
    except:
        return np.nan

def calculate_ap(targets, predictions,overlap=None):
    """Estimates the Average Precision"""
    if overlap is not None:
        idx_not_overlapped = np.invert(overlap)  # keep the non overlapping sequences results
    else:
        idx_not_overlapped = np.ones(len(targets)).astype(bool)  # keep all sequences
    targets = targets.to_numpy()[idx_not_overlapped]
    predictions = predictions.to_numpy()[idx_not_overlapped]
    try:
        # if len(unique_vals) > 2:
        #     binary_predictions = np.where(predictions >= 0.5, 1,0)  # for the rank this is useless, but I want to preserve the error
        #
        # else:
        #     binary_predictions = predictions
        # average_precision = average_precision_score(targets, binary_predictions)
        predictions_nan = np.isnan(predictions)  # some predictions are not made ...
        targets = targets[~predictions_nan]
        predictions = predictions[~predictions_nan]
        average_precision = average_precision_score(targets, predictions,average="weighted")
        return average_precision
    except:
        return np.nan

def calculate_pval(targets,predictions,overlap=None):
    if overlap is not None:
        idx_not_overlapped = np.invert(overlap)  # keep the non overlapping sequences results
    else:
        idx_not_overlapped = np.ones(len(targets)).astype(bool)  # keep all sequences
    targets = targets.to_numpy()[idx_not_overlapped]
    predictions = predictions.to_numpy()[idx_not_overlapped]
    try:
        lrm = sm.Logit(targets, predictions).fit(disp=0)
        pval = lrm.pvalues.item()
    except:
        pval = np.nan
    return pval

def calculate_precision_recall(targets,predictions,r="precision",overlap=None):
    """Calculates maximum precision & recall"""
    if overlap is not None:
        idx_not_overlapped = np.invert(overlap)  # keep the non overlapping sequences results
    else:
        idx_not_overlapped = np.ones(len(targets)).astype(bool)  # keep all sequences
    targets = targets.to_numpy()[idx_not_overlapped]
    predictions = predictions.to_numpy()[idx_not_overlapped]
    try:

        precision, recall, _ = precision_recall_curve(targets, predictions)
        
        precision_recall_joint = np.concatenate([precision[None,:],recall[None,:]],axis=0)
        max_precision_recall_combo = precision_recall_joint[:,precision_recall_joint.sum(axis=0).argmax()]
        if r == "precision":
            return max_precision_recall_combo[0]
        elif r == "recall":
            return max_precision_recall_combo[1]
    except:
        return np.nan

def process_nnalign(results_path, seqs_df,stress_dataset,mode="train",save_plot=True):
    nnalign_results_full = pd.read_csv(results_path, sep="\t", header=0)  # ["true_samples"]
    nnalign_results_full = nnalign_results_full[["Peptide", "Prediction","Measure"]]
    nnalign_results_full.columns = ["Icore", "Prediction","targets"]

    if any(x in stress_dataset for x in ["random","shuffled_targets","shuffled"]): #use the targets in the model because the random seed was different
        nnalign_results_full["targets"] = nnalign_results_full["targets"].astype(int)
    else:
        nnalign_results_full.drop("targets",axis=1,inplace=True)
        #nnalign_results_full1 = nnalign_results_full.merge(seqs_df, on="Icore", how="left")
        nnalign_results_full = nnalign_results_full.merge(seqs_df, on="Icore", how="inner")
        nnalign_results_full = nnalign_results_full[nnalign_results_full["targets"] != 2] #apparently some funky supposedly unlabelled data point is in the labelled data (removing duplicates failed), it is only 1

    auc_results = calculate_auc(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    roc_auc_dict = {"NNAlign2.1": auc_results[0]}
    auc01_dict = {"NNAlign2.1": auc_results[1]}

    ppv_results = calculate_ppv_modified(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    ppv_dict = {"NNAlign2.1": ppv_results}

    ap_results = calculate_ap(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    ap_dict = {"NNAlign2.1": ap_results}
    
    pval_results = calculate_pval(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    pval_dict = {"NNAlign2.1": pval_results}

    precision = defaultdict()
    recall = defaultdict()
    average_precision = defaultdict()

    precision["ordinary"], recall["ordinary"], _ = precision_recall_curve(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    average_precision["ordinary"] = average_precision_score(nnalign_results_full["targets"], nnalign_results_full["Prediction"],average="macro")


    #if save_plot:
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        nnalign_results_full["targets"].to_numpy().ravel(), nnalign_results_full["Prediction"].to_numpy().ravel()
    )
    precision_recall_joint = np.concatenate([precision["ordinary"][None,:],recall["ordinary"][None,:]],axis=0)
    max_precision_recall_combo = precision_recall_joint[:,precision_recall_joint.sum(axis=0).argmax()]
    max_precision_dict = {"NNAlign2.1":max_precision_recall_combo[0]}
    max_recall_dict = {"NNAlign2.1":max_precision_recall_combo[1]}

    average_precision["micro"] = average_precision_score(nnalign_results_full["targets"], nnalign_results_full["Prediction"], average="micro")
    fig = plt.figure()
    #Highlight: same results
    plt.plot(recall["micro"], precision["micro"],label="Average Precision (AP): {}".format(average_precision["micro"]))
    #plt.plot(recall["ordinary"], precision["ordinary"],label="Average Precision (AP): {}".format(average_precision["ordinary"]))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    plt.legend(loc='lower right', prop={'size': 15})
    plt.title("Average Precision curves \n NNAlign ({})".format(mode))
    plt.savefig("Benchmark/Plots/NNAlign_PrecisionRecall_curves_{}".format(mode))
    plt.clf()
    plt.close(fig)



    return roc_auc_dict,auc01_dict, ppv_dict, ap_dict, pval_dict,max_precision_dict,max_recall_dict

def process_nnalign_report_CV(results_path, seqs_df, mode="train", save_plot=True):
    """

    """

    nnalign_results_full = pd.read_csv(results_path, sep="\s+", header=0)  # ["true_samples"]

    dataset_name = "variable_length_Icore_sequences_viral_dataset9"
    nnalign_results_full = nnalign_results_full.loc[nnalign_results_full['DATASET'] == dataset_name]
    nnalign_results_full = nnalign_results_full[["AP", "AN", "AUC01_train", "AUC_train", "PPV_train"]]
    nnalign_results_full = nnalign_results_full.mean(0)


    auc_results = calculate_auc(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    roc_auc_dict = {"NNAlign2.1": nnalign_results_full["AUC_{]".format(mode)].item()}
    auc01_dict = {"NNAlign2.1": nnalign_results_full["AUC01_{]".format(mode)].item()}

    ppv_results = calculate_ppv_modified(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    ppv_dict = {"NNAlign2.1": nnalign_results_full["PPV_{]".format(mode)]}

    ap_results = calculate_ap(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    ap_dict = {"NNAlign2.1": nnalign_results_full["AUC_{]".format(mode)]}

    pval_results = calculate_pval(nnalign_results_full["targets"], nnalign_results_full["Prediction"])
    pval_dict = {"NNAlign2.1": pval_results}

    precision = defaultdict()
    recall = defaultdict()
    average_precision = defaultdict()
    precision["ordinary"], recall["ordinary"], _ = precision_recall_curve(nnalign_results_full["targets"],
                                                                          nnalign_results_full["Prediction"])
    average_precision["ordinary"] = average_precision_score(nnalign_results_full["targets"],
                                                            nnalign_results_full["Prediction"], average="macro")

    # if save_plot:
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        nnalign_results_full["targets"].ravel(), nnalign_results_full["Prediction"].ravel()
    )
    precision_recall_joint = np.concatenate([precision["ordinary"][None, :], recall["ordinary"][None, :]], axis=0)
    max_precision_recall_combo = precision_recall_joint[:, precision_recall_joint.sum(axis=0).argmax()]
    max_precision_dict = {"NNAlign2.1": max_precision_recall_combo[0]}
    max_recall_dict = {"NNAlign2.1": max_precision_recall_combo[1]}

    average_precision["micro"] = average_precision_score(nnalign_results_full["targets"],
                                                         nnalign_results_full["Prediction"], average="micro")
    fig = plt.figure()
    # Highlight: same results
    plt.plot(recall["micro"], precision["micro"], label="Average Precision (AP): {}".format(average_precision["micro"]))
    # plt.plot(recall["ordinary"], precision["ordinary"],label="Average Precision (AP): {}".format(average_precision["ordinary"]))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    plt.legend(loc='lower right', prop={'size': 15})
    plt.title("Average Precision curves \n NNAlign ({})".format(mode))
    plt.savefig("/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Plots/NNAlign_PrecisionRecall_curves_{}".format(mode))
    plt.clf()
    plt.close(fig)

    return roc_auc_dict, auc01_dict, ppv_dict, ap_dict, pval_dict, max_precision_dict, max_recall_dict

def plot_benchmark_vegvisir_helper(vegvisir_folder,overlap_idx,kfolds=5,aggregated_not_overlap=True):

    metrics_results_dict = defaultdict(lambda: defaultdict(list))

    for mode in ["Train","Valid","Test"]:
        for fold in range(kfolds):
            results = pd.read_csv(f"{vegvisir_folder}/{mode}_fold_{fold}/Epitopes_predictions_{mode}_fold_{fold}.tsv", sep="\t")
            results = results.merge(overlap_idx,on="Icore",how="left")
            if aggregated_not_overlap:
                results = results[results["Aggregated_overlap"] == False]
                print("--------------------------")
                print(mode)
                print(results.shape)
                print(results["Target_corrected"].sum())
                print("--------------------------")
            targets = np.array(results["Target_corrected"].tolist())
            onehot_targets = np.zeros((targets.shape[0], 2))
            onehot_targets[np.arange(0, targets.shape[0]), targets.astype(int)] = 1
            target_scores = results[["Vegvisir_negative_prob", "Vegvisir_positive_prob"]].to_numpy().astype(float)

            fpr=dict()
            tpr=dict()
            roc_auc=dict()
            precision = dict()
            recall=dict()
            average_precision=dict()
            ppv_mod=dict()
            pvals=dict()
            # ROC AUC per class
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(onehot_targets[:, i], target_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                roc_auc["auc01_class_{}".format(i)] = roc_auc_score(onehot_targets[:, i], target_scores[:, i],average="weighted", max_fpr=0.1)
                precision[i], recall[i], thresholds = precision_recall_curve(onehot_targets[:, i], target_scores[:, i])
                average_precision[i] = average_precision_score(onehot_targets[:, i], target_scores[:, i])
                ppv_mod[i] = calculate_ppv_modified(onehot_targets[:, i], target_scores[:, i])
                lrm = sm.Logit(onehot_targets[:, i], target_scores[:, i]).fit(disp=0)
                pvals[i] = lrm.pvalues.item()

            # metrics_results_dict[mode]["fpr_0"].append(fpr[0])
            # metrics_results_dict[mode]["fpr_1"].append(fpr[1])
            # metrics_results_dict[mode]["tpr_0"].append(tpr[0])
            # metrics_results_dict[mode]["tpr_1"].append(tpr[1])
            metrics_results_dict[mode.lower()]["roc_auc_class_0"].append(roc_auc[0])
            metrics_results_dict[mode.lower()]["roc_auc_class_1"].append(roc_auc[1])
            metrics_results_dict[mode.lower()]["auc01_class_0"].append(roc_auc[f"auc01_class_{0}"])
            metrics_results_dict[mode.lower()]["auc01_class_1"].append(roc_auc[f"auc01_class_{1}"])
            metrics_results_dict[mode.lower()]["pval_class_0"].append(pvals[0])
            metrics_results_dict[mode.lower()]["pval_class_1"].append(pvals[1])
            metrics_results_dict[mode.lower()]["ap_class_0"].append(average_precision[0])
            metrics_results_dict[mode.lower()]["ap_class_1"].append(average_precision[1])
            metrics_results_dict[mode.lower()]["ppv_mod_class_0"].append(ppv_mod[0])
            metrics_results_dict[mode.lower()]["ppv_mod_class_1"].append(ppv_mod[1])
            metrics_results_dict[mode.lower()]["precision_class_0"].append(precision[0])
            metrics_results_dict[mode.lower()]["precision_class_1"].append(precision[1])
            metrics_results_dict[mode.lower()]["recall_class_0"].append(recall[0])
            metrics_results_dict[mode.lower()]["recall_class_1"].append(recall[1])

    # print(metrics_results_dict["train"]["pval_class_0"])
    # print(metrics_results_dict["train"]["pval_class_1"])
    # print(metrics_results_dict["test"]["pval_class_0"])
    # print(metrics_results_dict["test"]["pval_class_1"])
    #
    # exit()

    return metrics_results_dict

def plot_benchmark_vegvisir_helper2(args,vegvisir_folder,overlap_idx,kfolds=5,aggregated_not_overlap=True):

    metrics_results_dict = defaultdict(lambda: defaultdict(list))

    for mode in ["Train","Valid","Test"]:
        for fold in range(kfolds):
            results = pd.read_csv(f"{vegvisir_folder}/{mode}_fold_{fold}/Epitopes_predictions_{mode}_fold_{fold}.tsv", sep="\t")
            results = results.merge(overlap_idx,on="Icore",how="left")
            if aggregated_not_overlap:
                results = results[results["Aggregated_overlap"] == False]
                print("--------------------------")
                print(mode)
                print(results.shape)
                print(results["Target_corrected"].sum())
                print("--------------------------")
            targets = np.array(results["Target_corrected"].tolist())
            target_scores = results[["Vegvisir_negative_prob", "Vegvisir_positive_prob"]].to_numpy().astype(float)
            target_scores =  torch.nn.functional.softmax(torch.from_numpy(target_scores),dim=-1)[:,1].numpy()
            fpr=dict()
            tpr=dict()
            roc_auc=dict()
            precision = dict()
            recall=dict()
            average_precision=dict()
            ppv_mod=dict()
            pvals=dict()
            # ROC AUC per class
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(targets, target_scores)
                roc_auc[i] = auc(fpr[i], tpr[i])
                roc_auc["auc01_class_{}".format(i)] = roc_auc_score(targets, target_scores,average="weighted", max_fpr=0.1)
                precision[i], recall[i], thresholds = precision_recall_curve(targets, target_scores)
                average_precision[i] = average_precision_score(targets, target_scores)
                ppv_mod[i] = calculate_ppv_modified(targets, target_scores)
                lrm = sm.Logit(targets, target_scores).fit(disp=0)
                pvals[i] = lrm.pvalues.item()

            # metrics_results_dict[mode]["fpr_0"].append(fpr[0])
            # metrics_results_dict[mode]["fpr_1"].append(fpr[1])
            # metrics_results_dict[mode]["tpr_0"].append(tpr[0])
            # metrics_results_dict[mode]["tpr_1"].append(tpr[1])
            metrics_results_dict[mode.lower()]["roc_auc_class_0"].append(roc_auc[0])
            metrics_results_dict[mode.lower()]["roc_auc_class_1"].append(roc_auc[1])
            metrics_results_dict[mode.lower()]["auc01_class_0"].append(roc_auc[f"auc01_class_{0}"])
            metrics_results_dict[mode.lower()]["auc01_class_1"].append(roc_auc[f"auc01_class_{1}"])
            metrics_results_dict[mode.lower()]["pval_class_0"].append(pvals[0])
            metrics_results_dict[mode.lower()]["pval_class_1"].append(pvals[1])
            metrics_results_dict[mode.lower()]["ap_class_0"].append(average_precision[0])
            metrics_results_dict[mode.lower()]["ap_class_1"].append(average_precision[1])
            metrics_results_dict[mode.lower()]["ppv_mod_class_0"].append(ppv_mod[0])
            metrics_results_dict[mode.lower()]["ppv_mod_class_1"].append(ppv_mod[1])
            metrics_results_dict[mode.lower()]["precision_class_0"].append(precision[0])
            metrics_results_dict[mode.lower()]["precision_class_1"].append(precision[1])
            metrics_results_dict[mode.lower()]["recall_class_0"].append(recall[0])
            metrics_results_dict[mode.lower()]["recall_class_1"].append(recall[1])

    return metrics_results_dict

def plot_benchmark_vegvisir_helper3(args,vegvisir_folder,overlap_idx,kfolds=5,aggregated_not_overlap=True):
    """Compute Ensembl metrics"""
    metrics_results_dict = defaultdict(lambda: defaultdict(list))


    for mode in ["Train","Test"]:
        target_scores_list = []
        for fold in range(kfolds):
            results = pd.read_csv(f"{vegvisir_folder}/{mode}_fold_{fold}/Epitopes_predictions_{mode}_fold_{fold}.tsv", sep="\t")
            if overlap_idx is not None:
                results = results.merge(overlap_idx,on="Icore",how="left")
            if aggregated_not_overlap:
                results = results[results["Aggregated_overlap"] == False]
            targets = np.array(results["Target_corrected"].tolist())
            onehot_targets = np.zeros((targets.shape[0], args.num_classes))
            onehot_targets[np.arange(0, targets.shape[0]), targets.astype(int)] = 1
            target_scores = results[["Vegvisir_negative_prob", "Vegvisir_positive_prob"]].to_numpy().astype(float)
            target_scores_list.append(target_scores)

        target_scores = sum(target_scores_list)/kfolds

        fpr=dict()
        tpr=dict()
        roc_auc=dict()
        precision = dict()
        recall=dict()
        average_precision=dict()
        ppv_mod=dict()
        pvals=dict()
        # ROC AUC per class
        for i in range(args.num_classes):
            if i <= 1:
                fpr[i], tpr[i], _ = roc_curve(onehot_targets[:, i], target_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                roc_auc["auc01_class_{}".format(i)] = roc_auc_score(onehot_targets[:, i], target_scores[:, i],average="weighted", max_fpr=0.1)
                precision[i], recall[i], thresholds = precision_recall_curve(onehot_targets[:, i], target_scores[:, i])
                average_precision[i] = average_precision_score(onehot_targets[:, i], target_scores[:, i])
                ppv_mod[i] = calculate_ppv_modified(onehot_targets[:, i], target_scores[:, i])
                lrm = sm.Logit(onehot_targets[:, i], target_scores[:, i]).fit(disp=0)
                pvals[i] = lrm.pvalues.item()

        # metrics_results_dict[mode]["fpr_0"].append(fpr[0])
        # metrics_results_dict[mode]["fpr_1"].append(fpr[1])
        # metrics_results_dict[mode]["tpr_0"].append(tpr[0])
        # metrics_results_dict[mode]["tpr_1"].append(tpr[1])
        metrics_results_dict[mode.lower()]["roc_auc_class_0"].append(roc_auc[0])
        metrics_results_dict[mode.lower()]["roc_auc_class_1"].append(roc_auc[1])
        metrics_results_dict[mode.lower()]["auc01_class_0"].append(roc_auc[f"auc01_class_{0}"])
        metrics_results_dict[mode.lower()]["auc01_class_1"].append(roc_auc[f"auc01_class_{1}"])
        metrics_results_dict[mode.lower()]["pval_class_0"].append(pvals[0])
        metrics_results_dict[mode.lower()]["pval_class_1"].append(pvals[1])
        metrics_results_dict[mode.lower()]["ap_class_0"].append(average_precision[0])
        metrics_results_dict[mode.lower()]["ap_class_1"].append(average_precision[1])
        metrics_results_dict[mode.lower()]["ppv_mod_class_0"].append(ppv_mod[0])
        metrics_results_dict[mode.lower()]["ppv_mod_class_1"].append(ppv_mod[1])
        metrics_results_dict[mode.lower()]["precision_class_0"].append(precision[0])
        metrics_results_dict[mode.lower()]["precision_class_1"].append(precision[1])
        metrics_results_dict[mode.lower()]["recall_class_0"].append(recall[0])
        metrics_results_dict[mode.lower()]["recall_class_1"].append(recall[1])

    # print(metrics_results_dict["train"]["pval_class_0"])
    # print(metrics_results_dict["train"]["pval_class_1"])
    # print(metrics_results_dict["test"]["pval_class_0"])
    # print(metrics_results_dict["test"]["pval_class_1"])
    #
    # exit()

    return metrics_results_dict

def plot_benchmarking_results(dict_results_vegvisir,script_dir,keyname="",folder="Benchmark",title="",keep_only_overlapped=False,aggregated_not_overlap=False,keep_all=True,only_class1=True,ensemble=False):
    """Compare results across different programns on the -golden- dataset that is built from the Icore sequence and sequences of variable length 8-11

    :param keep_overlapped : computes metrics for the sequences present in the original training dataset's --> False for benchmark
    :param aggregated_not_overlap : True #removes any sequence present in any of the other model's datasets --> True for benchmark
    -Notes:
        https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
        https://towardsdatascience.com/imbalanced-data-stop-using-roc-auc-and-use-auprc-instead-46af4910a494
        Class imbalance metrics: https://www.kaggle.com/code/marcinrutecki/best-techniques-and-metrics-for-imbalanced-dataset
    """

    #Highlight: Vegvisir results

    metrics_keys = ["ppv","fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1","pval_class_0","pval_class_1"]

    vegvisir_folder = dict_results_vegvisir["Icore"][keyname]
    train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(vegvisir_folder, 0))
    #valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(vegvisir_folder, 0))
    test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(vegvisir_folder, 0))
    args = train_out["args"]

    #Highlight: extract the original sequences and the true labels to use them with NNAlign
    dataset_info = train_out["dataset_info"]
    custom_features_dicts = VegvisirUtils.build_features_dicts(dataset_info)
    aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]

    train_sequences = train_out["summary_dict"]["data_int_samples"][:, 1:].squeeze(1)
    train_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(train_sequences)
    train_sequences_raw = list(map(lambda seq: "".join(seq).replace("#",""),train_sequences_raw))

    train_labels = train_out["summary_dict"]["true_samples"]
    train_df = pd.DataFrame({"Icore":train_sequences_raw,"targets":train_labels})

    test_sequences = test_out["summary_dict"]["data_int_samples"][:, 1:].squeeze(1)
    test_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(test_sequences)
    test_sequences_raw = list(map(lambda seq: "".join(seq).replace("#",""),test_sequences_raw))
    test_labels = test_out["summary_dict"]["true_samples"]
    test_df = pd.DataFrame({"Icore":test_sequences_raw,"targets":test_labels})
    #
    # metrics_results_dict = plot_kfold_comparison_helper(metrics_keys, script_dir, folder=vegvisir_folder, overwrite=False, kfolds=5)

    #Highlight: Find the sequence overlap with the train datasets from the other programs to take it into account for the metrics. This file indicates as True if the sequence is present in the dataset
    other_programs_sequence_overlap = pd.read_csv("{}/Benchmark/Other_Programs/dataset_partitioned_overlap_benchmark.tsv".format(script_dir),sep="\t")

    # #Highlight: invert the dataframe to see the performance on the sequences that have been used for training the benchmarking programs
    if keep_only_overlapped:#keep the sequences in the original training datasets that overlap with Vegvisir's
        #other_programs_sequence_overlap_inverted = other_programs_sequence_overlap.applymap(lambda x: not x).astype(bool) #invert to check results on the overlapped sequences
        other_programs_sequence_overlap_inverted = other_programs_sequence_overlap.map(lambda x: not x).astype(bool) #invert booleans to check results on the overlapped sequences
        other_programs_sequence_overlap_inverted["Icore"] = other_programs_sequence_overlap["Icore"]
        other_programs_sequence_overlap = other_programs_sequence_overlap_inverted

    other_programs_sequence_overlap["MixMHCPred"] = False
    other_programs_sequence_overlap["NetMHC"] = False #assign to False, since we keep everything with False

    if aggregated_not_overlap:#use the sequences found in all the datasets (overlapped_per_model= True) or not (overlapped_per_model=False)
        aggregated_overlap_idx = other_programs_sequence_overlap.loc[:, ~other_programs_sequence_overlap.columns.isin(['Icore',"target","partition"])].to_numpy()
        aggregated_overlap_idx = pd.DataFrame({"Aggregated_overlap":np.mean(aggregated_overlap_idx,axis=1).astype(bool)}) #track all the sequences not present in any of the benchmark training datasets
        #print(aggregated_overlap_idx["Aggregated_overlap"].sum())
        other_programs_sequence_overlap = pd.concat([other_programs_sequence_overlap["Icore"],aggregated_overlap_idx],axis=1)

    train_df = train_df.merge(other_programs_sequence_overlap,on="Icore",how="left")
    test_df = test_df.merge(other_programs_sequence_overlap,on="Icore",how="left")

    if aggregated_not_overlap:
        train_df = train_df[train_df["Aggregated_overlap"] == False]
        test_df = test_df[test_df["Aggregated_overlap"] == False]

    if ensemble:
        metrics_results_dict = plot_benchmark_vegvisir_helper3(args,vegvisir_folder,other_programs_sequence_overlap,kfolds=5,aggregated_not_overlap=aggregated_not_overlap)
    else:
        metrics_results_dict = plot_benchmark_vegvisir_helper2(args,vegvisir_folder,other_programs_sequence_overlap,kfolds=5,aggregated_not_overlap=aggregated_not_overlap)

    metrics_results_train = metrics_results_dict["train"]
    #metrics_results_valid = metrics_results_dict["valid"]
    metrics_results_test = metrics_results_dict["test"]
    #aggregated_not_overlap = False #removes any sequence present in any of the other model's datasets --> True for benchmark

    vegvisir_results_roc_auc_train = {"Vegvisir":np.round((np.mean(np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train["roc_auc_class_1"])))/2, 2)}
    #vegvisir_results_roc_auc_train_std = {"Vegvisir":np.round((np.std(np.array(metrics_results_train["roc_auc_class_0"])) + np.std(np.array(metrics_results_train["roc_auc_class_1"])))/2, 2)}

    vegvisir_results_ap_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["ap_class_0"])) + np.mean(np.array(metrics_results_train["ap_class_1"]))) / 2, 2)}
    vegvisir_results_pval_train = {"Vegvisir":np.mean(np.array(metrics_results_train["pval_class_0"])) + np.mean(np.array(metrics_results_train["pval_class_1"]))}

    if only_class1: #some statistics are uni-class
        vegvisir_results_auc01_train = {"Vegvisir":np.round(np.mean(np.array(metrics_results_train["auc01_class_1"])), 2)}
        vegvisir_results_ppv_train = {"Vegvisir":np.round(np.mean(np.array(metrics_results_train["ppv_mod_class_1"])), 2)}
        if keep_only_overlapped and aggregated_not_overlap:
            vegvisir_results_ap_train = {"Vegvisir":np.round(np.mean(np.array(metrics_results_train["ap_class_1"])), 2)}
        vegvisir_results_pval_train = {"Vegvisir":np.mean(np.array(metrics_results_train["pval_class_1"]))}

    else:
        vegvisir_results_auc01_train = {"Vegvisir": np.round((np.mean(
            np.array(metrics_results_train["auc01_class_0"])) + np.mean(
            np.array(metrics_results_train["auc01_class_1"]))) / 2, 2)}
        vegvisir_results_ppv_train = {"Vegvisir": np.round((np.mean(
            np.array(metrics_results_train["ppv_mod_class_0"])) + np.mean(
            np.array(metrics_results_train["ppv_mod_class_1"]))) / 2, 2)}

    # vegvisir_results_precision_train = {"Vegvisir":np.round((np.mean(np.array(metrics_results_train["precision_class_0"])) + np.mean(np.array(metrics_results_train["precision_class_1"])))/2, 3)}
    # vegvisir_results_recall_train = {"Vegvisir":np.round((np.mean(np.array(metrics_results_train["recall_class_0"])) + np.mean(np.array(metrics_results_train["recall_class_1"])))/2, 3)}
    vegvisir_results_precision_train = {"Vegvisir":np.round(np.mean(np.array(metrics_results_train["precision_class_1"])), 3)}
    vegvisir_results_recall_train = {"Vegvisir":np.round(np.mean(np.array(metrics_results_train["recall_class_0"])), 3)}

    #############################################################################################################################################################################

    vegvisir_results_roc_auc_test = {"Vegvisir":np.round((np.mean(np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test["roc_auc_class_1"])))/2, 2)}
    #vegvisir_results_roc_auc_test_std = {"Vegvisir":np.round((np.std(np.array(metrics_results_test["roc_auc_class_0"])) + np.std(np.array(metrics_results_test["roc_auc_class_1"])))/2, 3)}

    #if aggregated_not_overlap and keep_only_overlapped:#the majority of the sequences are +
    vegvisir_results_ap_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["ap_class_0"])) + np.mean(np.array(metrics_results_test["ap_class_1"]))) / 2, 3)}
    vegvisir_results_pval_test = {"Vegvisir":(np.mean(np.array(metrics_results_test["pval_class_0"])) + np.mean(np.array(metrics_results_test["pval_class_1"])))/2}

    if only_class1:
        vegvisir_results_auc01_test = {"Vegvisir":np.round(np.mean(np.array(metrics_results_test["auc01_class_1"])), 3)}
        vegvisir_results_ppv_test = {"Vegvisir":np.round(np.mean(np.array(metrics_results_test["ppv_mod_class_1"])), 3)}
        if keep_only_overlapped and aggregated_not_overlap:
            vegvisir_results_ap_test = {"Vegvisir":np.round(np.array(metrics_results_test["ap_class_1"]), 3)}
        vegvisir_results_pval_test = {"Vegvisir":np.mean(np.array(metrics_results_test["pval_class_1"]))}

    else:
        vegvisir_results_auc01_test = {"Vegvisir": np.round((np.mean(
            np.array(metrics_results_test["auc01_class_0"])) + np.mean(
            np.array(metrics_results_test["auc01_class_1"]))) / 2, 2)}
        vegvisir_results_ppv_test = {"Vegvisir": np.round((np.mean(
            np.array(metrics_results_test["ppv_mod_class_0"])) + np.mean(
            np.array(metrics_results_test["ppv_mod_class_1"]))) / 2, 2)}


    # vegvisir_results_precision_test = {"Vegvisir":np.round((np.mean(np.array(metrics_results_test["precision_class_0"])) + np.mean(np.array(metrics_results_test["precision_class_1"])))/2, 3)}
    # vegvisir_results_recall_test = {"Vegvisir":np.round((np.mean(np.array(metrics_results_test["recall_class_0"])) + np.mean(np.array(metrics_results_test["recall_class_1"])))/2, 3)}
    vegvisir_results_precision_test = {"Vegvisir":np.round(np.mean(np.array(metrics_results_test["precision_class_1"])), 3)}
    vegvisir_results_recall_test = {"Vegvisir":np.round(np.mean(np.array(metrics_results_test["recall_class_0"])),3)}


    #Highlight: NNalign results for viral_dataset9
    if keyname == "raw-variable-length-vd9":
        nnalign_results_path_train_full = "Benchmark/Other_Programs/NNAlign_results_07_01_2024/Icore/variable_length_Icore_sequences_viral_dataset9/nnalign_peplen_8-11_iter_100_3369/nnalign_peplen_8-11_iter_100_3369.lg9.sorted.pred"
        nnalign_results_path_test_full = "Benchmark/Other_Programs/NNAlign_results_07_01_2024/Icore/variable_length_Icore_sequences_viral_dataset9/nnalign_peplen_8-11_iter_100_3369/test_Icore_variable_length_Icore_sequences_viral_dataset9_lg9_10424.evalset.txt"

    else: #Highlight: NNalign results for viral_dataset15
        nnalign_results_path_train_full = "Benchmark/Other_Programs/NNAlign_results_19_01_2024/Icore/variable_length_Icore_sequences_viral_dataset15/nnalign_peplen_8-11_iter_100_10523/nnalign_peplen_8-11_iter_100_10523.lg9.sorted.pred"
        nnalign_results_path_test_full = "Benchmark/Other_Programs/NNAlign_results_19_01_2024/Icore/variable_length_Icore_sequences_viral_dataset15/nnalign_peplen_8-11_iter_100_10523/test_Icore_variable_length_Icore_sequences_viral_dataset15_lg9_13809.evalset.txt"

    #train_test_df = pd.concat([train_df,test_df],axis=0)

    (nnalign_results_train_roc_auc_dict,nnalign_results_train_auc01_dict, nnalign_results_train_ppv_dict,
     nnalign_results_train_ap_dict,nnalign_results_train_pval_dict,nnalign_results_train_precision_dict,nnalign_results_train_recall_dict) = process_nnalign(nnalign_results_path_train_full,train_df,keyname,mode="train")
    (nnalign_results_test_roc_auc_dict,nnalign_results_test_auc01_dict, nnalign_results_test_ppv_dict,
     nnalign_results_test_ap_dict, nnalign_results_test_pval_dict,nnalign_results_test_precision_dict,nnalign_results_test_recall_dict) = process_nnalign(nnalign_results_path_test_full,test_df,keyname,mode="test")

    #Highlight: Other programs
    #other_programs_results_path = "{}/Benchmark/Other_Programs/sequences_viral_dataset9_predictors_other_models.tsv".format(script_dir)
    other_programs_results_path = "{}/Benchmark/Other_Programs/sequences_viral_dataset9_predictors_NEW.tsv".format(script_dir)
    other_programs_results = pd.read_csv(other_programs_results_path,sep="\t")


    #pd.set_option('display.max_columns', None)

    def process_results_option_a(seqs_df,training=True):
        """Aggregate results based on same Icore and remove the possible allele-encoding prediction effect"""
        #other_programs_results_mode= other_programs_results[other_programs_results["training"] == training]
        #missing_seqs = seqs_df[~seqs_df["Icore"].isin(other_programs_results["Icore"])]
        #other_programs_results_mode= other_programs_results[other_programs_results["training"] == training] #Highlight: This training flag corresponds solely to the viral_dataset9, not viral_dataset15

        other_programs_results_mode = other_programs_results[other_programs_results["Icore"].isin(seqs_df["Icore"].values.tolist())]

        other_programs_results_mode = other_programs_results_mode[["Icore","target_corrected","PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno","DeepNetBim_immuno_probability","BigMHC","NetMHCpan_binding"]]
        #Highlight: Grouping by alleles ---> To make similar to vegvisir's logic
        #a) Group non categorical outputs ("continuous")
        other_programs_results_mode_continuous = other_programs_results_mode.groupby("Icore",as_index=False)[["PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno_probability","BigMHC","NetMHCpan_binding"]].agg(lambda x: sum(list(x))/len(list(x)))
        #b) Process categorical outputs ("discrete")
        other_programs_results_mode_categorical = other_programs_results_mode.groupby("Icore",as_index=False)[["target_corrected","DeepNetBim_immuno"]].agg(pd.Series.mode)
        other_programs_results_mode = other_programs_results_mode_continuous.merge(other_programs_results_mode_categorical,on=["Icore"],how="left")

        return other_programs_results_mode

    def process_results_option_b(seqs_df,training=True):
        "Do not aggregate results based on Icore"
        #other_programs_results_mode= other_programs_results[other_programs_results["training"] == training] #Highlight: DO NOT USE: This training flag corresponds solely to the viral_dataset9, not viral_dataset15
        #missing_seqs = seqs_df[~seqs_df["Icore"].isin(other_programs_results["Icore"])]
        other_programs_results_mode = other_programs_results[other_programs_results["Icore"].isin(seqs_df["Icore"].values.tolist())]
        other_programs_results_mode = other_programs_results_mode[["Icore","target_corrected","PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno","DeepNetBim_immuno_probability","BigMHC","NetMHCpan_binding"]]

        return other_programs_results_mode

    other_programs_results_train = process_results_option_a(train_df,training=True)
    other_programs_results_test = process_results_option_a(test_df,training=False)
    #Highlight: Make sure that the information on the overlapped sequences has the same order
    #Before
    other_programs_sequence_overlap_train = VegvisirUtils.merge_in_left_order(other_programs_results_train[["Icore"]],other_programs_sequence_overlap,on="Icore") #guarantees the same order of Icore
    #After
    other_programs_sequence_overlap_test = VegvisirUtils.merge_in_left_order(other_programs_results_test[["Icore"]],other_programs_sequence_overlap,on="Icore") #guarantees the same order of Icore

    programs_list = ["PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno","DeepNetBim_immuno_probability","BigMHC","NetMHCpan_binding"]
    if aggregated_not_overlap:
        programs_list2 = ["Aggregated_overlap"]*10
    else:
        programs_list2 = ["PRIME", "PRIME", "MixMHCPred", "IEDB_immunogenicity", "DeepImmuno", "DeepNetBim","DeepNetBim", "DeepNetBim", "BigMHC", "NetMHC"]

    #Highlight: ROC-AUC & AUC-01
    if keep_all:
        auc_results_train = list(map(lambda program,program2: calculate_auc(other_programs_results_train["target_corrected"],other_programs_results_train[program],None),programs_list,programs_list2))
    else:
        auc_results_train = list(map(lambda program,program2: calculate_auc(other_programs_results_train["target_corrected"],other_programs_results_train[program],other_programs_sequence_overlap_train[program2]),programs_list,programs_list2 ))
    auc_results_train = list(zip(*auc_results_train))
    roc_auc_results_train = auc_results_train[0]
    auc01_results_train = auc_results_train[1]

    if keep_all:
        auc_results_test = list(map(lambda program,program2: calculate_auc(other_programs_results_test["target_corrected"],other_programs_results_test[program],None),programs_list,programs_list2 ))
    else:
        auc_results_test = list(map(lambda program,program2: calculate_auc(other_programs_results_test["target_corrected"],other_programs_results_test[program],other_programs_sequence_overlap_test[program2]),programs_list,programs_list2 ))

    auc_results_test = list(zip(*auc_results_test))
    roc_auc_results_test = auc_results_test[0]
    auc01_results_test = auc_results_test[1]

    roc_auc_results_train_dict = dict(zip(programs_list,roc_auc_results_train))
    roc_auc_results_test_dict = dict(zip(programs_list,roc_auc_results_test))
    
    auc01_results_train_dict = dict(zip(programs_list,auc01_results_train))
    auc01_results_test_dict = dict(zip(programs_list,auc01_results_test))

    #Highlight: PPV
    if keep_all:
        ppv_results_train = list(map(lambda program, program2: calculate_ppv_modified(other_programs_results_train["target_corrected"],
                                                                 other_programs_results_train[program],
                                                                 None),programs_list, programs_list2))
        ppv_results_test = list(map(lambda program, program2: calculate_ppv_modified(other_programs_results_test["target_corrected"],
                                                                 other_programs_results_test[program],
                                                                 None),programs_list, programs_list2))

    else:
        ppv_results_train = list(map(lambda program,program2: calculate_ppv_modified(other_programs_results_train["target_corrected"],other_programs_results_train[program],other_programs_sequence_overlap_train[program2]),programs_list,programs_list2 ))
        ppv_results_test = list(map(lambda program,program2: calculate_ppv_modified(other_programs_results_test["target_corrected"],other_programs_results_test[program],other_programs_sequence_overlap_test[program2]),programs_list,programs_list2 ))

    ppv_results_train_dict = dict(zip(programs_list,ppv_results_train))
    ppv_results_test_dict = dict(zip(programs_list,ppv_results_test))
    
    
    #Highlight: Pval

    if keep_all:
        pval_results_train = list(map(lambda program,program2: calculate_pval(other_programs_results_train["target_corrected"],other_programs_results_train[program],None), programs_list,programs_list2))
        pval_results_test = list(map(lambda program,program2: calculate_pval(other_programs_results_test["target_corrected"],other_programs_results_test[program],None), programs_list,programs_list2))

    else:
        pval_results_train = list(map(lambda program, program2: calculate_pval(other_programs_results_train["target_corrected"],other_programs_results_train[program],other_programs_sequence_overlap_train[program2]),programs_list, programs_list2))
        pval_results_test = list(map(lambda program, program2: calculate_pval(other_programs_results_test["target_corrected"],other_programs_results_test[program],other_programs_sequence_overlap_test[program2]), programs_list,programs_list2))

    pval_results_train_dict = dict(zip(programs_list, pval_results_train))
    pval_results_test_dict = dict(zip(programs_list, pval_results_test))

    #Highlight: AP
    if keep_all:
        ap_results_train = list(map(lambda program,program2: calculate_ap(other_programs_results_train["target_corrected"],other_programs_results_train[program],None),programs_list,programs_list2 ))
        ap_results_test = list(map(lambda program,program2: calculate_ap(other_programs_results_test["target_corrected"],other_programs_results_test[program],None),programs_list,programs_list2 ))
    else:
        ap_results_train = list(map(lambda program, program2: calculate_ap(other_programs_results_train["target_corrected"],other_programs_results_train[program],other_programs_sequence_overlap_train[program2]), programs_list,programs_list2))
        ap_results_test = list(map(lambda program, program2: calculate_ap(other_programs_results_test["target_corrected"],other_programs_results_test[program],other_programs_sequence_overlap_test[program2]), programs_list,programs_list2))

    ap_results_train_dict = dict(zip(programs_list,ap_results_train))
    ap_results_test_dict = dict(zip(programs_list,ap_results_test))
    
    
    #Highlight: Max Precision & Recall
    if keep_all:
        precision_results_train = list(map(lambda program,program2: calculate_precision_recall(other_programs_results_train["target_corrected"],other_programs_results_train[program],r="precision",overlap=None),programs_list,programs_list2 ))
        precision_results_test = list(map(lambda program,program2: calculate_precision_recall(other_programs_results_test["target_corrected"],other_programs_results_test[program],r="precision",overlap=None),programs_list,programs_list2 ))
    else:

        precision_results_train = list(map(lambda program, program2: calculate_precision_recall(other_programs_results_train["target_corrected"],other_programs_results_train[program],r="precision",overlap=other_programs_sequence_overlap_train[program2]), programs_list, programs_list2))
        precision_results_test = list(map(lambda program, program2: calculate_precision_recall(other_programs_results_test["target_corrected"],other_programs_results_test[program],r="precision",overlap=other_programs_sequence_overlap_test[program2]), programs_list, programs_list2))

    precision_results_train_dict = dict(zip(programs_list,precision_results_train))
    precision_results_test_dict = dict(zip(programs_list,precision_results_test))
    
    if keep_all:
        recall_results_train = list(map(lambda program,program2: calculate_precision_recall(other_programs_results_train["target_corrected"],other_programs_results_train[program],r="recall",overlap=None),programs_list,programs_list2 ))
        recall_results_test = list(map(lambda program,program2: calculate_precision_recall(other_programs_results_test["target_corrected"],other_programs_results_test[program],r="recall",overlap=None),programs_list,programs_list2 ))
    else:
        recall_results_train = list(map(lambda program, program2: calculate_precision_recall(other_programs_results_train["target_corrected"],other_programs_results_train[program], r="recall",overlap=other_programs_sequence_overlap_train[program2]),programs_list, programs_list2))
        recall_results_test = list(map(lambda program, program2: calculate_precision_recall(other_programs_results_test["target_corrected"],other_programs_results_test[program], r="recall",overlap=other_programs_sequence_overlap_test[program2]),programs_list, programs_list2))

    recall_results_train_dict = dict(zip(programs_list,recall_results_train))
    recall_results_test_dict = dict(zip(programs_list,recall_results_test))
    
    
    

    #Highlight: Merge all results
    roc_auc_results_train_dict = {**vegvisir_results_roc_auc_train,**nnalign_results_train_roc_auc_dict,**roc_auc_results_train_dict,}
    roc_auc_results_test_dict = {**vegvisir_results_roc_auc_test,**nnalign_results_test_roc_auc_dict,**roc_auc_results_test_dict}
    
    auc01_results_train_dict = {**vegvisir_results_auc01_train,**nnalign_results_train_auc01_dict,**auc01_results_train_dict,}
    auc01_results_test_dict = {**vegvisir_results_auc01_test,**nnalign_results_test_auc01_dict,**auc01_results_test_dict}
    

    ppv_results_train_dict = {**vegvisir_results_ppv_train,**nnalign_results_train_ppv_dict,**ppv_results_train_dict,}
    ppv_results_test_dict = {**vegvisir_results_ppv_test,**nnalign_results_test_ppv_dict,**ppv_results_test_dict}
    
    pval_results_train_dict = {**vegvisir_results_pval_train,**nnalign_results_train_pval_dict,**pval_results_train_dict,}
    pval_results_test_dict = {**vegvisir_results_pval_test,**nnalign_results_test_pval_dict,**pval_results_test_dict}

    ap_results_train_dict = {**vegvisir_results_ap_train, **nnalign_results_train_ap_dict,**ap_results_train_dict, }
    ap_results_test_dict = {**vegvisir_results_ap_test, **nnalign_results_test_ap_dict, **ap_results_test_dict}
    
    
    precision_results_train_dict = {**vegvisir_results_precision_train, **nnalign_results_train_precision_dict,**precision_results_train_dict, }
    precision_results_test_dict = {**vegvisir_results_precision_test, **nnalign_results_test_precision_dict, **precision_results_test_dict}
    
    recall_results_train_dict = {**vegvisir_results_recall_train, **nnalign_results_train_recall_dict,**recall_results_train_dict, }
    recall_results_test_dict = {**vegvisir_results_recall_test, **nnalign_results_test_recall_dict, **recall_results_test_dict}

    names_dict = {"Vegvisir":"Vegvisir",
                  "NNAlign2.1":"NNAlign2.1",
                  "PRIME_rank":"PRIME \n (rank)",
                  "PRIME_score":"PRIME \n (probability)",
                  "MixMHCpred_rank_binding":"MixMHCpred \n (MHC binding rank)",
                  "IEDB_immuno":"IEDB immuno",
                  "DeepImmuno":"DeepImmuno",
                  "DeepNetBim_binding":"DeepNetBim \n (MHC binding rank)",
                  "DeepNetBim_immuno":"DeepNetBim \n (binary prediction)",
                  "DeepNetBim_immuno_probability":"DeepNetBim \n (probability)",
                  "BigMHC":"BigMHC",
                  "NetMHCpan_binding":"NetMHCpan"}


    rank_programs = ["MixMHCpred_rank_binding","PRIME_rank","DeepNetBim_immuno","DeepNetBim_binding"]
    benchmark_programs_list = [program for program in names_dict.keys() if program not in rank_programs]

    roc_auc_results_train_dict = {key:val for key,val in roc_auc_results_train_dict.items() if key in benchmark_programs_list}
    roc_auc_results_test_dict = {key:val for key,val in roc_auc_results_test_dict.items() if key in benchmark_programs_list}

    auc01_results_train_dict = {key: val for key, val in auc01_results_train_dict.items() if
                                  key in benchmark_programs_list}
    auc01_results_test_dict = {key: val for key, val in auc01_results_test_dict.items() if
                                 key in benchmark_programs_list}

    ppv_results_train_dict = {key: val for key, val in ppv_results_train_dict.items() if key in benchmark_programs_list}
    ppv_results_test_dict = {key: val for key, val in ppv_results_test_dict.items() if key in benchmark_programs_list}
    
    pval_results_train_dict = {key: val for key, val in pval_results_train_dict.items() if key in benchmark_programs_list}
    pval_results_test_dict = {key: val for key, val in pval_results_test_dict.items() if key in benchmark_programs_list}

    ap_results_train_dict = {key: val for key, val in ap_results_train_dict.items() if key in benchmark_programs_list}
    ap_results_test_dict = {key: val for key, val in ap_results_test_dict.items() if key in benchmark_programs_list}
    
    precision_results_train_dict = {key: val for key, val in precision_results_train_dict.items() if key in benchmark_programs_list}
    precision_results_test_dict = {key: val for key, val in precision_results_test_dict.items() if key in benchmark_programs_list}
    
    recall_results_train_dict = {key: val for key, val in recall_results_train_dict.items() if key in benchmark_programs_list}
    recall_results_test_dict = {key: val for key, val in recall_results_test_dict.items() if key in benchmark_programs_list}

    colors_dict = {"Train":"skyblue","Test":"tomato"}


    plot_only_auc_ap = True
    fontsize= 20
    if plot_only_auc_ap:
        suffix = "auc_ap_ppv_mod"
        fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(20, 30))
        i = 0
        positions = []
        labels = []
        for (program_train, roc_auc_train), roc_auc_test,auc01_train,auc01_test,ap_train,ap_test,ppv_train,ppv_test in zip(roc_auc_results_train_dict.items(),
                                                                                            roc_auc_results_test_dict.values(),
                                                                                            auc01_results_train_dict.values(),
                                                                                            auc01_results_test_dict.values(),
                                                                                            ap_results_train_dict.values(),
                                                                                            ap_results_test_dict.values(),
                                                                                            ppv_results_train_dict.values(),
                                                                                            ppv_results_test_dict.values()):

            if np.isnan(roc_auc_train) or np.isnan(roc_auc_test):
                pass
            else:
                bar_train_roc_auc = ax1.barh(i, width=roc_auc_train, color="skyblue", height=0.2)
                bar_train_auc01 = ax2.barh(i, width=auc01_train, color="skyblue", height=0.2)
                bar_train_ap= ax3.barh(i,width=ap_train,color="skyblue",height=0.2)
                bar_train_ppv= ax4.barh(i,width=ppv_train,color="skyblue",height=0.2)

                bar_test_roc_auc = ax1.barh(i + 0.2, width=roc_auc_test, height=0.2, color="tomato")
                bar_test_auc01 = ax2.barh(i + 0.2, width=auc01_test, height=0.2, color="tomato")
                bar_test_ap = ax3.barh(i + 0.2, width=ap_test, height=0.2, color="tomato")
                bar_test_ppv = ax4.barh(i + 0.2, width=ppv_test, height=0.2, color="tomato")
                positions.append(i)
                i += 1
                labels.append(names_dict[program_train])
                for bar in [bar_train_roc_auc.patches, bar_test_roc_auc.patches]:
                    rect = bar[0]  # single rectangle
                    ax1.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=int(fontsize+3),weight='bold')
                for bar in [bar_train_auc01.patches, bar_test_auc01.patches]:
                    rect = bar[0]  # single rectangle
                    ax2.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=int(fontsize+3), weight='bold')
                for bar in [bar_train_ap.patches,bar_test_ap.patches]:
                    rect = bar[0] #single rectangle
                    ax3.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=int(fontsize+3),weight='bold')
                for bar in [bar_train_ppv.patches,bar_test_ppv.patches]:
                    rect = bar[0] #single rectangle
                    ax4.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=int(fontsize+3),weight='bold')


        ax1.axvline(x=0.5, color='goldenrod', linestyle='--',linewidth=4)
        ax1.set_yticks(positions, labels=labels, fontsize=fontsize + 10 ,weight='bold')
        ax1.tick_params(axis='x', labelsize=fontsize + 10 )
        transformation = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
        ax1.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
        ax1.set_title("ROC-AUC", fontsize=int(fontsize + 5))
        ax1.margins(x=0.15)

        ax2.set_yticks(positions, labels=labels, fontsize=fontsize + 10 ,weight='bold')
        ax2.tick_params(axis='x', labelsize=fontsize + 10)
        ax2.axvline(x=0.5, color='goldenrod', linestyle='--',linewidth=4)
        transformation = transforms.blended_transform_factory(ax2.get_yticklabels()[0].get_transform(), ax2.transData)
        ax2.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
        ax2.set_title("ROC-AUC-10%", fontsize=int(fontsize + 5))
        ax2.margins(x=0.15)

        ax3.axvline(x=0.5, color='goldenrod', linestyle='--',linewidth=4)
        ax3.tick_params(axis='x', labelsize=fontsize + 10)
        ax3.set_yticks(positions, labels=labels, fontsize=fontsize + 10, weight='bold')
        transformation = transforms.blended_transform_factory(ax3.get_yticklabels()[0].get_transform(), ax3.transData)
        ax3.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
        ax3.set_title("Average Precision Recall Curve", fontsize=int(fontsize + 5))
        ax3.margins(x=0.2)
        
        ax4.axvline(x=0.5, color='goldenrod', linestyle='--',linewidth=4)
        ax4.tick_params(axis='x', labelsize=fontsize + 10)
        ax4.set_yticks(positions, labels=labels, fontsize=fontsize + 10, weight='bold')
        transformation = transforms.blended_transform_factory(ax4.get_yticklabels()[0].get_transform(), ax4.transData)
        ax4.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
        ax4.set_title("Modified Positive Predictive Value", fontsize=int(fontsize + 5))
        ax4.margins(x=0.2)
        for ax in [ax1,ax2,ax2,ax3,ax4]:
            ax.get_yticklabels()[0].set_color("white")
            ax.get_yticklabels()[0].set_weight("bold")
            ax.get_yticklabels()[0].set_bbox(dict(facecolor="mediumseagreen", alpha=0.9))

        plt.subplots_adjust(left=0.17, wspace=0.55,right=0.9)
        legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in colors_dict.items()]
        fig.legend(handles=legends, prop={'size': int(fontsize + 8)}, loc='upper center', bbox_to_anchor=(0.5, 0.95))
        fig.suptitle("Benchmark metrics", fontsize=int(fontsize + 12))


    else:
        suffix = "auc_precision_recall"
        fig,[ax1,ax2,ax3,ax4] = plt.subplots(nrows=1,ncols=4,figsize=(28,16))
        i= 0
        positions = []
        labels = []
        for (program_train, roc_auc_train), roc_auc_test, precision_train, precision_test, recall_train,recall_test,pval_train,pval_test in zip(roc_auc_results_train_dict.items(),
                                                                             roc_auc_results_test_dict.values(),
                                                                             precision_results_train_dict.values(),
                                                                             precision_results_test_dict.values(),
                                                                             recall_results_train_dict.values(),
                                                                             recall_results_test_dict.values(),
                                                                             pval_results_train_dict.values(),
                                                                             pval_results_test_dict.values()):
            if np.isnan(roc_auc_train) or np.isnan(roc_auc_test):
                pass
            else:
                bar_train_auc= ax1.barh(i,width=roc_auc_train,color="skyblue",height=0.2)
                bar_train_pval= ax2.barh(i,width=pval_train,color="skyblue",height=0.2)
                bar_train_precision= ax3.barh(i,width=precision_train,color="skyblue",height=0.2)
                bar_train_recall= ax4.barh(i,width=recall_train,color="skyblue",height=0.2)

                bar_test_auc = ax1.barh(i + 0.2,width=roc_auc_test,height=0.2,color="tomato")
                bar_test_pval = ax2.barh(i + 0.2,width=pval_test,height=0.2,color="tomato")
                bar_test_precision = ax3.barh(i + 0.2,width=precision_test,height=0.2,color="tomato")
                bar_test_recall = ax4.barh(i + 0.2,width=recall_test,height=0.2,color="tomato")
                positions.append(i)
                i += 1
                labels.append(names_dict[program_train])
                for bar in [bar_train_auc.patches,bar_test_auc.patches]:
                    rect = bar[0] #single rectangle
                    ax1.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=int(fontsize - 3),weight='bold')
                for bar in [bar_train_pval.patches,bar_test_pval.patches]:
                    rect = bar[0] #single rectangle
                    ax2.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=int(fontsize - 3),weight='bold')
                for bar in [bar_train_precision.patches,bar_test_precision.patches]:
                    rect = bar[0] #single rectangle
                    ax3.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=int(fontsize - 3),weight='bold')
                for bar in [bar_train_recall.patches,bar_test_recall.patches]:
                    rect = bar[0] #single rectangle
                    ax4.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=int(fontsize - 3),weight='bold')

        ax1.set_yticks(positions,labels=labels,fontsize=fontsize,weight='bold')
        ax1.tick_params(axis='x', labelsize=fontsize)

        ax1.axvline(x=0.5, color='goldenrod', linestyle='--')
        transformation = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
        ax1.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center",transform=transformation,fontsize=fontsize)
        ax1.set_title("ROC-AUC",fontsize=int(fontsize + 5))
        ax1.margins(x=0.2)
        
        ax2.axvline(x=0.05, color='goldenrod', linestyle='--')
        ax2.set_yticks(positions,labels=labels,fontsize=15,weight='bold')
        transformation = transforms.blended_transform_factory(ax2.get_yticklabels()[0].get_transform(), ax2.transData)
        ax2.text(0.05, -0.30, "0.5", color="dimgrey", ha="right", va="center",transform=transformation,fontsize=fontsize)
        ax2.set_title("P-value",fontsize=int(fontsize + 5))
        ax2.set_xlim(0,0.4)
        ax2.margins(x=0.2)

        ax3.axvline(x=0.5, color='goldenrod', linestyle='--')
        ax3.set_yticks(positions,labels=labels,fontsize=fontsize,weight='bold')
        ax3.set_title("Precision (max)",fontsize=int(fontsize + 5))
        ax3.margins(x=0.2)

        ax4.axvline(x=0.5, color='goldenrod', linestyle='--')
        ax4.set_yticks(positions,labels=labels,fontsize=fontsize,weight='bold')
        transformation = transforms.blended_transform_factory(ax4.get_yticklabels()[0].get_transform(), ax4.transData)
        ax4.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center",transform=transformation,fontsize=fontsize)
        ax4.set_title("Recall (max)",fontsize=fontsize)
        ax4.margins(x=0.2)

        plt.subplots_adjust(wspace=0.5)

        legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in colors_dict.items()]
        fig.legend(handles=legends, prop={'size': int(fontsize + 5)}, loc='center right', bbox_to_anchor=(0.98, 0.5))
        fig.suptitle("Benchmark metrics",fontsize=int(fontsize + 5))

    #plt.show()
    plt.savefig("{}/{}/Benchmarking_{}_{}.jpg".format(script_dir,folder,title,suffix))
    plt.savefig("{}/{}/Benchmarking_{}_{}.pdf".format(script_dir, folder, title, suffix))

def plot_model_stressing_comparison1(dict_results_vegvisir,script_dir,results_folder="Benchmark/Plots",subtitle="",encoding="-blosum",keyname="viral_dataset15",ensemble=False):

    """
    :param str encoding: "-blosum-", "-onehot-" or ""
    """
    stress_mode_dict = {
                        "random{}variable-length".format(encoding):f"random_variable_length_Icore_sequences_{keyname}",
                        "random{}9mers".format(encoding):f"random_fixed_length_Icore_sequences_{keyname}",
                        "random{}7mers".format(encoding):f"random_fixed_length_Icore_sequences_{keyname}", #I keep Icore instead of Ocore_non_anchor for convenience  later
                        "shuffled{}variable-length".format(encoding):f"shuffled_variable_length_Icore_sequences_{keyname}",
                        "shuffled{}9mers".format(encoding):f"shuffled_fixed_length_Icore_sequences_{keyname}",
                        "shuffled{}7mers".format(encoding):f"shuffled_fixed_length_Icore_sequences_{keyname}",
                        "shuffled{}8mers".format(encoding):f"shuffled_fixed_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}variable-length".format(encoding): f"shuffled_labels_variable_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}9mers".format(encoding): f"shuffled_labels_fixed_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}8mers".format(encoding): f"shuffled_labels_fixed_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}7mers".format(encoding): f"shuffled_labels_fixed_length_Icore_sequences_{keyname}",
                        "raw{}variable-length".format(encoding):f"variable_length_Icore_sequences_{keyname}",
                        "raw{}7mers".format(encoding):f"fixed_length_Icore_sequences_{keyname}",
                        "raw{}8mers".format(encoding):f"fixed_length_Icore_sequences_{keyname}",
                        "raw{}9mers".format(encoding):f"fixed_length_Icore_sequences_{keyname}",
                         }

    # stress_testing_auc = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda :defaultdict())))
    # stress_testing_auc01 = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda :defaultdict())))
    # stress_testing_ppv = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: defaultdict())))
    # stress_testing_ap = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: defaultdict())))
    stress_testing_results_ICORE_dict = defaultdict(lambda :defaultdict(lambda : defaultdict()))
    stress_testing_results_ICORENONANCHOR_dict = defaultdict(lambda :defaultdict(lambda : defaultdict()))

    fontsize = 20
    metrics_keys = ["ppv", "fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1", "pval_class_0", "pval_class_1"]
    fig1, [ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    fig2, [ax3,ax4] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    fig3, [ax5,ax6] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    fig4, [ax7,ax8] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    i = 0
    positions = []
    labels = []
    tuples_idx_icore = []
    tuples_idx_icore_non_anchor = []
    for sequence_type in dict_results_vegvisir.keys():
        print("Analyzing {} datasets".format(sequence_type))
        for stress_mode in dict_results_vegvisir[sequence_type].keys():
            print(stress_mode)
            if encoding in stress_mode:
                # Highlight: Vegvisir results
                print("Analizing {}".format(stress_mode))
                vegvisir_folder = dict_results_vegvisir[sequence_type][stress_mode]
                train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(vegvisir_folder, 0))
                # valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(vegvisir_folder, 0))
                test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(vegvisir_folder, 0))

                # Highlight: extract the original sequences and the true labels to use them with NNAlign
                dataset_info = train_out["dataset_info"]
                custom_features_dicts = VegvisirUtils.build_features_dicts(dataset_info)
                aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]

                train_sequences = train_out["summary_dict"]["data_int_samples"][:, 1:].squeeze(1)
                train_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(train_sequences)
                train_sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), train_sequences_raw))

                train_labels = train_out["summary_dict"]["true_samples"]
                train_df = pd.DataFrame({"Icore": train_sequences_raw, "targets": train_labels})

                test_sequences = test_out["summary_dict"]["data_int_samples"][:, 1:].squeeze(1)
                test_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(test_sequences)
                test_sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), test_sequences_raw))
                test_labels = test_out["summary_dict"]["true_samples"]
                test_df = pd.DataFrame({"Icore": test_sequences_raw, "targets": test_labels})



                metrics_results_dict = plot_kfold_comparison_helper(metrics_keys, script_dir, folder=vegvisir_folder,overwrite=False, kfolds=5)
                metrics_results_train = metrics_results_dict["train"]
                # metrics_results_valid = metrics_results_dict["valid"]
                metrics_results_test = metrics_results_dict["test"]

                #Highlight: Train results
                vegvisir_results_auc_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train["roc_auc_class_1"]))) / 2, 2)}
                vegvisir_results_auc01_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["auc01_class_0"])) + np.mean(np.array(metrics_results_train["auc01_class_1"]))) / 2, 2)}
                vegvisir_results_ap_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["ap_class_0"])) + np.mean(np.array(metrics_results_train["ap_class_1"]))) / 2, 2)}
                vegvisir_results_ppv_train = {"Vegvisir": np.round(np.mean(np.array(metrics_results_train["ppv"])), 2)}
                if sequence_type == "Icore":
                    stress_testing_results_ICORE_dict[stress_mode]["Vegvisir"]["train"] = {"ROC-AUC" : vegvisir_results_auc_train["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_train["Vegvisir"],
                                                                        "PPV*":vegvisir_results_ppv_train["Vegvisir"],
                                                                        "AP":vegvisir_results_ap_train["Vegvisir"]}
                    tuples_idx_icore.append((stress_mode,"Vegvisir", "train"))
                else:
                    stress_testing_results_ICORENONANCHOR_dict[stress_mode]["Vegvisir"]["train"] = {"ROC-AUC" : vegvisir_results_auc_train["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_train["Vegvisir"],
                                                                        "PPV*":vegvisir_results_ppv_train["Vegvisir"],
                                                                        "AP":vegvisir_results_ap_train["Vegvisir"]}
                    tuples_idx_icore_non_anchor.append((stress_mode,"Vegvisir", "train"))
                #Highlight: TEST
                vegvisir_results_auc_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test["roc_auc_class_1"]))) / 2, 2)}
                vegvisir_results_auc01_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["auc01_class_0"])) + np.mean(np.array(metrics_results_test["auc01_class_1"]))) / 2, 2)}
                vegvisir_results_ap_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["ap_class_0"])) + np.mean(np.array(metrics_results_test["ap_class_1"]))) / 2, 2)}
                vegvisir_results_ppv_test = {"Vegvisir": np.round(np.mean(np.array(metrics_results_test["ppv"])), 2)}
                if sequence_type == "Icore":
                    stress_testing_results_ICORE_dict[stress_mode]["Vegvisir"]["test"] = {"ROC-AUC": vegvisir_results_auc_test["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_test["Vegvisir"],
                                                                        "PPV*": vegvisir_results_ppv_test["Vegvisir"],
                                                                        "AP": vegvisir_results_ap_test["Vegvisir"]}
                    tuples_idx_icore.append((stress_mode,"Vegvisir", "test"))
                else:
                    stress_testing_results_ICORENONANCHOR_dict[stress_mode]["Vegvisir"]["test"] = {"ROC-AUC": vegvisir_results_auc_test["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_test["Vegvisir"],
                                                                        "PPV*": vegvisir_results_ppv_test["Vegvisir"],
                                                                        "AP": vegvisir_results_ap_test["Vegvisir"]}
                    tuples_idx_icore_non_anchor.append((stress_mode,"Vegvisir", "test"))


                stress_dataset = stress_mode_dict[stress_mode].replace("Icore",sequence_type) #This is weird, but i do not feel like making it better
                if keyname == "viral_dataset9":
                    folders_list = glob("Benchmark/Other_Programs/NNAlign_results_07_01_2024/{}/{}/*/".format(sequence_type, stress_dataset), recursive=True)
                    folder_name = Path(folders_list[0]).parts[-1]
                    subfolders_list = glob(
                        "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Other_Programs/NNAlign_results_07_01_2024/{}/{}/{}/*".format(
                            sequence_type, stress_dataset, folder_name), recursive=True)

                else:
                    folders_list = glob("Benchmark/Other_Programs/NNAlign_results_19_01_2024/{}/{}/*/".format(sequence_type,stress_dataset),recursive=True)
                    folder_name = Path(folders_list[0]).parts[-1]
                    subfolders_list = glob("Benchmark/Other_Programs/NNAlign_results_19_01_2024/{}/{}/{}/*".format(sequence_type,stress_dataset,folder_name),recursive=True)


                nnalign_results_path_train_full = list(filter(lambda x: "lg9.sorted.pred" in x, subfolders_list))[0]
                nnalign_results_path_test_full = list(filter(lambda x: "lg9" in x and "evalset" in x, subfolders_list))[0]


                (nnalign_results_train_auc_dict, nnalign_results_train_auc01_dict, nnalign_results_train_ppv_dict, nnalign_results_train_ap_dict,
                 nnalign_results_train_pval_dict, nnalign_results_train_max_precision_dict,nnalign_results_train_max_recall_dict) = process_nnalign(nnalign_results_path_train_full, train_df,stress_dataset,mode="train")
                if sequence_type == "Icore":
                    stress_testing_results_ICORE_dict[stress_mode]["NNAlign"]["train"] = {"ROC-AUC" : nnalign_results_train_auc_dict["NNAlign2.1"],
                                                                       "AUC01":nnalign_results_train_auc01_dict["NNAlign2.1"],
                                                                       "PPV*": nnalign_results_train_ppv_dict["NNAlign2.1"],
                                                                       "AP":nnalign_results_train_ap_dict["NNAlign2.1"]}

                    tuples_idx_icore.append((stress_mode,"NNAlign", "train"))
                else:
                    stress_testing_results_ICORENONANCHOR_dict[stress_mode]["NNAlign"]["train"] = {
                        "ROC-AUC": nnalign_results_train_auc_dict["NNAlign2.1"],
                        "AUC01": nnalign_results_train_auc01_dict["NNAlign2.1"],
                        "PPV*": nnalign_results_train_ppv_dict["NNAlign2.1"],
                        "AP": nnalign_results_train_ap_dict["NNAlign2.1"]}

                    tuples_idx_icore_non_anchor.append((stress_mode, "NNAlign", "train"))
                (nnalign_results_test_auc_dict, nnalign_results_test_auc01_dict, nnalign_results_test_ppv_dict, nnalign_results_test_ap_dict,
                 nnalign_results_test_pval_dict, nnalign_results_test_max_precision_dict,nnalign_results_test_max_recall_dict)  = process_nnalign(nnalign_results_path_test_full, test_df,stress_dataset,mode="test")
                if sequence_type == "Icore":
                    stress_testing_results_ICORE_dict[stress_mode]["NNAlign"]["test"] = {
                                                                    "ROC-AUC": nnalign_results_test_auc_dict["NNAlign2.1"],
                                                                    "AUC01": nnalign_results_test_auc01_dict["NNAlign2.1"],
                                                                    "PPV*": nnalign_results_test_ppv_dict["NNAlign2.1"],
                                                                    "AP": nnalign_results_test_ap_dict["NNAlign2.1"]}
                    tuples_idx_icore.append((stress_mode,"NNAlign", "test"))
                else:
                    stress_testing_results_ICORENONANCHOR_dict[stress_mode]["NNAlign"]["test"] = {
                                                                    "ROC-AUC": nnalign_results_test_auc_dict["NNAlign2.1"],
                                                                    "AUC01": nnalign_results_test_auc01_dict["NNAlign2.1"],
                                                                    "PPV*": nnalign_results_test_ppv_dict["NNAlign2.1"],
                                                                    "AP": nnalign_results_test_ap_dict["NNAlign2.1"]}
                    tuples_idx_icore_non_anchor.append((stress_mode,"NNAlign", "test"))



                #Highlight: ROC-AUC
                bar_train_auc1 = ax1.barh(i, width=nnalign_results_train_auc_dict["NNAlign2.1"], color="plum", height=0.2)
                bar_train_auc2 = ax1.barh(i + 0.4, width=vegvisir_results_auc_train["Vegvisir"], color="darkturquoise", height=0.2)

                bar_test_auc1 = ax2.barh(i, width=nnalign_results_test_auc_dict["NNAlign2.1"], height=0.2, color="plum")
                bar_test_auc2 = ax2.barh(i + 0.4, width=vegvisir_results_auc_test["Vegvisir"], height=0.2, color="darkturquoise")

                #Highlight: AUC01
                bar_train_auc011 = ax3.barh(i, width=nnalign_results_train_auc01_dict["NNAlign2.1"], color="plum",height=0.2)
                bar_train_auc012 = ax3.barh(i + 0.4, width=vegvisir_results_auc01_train["Vegvisir"], color="darkturquoise",height=0.2)

                bar_test_auc011 = ax4.barh(i, width=nnalign_results_test_auc01_dict["NNAlign2.1"], height=0.2, color="plum")
                bar_test_auc012 = ax4.barh(i + 0.4, width=vegvisir_results_auc01_test["Vegvisir"], height=0.2,color="darkturquoise")

                # #Highlight: PPV
                bar_train_ppv1 = ax5.barh(i, width=nnalign_results_train_ppv_dict["NNAlign2.1"], color="plum",height=0.2)
                bar_train_ppv2 = ax5.barh(i + 0.4, width=vegvisir_results_ppv_train["Vegvisir"],color="darkturquoise", height=0.2)

                bar_test_ppv1 = ax6.barh(i, width=nnalign_results_test_ppv_dict["NNAlign2.1"], height=0.2, color="plum")
                bar_test_ppv2 = ax6.barh(i + 0.4, width=vegvisir_results_ppv_test["Vegvisir"], height=0.2,color="darkturquoise")
                #
                # # Highlight: AP
                bar_train_ap1 = ax7.barh(i, width=nnalign_results_train_ap_dict["NNAlign2.1"], color="plum",height=0.2)
                bar_train_ap2 = ax7.barh(i + 0.4, width=vegvisir_results_ap_train["Vegvisir"], color="darkturquoise",height=0.2)

                bar_test_ap1 = ax8.barh(i, width=nnalign_results_test_ap_dict["NNAlign2.1"], height=0.2, color="plum")
                bar_test_ap2 = ax8.barh(i + 0.4, width=vegvisir_results_ap_test["Vegvisir"], height=0.2,color="darkturquoise")

                positions.append(i)
                labels.append("{}\n{}".format(sequence_type.replace("_","-"),stress_mode.replace("{}-".format(encoding),"")))
                i += 1
                for bar in [bar_train_auc1.patches,bar_train_auc2.patches]:
                    rect = bar[0]  # single rectangle
                    ax1.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                for bar in [bar_test_auc1.patches,bar_test_auc2.patches]:
                    rect = bar[0]  # single rectangle
                    ax2.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')

                for bar in [bar_train_auc011.patches, bar_train_auc012.patches]:
                    rect = bar[0]  # single rectangle
                    ax3.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                for bar in [bar_test_auc011.patches, bar_test_auc012.patches]:
                    rect = bar[0]  # single rectangle
                    ax4.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                for bar in [bar_train_ppv1.patches, bar_train_ppv2.patches]:
                    rect = bar[0]  # single rectangle
                    ax5.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                for bar in [bar_test_ppv1.patches, bar_test_ppv2.patches]:
                    rect = bar[0]  # single rectangle
                    ax6.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                for bar in [bar_train_ap1.patches, bar_train_ap2.patches]:
                    rect = bar[0]  # single rectangle
                    ax7.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                for bar in [bar_test_ap1.patches, bar_test_ap2.patches]:
                    rect = bar[0]  # single rectangle
                    ax8.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize, weight='bold')




    #Highlight: Saving dataframe
    def process_dict(metric_dict,tuples_idx, title="", subtitle=""):
        """NOTES: https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex"""
        metric_dict = {key.replace(r"\textbf{", "").replace("}", ""): val for key, val in metric_dict.items()}
        df = pd.DataFrame.from_dict(
            {(i, j, k): metric_dict[i][j][k] for i in metric_dict.keys() for j in metric_dict[i].keys() for k in
             metric_dict[i][j]}, orient='index')


        colors =  {"random{}variable-length".format(encoding):matplotlib.colors.to_rgba('yellow'),
                        "random{}9mers".format(encoding):matplotlib.colors.to_rgba('yellow'),
                        "random{}7mers".format(encoding):matplotlib.colors.to_rgba('yellow'),
                        "shuffled{}variable-length".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled{}9mers".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled{}7mers".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled{}8mers".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled-labels{}variable-length".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "shuffled-labels{}9mers".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "shuffled-labels{}8mers".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "shuffled-labels{}7mers".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "raw{}variable-length".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                        "raw{}7mers".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                        "raw{}8mers".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                        "raw{}9mers".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                         }

        c = {k: matplotlib.colors.rgb2hex(v) for k, v in colors.items()}
        idx = df.index.get_level_values(0)
        css = [{'selector': f'.row{i}.level0', 'props': [('background-color', c[v])]} for i, v in enumerate(idx)]
        new_index = pd.MultiIndex.from_tuples(tuples_idx)
        df = df.reindex(index=new_index)  # guarantees the same order as the dictionary

        df_styled = df.style.format(na_rep="-", escape="latex", precision=2).background_gradient(axis=None,
                                                                                                 cmap="YlOrBr").set_table_styles(css)  # TODO: Switch to escape="latex-math" when pandas 2.1 arrives

        dfi.export(df_styled, '{}/{}/{}_DATAFRAME_{}.png'.format(script_dir, results_folder, title, subtitle),
                   max_cols=-1,
                   max_rows=-1)

        return df

    def latex_with_lines(df, *args, **kwargs):
        kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels + ['r'] * df.shape[1] + [''])
        kwargs["na_rep"] = ""
        kwargs["float_format"] = "%.2f"
        res = df.to_latex(*args, **kwargs)
        res = res.replace("\multirow[t]", "\multirow")
        res = res.replace("\multirow{4}{*}{raw-blosum-variable-length}",r"\multirow{4}{*}{\colorbox{green!20}{\makebox[4cm]{\textbf{raw-blosum-variable-length}}}}")
        res = res.replace(r"\begin{tabular}{|l|l|l|r|r|r|r|}",r"\begin{tabular}{|p{4.5cm}|l|l|W|W|W|W|}")
        res = res.replace("\multirow{2}{*}{Vegvisir}","\multirow{2}{*}{\colorbox{green!20}{\makebox[1cm]{Vegvisir}}}")
        res = res.replace(r"\multirow{4}{*}{",r"\multirow{4}{*}{\large ")
        res = res.replace(r"\bottomrule","")
        return res

    icore_df_latex = process_dict(stress_testing_results_ICORE_dict,tuples_idx_icore,"STRESS_testing_ICORE",subtitle)
    icorenonnachor_df_latex = process_dict(stress_testing_results_ICORENONANCHOR_dict,tuples_idx_icore_non_anchor,"STRESS_testing_ICORENONANCHOR",subtitle)



    # df_latex.style.format(na_rep="0",precision=2).to_latex('{}/{}/Pearson_coefficients_LATEX_{}.tex'.format(script_dir, results_folder,subtitle),hrules=True)
    icore_df_latex = latex_with_lines(icore_df_latex)
    icore_latex_file = open('{}/{}/Stress_testing_ICORE_LATEX_{}.tex'.format(script_dir, results_folder, subtitle), "w+")
    icore_latex_file.write(icore_df_latex)

    icorenonanchor_df_latex = latex_with_lines(icorenonnachor_df_latex)
    icorenonanchor_latex_file = open('{}/{}/Stress_testing_ICORENONANCHOR_LATEX_{}.tex'.format(script_dir, results_folder, subtitle), "w+")
    icorenonanchor_latex_file.write(icorenonanchor_df_latex)

    #TODO: plotting the axes in a in for loop does not seem to work
    #Highlight: ROC-AUC
    ax1.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax1.tick_params(axis='x', labelsize=fontsize)

    ax1.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
    ax1.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax1.set_title("Train", fontsize=fontsize)
    ax1.margins(x=0.15)

    ax2.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax2.get_yticklabels()[0].get_transform(), ax2.transData)
    ax2.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.set_yticks([])
    ax2.set_title("Test", fontsize=fontsize)
    ax2.margins(x=0.15)

    fig1.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in {"Vegvisir":"darkturquoise","NNAlign2.1":"plum"}.items()]
    fig1.legend(handles=legends, prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig1.suptitle("Stress testing: ROC-AUC", fontsize=int(fontsize + 5))
    fig1.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"ROC_AUC",subtitle),dpi=600)
    plt.close(fig1)
    fig1.clf()

    #Highlight: AUC01
    ax3.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax3.get_yticklabels()[0].get_transform(), ax3.transData)
    ax3.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax3.set_title("Train", fontsize=fontsize)
    ax3.margins(x=0.15)

    ax4.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax4.get_yticklabels()[0].get_transform(), ax4.transData)
    ax4.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.set_yticks([])
    ax4.set_title("Test", fontsize=fontsize)
    ax4.margins(x=0.15)


    fig2.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in {"Vegvisir":"darkturquoise","NNAlign2.1":"plum"}.items()]
    fig2.legend(handles=legends, prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig2.suptitle("Stress testing: AUC01", fontsize=int(fontsize + 5))
    fig2.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"AUC01",subtitle),dpi=600)
    plt.close(fig2)
    fig2.clf()
    #
    #Highlight: PPV
    ax5.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax5.tick_params(axis='x', labelsize=fontsize)
    ax5.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax5.get_yticklabels()[0].get_transform(), ax5.transData)
    ax5.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax5.set_title("Train", fontsize=fontsize)
    ax5.margins(x=0.15)

    ax6.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax6.get_yticklabels()[0].get_transform(), ax6.transData)
    ax6.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax6.tick_params(axis='x', labelsize=fontsize)
    ax6.set_yticks([])
    ax6.set_title("Test", fontsize=fontsize)
    ax6.margins(x=0.15)


    fig3.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in {"Vegvisir":"darkturquoise","NNAlign2.1":"plum"}.items()]
    fig3.legend(handles=legends, prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig3.suptitle("Stress testing: PPV*", fontsize=int(fontsize + 5))
    fig3.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"PPV",subtitle),dpi=600)
    plt.close(fig3)
    fig3.clf()

    #Highlight: AP
    ax7.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax7.tick_params(axis='x', labelsize=fontsize)
    ax7.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax7.get_yticklabels()[0].get_transform(), ax7.transData)
    ax7.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax7.set_title("Train", fontsize=fontsize)
    ax7.margins(x=0.15)

    ax8.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax8.get_yticklabels()[0].get_transform(), ax8.transData)
    ax8.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax8.tick_params(axis='x', labelsize=fontsize)
    ax8.set_yticks([])
    ax8.set_title("Test", fontsize=fontsize)
    ax8.margins(x=0.15)

    fig4.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in {"Vegvisir":"darkturquoise","NNAlign2.1":"plum"}.items()]
    fig4.legend(handles=legends, prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig4.suptitle("Stress testing: Average Precision", fontsize=int(fontsize + 5))
    fig4.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"AP",subtitle),dpi=600)
    plt.close(fig4)
    fig4.clf()

def plot_model_stressing_comparison2(dict_results_vegvisir,script_dir,results_folder="Benchmark/Plots",subtitle="",encoding="-blosum",keyname="viral_dataset15",ensemble=False):

    """
    :param str encoding: "-blosum-", "-onehot-" or ""
    """
    stress_mode_dict = {
                        "random{}variable-length".format(encoding):f"random_variable_length_Icore_sequences_{keyname}",
                        "random{}9mers".format(encoding):f"random_fixed_length_Icore_sequences_{keyname}",
                        "random{}7mers".format(encoding):f"random_fixed_length_Icore_sequences_{keyname}", #I keep Icore instead of Ocore_non_anchor for convenience  later
                        "shuffled{}variable-length".format(encoding):f"shuffled_variable_length_Icore_sequences_{keyname}",
                        "shuffled{}9mers".format(encoding):f"shuffled_fixed_length_Icore_sequences_{keyname}",
                        "shuffled{}7mers".format(encoding):f"shuffled_fixed_length_Icore_sequences_{keyname}",
                        "shuffled{}8mers".format(encoding):f"shuffled_fixed_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}variable-length".format(encoding): f"shuffled_labels_variable_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}9mers".format(encoding): f"shuffled_labels_fixed_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}8mers".format(encoding): f"shuffled_labels_fixed_length_Icore_sequences_{keyname}",
                        "shuffled-labels{}7mers".format(encoding): f"shuffled_labels_fixed_length_Icore_sequences_{keyname}",
                        "raw{}variable-length".format(encoding):f"variable_length_Icore_sequences_{keyname}",
                        "raw{}7mers".format(encoding):f"fixed_length_Icore_sequences_{keyname}",
                        "raw{}8mers".format(encoding):f"fixed_length_Icore_sequences_{keyname}",
                        "raw{}9mers".format(encoding):f"fixed_length_Icore_sequences_{keyname}",
                         }

    # stress_testing_auc = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda :defaultdict())))
    # stress_testing_auc01 = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda :defaultdict())))
    # stress_testing_ppv = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: defaultdict())))
    # stress_testing_ap = defaultdict(lambda :defaultdict(lambda : defaultdict(lambda: defaultdict())))
    stress_testing_results_ICORE_dict = defaultdict(lambda :defaultdict())
    stress_testing_results_ICORENONANCHOR_dict = defaultdict(lambda :defaultdict())

    fontsize = 20
    metrics_keys = ["ppv", "fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1", "pval_class_0", "pval_class_1"]
    fig1, [ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    fig2, [ax3,ax4] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    fig3, [ax5,ax6] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    fig4, [ax7,ax8] = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
    i = 0
    positions = []
    labels = []
    tuples_idx_icore = []
    tuples_idx_icore_non_anchor = []
    filled=False
    for sequence_type in dict_results_vegvisir.keys():
        print("Analyzing {} datasets".format(sequence_type))
        for stress_mode in dict_results_vegvisir[sequence_type].keys():
            if encoding in stress_mode:
                # Highlight: Vegvisir results
                print("Analizing {}".format(stress_mode))
                vegvisir_folder = dict_results_vegvisir[sequence_type][stress_mode]
                train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(vegvisir_folder, 0))
                # valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(vegvisir_folder, 0))
                test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(vegvisir_folder, 0))

                # Highlight: extract the original sequences and the true labels to use them with NNAlign
                dataset_info = train_out["dataset_info"]
                custom_features_dicts = VegvisirUtils.build_features_dicts(dataset_info)
                aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]

                train_sequences = train_out["summary_dict"]["data_int_samples"][:, 1:].squeeze(1)
                train_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(train_sequences)
                train_sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), train_sequences_raw))

                train_labels = train_out["summary_dict"]["true_samples"]
                train_df = pd.DataFrame({"Icore": train_sequences_raw, "targets": train_labels})

                test_sequences = test_out["summary_dict"]["data_int_samples"][:, 1:].squeeze(1)
                test_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(test_sequences)
                test_sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), test_sequences_raw))
                test_labels = test_out["summary_dict"]["true_samples"]
                test_df = pd.DataFrame({"Icore": test_sequences_raw, "targets": test_labels})


                if ensemble:
                    metrics_results_dict = plot_benchmark_vegvisir_helper3(vegvisir_folder,None,kfolds=5,aggregated_not_overlap=False)

                else:
                    metrics_results_dict = plot_kfold_comparison_helper(metrics_keys, script_dir, folder=vegvisir_folder,overwrite=False, kfolds=5)


                metrics_results_train = metrics_results_dict["train"]
                # metrics_results_valid = metrics_results_dict["valid"]
                metrics_results_test = metrics_results_dict["test"]

                #Highlight: Train results
                vegvisir_results_auc_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train["roc_auc_class_1"]))) / 2, 2)}
                vegvisir_results_auc01_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["auc01_class_0"])) + np.mean(np.array(metrics_results_train["auc01_class_1"]))) / 2, 2)}
                vegvisir_results_ap_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["ap_class_0"])) + np.mean(np.array(metrics_results_train["ap_class_1"]))) / 2, 2)}
                if ensemble:
                    vegvisir_results_ppv_train = {"Vegvisir": np.round((np.mean(np.array(metrics_results_train["ppv_mod_class_0"])) + np.mean(np.array(metrics_results_train["ppv_mod_class_1"]))) / 2, 2)}
                else:
                    vegvisir_results_ppv_train = {"Vegvisir": np.round(np.mean(np.array(metrics_results_train["ppv_class_0"])), 2)}

                if sequence_type == "Icore":
                    stress_testing_results_ICORE_dict[stress_mode]["train"] = {"ROC-AUC" : vegvisir_results_auc_train["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_train["Vegvisir"],
                                                                        "PPV*":vegvisir_results_ppv_train["Vegvisir"],
                                                                        "AP":vegvisir_results_ap_train["Vegvisir"]}
                    tuples_idx_icore.append((stress_mode, "train"))
                else:
                    stress_testing_results_ICORENONANCHOR_dict[stress_mode]["train"] = {"ROC-AUC" : vegvisir_results_auc_train["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_train["Vegvisir"],
                                                                        "PPV*":vegvisir_results_ppv_train["Vegvisir"],
                                                                        "AP":vegvisir_results_ap_train["Vegvisir"]}
                    tuples_idx_icore_non_anchor.append((stress_mode, "train"))
                #Highlight: TEST
                vegvisir_results_auc_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test["roc_auc_class_1"]))) / 2, 2)}
                vegvisir_results_auc01_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["auc01_class_0"])) + np.mean(np.array(metrics_results_test["auc01_class_1"]))) / 2, 2)}
                vegvisir_results_ap_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["ap_class_0"])) + np.mean(np.array(metrics_results_test["ap_class_1"]))) / 2, 2)}
                if ensemble:
                    vegvisir_results_ppv_test = {"Vegvisir": np.round((np.mean(np.array(metrics_results_test["ppv_mod_class_0"])) + np.mean(np.array(metrics_results_test["ppv_mod_class_1"]))) / 2, 2)}
                else:
                    vegvisir_results_ppv_test = {"Vegvisir": np.round(np.mean(np.array(metrics_results_test["ppv"])), 2)}

                if sequence_type == "Icore":
                    stress_testing_results_ICORE_dict[stress_mode]["test"] = {"ROC-AUC": vegvisir_results_auc_test["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_test["Vegvisir"],
                                                                        "PPV*": vegvisir_results_ppv_test["Vegvisir"],
                                                                        "AP": vegvisir_results_ap_test["Vegvisir"]}
                    tuples_idx_icore.append((stress_mode, "test"))

                else:
                    stress_testing_results_ICORENONANCHOR_dict[stress_mode]["test"] = {"ROC-AUC": vegvisir_results_auc_test["Vegvisir"],
                                                                        "AUC01": vegvisir_results_auc01_test["Vegvisir"],
                                                                        "PPV*": vegvisir_results_ppv_test["Vegvisir"],
                                                                        "AP": vegvisir_results_ap_test["Vegvisir"]}
                    tuples_idx_icore_non_anchor.append((stress_mode, "test"))


                stress_dataset = stress_mode_dict[stress_mode].replace("Icore",sequence_type) #This is weird, but i do not feel like making it better

                #Highlight: ROC-AUC
                bar_train_auc2 = ax1.barh(i, width=vegvisir_results_auc_train["Vegvisir"], color="skyblue", height=0.2)

                bar_test_auc2 = ax2.barh(i, width=vegvisir_results_auc_test["Vegvisir"], height=0.2, color="tomato")

                #Highlight: AUC01
                bar_train_auc012 = ax3.barh(i, width=vegvisir_results_auc01_train["Vegvisir"], color="skyblue",height=0.2)

                bar_test_auc012 = ax4.barh(i, width=vegvisir_results_auc01_test["Vegvisir"], height=0.2,color="tomato")

                #Highlight: PPV
                bar_train_ppv2 = ax5.barh(i, width=vegvisir_results_ppv_train["Vegvisir"],color="skyblue", height=0.2)

                bar_test_ppv2 = ax6.barh(i, width=vegvisir_results_ppv_test["Vegvisir"], height=0.2,color="tomato")

                #Highlight: AP
                bar_train_ap2 = ax7.barh(i, width=vegvisir_results_ap_train["Vegvisir"], color="skyblue",height=0.2)

                bar_test_ap2 = ax8.barh(i, width=vegvisir_results_ap_test["Vegvisir"], height=0.2,color="tomato")
                if sequence_type == "Icore_non_anchor" and not filled:
                    position_line = i -0.5
                    ax1.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax2.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax3.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax4.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax5.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax6.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax7.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    ax8.hlines(y=position_line,xmin=0,xmax=1, color='black', linestyle='-', linewidth=4)
                    filled = True
                positions.append(i)
                labels.append("{}\n{}\n".format(sequence_type.replace("_","-").upper(),stress_mode.replace("{}-".format(encoding),"").upper()))
                i += 1

                for bar in [bar_train_auc2.patches]:
                    rect = bar[0]  # single rectangle
                    ax1.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_test_auc2.patches]:
                    rect = bar[0]  # single rectangle
                    ax2.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_train_auc012.patches]:
                    rect = bar[0]  # single rectangle
                    ax3.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_test_auc012.patches]:
                    rect = bar[0]  # single rectangle
                    ax4.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_train_ppv2.patches]:
                    rect = bar[0]  # single rectangle
                    ax5.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_test_ppv2.patches]:
                    rect = bar[0]  # single rectangle
                    ax6.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_train_ap2.patches]:
                    rect = bar[0]  # single rectangle
                    ax7.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')
                for bar in [bar_test_ap2.patches]:
                    rect = bar[0]  # single rectangle
                    ax8.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=fontsize + 5, weight='bold')




    #Highlight: Saving dataframe
    def process_dict(metric_dict,tuples_idx, title="", subtitle=""):
        """NOTES: https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex"""
        metric_dict = {key.replace(r"\textbf{", "").replace("}", ""): val for key, val in metric_dict.items()}
        #df = pd.DataFrame.from_dict({(i, j, k): metric_dict[i][j][k] for i in metric_dict.keys() for j in metric_dict[i].keys() for k in metric_dict[i][j]}, orient='index')
        df = pd.DataFrame.from_dict({(i, j): metric_dict[i][j] for i in metric_dict.keys() for j in metric_dict[i] }, orient='index')

        colors =  {"random{}variable-length".format(encoding):matplotlib.colors.to_rgba('yellow'),
                        "random{}9mers".format(encoding):matplotlib.colors.to_rgba('yellow'),
                        "random{}7mers".format(encoding):matplotlib.colors.to_rgba('yellow'),
                        "shuffled{}variable-length".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled{}9mers".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled{}7mers".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled{}8mers".format(encoding):matplotlib.colors.to_rgba('orange'),
                        "shuffled-labels{}variable-length".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "shuffled-labels{}9mers".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "shuffled-labels{}8mers".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "shuffled-labels{}7mers".format(encoding): matplotlib.colors.to_rgba('lime'),
                        "raw{}variable-length".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                        "raw{}7mers".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                        "raw{}8mers".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                        "raw{}9mers".format(encoding):matplotlib.colors.to_rgba('paleturquoise'),
                         }

        c = {k: matplotlib.colors.rgb2hex(v) for k, v in colors.items()}
        idx = df.index.get_level_values(0)
        css = [{'selector': f'.row{i}.level0', 'props': [('background-color', c[v])]} for i, v in enumerate(idx)]
        new_index = pd.MultiIndex.from_tuples(tuples_idx)
        df = df.reindex(index=new_index)  # guarantees the same order as the dictionary

        df_styled = df.style.format(na_rep="-", escape="latex", precision=2).background_gradient(axis=None,cmap="YlOrBr").set_table_styles(css)  # TODO: Switch to escape="latex-math" when pandas 2.1 arrives

        # html = df_styled.to_html()
        # imgkit.from_string(html,'{}/{}/{}_DATAFRAME_{}.jpg'.format(script_dir, results_folder, title, subtitle))
        # imgkit.from_string(html,'{}/{}/{}_DATAFRAME_{}.pdf'.format(script_dir, results_folder, title, subtitle))


        #dfi.export(df_styled, '{}/{}/{}_DATAFRAME_{}.png'.format(script_dir, results_folder, title, subtitle),max_cols=-1,max_rows=-1)

        return df

    def latex_with_lines(df, *args, **kwargs):
        kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels + ['r'] * df.shape[1] + [''])
        kwargs["na_rep"] = ""
        kwargs["float_format"] = "%.2f"
        res = df.to_latex(*args, **kwargs)
        res = res.replace("\multirow[t]", "\multirow")
        res = res.replace("\multirow{4}{*}{raw-blosum-variable-length}",r"\multirow{4}{*}{\colorbox{green!20}{\makebox[4cm]{\textbf{raw-blosum-variable-length}}}}")
        res = res.replace(r"\begin{tabular}{|l|l|l|r|r|r|r|}",r"\begin{tabular}{|p{4.5cm}|l|l|W|W|W|W|}")
        res = res.replace("\multirow{2}{*}{Vegvisir}","\multirow{2}{*}{\colorbox{green!20}{\makebox[1cm]{Vegvisir}}}")
        res = res.replace(r"\multirow{4}{*}{",r"\multirow{4}{*}{\large ")
        res = res.replace(r"\bottomrule","")
        return res

    icore_df_latex = process_dict(stress_testing_results_ICORE_dict,tuples_idx_icore,"STRESS_testing_ICORE",subtitle)
    icorenonnachor_df_latex = process_dict(stress_testing_results_ICORENONANCHOR_dict,tuples_idx_icore_non_anchor,"STRESS_testing_ICORENONANCHOR",subtitle)



    # df_latex.style.format(na_rep="0",precision=2).to_latex('{}/{}/Pearson_coefficients_LATEX_{}.tex'.format(script_dir, results_folder,subtitle),hrules=True)
    icore_df_latex = latex_with_lines(icore_df_latex)
    icore_latex_file = open('{}/{}/Stress_testing_ICORE_LATEX_{}.tex'.format(script_dir, results_folder, subtitle), "w+")
    icore_latex_file.write(icore_df_latex)

    icorenonanchor_df_latex = latex_with_lines(icorenonnachor_df_latex)
    icorenonanchor_latex_file = open('{}/{}/Stress_testing_ICORENONANCHOR_LATEX_{}.tex'.format(script_dir, results_folder, subtitle), "w+")
    icorenonanchor_latex_file.write(icorenonanchor_df_latex)

    #TODO: plotting the axes in a in for loop does not seem to work
    #Highlight: ROC-AUC
    ax1.set_yticks(positions, labels=labels, fontsize=fontsize,  weight='bold')
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
    ax1.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax1.set_title("Train", fontsize=fontsize+5)
    ax1.margins(x=0.15)
    ax1.set_xlim(0,1)

    ax2.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax2.get_yticklabels()[0].get_transform(), ax2.transData)
    ax2.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.set_yticks([])
    ax2.set_title("Test", fontsize=fontsize+5)
    ax2.margins(x=0.15)
    ax2.set_xlim(0, 1)

    fig1.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    legends = [mpatches.Patch(color=color, label='{}'.format(label.upper())) for label, color in {"Train":"skyblue","Test":"tomato"}.items()]
    fig1.legend(handles=legends, prop={'size': 25}, loc='upper center', bbox_to_anchor=(0.55, 0.95),ncol=2)
    fig1.suptitle("Stress testing: ROC-AUC", fontsize=int(fontsize + 10))
    #fig1.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"ROC_AUC",subtitle),dpi=600)
    fig1.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.pdf".format(script_dir,results_folder,"ROC_AUC",subtitle))
    plt.close(fig1)
    fig1.clf()

    #Highlight: AUC01
    ax3.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax3.get_yticklabels()[0].get_transform(), ax3.transData)
    ax3.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax3.set_title("Train", fontsize=fontsize+5)
    ax3.margins(x=0.15)
    ax3.set_xlim(0,1)

    ax4.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax4.get_yticklabels()[0].get_transform(), ax4.transData)
    ax4.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.set_yticks([])
    ax4.set_title("Test", fontsize=fontsize+5)
    ax4.margins(x=0.15)
    ax4.set_xlim(0, 1)


    fig2.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    fig2.legend(handles=legends, prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.55, 0.95),ncol=2)
    fig2.suptitle("Stress testing: ROC-AUC-10%", fontsize=int(fontsize + 10))
    #fig2.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"AUC01",subtitle),dpi=600)
    fig2.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.pdf".format(script_dir,results_folder,"AUC01",subtitle))
    plt.close(fig2)
    fig2.clf()
    #
    #Highlight: PPV
    ax5.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax5.tick_params(axis='x', labelsize=fontsize)
    ax5.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax5.get_yticklabels()[0].get_transform(), ax5.transData)
    ax5.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax5.set_title("Train", fontsize=fontsize+5)
    ax5.margins(x=0.15)
    ax5.set_xlim(0,1)

    ax6.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax6.get_yticklabels()[0].get_transform(), ax6.transData)
    ax6.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax6.tick_params(axis='x', labelsize=fontsize)
    ax6.set_yticks([])
    ax6.set_title("Test", fontsize=fontsize+5)
    ax6.margins(x=0.15)
    ax6.set_xlim(0, 1)


    fig3.subplots_adjust(left=0.25, wspace=0.2, right=0.84)
    fig3.legend(handles=legends, prop={'size': fontsize}, loc='upper center', bbox_to_anchor=(0.55, 0.95),ncol=2)
    fig3.suptitle("Stress testing: Modified Positive Predictive Value", fontsize=int(fontsize + 10))
    #fig3.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"PPV",subtitle),dpi=600)
    fig3.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.pdf".format(script_dir,results_folder,"PPV",subtitle))
    plt.close(fig3)
    fig3.clf()

    #Highlight: AP
    ax7.set_yticks(positions, labels=labels, fontsize=fontsize, weight='bold')
    ax7.tick_params(axis='x', labelsize=fontsize)
    ax7.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax7.get_yticklabels()[0].get_transform(), ax7.transData)
    ax7.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax7.set_title("Train", fontsize=fontsize+5)
    ax7.margins(x=0.15)
    ax7.set_xlim(0, 1)

    ax8.axvline(x=0.5, color='goldenrod', linestyle='--', linewidth=4)
    transformation = transforms.blended_transform_factory(ax8.get_yticklabels()[0].get_transform(), ax8.transData)
    ax8.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=fontsize)
    ax8.tick_params(axis='x', labelsize=fontsize)
    ax8.set_yticks([])
    ax8.set_title("Test", fontsize=fontsize+5)
    ax8.margins(x=0.15)
    ax8.set_xlim(0, 1)

    fig4.subplots_adjust(left=0.25, wspace=0.05, right=0.84)
    fig4.legend(handles=legends, prop={'size': 25}, loc='upper center', bbox_to_anchor=(0.55, 0.95),ncol=2)
    fig4.suptitle("Stress testing: Average Precision Recall Curve", fontsize=int(fontsize + 10))
    #fig4.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.jpg".format(script_dir,results_folder,"AP",subtitle),dpi=600)
    fig4.savefig("{}/{}/Benchmarking_stress_testing_{}_{}.pdf".format(script_dir,results_folder,"AP",subtitle))
    plt.close(fig4)
    fig4.clf()

def plot_hierarchical_clustering(vegvisir_folder,external_paths_dict,folder,title="",keyname="viral_dataset9"):

    fold = 1 if keyname == "viral_dataset9" else 3#3
    if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(vegvisir_folder, fold)):
        train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(vegvisir_folder, fold)) #should be the same training dataset all the time
        valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(vegvisir_folder, fold)) #should be the same training dataset all the time
        test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(vegvisir_folder, fold))
    else:
        train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_test.p".format(vegvisir_folder)) #should be the same training dataset all the time
        valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid.p".format(vegvisir_folder)) #should be the same training dataset all the time
        test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test.p".format(vegvisir_folder))

    # train_data_int = train_out["summary_dict"]["data_int_samples"]
    # train_latent = train_out["latent_space"]
    # corrected_aa_types = train_out["dataset_info"].corrected_aa_types

    if keyname == "viral_dataset15":
        test_data_int = test_out["summary_dict"]["data_int_samples"]
        test_latent = test_out["latent_space"]
    else:
        test_data_int = valid_out["summary_dict"]["data_int_samples"]
        test_latent = valid_out["latent_space"]

    corrected_aa_types = valid_out["dataset_info"].corrected_aa_types


    #Highlight: Slice out the sequences
    data_int = test_data_int
    #data_int = np.concatenate([train_data_int,test_data_int],axis=0)
    sequences_int = data_int[:,1]
    sequences_len = np.sum(sequences_int.astype(bool),axis=-1)


    idx_dict = {"9mers":np.array((sequences_len == 9)),"all":np.ones_like(sequences_len).astype(bool)}
    idx = idx_dict["9mers"]
    #Highlight: Transform to blosum encoding
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types, "BLOSUM62",
                                                                               zero_characters=["#"],
                                                                               include_zero_characters=True)
    sequences_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(sequences_int)

    latent = test_latent[:,6:]
    #latent = np.concatenate([train_latent,test_latent],axis=0)[:,6:]
    latent = latent[idx]
    sequences_blosum = sequences_blosum[idx][:,:9]
    sequences_int = sequences_int[idx][:,:9]
    sequences_blosum = sequences_blosum.reshape(sequences_blosum.shape[0],-1)


    aminoacids_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types,zero_characters = ["#"])
    aminoacids_dict_reversed = {val:key for key,val in aminoacids_dict.items()}
    sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(sequences_int)
    sequences_list = sequences_raw.tolist()

    #Highlight: Retrieve the pre-calculated peptide properties
    sequences_list = list(map(lambda seq: ("".join(seq)).replace("#",""),sequences_list))
    latent_df = pd.DataFrame({"Icore":sequences_list,"latent":latent.tolist()})
    features_df = pd.read_csv(external_paths_dict["embedded_epitopes"],sep="\t")
    combined_df = latent_df.merge(features_df,on=["Icore"],how="left") #keeps the order of the labels of the latent_df dataframe
    embedding = combined_df["Embedding"].apply(eval).apply(np.array).tolist()
    embedding = np.array(embedding)

    #Highlight: Retrieve the ESMB1 values

    esm1b_df = pd.read_csv(external_paths_dict["esmb1_path"],sep=",",low_memory=False)
    esm1b_df_cols = esm1b_df.columns[(esm1b_df.columns.str.contains("Icore")) | (esm1b_df.columns.str.contains(pat='esm1b_'))]
    esm1b_df = esm1b_df[esm1b_df_cols]
    combined_df = combined_df.merge(esm1b_df,on=["Icore"],how="left")

    combined_df_cols = combined_df.columns[(combined_df.columns.str.contains(pat='esm1b_'))]
    esm1b_array = combined_df[combined_df_cols].values

    labels = combined_df["target_corrected"]


    colors_labels = np.vectorize(colors_dict_labels.get)(labels)
    distance_treshold = 0.5
    #Highlight: Latent space
    g1 = sns.clustermap(latent,metric="cosine",row_colors=colors_labels,cmap="crest",vmin=0,vmax=10,z_score=0)
    labels_clustered = labels[g1.dendrogram_row.reordered_ind]
    clustering_latent = AgglomerativeClustering(linkage="average",metric="cosine",n_clusters=None,distance_threshold=distance_treshold).fit(latent)
    clustered_labels = clustering_latent.labels_
    clustering_significance = np.round(adjusted_mutual_info_score(labels,clustered_labels),4)
    g1.ax_heatmap.tick_params(tick2On=False, labelsize=False,labelbottom=False,labelright=False)
    #g1.ax_col_dendrogram.set_title("Latent representations",fontsize=20,weight="bold")
    g1.ax_col_dendrogram.set_title("Latent representations: \n {}".format(clustering_significance),fontsize=20,weight="bold")

    #Highlight: Feature embeddings
    g2 = sns.clustermap(embedding,metric="cosine",row_colors=colors_labels,cmap="crest",vmin=0,vmax=10,z_score=0)
    labels_clustered = labels[g2.dendrogram_row.reordered_ind]
    clustering_embedding = AgglomerativeClustering(linkage="average",metric="cosine",n_clusters=None,distance_threshold=distance_treshold).fit(embedding)
    clustered_labels = clustering_embedding.labels_
    clustering_significance = np.round(adjusted_mutual_info_score(labels,clustered_labels),4)
    g2.ax_heatmap.tick_params(tick2On=False, labelsize=False,labelbottom=False,labelright=False)
    #g2.ax_col_dendrogram.set_title("Feature embeddings",fontsize=20,weight="bold")
    g2.ax_col_dendrogram.set_title("Feature embeddings: \n {}".format(clustering_significance),fontsize=20,weight="bold")

    #Highlight: Blosum encoding
    g3 = sns.clustermap(sequences_blosum,metric="cosine",row_colors=colors_labels,cmap="crest",vmin=0,vmax=10,z_score=0)
    labels_clustered = labels[g3.dendrogram_row.reordered_ind]
    clustering_blosum = AgglomerativeClustering(linkage="average",metric="cosine",n_clusters=None,distance_threshold=distance_treshold).fit(sequences_blosum)
    clustered_labels = clustering_blosum.labels_
    clustering_significance = np.round(adjusted_mutual_info_score(labels,clustered_labels),4)
    g3.ax_heatmap.tick_params(tick2On=False, labelsize=False,labelbottom=False,labelright=False)
    #g3.ax_col_dendrogram.set_title("Blosum encoding",fontsize=20,weight="bold")
    g3.ax_col_dendrogram.set_title("Blosum encoding: \n {}".format(clustering_significance),fontsize=20,weight="bold")

    #Highlight: ESM1B
    g4 = sns.clustermap(esm1b_array,metric="cosine",row_colors=colors_labels,cmap="crest",vmin=0,vmax=10,z_score=0)
    labels_clustered = labels[g4.dendrogram_row.reordered_ind]
    clustering_esmb1 = AgglomerativeClustering(linkage="average",metric="cosine",n_clusters=None,distance_threshold=distance_treshold).fit(esm1b_array)
    clustered_labels = clustering_esmb1.labels_
    clustering_significance = np.round(adjusted_mutual_info_score(labels,clustered_labels),4)
    g4.ax_heatmap.tick_params(tick2On=False, labelsize=False,labelbottom=False,labelright=False)
    #g4.ax_col_dendrogram.set_title("ESM1B",fontsize=20,weight="bold")
    g4.ax_col_dendrogram.set_title("ESM1B: \n {}".format(clustering_significance),fontsize=20,weight="bold")


    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(2, 2)

    mg0 = VegvisirUtils.SeabornFig2Grid(g1, fig, gs[0, 0])
    mg1 = VegvisirUtils.SeabornFig2Grid(g2, fig, gs[0, 1])
    mg2 = VegvisirUtils.SeabornFig2Grid(g3, fig, gs[1, 0])
    mg3 = VegvisirUtils.SeabornFig2Grid(g4, fig, gs[1, 1])

    fig.suptitle("Cluster heatmaps (cosine similarity)",fontsize=20,weight="bold")

    plt.savefig("{}/Clustermaps_{}.jpg".format(folder,title))

def plot_ablation_study(ablation_dict,script_dir,folder,subtitle,ensemble=False):
    """"""

    metrics_keys = ["ppv", "fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1", "pval_class_0", "pval_class_1"]
    roc_auc_results_dict = defaultdict(lambda: defaultdict())
    for key,vegvisir_folder in ablation_dict.items():
        if ensemble:
            metrics_results_dict = plot_benchmark_vegvisir_helper3(vegvisir_folder, None, kfolds=5,aggregated_not_overlap=False)

        else:
            metrics_results_dict = plot_kfold_comparison_helper(metrics_keys, script_dir, folder=vegvisir_folder,overwrite=False, kfolds=5)

        metrics_results_train = metrics_results_dict["train"]
        metrics_results_test = metrics_results_dict["test"] #I cannot do an ensemble with the valid data points

        # Highlight: Train results
        vegvisir_results_auc_train =  np.round((np.mean(
            np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(
            np.array(metrics_results_train["roc_auc_class_1"]))) / 2, 2)
        vegvisir_results_auc_test = np.round((np.mean(
            np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(
            np.array(metrics_results_test["roc_auc_class_1"]))) / 2, 2)
        roc_auc_results_dict[key]["train"] = vegvisir_results_auc_train
        roc_auc_results_dict[key]["test"] = vegvisir_results_auc_test


    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 15),facecolor='white')



    x = list(roc_auc_results_dict.keys())

    y_train = [v['train'] for v in roc_auc_results_dict.values()]
    y_test= [v['test'] for v in roc_auc_results_dict.values()]

    ax1.plot(x,y_train,c= "skyblue",marker = "o",markersize=30)
    ax1.plot(x,y_test,c= "darkgreen",marker = "o",markersize=30)
    ax1.set_facecolor("white")
    ax1.tick_params(axis='x', labelsize=30)
    ax1.tick_params(axis='y', labelsize=30)
    ax1.set_xlabel(r"$\beta$ values",fontsize=30)
    ax1.set_ylabel(r"ROC-AUC",fontsize=30)
    fig1.suptitle(r"Ablation study: $\beta$ parameter",fontsize=30)
    legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in {"Train": "skyblue", "Test": "darkgreen"}.items()]
    fig1.legend(handles=legends, prop={'size': 30}, loc='upper center', bbox_to_anchor=(0.55, 0.95))

    fig1.savefig("{}/Ablation_study_betaparam{}.pdf".format(folder,subtitle))

def plot_benchmark_vegvisir_helper4(vegvisir_folder,overlap_idx,kfolds=5,aggregated_not_overlap=True):
    """Compute Ensembl metrics per sequence length group"""
    metrics_results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))


    for mode in ["Train","Test"]:
        target_scores_dict = defaultdict(lambda:defaultdict(list))
        for fold in range(kfolds):
            results = pd.read_csv(f"{vegvisir_folder}/{mode}_fold_{fold}/Epitopes_predictions_{mode}_fold_{fold}.tsv", sep="\t")

            results["Length"] = np.array(list(map(len, results["Icore"].tolist())))

            unique_lens = list(set(results["Lengths"].tolist()))

            for seq_len in unique_lens:
                results_subset = results[results["Length"] == seq_len]
                if overlap_idx is not None:
                    results_subset = results_subset.merge(overlap_idx,on="Icore",how="left")
                if aggregated_not_overlap:
                    results_subset = results_subset[results_subset["Aggregated_overlap"] == False]
                targets = np.array(results_subset["Target_corrected"].tolist())
                onehot_targets = np.zeros((targets.shape[0], 2))
                onehot_targets[np.arange(0, targets.shape[0]), targets.astype(int)] = 1
                target_scores = results_subset[["Vegvisir_negative_prob", "Vegvisir_positive_prob"]].to_numpy().astype(float)
                target_scores_dict[seq_len].append(target_scores)


        #TODO: Per sequence length calculate ....
        target_scores = sum(target_scores_list)/kfolds

        fpr=dict()
        tpr=dict()
        roc_auc=dict()
        precision = dict()
        recall=dict()
        average_precision=dict()
        ppv_mod=dict()
        pvals=dict()
        # ROC AUC per class
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(onehot_targets[:, i], target_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc["auc01_class_{}".format(i)] = roc_auc_score(onehot_targets[:, i], target_scores[:, i],average="weighted", max_fpr=0.1)
            precision[i], recall[i], thresholds = precision_recall_curve(onehot_targets[:, i], target_scores[:, i])
            average_precision[i] = average_precision_score(onehot_targets[:, i], target_scores[:, i])
            ppv_mod[i] = calculate_ppv_modified(onehot_targets[:, i], target_scores[:, i])
            lrm = sm.Logit(onehot_targets[:, i], target_scores[:, i]).fit(disp=0)
            pvals[i] = lrm.pvalues.item()

        # metrics_results_dict[mode]["fpr_0"].append(fpr[0])
        # metrics_results_dict[mode]["fpr_1"].append(fpr[1])
        # metrics_results_dict[mode]["tpr_0"].append(tpr[0])
        # metrics_results_dict[mode]["tpr_1"].append(tpr[1])
        metrics_results_dict[mode.lower()]["roc_auc_class_0"].append(roc_auc[0])
        metrics_results_dict[mode.lower()]["roc_auc_class_1"].append(roc_auc[1])
        metrics_results_dict[mode.lower()]["auc01_class_0"].append(roc_auc[f"auc01_class_{0}"])
        metrics_results_dict[mode.lower()]["auc01_class_1"].append(roc_auc[f"auc01_class_{1}"])
        metrics_results_dict[mode.lower()]["pval_class_0"].append(pvals[0])
        metrics_results_dict[mode.lower()]["pval_class_1"].append(pvals[1])
        metrics_results_dict[mode.lower()]["ap_class_0"].append(average_precision[0])
        metrics_results_dict[mode.lower()]["ap_class_1"].append(average_precision[1])
        metrics_results_dict[mode.lower()]["ppv_mod_class_0"].append(ppv_mod[0])
        metrics_results_dict[mode.lower()]["ppv_mod_class_1"].append(ppv_mod[1])
        metrics_results_dict[mode.lower()]["precision_class_0"].append(precision[0])
        metrics_results_dict[mode.lower()]["precision_class_1"].append(precision[1])
        metrics_results_dict[mode.lower()]["recall_class_0"].append(recall[0])
        metrics_results_dict[mode.lower()]["recall_class_1"].append(recall[1])

    # print(metrics_results_dict["train"]["pval_class_0"])
    # print(metrics_results_dict["train"]["pval_class_1"])
    # print(metrics_results_dict["test"]["pval_class_0"])
    # print(metrics_results_dict["test"]["pval_class_1"])
    #
    # exit()

    return metrics_results_dict

def plot_metrics_per_length(vegvisir_dict,script_dir,folder,subtitle):
    """"""

    metrics_keys = ["ppv", "fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1", "pval_class_0", "pval_class_1"]
    roc_auc_results_dict = defaultdict(lambda: defaultdict())
    for key,vegvisir_folder in vegvisir_dict.items():

        metrics_results_dict = plot_benchmark_vegvisir_helper4(vegvisir_folder, None, kfolds=5,aggregated_not_overlap=False)









