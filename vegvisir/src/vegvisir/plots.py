"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import functools
import gc
import json
import math
import operator
import pickle
import warnings
from collections import defaultdict

import dill
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
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
from sklearn.metrics import auc,roc_auc_score,roc_curve,confusion_matrix,matthews_corrcoef,precision_recall_curve,average_precision_score
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import multiprocessing
import os
from functools import partial
from scipy import stats
import vegvisir.similarities as VegvisirSimilarities
from collections import namedtuple
import dataframe_image as dfi
import statsmodels.api as sm
MAX_WORKERs = ( multiprocessing. cpu_count() - 1 )
plt.style.use('ggplot')
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

PlotSettings = namedtuple("PlotSettings",["colormap_unique","colors_feature","unique_values"])

def plot_data_information(data, filters_dict, storage_folder, args, name_suffix):
    """"""
    ndata = data.shape[0]
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


def plot_data_information_reduced(data, filters_dict, storage_folder, args, name_suffix):
    """"""
    ndata = data.shape[0]
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15, 10))
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
    ax[0][0].set_title('Binary targets counts',fontsize=12)
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
    ax[0][1].set_title('Corrected binary targets counts',fontsize=12)
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
            textcoords='offset points',weight='bold')
    ####### Immunodominance scores ###################
    ax[0][3].hist(data["immunodominance_score_scaled"].to_numpy(), num_bins, density=True)
    ax[0][3].set_xlabel('Minmax scaled \n immunodominance score \n (N + / Total)',fontsize=12)
    ax[0][3].set_title('Histogram of \n immunodominance scores',fontsize=12)
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
                              size=12, xytext=(0, 8),
                              textcoords='offset points',weight='bold')
        ax[1][0].margins(y=0.1)
        ax[1][0].xaxis.set_ticks([0])
        ax[1][0].set_xticklabels(["Test proportions"], fontsize=12)
        ax[1][0].set_title('Test dataset. \n Class proportions', fontsize=12)
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
                    textcoords='offset points',weight='bold')

        partitions_names.append(name)
    ax[0][2].margins(y=0.1)
    ax[0][2].xaxis.set_ticks([0, 0.4, 0.8, 1.2, 1.6])
    ax[0][2].set_xticklabels(["Part. {}".format(int(i)) for i in partitions_names])
    ax[0][2].set_title('Train-validation dataset. \n Class proportions per partition',fontsize=12)
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
    ax[1][1].set_title("Sequence length distribution of \n  the Train-validation dataset",fontsize=12)

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
    ax[1][2].set_title("Sequence length distribution of \n  the test dataset",fontsize=12)

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
    fig.legend(handles=legends, prop={'size': 12}, loc='center right', bbox_to_anchor=(0.85, 0.2))
    fig.tight_layout(pad=0.1)
    fig.suptitle("Dataset distributions")
    plt.subplots_adjust(right=0.9,top=0.9,hspace=0.35,wspace=0.3)
    plt.savefig("{}/{}/Viruses_histograms_{}".format(storage_folder, args.dataset_name, name_suffix), dpi=500)
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
    ax4.set_title("Immunodominance scores", fontsize=20)
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

def plot_aa_frequencies(data_array,aa_types,aa_dict,max_len,storage_folder,args,analysis_mode,mode):

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

def plot_heatmap(array, title,file_name):
    fig = plt.figure(figsize=(20, 20))
    sns.heatmap(array, cmap='RdYlGn_r',yticklabels=False,xticklabels=False)
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()
    plt.close(fig)

def plot_umap1(array,labels,storage_folder,args,title_name,file_name):
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

def plot_loss(train_loss,valid_loss,epochs_list,fold,results_dir):
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

def plot_accuracy(train_accuracies,valid_accuracies,epochs_list,mode,results_dir):
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

def plot_classification_score(train_auc,valid_auc,epochs_list,fold,results_dir,method):
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

def plot_latent_vector(latent_space,predictions_dict,fold,results_dir,method):

    print("Plotting latent vector...")
    latent_vectors = latent_space[:,5:]
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

def plot_clusters_features_distributions(dataset_info,cluster_assignments,n_clusters,predictions_dict,sample_mode,results_dir,method,vector_name):
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
                "clusters_info":clusters_info}
    
    return features_dict,sequences_raw

def define_colormap(feature,cmap_name):
    """"""

    unique_values = np.unique(feature)
    feature, unique_values = VegvisirUtils.replace_nan(feature, unique_values)
    colormap_unique = matplotlib.cm.get_cmap(cmap_name, len(unique_values.tolist()))

    colors_dict = dict(zip(unique_values, colormap_unique.colors))
    colors_feature = np.vectorize(colors_dict.get, signature='()->(n)')(feature)

    return PlotSettings(colormap_unique=colormap_unique,
                        colors_feature=colors_feature,
                        unique_values=unique_values)

def plot_preprocessing(umap_proj,dataset_info,predictions_dict,sample_mode,results_dir,method,vector_name="latent_space_z",n_clusters=4):
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
            "extintion_coefficients_settings":extintion_coefficients_settings
            }

def plot_scatter(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name="latent_space_z",n_clusters=4):
    print("Plotting scatter UMAP of {}...".format(vector_name))

    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:, 0])
    if method == "_single_sample":
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(
            predictions_dict["class_binary_prediction_single_sample"])
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
    frequency_class0_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:, 0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn',
                                                       len(frequency_class0_unique))  # This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array = np.array(
        [colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_prediction_samples_frequencies"][:, 0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:, 1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array = np.array(
        [colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_prediction_samples_frequencies"][:, 1])
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
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=latent_space[:, 2], alpha=alpha, s=size)

    ax1.set_title("Binary targets", fontsize=20)
    if method == "_single_sample":
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

    #del confidence_scores,immunodominance_scores,gravy_scores,volume_scores,side_chain_pka_scores,frequency_class1_unique,frequency_class0_unique,sequences_lens,radius_scores,molecular_weight_scores,aromaticity_scores,bulkiness_scores
    #gc.collect()

def colorbar(mappable):
    """Places a figure color bar without squeezing the plot"""
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def plot_scatter_reduced(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name="latent_space_z",n_clusters=4):
    print("Plotting (reduced) scatter UMAP of {}...".format(vector_name))

    title_dict = {"latent_space_z": "Latent representation (z)",
                  "encoder_final_hidden_state":"Encoder Hf",
                  "decoder_final_hidden_state": "Decoder Hf"}
    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:, 0])
    if method == "_single_sample":
        predictions_binary = predictions_dict["class_binary_prediction_single_sample"]
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_binary)
    else:
        predictions_binary = predictions_dict["class_binary_predictions_samples_mode"]
        #predictions_binary = predictions_dict["class_binary_predictions_samples_logits_average_argmax"]
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_binary)

    dataframe = pd.DataFrame({"UMAP_x":umap_proj[:,0],
                              "UMAP_y": umap_proj[:, 1],
                              "Binary targets":latent_space[:, 0],
                              "predictions_binary":predictions_binary,
                              "Immunodominance":latent_space[:, 3],
                              "frequency_0":predictions_dict["class_binary_prediction_samples_frequencies"][:, 0],
                              "frequency_1":predictions_dict["class_binary_prediction_samples_frequencies"][:, 1]
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
    frequency_class0_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:, 0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn',len(frequency_class0_unique))  # This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array = np.array([colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_prediction_samples_frequencies"][:, 0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:, 1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array = np.array(
        [colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(predictions_dict["class_binary_prediction_samples_frequencies"][:, 1])

    alpha = 0.7
    size = 4
    # Highlight: Scatter and density plot
    g0 = sns.jointplot(data=dataframe, x="UMAP_x", y="UMAP_y", hue="Binary targets", alpha=alpha, s=8,
                        palette=list(colors_dict_labels.values()),
                        hue_order=[0,1]
                        )
    g0.ax_joint.legend(fancybox=True, framealpha=0.5)
    g0.ax_joint.axis("off")
    g0.ax_marg_x.axis("off")
    g0.ax_marg_y.axis("off")

    #Highlight: Predictions scatter plot
    g1 = sns.FacetGrid(dataframe,hue="predictions_binary", subplot_kws={"fc": "white"}, margin_titles=True,
    palette = list(colors_dict_labels.values()),
    hue_order = [0, 1]
    )
    g1.set(yticks=[])
    g1.set(xticks=[])
    g1_axes = g1.axes.flatten()
    g1_axes[0].set_title("Predicted targets")
    g1.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
    g1_axes[0].set_xlabel("")
    g1_axes[0].set_ylabel("")

    #Highlight: Immunodominace scatter plot
    g2 = sns.FacetGrid(dataframe, hue="Immunodominance", subplot_kws={"fc": "white"},
                       palette=colormap_immunodominance.colors,
                       hue_order=immunodominance_scores_unique)
    g2.set(yticks=[])
    g2.set(xticks=[])
    g2_axes = g2.axes.flatten()
    g2_axes[0].set_title("Immunodominance")
    g2.map(plt.scatter, "UMAP_x", "UMAP_y", alpha=alpha, s=size)
    g2_axes[0].set_xlabel("")
    g2_axes[0].set_ylabel("")

    g3 = sns.FacetGrid(dataframe, hue="frequency_0", subplot_kws={"fc": "white"},
                       palette=colormap_frequency_class0_array,
                       hue_order=frequency_class0_unique
                       )
    g3.set(yticks=[])
    g3.set(xticks=[])
    g3_axes = g3.axes.flatten()
    g3_axes[0].set_title("Posterior predictive (class 0)")
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
    g4_axes[0].set_title("Posterior predictive (class 1)")
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
    # gs.update(right=0.4)

    # Following: https://www.sc.eso.org/~bdias/pycoffee/codes/20160407/gridspec_demo.html
    cbax2 = plt.subplot(gs[0, 5])  # Place it where it should be.
    cbax3 = plt.subplot(gs[1, 2])  # Place it where it should be.
    cbax4 = plt.subplot(gs[1, 5])  # Place it where it should be.

    cb2 = Colorbar(ax=cbax2, mappable=plt.cm.ScalarMappable(cmap=colormap_immunodominance))
    cb3 = Colorbar(ax=cbax3, mappable=plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap_frequency_class0))
    cb4 = Colorbar(ax=cbax4, mappable=plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap_frequency_class1))

    fig.suptitle("UMAP latent space (z) projections")

    plt.savefig("{}/{}/umap_SCATTER_reduced_{}_{}".format(results_dir, method, vector_name, sample_mode))
    plt.clf()
    plt.close(fig)

    #del confidence_scores,immunodominance_scores,gravy_scores,volume_scores,side_chain_pka_scores,frequency_class1_unique,frequency_class0_unique,sequences_lens,radius_scores,molecular_weight_scores,aromaticity_scores,bulkiness_scores
    #gc.collect()

def plot_scatter_quantiles(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name="latent_space_z",n_clusters=4):
    print("Plotting scatter (quantiles) UMAP of {}...".format(vector_name))

    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:, 0])
    if method == "_single_sample":
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(
            predictions_dict["class_binary_prediction_single_sample"])
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
    frequency_class0_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:, 0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn',
                                                       len(frequency_class0_unique))  # This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array = np.array(
        [colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_prediction_samples_frequencies"][:, 0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:, 1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array = np.array(
        [colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(
        predictions_dict["class_binary_prediction_samples_frequencies"][:, 1])
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
    if method == "_single_sample":
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

def plot_latent_correlations(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,vector_name):
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

    cosine_dist = 1- VegvisirSimilarities.cosine_similarity(umap_proj[:,None], umap_proj[:,None], correlation_matrix=False, parallel=False).squeeze(-1).squeeze(-1)
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

def plot_latent_correlations_1d(umap_proj_1d,args,settings,dataset_info,latent_space,sample_mode,results_dir,method,vector_name,plot_scatter_correlations=False,plot_covariance=True,save_plot=True,filter_correlations=True):
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
    features_dict["immunodominance_scores"] = latent_space[:,3]
    true_labels = latent_space[:,0]
    features_dict["binary_targets"] = true_labels
    # #Highlight: Bring the pre-calculate peptide features back. PRESERVING THE ORDER OF THE SEQUENCES!
    if (not args.shuffle_sequence) and (not args.random_sequences) and (not args.num_mutations != 0) and (args.sequence_type == "Icore"):
        if (args.num_classes == args.num_obs_classes):
            try:
                sequences_raw = settings["sequences_raw"]  # the sequences are following the order from the data loader
                sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), sequences_raw))
                sequences_raw = pd.DataFrame({"Icore": sequences_raw})
                all_feats = pd.read_csv("{}/common_files/dataset_all_features.tsv".format(dataset_info.storage_folder),sep="\s+",index_col=0)
                peptide_feats_cols = all_feats.columns[(all_feats.columns.str.contains("Icore")) | (all_feats.columns.str.contains(pat = 'pep_'))]
                peptide_feats = all_feats[peptide_feats_cols]
                sequences_feats = VegvisirUtils.merge_in_left_order(sequences_raw, peptide_feats, "Icore")
                sequences_feats = sequences_feats.groupby('Icore', as_index=False, sort=False)[peptide_feats_cols[peptide_feats_cols != "Icore"]].agg(lambda x: sum(list(x)) / len(list(x))) #sort Falsse to not mess up the order in which the sequences come out from the model
                sequences_feats = sequences_feats[~sequences_feats[peptide_feats_cols[1]].isna()]

                sequences_feats = sequences_feats.to_dict(orient="list")
                sequences_feats.pop('Icore', None)
                #Highlight: Merge both features dict if there is useful information
                if sequences_feats[peptide_feats_cols[1]] and len(sequences_feats[peptide_feats_cols[1]]) == umap_proj_1d.shape:
                    features_dict = {**features_dict, **sequences_feats}
                else:
                    print("Could not find information about all the sequences")
            except:
                print("Not all of the sequences are in the pre-computed features, skipping")
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
    if plot_covariance:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 20))
        features_names = ["UMAP-1D"] + list(features_dict.keys())
        features_matrix = np.array(list(features_dict.values()))
        features_matrix = np.vstack([umap_proj_1d[None, :], features_matrix])
        features_covariance = np.cov(features_matrix)
        features_correlations = np.corrcoef(features_matrix)
        if save_plot:
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
                "spearman_coefficients":spearman_coefficients}

def plot_latent_space(args,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,vector_name="latent_space_z",n_clusters=4,plot_correlations=True):
    """
    -Notes on UMAP: https://www.arxiv-vanity.com/papers/2009.12981/
    """
    reducer = umap.UMAP()
    umap_proj = reducer.fit_transform(latent_space[:, 5:])
    settings =plot_preprocessing(umap_proj, dataset_info, predictions_dict, sample_mode, results_dir, method,
                       vector_name="latent_space_z", n_clusters=4)

    plot_scatter(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name=vector_name,n_clusters=n_clusters)
    plot_scatter_reduced(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name=vector_name,n_clusters=n_clusters)
    if vector_name == "latent_space_z":
        plot_scatter_quantiles(umap_proj,dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,settings,vector_name=vector_name,n_clusters=n_clusters)
    if plot_correlations and vector_name == "latent_space_z":
        reducer = umap.UMAP(n_components=1)
        umap_proj_1d = reducer.fit_transform(latent_space[:, 5:]).squeeze(-1)
        plot_latent_correlations_1d(umap_proj_1d,args, settings,dataset_info, latent_space, sample_mode, results_dir, method,
                                 vector_name)
    del umap_proj
    gc.collect()

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

def micro_auc(args,onehot_labels,y_prob,idx):
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

        if save_plot:
            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                onehot_targets.ravel(), target_scores.ravel()
            )
            average_precision["micro"] = average_precision_score(onehot_targets, target_scores, average="micro")
            fig = plt.figure()
            plt.plot(recall["micro"],precision["micro"], label="Average Precision (AP): {}".format(average_precision["micro"]))
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
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pvals = dict()
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
    try:
        for i in range(args.num_obs_classes):
            lrm = sm.Logit(onehot_targets[:, i], target_scores[:, i]).fit(disp=0)
            pvals[i] = lrm.pvalues.item()
    except:
        print("Regression failed")
        pvals[0] = np.nan
        pvals[1] = np.nan
    return fpr,tpr,roc_auc,pvals

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
    probability_modes = ["class_probs_predictions_samples_average","class_probs_prediction_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_probs_predictions_samples_average"]
    binary_modes = ["class_binary_predictions_samples_mode","class_binary_prediction_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_mode"]
    #binary_modes = ["class_binary_predictions_samples_logits_average_argmax","class_binary_prediction_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_logits_average_argmax"]

    metrics_summary_dict = defaultdict(lambda:defaultdict(lambda : defaultdict()))
    for sample_mode,prob_mode,binary_mode in zip(evaluation_modes,probability_modes,binary_modes):
    #for sample_mode,prob_mode,binary_mode in zip(["samples","single_sample"],["class_probs_predictions_samples_average","class_probs_prediction_single_sample"],["class_binary_predictions_samples_mode","class_binary_prediction_single_sample"]):
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


            #for key_name_1,stats_name_1 in zip(["samples_average_prob","single_sample_prob"],["class_probs_predictions_samples_average","class_probs_prediction_single_sample"]):
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

                fpr,tpr,roc_auc,pvals = plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name)
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



            #for key_name_2,stats_name_2 in zip(["samples_mode","single_sample"],["class_binary_predictions_samples_mode","class_binary_prediction_single_sample"]):
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
    binary_modes = ["class_binary_predictions_samples_mode","class_binary_prediction_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_mode"]
    #binary_modes = ["class_binary_predictions_samples_logits_average_argmax","class_binary_prediction_single_sample"] if predictions_dict["true_single_sample"] is not None else ["class_binary_predictions_samples_logits_average_argmax"]

    #for sample_mode,prob_mode,binary_mode in zip(["samples","single_sample"],["class_probs_predictions_samples_average","class_probs_prediction_single_sample"],["class_binary_predictions_samples_mode","class_binary_prediction_single_sample"]):
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


                    fpr,tpr,roc_auc,pvals = plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name,save=False)
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
                    fig.legend(bbox_to_anchor=(0.71, 0.35), prop={'size': 15})  # loc='lower right'
                    fig.suptitle("ROC curve. AUC_micro_ovr_average: {}".format(average_micro_auc/args.num_samples),fontsize=12)
                    plt.savefig("{}/{}/ROC_curves_PER_SAMPLE_{}".format(results_dir, mode, "{}".format(idx_name)))
                    plt.clf()
                    plt.close(fig)
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
        print("Plotting Hidden dimensions latent space")
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
                            ax2.set_title("Information shift by amino acid type")
                            # Highlight: Aminoacids coloured by functional group (i.e positive, negative ...)
                            sns.heatmap(aminoacids_masked, ax=ax4, cbar=False, cmap=aa_groups_colormap)
                            ax4.set_xticks(np.arange(max_len) + 0.5,
                                           labels=["{}".format(i) for i in range(max_len)])
                            ax4.spines['left'].set_visible(False)
                            ax4.yaxis.set_ticklabels([])
                            ax4.set_title("Information shift by amino acid group")

                            ax3.axis("off")
                            ax5.axis("off")
                            ax6.axis("off")

                            # legend1 = plt.legend(handles=aa_patches, prop={'size': 8}, loc='best',
                            #                      bbox_to_anchor=(2.1, -7.4), ncol=2)
                            # plt.legend(handles=aa_groups_patches, prop={'size': 8}, loc='best',
                            #            bbox_to_anchor=(1.6, -7.4), ncol=1)
                            # plt.gca().add_artist(legend1)

                            legend1 = plt.legend(handles=aa_patches, prop={'size': 10}, loc='best',
                                                 bbox_to_anchor=(24, -0.3), ncol=2)
                            plt.legend(handles=aa_groups_patches, prop={'size': 10}, loc='best',
                                       bbox_to_anchor=(14, -0.4), ncol=1)
                            plt.gca().add_artist(legend1)

                            #fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
                            plt.subplots_adjust(left=0.1,hspace=0.3,wspace=0.3)
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

    label_names = {"_immunodominance_scores":"Immunodominance",
                   "_binary_labels":"Binary targets"}
    if (not args.shuffle_sequence) and (not args.random_sequences) and (not args.num_mutations != 0) and (args.sequence_type == "Icore"):
        if (args.num_classes == args.num_obs_classes) and use_precomputed_features:
            sequences_raw = list(map(lambda seq: "".join(seq).replace("#", ""), sequences_raw))
            sequences_raw = pd.DataFrame({"Icore": sequences_raw})
            all_feats = pd.read_csv("{}/common_files/dataset_all_features.tsv".format(storage_folder),sep="\s+",index_col=0)
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


    if tag == "_immunodominance_scores" and features_dict["volume"] is not None:
        pearson_correlations = list(map(lambda feat1,feat2: VegvisirUtils.calculate_correlations(feat1, feat2),[labels]*len(features_dict.keys()),list(features_dict.values())))
        pearson_correlations = list(zip(*pearson_correlations))
        pearson_coefficients = np.array(pearson_correlations[0])
        pearson_coefficients = np.round(pearson_coefficients,2)
        pearson_pvalues = np.array(pearson_correlations[1])
        pearson_pvalues = np.round(pearson_pvalues,3)
    else:
        if features_dict["volume"] is not None:
            pearson_correlations = list(map(lambda feat1,feat2: VegvisirUtils.calculate_correlations(feat2, feat1),[labels]*len(features_dict.keys()),list(features_dict.values())))
            pearson_correlations = list(zip(*pearson_correlations))
            pearson_coefficients = np.array(pearson_correlations[0])
            pearson_coefficients = np.round(pearson_coefficients,2)
            pearson_pvalues = np.array(pearson_correlations[1])
            pearson_pvalues = np.round(pearson_pvalues,3)

    if features_dict["volume"] is not None:
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
        ax1.set_xticks(np.arange(len(features_names)) ,labels=features_names,rotation=45,fontsize=18)
        ax1.spines['left'].set_visible(False)
        #ax1.yaxis.set_ticklabels([])
        ax1.set_yticks(np.arange(len(features_names)) + 0.5,labels=features_names,rotation=360,fontsize=18)
        ymax=len(features_names)-1
        xpos=0
        ax1.add_patch(matplotlib.patches.Rectangle((ymax, xpos), 1, len(features_names), fill=False, edgecolor='green', lw=3))
        ax1.set_title("Covariance matrix features ({})".format(subfolders.replace("/",",")),fontsize=20)

        ax2.axis("off")
        fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.2)
        #fig.suptitle("Features covariance")
        plt.savefig("{}/{}/similarities/{}/HEATMAP_features_covariance{}.png".format(storage_folder,args.dataset_name,subfolders,tag))
        plt.clf()
        plt.close(fig)

        #Highlight: Plot correlations

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 12)) #gridspec_kw={'width_ratios': [4.5, 0.5]}

        i=0
        j = 0
        position_labels = []
        for feat_name,coeff,pval in zip(features_dict.keys(),pearson_coefficients,pearson_pvalues):
            position_labels.append(i)
            ax1.bar(i,coeff,label=feat_name,width = 0.1,color=colors_list_aa[j])
            i += 0.2
            j +=1

        ax1.xaxis.set_ticks(position_labels)
        def clean_labels(label):
            if label == "extintion_coefficient_cystines":
                label = "extintion coefficient \n (cystines)"
            elif label == "extintion_coefficient_cysteines":
                label = "extintion coefficient \n (cysteines)"
            else:
                label = label.replace("_"," ")
            return label
        labels_names = list(map(lambda label: clean_labels(label), list(features_dict.keys())))
        ax1.set_xticklabels(labels_names,fontsize=25,rotation=80)
        ax1.tick_params(axis="y",labelsize=30)
        plt.subplots_adjust(top=0.9,bottom=0.35)

        fig.suptitle("Correlation coefficients: Features vs {}".format(label_names[tag]),fontsize=30)

        plt.savefig("{}/{}/similarities/{}/HISTOGRAM_features_correlations{}.png".format(storage_folder,args.dataset_name,subfolders,tag))

def calculate_species_roc_auc_helper(summary_dict,args,script_dir,idx_all,fold,prob_mode,sample_mode,mode="train_species"):
    
    
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

            fpr, tpr, roc_auc,pvals = plot_ROC_curves(labels, onehot_labels, summary_dict, args, script_dir, mode, fold,
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
        train_folds_ap_class_0,train_folds_ap_class_1,train_folds_ppv,train_folds_fpr, train_folds_tpr, train_folds_roc_auc_class_0, train_folds_roc_auc_class_1, train_folds_pvals_class_0, train_folds_pvals_class_1 = metrics_results_train["ap_class_0"],metrics_results_train["ap_class_1"],metrics_results_train["ppv"],metrics_results_train["fpr"], metrics_results_train["tpr"], metrics_results_train["roc_auc_class_0"], metrics_results_train["roc_auc_class_1"], metrics_results_train["pval_class_0"], metrics_results_train["pval_class_1"]

        metrics_results_train_species = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_train_species.p".format(folder), "rb"))
        train_species_folds_ap_class_0,train_species_folds_ap_class_1,train_species_folds_ppv,train_species_folds_fpr, train_species_folds_tpr, train_species_folds_roc_auc_class_0, train_species_folds_roc_auc_class_1,train_species_folds_pvals_class_0, train_species_folds_pvals_class_1 = metrics_results_train_species["ap_class_0"],metrics_results_train_species["ap_class_1"],metrics_results_train_species["ppv"],metrics_results_train_species["fpr"], metrics_results_train_species["tpr"], metrics_results_train_species["roc_auc_class_0"], metrics_results_train_species["roc_auc_class_1"], metrics_results_train_species["pval_class_0"], metrics_results_train_species["pval_class_1"]

        if os.path.exists("{}/Vegvisir_checkpoints/roc_auc_valid.p".format(folder)):
            metrics_results_valid = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_valid.p".format(folder), "rb"))
            valid_folds_ap_class_0,valid_folds_ap_class_1,valid_folds_ppv,valid_folds_fpr, valid_folds_tpr, valid_folds_roc_auc_class_0, valid_folds_roc_auc_class_1, train_folds_pvals_class_0, train_folds_pvals_class_1 = \
            metrics_results_valid["ap_class_0"],metrics_results_valid["ap_class_1"],metrics_results_valid["ppv"],metrics_results_valid["fpr"], metrics_results_valid["tpr"], metrics_results_valid["roc_auc_class_0"], \
            metrics_results_valid["roc_auc_class_1"], metrics_results_valid["pval_class_0"], metrics_results_valid["pval_class_1"]
            metrics_results_valid_species = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_valid_species.p".format(folder), "rb"))
            valid_species_folds_ap_class_0,valid_species_folds_ap_class_1,valid_species_folds_ppv,valid_species_folds_fpr, valid_species_folds_tpr, valid_species_folds_roc_auc_class_0, valid_species_folds_roc_auc_class_1, valid_species_folds_pvals_class_0, valid_species_folds_pvals_class_1 =metrics_results_valid_species["ap_class_0"],metrics_results_valid_species["ap_class_1"],metrics_results_valid_species["ppv"], metrics_results_valid_species["fpr"], metrics_results_valid_species["tpr"],metrics_results_valid_species["roc_auc_class_0"], metrics_results_valid_species["roc_auc_class_1"], metrics_results_valid_species["pval_class_0"], metrics_results_valid_species["pval_class_1"]

        if os.path.exists("{}/Vegvisir_checkpoints/roc_auc_test.p".format(folder)):
            metrics_results_test = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_test.p".format(folder), "rb"))
            test_folds_ap_class_0,test_folds_ap_class_1,test_folds_ppv,test_folds_fpr, test_folds_tpr, test_folds_roc_auc_class_0, test_folds_roc_auc_class_1, test_folds_pvals_class_0, test_folds_pvals_class_1 = \
                metrics_results_test["ap_class_0"],metrics_results_test["ap_class_1"],metrics_results_test["ppv"],metrics_results_test["fpr"], metrics_results_test["tpr"], metrics_results_test["roc_auc_class_0"], \
                    metrics_results_test["roc_auc_class_1"], metrics_results_test["pval_class_0"], metrics_results_test[
                    "pval_class_1"]

            metrics_results_test_species = pickle.load(open("{}/Vegvisir_checkpoints/roc_auc_test_species.p".format(folder), "rb"))
            test_species_folds_ap_class_0,test_species_folds_ap_class_1,test_species_folds_ppv,test_species_folds_fpr, test_species_folds_tpr, test_species_folds_roc_auc_class_0, test_species_folds_roc_auc_class_1, test_species_folds_pvals_class_0, test_species_folds_pvals_class_1 = \
                metrics_results_test_species["ap_class_0"],metrics_results_test_species["ap_class_1"],metrics_results_test_species["ppv"],metrics_results_test_species["fpr"], metrics_results_test_species["tpr"], metrics_results_test_species[
                    "roc_auc_class_0"], metrics_results_test_species["roc_auc_class_1"], metrics_results_test_species["pval_class_0"], metrics_results_test_species["pval_class_1"]


    else:
        print("calculating ROC-AUC values")
        train_folds_ap_class_0,train_folds_ap_class_1,train_folds_ppv,train_folds_fpr, train_folds_tpr, train_folds_roc_auc_class_0, train_folds_roc_auc_class_1, train_folds_pvals_class_0, train_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[]
        valid_folds_ap_class_0,valid_folds_ap_class_1,valid_folds_ppv,valid_folds_fpr, valid_folds_tpr, valid_folds_roc_auc_class_0, valid_folds_roc_auc_class_1, valid_folds_pvals_class_0, valid_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[]
        test_folds_ap_class_0,test_folds_ap_class_1,test_folds_ppv,test_folds_fpr, test_folds_tpr, test_folds_roc_auc_class_0, test_folds_roc_auc_class_1, test_folds_pvals_class_0, test_folds_pvals_class_1 =[], [], [], [], [], [], [],[],[]
        train_species_folds_ap_class_0,train_species_folds_ap_class_1,train_species_folds_ppv,train_species_folds_fpr, train_species_folds_tpr, train_species_folds_roc_auc_class_0, train_species_folds_roc_auc_class_1, train_species_folds_pvals_class_0, train_species_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[]
        valid_species_folds_ap_class_0,valid_species_folds_ap_class_1,valid_species_folds_ppv,valid_species_folds_fpr, valid_species_folds_tpr, valid_species_folds_roc_auc_class_0, valid_species_folds_roc_auc_class_1, valid_species_folds_pvals_class_0, valid_species_folds_pvals_class_1 = [],[], [], [], [], [], [],[],[]
        test_species_folds_ap_class_0,test_species_folds_ap_class_1,test_species_folds_ppv,test_species_folds_fpr, test_species_folds_tpr, test_species_folds_roc_auc_class_0, test_species_folds_roc_auc_class_1, test_species_folds_pvals_class_0, test_species_folds_pvals_class_1 = [], [], [],[], [], [], [],[],[]

        for fold in range(kfolds):
            print("-------------FOLD {}--------------".format(fold))
            if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold)):
                train_out = torch.load(
                    "{}/Vegvisir_checkpoints/model_outputs_train_test_fold_{}.p".format(folder, fold))
                args = train_out["args"]
                train_out = train_out["summary_dict"]
            else:
                train_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_train_valid_fold_{}.p".format(folder, fold))["summary_dict"]
            valid_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_valid_fold_{}.p".format(folder, fold))["summary_dict"]
            if os.path.exists("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold)):
                test_out = torch.load("{}/Vegvisir_checkpoints/model_outputs_test_fold_{}.p".format(folder, fold))["summary_dict"]
            else:
                test_out = None

            for mode, summary_dict in zip(["train", "valid", "test"], [train_out, valid_out, test_out]):

                if summary_dict is not None:
                    labels = summary_dict["true_samples"]
                    onehot_labels = summary_dict["true_onehot_samples"]
                    # confidence_scores = summary_dict["confidence_scores_samples"]
                    prob_mode = "class_probs_predictions_samples_average"
                    idx_all = np.ones_like(labels).astype(bool)
                    if args.num_classes > args.num_obs_classes:
                        idx_all = (labels[..., None] != 2).any(
                            -1)  # Highlight: unlabelled data has been assigned labelled 2,we skip it
                    idx_name = "all"
                    sample_mode = "samples"

                    fpr, tpr, roc_auc, pvals = plot_ROC_curves(labels, onehot_labels, summary_dict, args, script_dir,
                                                               mode,
                                                               fold, sample_mode,
                                                               prob_mode, idx_all, idx_name, save=False)

                    binary_predictions = np.argmax(summary_dict[prob_mode],axis=1)
                    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=binary_predictions).ravel()
                    precision = tp / (tp + fp)

                    ap_dict = plot_precision_recall_curve(labels, onehot_labels, summary_dict, args, script_dir, mode, fold,
                                                "", prob_mode, idx_all, idx_name, save_plot=False)


                    if mode == "train":

                        train_folds_ap_class_0.append(ap_dict["average_precision"][0])
                        train_folds_ap_class_1.append(ap_dict["average_precision"][1])
                        train_folds_ppv.append(precision)
                        train_folds_fpr.append(fpr)
                        train_folds_tpr.append(tpr)
                        train_folds_roc_auc_class_0.append(roc_auc[0])
                        train_folds_roc_auc_class_1.append(roc_auc[1])
                        train_folds_pvals_class_0.append(pvals[0])
                        train_folds_pvals_class_1.append(pvals[1])
                        species_results = calculate_species_roc_auc_helper(summary_dict, args, script_dir, idx_all,
                                                                           fold, prob_mode, sample_mode,
                                                                           mode="{}_species".format(mode))
                        
                        train_species_folds_ap_class_0.append(np.nan)
                        train_species_folds_ap_class_1.append(np.nan)
                        train_species_folds_ppv.append(np.nan)
                        train_species_folds_fpr.append(np.nan)
                        train_species_folds_tpr.append(np.nan)
                        train_species_folds_roc_auc_class_0.append(species_results[0])
                        train_species_folds_roc_auc_class_1.append(species_results[1])
                        train_species_folds_pvals_class_0.append(species_results[2])
                        train_species_folds_pvals_class_1.append(species_results[3])

                    elif mode == "valid":
                        valid_folds_ap_class_0.append(ap_dict["average_precision"][0])
                        valid_folds_ap_class_1.append(ap_dict["average_precision"][1])
                        valid_folds_ppv.append(precision)
                        valid_folds_fpr.append(fpr)
                        valid_folds_tpr.append(tpr)
                        valid_folds_roc_auc_class_0.append(roc_auc[0])
                        valid_folds_roc_auc_class_1.append(roc_auc[1])
                        valid_folds_pvals_class_0.append(pvals[0])
                        valid_folds_pvals_class_1.append(pvals[1])
                        species_results = calculate_species_roc_auc_helper(summary_dict, args, script_dir, idx_all,
                                                                           fold, prob_mode, sample_mode,
                                                                           mode="{}_species".format(mode))
                        valid_species_folds_ap_class_0.append(np.nan)
                        valid_species_folds_ap_class_1.append(np.nan)
                        valid_species_folds_ppv.append(np.nan)
                        valid_species_folds_fpr.append(np.nan)
                        valid_species_folds_tpr.append(np.nan)
                        valid_species_folds_roc_auc_class_0.append(species_results[0])
                        valid_species_folds_roc_auc_class_1.append(species_results[1])
                        valid_species_folds_pvals_class_0.append(species_results[2])
                        valid_species_folds_pvals_class_1.append(species_results[3])

                    else:
                        test_folds_ap_class_0.append(ap_dict["average_precision"][0])
                        test_folds_ap_class_1.append(ap_dict["average_precision"][1])
                        test_folds_ppv.append(precision)
                        test_folds_fpr.append(fpr)
                        test_folds_tpr.append(tpr)
                        test_folds_roc_auc_class_0.append(roc_auc[0])
                        test_folds_roc_auc_class_1.append(roc_auc[1])
                        test_folds_pvals_class_0.append(pvals[0])
                        test_folds_pvals_class_1.append(pvals[1])
                        
                        test_species_folds_ap_class_0.append(np.nan)
                        test_species_folds_ap_class_1.append(np.nan)
                        test_species_folds_ppv.append(np.nan)
                        test_species_folds_fpr.append(np.nan)
                        test_species_folds_tpr.append(np.nan)
                        test_species_folds_roc_auc_class_0.append(np.nan)
                        test_species_folds_roc_auc_class_1.append(np.nan)
                        test_species_folds_pvals_class_0.append(np.nan)
                        test_species_folds_pvals_class_1.append(np.nan)

                else:
                    if mode == "valid":
                        valid_folds_ap_class_0.append(np.nan)
                        valid_folds_ap_class_1.append(np.nan)
                        valid_folds_fpr.append(np.nan)
                        valid_folds_tpr.append(np.nan)
                        valid_folds_roc_auc_class_0.append(np.nan)
                        valid_folds_roc_auc_class_1.append(np.nan)
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
                        test_folds_roc_auc_class_0.append(np.nan)
                        test_folds_roc_auc_class_1.append(np.nan)
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
        metrics_results_train["fpr"] = train_folds_fpr
        metrics_results_train["tpr"] = train_folds_tpr
        metrics_results_train["roc_auc_class_0"] = train_folds_roc_auc_class_0
        metrics_results_train["roc_auc_class_1"] = train_folds_roc_auc_class_1
        train_folds_pvals_class_0 = np.array(train_folds_pvals_class_0)
        metrics_results_train["pval_class_0"] = train_folds_pvals_class_0[~np.isnan(train_folds_pvals_class_0)]
        train_folds_pvals_class_1 = np.array(train_folds_pvals_class_1)
        metrics_results_train["pval_class_1"] = train_folds_pvals_class_1[~np.isnan(train_folds_pvals_class_1)]

        
        metrics_results_valid["ap_class_0"] = valid_folds_ap_class_0
        metrics_results_valid["ap_class_1"] = valid_folds_ap_class_1
        metrics_results_valid["ppv"] = valid_folds_ppv
        metrics_results_valid["fpr"] = valid_folds_fpr
        metrics_results_valid["tpr"] = valid_folds_tpr
        metrics_results_valid["roc_auc_class_0"] = valid_folds_roc_auc_class_0
        metrics_results_valid["roc_auc_class_1"] = valid_folds_roc_auc_class_1
        valid_folds_pvals_class_0 = np.array(valid_folds_pvals_class_0)
        metrics_results_valid["pval_class_0"] = valid_folds_pvals_class_0[~np.isnan(valid_folds_pvals_class_0)]
        valid_folds_pvals_class_1 = np.array(valid_folds_pvals_class_1)
        metrics_results_valid["pval_class_1"] = valid_folds_pvals_class_1[~np.isnan(valid_folds_pvals_class_1)]

        metrics_results_test["ap_class_0"] = test_folds_ap_class_0
        metrics_results_test["ap_class_1"] = test_folds_ap_class_1
        metrics_results_test["ppv"] = test_folds_ppv
        metrics_results_test["fpr"] = test_folds_fpr
        metrics_results_test["tpr"] = test_folds_tpr
        metrics_results_test["roc_auc_class_0"] = test_folds_roc_auc_class_0
        metrics_results_test["roc_auc_class_1"] = test_folds_roc_auc_class_1
        test_folds_pvals_class_0 = np.array(test_folds_pvals_class_0)
        metrics_results_test["pval_class_0"] = test_folds_pvals_class_0[~np.isnan(test_folds_pvals_class_0)]
        test_folds_pvals_class_1 = np.array(test_folds_pvals_class_1)
        metrics_results_test["pval_class_1"] = test_folds_pvals_class_1[~np.isnan(test_folds_pvals_class_1)]

        
        
        metrics_results_train_species["ap_class_0"] = train_species_folds_ap_class_0
        metrics_results_train_species["ap_class_1"] = train_species_folds_ap_class_1
        metrics_results_train_species["ppv"] = train_species_folds_ppv
        metrics_results_train_species["fpr"] = train_species_folds_fpr
        metrics_results_train_species["tpr"] = train_species_folds_tpr
        metrics_results_train_species["roc_auc_class_0"] = train_species_folds_roc_auc_class_0
        metrics_results_train_species["roc_auc_class_1"] = train_species_folds_roc_auc_class_1
        train_species_folds_pvals_class_0 = np.array(train_species_folds_pvals_class_0)
        metrics_results_train_species["pval_class_0"] = train_species_folds_pvals_class_0[~np.isnan(train_species_folds_pvals_class_0)]
        train_species_folds_pvals_class_1 = np.array(train_species_folds_pvals_class_1)
        metrics_results_train_species["pval_class_1"] = train_species_folds_pvals_class_1[~np.isnan(train_species_folds_pvals_class_1)]

        
        metrics_results_valid_species["ap_class_0"] = valid_species_folds_ap_class_0
        metrics_results_valid_species["ap_class_1"] = valid_species_folds_ap_class_1
        metrics_results_valid_species["ppv"] = valid_species_folds_ppv
        metrics_results_valid_species["fpr"] = valid_species_folds_fpr
        metrics_results_valid_species["tpr"] = valid_species_folds_tpr
        metrics_results_valid_species["roc_auc_class_0"] = valid_species_folds_roc_auc_class_0
        metrics_results_valid_species["roc_auc_class_1"] = valid_species_folds_roc_auc_class_1
        valid_species_folds_pvals_class_0 = np.array(valid_species_folds_pvals_class_0)
        metrics_results_valid_species["pval_class_0"] = valid_species_folds_pvals_class_0[~np.isnan(valid_species_folds_pvals_class_0)]
        valid_species_folds_pvals_class_1 = np.array(valid_species_folds_pvals_class_1)
        metrics_results_valid_species["pval_class_1"] = valid_species_folds_pvals_class_1[~np.isnan(valid_species_folds_pvals_class_1)]

        
        metrics_results_test_species["ap_class_0"] = test_species_folds_ap_class_0
        metrics_results_test_species["ap_class_1"] = test_species_folds_ap_class_1
        metrics_results_test_species["ppv"] = test_species_folds_ppv
        metrics_results_test_species["fpr"] = test_species_folds_fpr
        metrics_results_test_species["tpr"] = test_species_folds_tpr
        metrics_results_test_species["roc_auc_class_0"] = test_species_folds_roc_auc_class_0
        metrics_results_test_species["roc_auc_class_1"] = test_species_folds_roc_auc_class_1
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
        "test-auc-1": test_folds_roc_auc_class_1,

            }

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
                    np.array(metrics_results_test["ap_class_1"]))) / 2, 2)) if metrics_results_test[
                                                                                     "ap_class_0"] is not None else metrics_results_test["ap_class_0"]
            
                                                                                     
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
        dfi.export(df_styled, '{}/{}/metrics_comparison_{}.png'.format(script_dir, results_folder,title), max_cols=-1,max_rows=-1)


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
                umap_proj_1d = reducer.fit_transform(latent_space[:, 5:]).squeeze(-1)
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

            isoelectric_scores = np.array(
                list(map(lambda seq: VegvisirUtils.calculate_isoelectric(seq), sequences_list)))
            aromaticity_scores = np.array(
                list(map(lambda seq: VegvisirUtils.calculate_aromaticity(seq), sequences_list)))
            gravy_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_gravy(seq), sequences_list)))
            molecular_weight_scores = np.array(
                list(map(lambda seq: VegvisirUtils.calculate_molecular_weight(seq), sequences_list)))
            extintion_coefficient_reduced_scores = np.array(
                list(map(lambda seq: VegvisirUtils.calculate_extintioncoefficient(seq)[0], sequences_list)))
            extintion_coefficient_cystines_scores = np.array(
                list(map(lambda seq: VegvisirUtils.calculate_extintioncoefficient(seq)[1], sequences_list)))

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
                                                            plot_covariance=True,
                                                            save_plot=False,
                                                            filter_correlations=False)

            if mode == "train":
                covariances_dict_train[learning_type][name]["covariance"].append(np.abs(correlations_dict["features_covariance"])) #make positive because umap vector can be orientated anywhere
                covariances_dict_train[learning_type][name]["features_names"].append(correlations_dict["features_names"])
                covariances_dict_train[learning_type][name]["pearson_coefficients"].append(np.abs(correlations_dict["pearson_coefficients"]))
                covariances_dict_train[learning_type][name]["pearson_pvalues"].append(correlations_dict["pearson_pvalues"])
                covariances_dict_train[learning_type][name]["umap_1d"].append(umap_proj_1d)
            elif mode == "valid":
                covariances_dict_valid[learning_type][name]["covariance"].append(np.abs(correlations_dict["features_covariance"]))
                covariances_dict_valid[learning_type][name]["features_names"].append(correlations_dict["features_names"])
                covariances_dict_valid[learning_type][name]["pearson_coefficients"].append(np.abs(correlations_dict["pearson_coefficients"]))
                covariances_dict_valid[learning_type][name]["pearson_pvalues"].append(correlations_dict["pearson_pvalues"])
                covariances_dict_valid[learning_type][name]["umap_1d"].append(umap_proj_1d)
            elif mode == "test":
                covariances_dict_test[learning_type][name]["covariance"].append(np.abs(correlations_dict["features_covariance"]))
                covariances_dict_test[learning_type][name]["features_names"].append(correlations_dict["features_names"])
                covariances_dict_test[learning_type][name]["pearson_coefficients"].append(np.abs(correlations_dict["pearson_coefficients"]))
                covariances_dict_test[learning_type][name]["pearson_pvalues"].append(correlations_dict["pearson_pvalues"])
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
                          "extintion_coefficients_reduced":"Extintion coefficients (reduced)",
                          "extintion_coefficients_cystines":"Extintion coefficients (oxidized)",
                          "immunodominance_scores":"Immunodominance",
                          "binary_targets":"Binary targets"}
     covariances_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     pearson_coefficients_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))
     pearson_pvalues_all = defaultdict(lambda: defaultdict(lambda: defaultdict()))

     covariances_dict_train= defaultdict(lambda :defaultdict(lambda : defaultdict(lambda : [])))
     covariances_dict_valid= defaultdict(lambda :defaultdict(lambda : defaultdict(lambda : [])))
     covariances_dict_test= defaultdict(lambda :defaultdict(lambda : defaultdict(lambda : [])))
     reducer = umap.UMAP(n_components=1)
     tuples_idx = []
     plot_all = False
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


             if covariances_dict_train[learning_type][name]["features_names"]: #UMAP-1D
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

             # covariances_all_latex[learning_type][name]["train"] = np.mean(covariances_dict_train[learning_type][name]["covariance"]) +"$\pm$" + np.mean(covariances_dict_train[learning_type][name]["covariance"])
             # covariances_all_latex[learning_type][name]["valid"] = np.mean(covariances_dict_valid[learning_type][name]["covariance"]) +"$\pm$" + np.mean(covariances_dict_valid[learning_type][name]["covariance"])
             # covariances_all_latex[learning_type][name]["test"] = np.mean(covariances_dict_test[learning_type][name]["covariance"]) +"$\pm$" + np.mean(covariances_dict_test[learning_type][name]["covariance"])



     print("Finished loop, building dataframe")

     # df_latex = convert_dict(covariances_all_latex)
     # df_latex.style.format(na_rep="-").to_latex('{}/{}/methods_comparison_LATEX.tex'.format(script_dir, results_folder))

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
         df_styled = df.style.format(na_rep="-", escape="latex",precision=3).background_gradient(axis=None,cmap="YlOrBr").set_table_styles(css)  # TODO: Switch to escape="latex-math" when pandas 2.1 arrives

         dfi.export(df_styled, '{}/{}/{}_DATAFRAME_{}.png'.format(script_dir, results_folder,title,subtitle), max_cols=-1,max_rows=-1)


     process_dict(covariances_all,"Latent_covariances",subtitle)
     process_dict(pearson_pvalues_all,"Latent_pearson_pvalues",subtitle)
     process_dict(pearson_coefficients_all,"Latent_pearson_coefficients",subtitle)

def plot_benchmarking_results(dict_results_vegvisir,script_dir,folder="Benchmark"):
    """"""

    #Highlight: vegvisir results

    metrics_keys = ["ppv","fpr", "tpr", "roc_auc_class_0", "roc_auc_class_1","pval_class_0","pval_class_1"]

    metrics_results_dict = plot_kfold_comparison_helper(metrics_keys, script_dir, folder=dict_results_vegvisir["viral-dataset9-likelihood-40"], overwrite=False, kfolds=5)

    metrics_results_train = metrics_results_dict["train"]
    #metrics_results_valid = metrics_results_dict["valid"]
    metrics_results_test = metrics_results_dict["test"]

    vegvisir_results_auc_train = {"Vegvisir":np.round((np.mean(np.array(metrics_results_train["roc_auc_class_0"])) + np.mean(np.array(metrics_results_train["roc_auc_class_1"])))/2, 2)}
    vegvisir_results_ppv_train = {"Vegvisir":np.round(np.mean(np.array(metrics_results_train["ppv"])), 2)}
    vegvisir_results_auc_test = {"Vegvisir":np.round((np.mean(np.array(metrics_results_test["roc_auc_class_0"])) + np.mean(np.array(metrics_results_test["roc_auc_class_1"])))/2, 2)}
    vegvisir_results_ppv_test = {"Vegvisir":np.round(np.mean(np.array(metrics_results_test["ppv"])), 2)}


    #Highlight: NNalign results
    nnalign_results_path = "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Other_Programs/report_models"

    nnalign_results = pd.read_csv(nnalign_results_path,sep="\s+")
    nnalign_results = nnalign_results.groupby('DATASET', as_index=False)[["AUC01_train", "AUC_train","PPV_train","AUC01_eval","AUC_eval","PPV_eval"]].agg(lambda x: sum(list(x))/len(list(x)))
    nnalign_results =nnalign_results[nnalign_results["DATASET"]=="sequences_viral_dataset9"]
    nnalign_results_train_auc_dict = {"NNAlign2.1":nnalign_results["AUC_train"].item()}
    nnalign_results_train_ppv_dict = {"NNAlign2.1":nnalign_results["PPV_train"].item()}
    nnalign_results_test_auc_dict = {"NNAlign2.1":nnalign_results["AUC_eval"].item()}
    nnalign_results_test_ppv_dict = {"NNAlign2.1":nnalign_results["PPV_eval"].item()}


    other_programs_results_path = "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Other_Programs/sequences_viral_dataset9_predictors.tsv"

    other_programs_results = pd.read_csv(other_programs_results_path,sep="\t")

    pd.set_option('display.max_columns', None)

    def process_results(training=True):

        other_programs_results_mode= other_programs_results[other_programs_results["training"] == training]
        other_programs_results_mode = other_programs_results_mode[["Icore","target_corrected","PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno","DeepNetBim_immuno_probability"]]
        #Highlight: Grouping by alleles
        #a) Group non categorical outputs ("continuous")
        other_programs_results_mode_continuous = other_programs_results_mode.groupby("Icore",as_index=False)[["PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno_probability"]].agg(lambda x: sum(list(x))/len(list(x)))
        #b) Process categorical outputs ("discrete")
        other_programs_results_mode_categorical = other_programs_results_mode.groupby("Icore",as_index=False)[["target_corrected","DeepNetBim_immuno"]].agg(pd.Series.mode)

        other_programs_results_mode = other_programs_results_mode_continuous.merge(other_programs_results_mode_categorical,on=["Icore"],how="left")
        return other_programs_results_mode

    other_programs_results_train = process_results(training=True)
    other_programs_results_test = process_results(training=False)


    programs_list = ["PRIME_rank","PRIME_score","MixMHCpred_rank_binding","IEDB_immuno","DeepImmuno","DeepNetBim_binding","DeepNetBim_immuno","DeepNetBim_immuno_probability"]


    def calculate_auc(targets,predictions):
        try:
            targets = targets.to_numpy()
            predictions = predictions.to_numpy()
            predictions_nan = np.isnan(predictions)
            targets = targets[~predictions_nan]
            predictions = predictions[~predictions_nan]
            fpr, tpr, _ = roc_curve(targets, predictions)
            roc_auc = auc(fpr, tpr)
            return roc_auc
        except:
            return np.nan



    def calculate_ppv(targets,predictions):
        targets = targets.to_numpy()
        predictions = predictions.to_numpy()
        try:
            try:
                binary_predictions = np.where(predictions >=0.5 , 1, 0) #for the rank this is useless, but I want to preserve the error
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

    ppv_results_train = list(map(lambda program: calculate_ppv(other_programs_results_train["target_corrected"],other_programs_results_train[program]),programs_list ))
    ppv_results_test = list(map(lambda program: calculate_ppv(other_programs_results_test["target_corrected"],other_programs_results_test[program]),programs_list ))

    ppv_results_train_dict = dict(zip(programs_list,ppv_results_train))
    ppv_results_test_dict = dict(zip(programs_list,ppv_results_test))


    auc_results_train = list(map(lambda program: calculate_auc(other_programs_results_train["target_corrected"],other_programs_results_train[program]),programs_list ))
    auc_results_test = list(map(lambda program: calculate_auc(other_programs_results_test["target_corrected"],other_programs_results_test[program]),programs_list ))

    auc_results_train_dict = dict(zip(programs_list,auc_results_train))
    auc_results_test_dict = dict(zip(programs_list,auc_results_test))

    ppv_results_train_dict = dict(zip(programs_list, ppv_results_train_dict))
    ppv_results_test_dict = dict(zip(programs_list, ppv_results_test_dict))


    auc_results_train_dict = {**vegvisir_results_auc_train,**nnalign_results_train_auc_dict,**auc_results_train_dict,}
    auc_results_test_dict = {**vegvisir_results_auc_test,**nnalign_results_test_auc_dict,**auc_results_test_dict}


    ppv_results_train_dict = {**vegvisir_results_ppv_train,**nnalign_results_train_ppv_dict,**ppv_results_train_dict,}
    ppv_results_test_dict = {**vegvisir_results_ppv_test,**nnalign_results_test_ppv_dict,**ppv_results_test_dict}


    names_dict = {"Vegvisir":"Vegvisir",
                  "NNAlign2.1":"NNAlign2.1",
                  "PRIME_rank":"PRIME \n (MHC binding rank)",
                  "PRIME_score":"PRIME \n (probability)",
                  "MixMHCpred_rank_binding":"MixMHCpred \n (MHC binding rank)",
                  "IEDB_immuno":"IEDB immuno",
                  "DeepImmuno":"DeepImmuno",
                  "DeepNetBim_binding":"DeepNetBim \n (MHC binding rank)",
                  "DeepNetBim_immuno":"DeepNetBim \n (binary prediction)",
                  "DeepNetBim_immuno_probability":"DeepNetBim \n (probability)"}

    plot_only_auc = True
    if plot_only_auc:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 18))
        i = 0
        positions = []
        labels = []
        for (program_train, auc_train), auc_test, ppv_train, ppv_test in zip(auc_results_train_dict.items(),
                                                                             auc_results_test_dict.values(),
                                                                             ppv_results_train_dict.values(),
                                                                             ppv_results_test_dict.values()):

            if np.isnan(auc_train) or np.isnan(auc_test):
                pass
            else:
                bar_train_auc = ax1.barh(i, width=auc_train, color="skyblue", height=0.2)
                bar_test_auc = ax1.barh(i + 0.2, width=auc_test, height=0.2, color="tomato")
                positions.append(i)
                i += 1
                labels.append(names_dict[program_train])
                for bar in [bar_train_auc.patches, bar_test_auc.patches]:
                    rect = bar[0]  # single rectangle
                    ax1.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(), 2)),
                             ha='center', va='center', fontsize=12)


        ax1.set_yticks(positions, labels=labels, fontsize=15)
        ax1.tick_params(axis='x', labelsize=15)

        ax1.axvline(x=0.5, color='goldenrod', linestyle='--')
        transformation = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
        ax1.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center", transform=transformation, fontsize=15)
        ax1.set_title("ROC-AUC", fontsize=18)
        ax1.margins(x=0.2)

        plt.subplots_adjust(left=0.2, wspace=0.4)

    else:

        fig,[ax1,ax2] = plt.subplots(nrows=1,ncols=2,figsize=(20,18))
        i= 0
        positions = []
        labels = []
        for (program_train,auc_train),auc_test,ppv_train,ppv_test in zip(auc_results_train_dict.items(),auc_results_test_dict.values(),ppv_results_train_dict.values(),ppv_results_test_dict.values()):

            if np.isnan(auc_train) or np.isnan(auc_test):
                pass
            else:
                bar_train_auc= ax1.barh(i,width=auc_train,color="skyblue",height=0.2)
                #bar_train_ppv= ax2.barh(i,width=ppv_train,color="skyblue",height=0.2)
                bar_test_auc = ax1.barh(i + 0.2,width=auc_test,height=0.2,color="tomato")
                #bar_test_ppv = ax2.barh(i + 0.2,width=ppv_test,height=0.2,color="tomato")
                positions.append(i)
                i += 1
                labels.append(names_dict[program_train])
                for bar in [bar_train_auc.patches,bar_test_auc.patches]:
                    rect = bar[0] #single rectangle
                    ax1.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                             '{}'.format(round(rect.get_width(),2)),
                             ha='center', va='center',fontsize=12)
                # for bar in [bar_train_ppv.patches,bar_test_ppv.patches]:
                #     rect = bar[0] #single rectangle
                #     ax2.text(1.05 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                #              '{}'.format(round(rect.get_width(),2)),
                #              ha='center', va='center',fontsize=12)

        ax1.set_yticks(positions,labels=labels,fontsize=15)
        ax1.tick_params(axis='x', labelsize=15)

        ax1.axvline(x=0.5, color='goldenrod', linestyle='--')
        transformation = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
        ax1.text(0.5, -0.30, "0.5", color="dimgrey", ha="right", va="center",transform=transformation,fontsize=15)
        ax1.set_title("ROC-AUC",fontsize=18)
        ax1.margins(x=0.2)


        ax2.set_yticks(positions,labels=labels,fontsize=15)
        ax2.set_title("Precision(PPV)",fontsize=18)
        ax2.margins(x=0.2)

        plt.subplots_adjust(left=0.2,wspace=0.4)

    colors_dict = {"Train":"skyblue","Test":"tomato"}
    legends = [mpatches.Patch(color=color, label='{}'.format(label)) for label, color in colors_dict.items()]
    fig.legend(handles=legends, prop={'size': 12}, loc='center right', bbox_to_anchor=(0.9, 0.5))
    fig.suptitle("Benchmarking",fontsize=20)


    plt.savefig("{}/{}/Benchmarking".format(script_dir,folder),dpi=600)









