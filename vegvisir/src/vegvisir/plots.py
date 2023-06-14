"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import json
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import  matplotlib
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
MAX_WORKERs = ( multiprocessing. cpu_count() - 1 )
plt.style.use('ggplot')
colors_dict = {0: "green", 1: "red",2:"navajowhite"}
colors_cluster_dict = {0: "seagreen", 1: "crimson",2:"gold",3:"mediumslateblue"}
colors_dict_labels = {0: "mediumaquamarine", 1: "orangered",2:"navajowhite"}
labels_dict = {0:"negative",1:"positive",2:"unobserved"}
colors_list_aa = ["black", "plum", "lime", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange",
               "yellow", "green",
               "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal",
               "goldenrod", "chocolate", "cornflowerblue", "pink", "darkgrey", "indianred",
               "mediumspringgreen"]


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
        group_counts = dict( sorted(group_counts.items(), key=operator.itemgetter(1),reverse=True)) #sort dict by values
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

def plot_clusters_features_distributions(dataset_info,cluster_assignments,n_clusters,predictions_dict,sample_mode,results_dir,method):
    """
    Notes:
        - https://www.researchgate.net/publication/15556561_Global_Fold_Determination_from_a_Small_Number_of_Distance_Restraints/figures?lo=1
        - Radius table: https://www.researchgate.net/publication/15556561_Global_Fold_Determination_from_a_Small_Number_of_Distance_Restraints/figures?lo=1
    """
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository

    aminoacid_properties = pd.read_csv("{}/aminoacid_properties.txt".format(storage_folder),sep = "\s+")
    hydropathy_dict= dict(zip(aminoacid_properties["1letter"].values.tolist(),aminoacid_properties["Hydropathy_index"].values.tolist()))
    hydropathy_dict["#"] =0
    volume_dict= dict(zip(aminoacid_properties["1letter"].values.tolist(),aminoacid_properties["Volume(A3)"].values.tolist()))
    volume_dict["#"] =0
    radius_dict= dict(zip(aminoacid_properties["1letter"].values.tolist(),aminoacid_properties["Radius"].values.tolist()))
    radius_dict["#"] =0
    side_chain_pka_dict= dict(zip(aminoacid_properties["1letter"].values.tolist(),aminoacid_properties["side_chain_pka"].values.tolist()))
    side_chain_pka_dict["#"] =0
    isoelectric_dict= dict(zip(aminoacid_properties["1letter"].values.tolist(),aminoacid_properties["isoelectric_point"].values.tolist()))
    isoelectric_dict["#"] =0
    if dataset_info.corrected_aa_types == 20:
        aminoacids_dict = VegvisirUtils.aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=[])
    else:
        aminoacids_dict = VegvisirUtils.aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=["#"])

    aminoacids_dict_reversed = {val:key for key,val in aminoacids_dict.items()}
    hydropathy_dict = {aminoacids_dict[key]:value for key,value in hydropathy_dict.items()}
    volume_dict = {aminoacids_dict[key]:value for key,value in volume_dict.items()}
    radius_dict = {aminoacids_dict[key]:value for key,value in radius_dict.items()}
    side_chain_pka_dict = {aminoacids_dict[key]:value for key,value in side_chain_pka_dict.items()}
    isoelectric_dict = {aminoacids_dict[key]:value for key,value in isoelectric_dict.items()}


    data_int = predictions_dict["data_int_{}".format(sample_mode)]
    sequences = data_int[:,1:].squeeze(1)
    sequences_mask = np.array((sequences == 0))
    sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(sequences)
    sequences_list = sequences_raw.tolist()

    # hydropathy_scores = np.vectorize(hydropathy_dict.get)(sequences)
    # hydropathy_scores = np.ma.masked_array(hydropathy_scores, mask=sequences_mask, fill_value=0)
    # hydropathy_scores = np.ma.mean(hydropathy_scores, axis=1)
    
    volume_scores = np.vectorize(volume_dict.get)(sequences)
    volume_scores = np.ma.masked_array(volume_scores, mask=sequences_mask, fill_value=0)
    volume_scores = np.ma.sum(volume_scores, axis=1)
    
    radius_scores = np.vectorize(radius_dict.get)(sequences)
    radius_scores = np.ma.masked_array(radius_scores, mask=sequences_mask, fill_value=0)
    radius_scores = np.ma.sum(radius_scores, axis=1)

    side_chain_pka_scores = np.vectorize(side_chain_pka_dict.get)(sequences)
    side_chain_pka_scores = np.ma.masked_array(side_chain_pka_scores, mask=sequences_mask, fill_value=0)
    side_chain_pka_scores = np.ma.sum(side_chain_pka_scores, axis=1)

    isoelectric_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_isoelectric(seq), sequences_list)))
    aromaticity_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_aromaticity(seq), sequences_list)))
    hydropathy_scores = np.array(list(map(lambda seq: VegvisirUtils.calculate_hydropathy(seq), sequences_list)))


    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2, 3,figsize=(22,22))

    clusters_info = defaultdict(lambda : defaultdict(lambda : defaultdict()))
    i = 0
    labels = []

    all_hydropathy = []
    all_volumes = []
    all_radius = []
    all_isoelectric = []
    all_aromaticity = []
    all_side_chain_pka = []
    all_colors = []
    label_locations = []
    for cluster in range(n_clusters):
        idx_cluster = (cluster_assignments[..., None] == cluster).any(-1)
        data_int_cluster = data_int[idx_cluster]
        idx_observed = (data_int_cluster[:, 0, 0][..., None] != 2).any(-1)
        for mode,idx in zip(["observed","unobserved"],[idx_observed,np.invert(idx_observed)]):
            sequences_cluster = data_int_cluster[idx][:, 1:].squeeze(1)
            sequences_raw_cluster = np.vectorize(aminoacids_dict_reversed.get)(sequences)
            sequences_raw_list = sequences_raw.tolist()
            if sequences_cluster.size != 0:
                sequences_mask = np.array((sequences_cluster == 0))
                # hydropathy = np.vectorize(hydropathy_dict.get)(sequences_cluster)
                # hydropathy = np.ma.masked_array(hydropathy,mask=sequences_mask,fill_value=0)
                # hydropathy = np.ma.sum(hydropathy,axis=1)
                hydropathy = np.array(list(map(lambda seq: VegvisirUtils.calculate_hydropathy(seq), sequences_raw_list)))
                aromaticity = np.array(list(map(lambda seq: VegvisirUtils.calculate_aromaticity(seq), sequences_raw_list)))

                volumes = np.vectorize(volume_dict.get)(sequences_cluster)
                volumes = np.ma.masked_array(volumes,mask=sequences_mask,fill_value=0)
                volumes = np.ma.sum(volumes,axis=1)
                radius = np.vectorize(radius_dict.get)(sequences_cluster)
                radius = np.ma.masked_array(radius,mask=sequences_mask,fill_value=0)
                radius = np.ma.sum(radius,axis=1)
                side_chain_pka = np.vectorize(side_chain_pka_dict.get)(sequences_cluster)
                side_chain_pka = np.ma.masked_array(side_chain_pka,mask=sequences_mask,fill_value=0)
                side_chain_pka = np.ma.sum(side_chain_pka,axis=1)

                # isoelectric = np.vectorize(isoelectric_dict.get)(sequences_cluster)
                # isoelectric = np.ma.masked_array(isoelectric,mask=sequences_mask,fill_value=0)
                # isoelectric = np.ma.sum(isoelectric,axis=1)
                isoelectric = np.array(list(map(lambda seq: VegvisirUtils.calculate_isoelectric(seq), sequences_list)))

                clusters_info["Cluster_{}".format(cluster)][mode]["hydrophobicity"] = hydropathy.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["volumes"] = volumes.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["radius"] = radius.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["side_chain_pka"] = side_chain_pka.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["isoelectric"] = isoelectric.mean()
                clusters_info["Cluster_{}".format(cluster)][mode]["aromaticity"] = aromaticity.mean()

                all_side_chain_pka.append(side_chain_pka)
                all_isoelectric.append(isoelectric)
                all_hydropathy.append(hydropathy)
                all_volumes.append(volumes)
                all_radius.append(radius)
                all_aromaticity.append(aromaticity)
                all_colors.append(colors_cluster_dict[cluster])
                labels.append("Cluster {}, \n {}".format(cluster,mode))
                label_locations.append(i + 0.2)
                i+=0.4

    boxplot0 = ax0.boxplot( all_hydropathy,
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
    boxplot5 = ax4.boxplot( all_aromaticity,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels
                     )  # olor=colors_cluster_dict[cluster])  # color=colors_cluster_dict[cluster]


    # fill with colors
    #colors = ['pink', 'lightblue', 'lightgreen',"gold"]
    for bplot in (boxplot0, boxplot1,boxplot2,boxplot3,boxplot4,boxplot5):
        for patch, color in zip(bplot['boxes'], all_colors):
            patch.set_facecolor(color)

    ax0.set_title("Hydrophobicity")
    ax1.set_title("Volumes")
    ax2.set_title("Radius")
    ax3.set_title("Isoelectric")
    ax4.set_title("Side chain PKA")
    ax5.set_title("Aromaticity")



    ax0.set_xticklabels(rotation=90,labels=labels)
    ax1.set_xticklabels(rotation=90,labels=labels)
    ax2.set_xticklabels(rotation=90,labels=labels)
    ax3.set_xticklabels(rotation=90,labels=labels)
    # ax4.set_xticklabels(rotation=90,labels=labels)
    # ax5.set_xticklabels(rotation=90,labels=labels)



    #ax1.set_xticks(label_locations)
    #ax1.set_xticklabels(labels=labels1,rotation=45,fontsize=8)
    #plt.legend(handles=[negative_patch,positive_patch], prop={'size': 20},loc= 'center right',bbox_to_anchor=(1,0.5),ncol=1)
    plt.savefig("{}/{}/clusters_features_{}".format(results_dir,method,sample_mode))
    plt.clf()
    plt.close(fig)

    return hydropathy_scores,volume_scores,radius_scores,side_chain_pka_scores,isoelectric_scores,aromaticity_scores,clusters_info

def plot_latent_space(dataset_info,latent_space,predictions_dict,sample_mode,results_dir,method,vector_name="latent_space_z",n_clusters=4):

    print("Plotting UMAP of {}...".format(vector_name))
    reducer = umap.UMAP()
    umap_proj = reducer.fit_transform(latent_space[:, 5:])

    if vector_name == "latent_space_z":
        cluster_assignments = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=100, max_iter=10, reassignment_ratio=0).fit_predict(umap_proj)
        colors_clusters = np.vectorize(colors_cluster_dict.get)(cluster_assignments)

        hydropathy_scores,volume_scores,radius_scores,side_chain_pka_scores,isoelectric_scores,aromaticity_scores,clusters_info=plot_clusters_features_distributions(dataset_info,cluster_assignments,n_clusters,predictions_dict,sample_mode,results_dir,method)

        #Highlight: Hydropathy
        hydropathy_scores_unique = np.unique(hydropathy_scores)
        hydropathy_scores, hydropathy_scores_unique = VegvisirUtils.replace_nan(hydropathy_scores,hydropathy_scores_unique)
        colormap_hydropathy = matplotlib.cm.get_cmap('viridis', len(hydropathy_scores_unique.tolist()))
        colors_dict = dict(zip(hydropathy_scores_unique, colormap_hydropathy.colors))
        colors_hydropathy = np.vectorize(colors_dict.get, signature='()->(n)')(hydropathy_scores)
        # Highlight: Peptide volumes
        volume_scores_unique = np.unique(volume_scores)
        volume_scores, volume_scores_unique = VegvisirUtils.replace_nan(volume_scores,volume_scores_unique)
        colormap_volume = matplotlib.cm.get_cmap('magma', len(volume_scores_unique.tolist()))
        colors_dict = dict(zip(volume_scores_unique, colormap_volume.colors))
        colors_volume = np.vectorize(colors_dict.get, signature='()->(n)')(volume_scores)
        #Highlight: Radius
        radius_scores_unique = np.unique(radius_scores)
        radius_scores, radius_scores_unique = VegvisirUtils.replace_nan(radius_scores,radius_scores_unique)
        colormap_radius = matplotlib.cm.get_cmap('magma', len(radius_scores_unique.tolist()))
        colors_dict = dict(zip(radius_scores_unique, colormap_radius.colors))
        colors_radius = np.vectorize(colors_dict.get, signature='()->(n)')(radius_scores)

        #Highlight: Side chain Pka
        side_chain_pka_scores_unique = np.unique(side_chain_pka_scores)
        side_chain_pka_scores, side_chain_pka_scores_unique = VegvisirUtils.replace_nan(side_chain_pka_scores,side_chain_pka_scores_unique)
        colormap_side_chain_pka = matplotlib.cm.get_cmap('magma', len(side_chain_pka_scores_unique.tolist()))
        colors_dict = dict(zip(side_chain_pka_scores_unique, colormap_side_chain_pka.colors))
        colors_side_chain_pka = np.vectorize(colors_dict.get, signature='()->(n)')(side_chain_pka_scores)

        #Highlight: Isoelectric scores
        isoelectric_scores_unique = np.unique(isoelectric_scores)
        isoelectric_scores, isoelectric_scores_unique = VegvisirUtils.replace_nan(isoelectric_scores,isoelectric_scores_unique)
        colormap_isoelectric = matplotlib.cm.get_cmap('magma', len(isoelectric_scores_unique.tolist()))
        colors_dict = dict(zip(isoelectric_scores_unique, colormap_isoelectric.colors))
        colors_isoelectric = np.vectorize(colors_dict.get, signature='()->(n)')(isoelectric_scores)
        
        #Highlight: aromaticity scores
        aromaticity_scores_unique = np.unique(aromaticity_scores)
        aromaticity_scores, aromaticity_scores_unique = VegvisirUtils.replace_nan(aromaticity_scores,aromaticity_scores_unique)
        colormap_aromaticity = matplotlib.cm.get_cmap('cividis', len(aromaticity_scores_unique.tolist()))
        colors_dict = dict(zip(aromaticity_scores_unique, colormap_aromaticity.colors))
        colors_aromaticity = np.vectorize(colors_dict.get, signature='()->(n)')(aromaticity_scores)

    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:,0])
    if method == "_single_sample":
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_dict["class_binary_prediction_single_sample"])
    else:
        colors_predicted_binary = np.vectorize(colors_dict_labels.get)(predictions_dict["class_binary_predictions_samples_mode"])

    #Highlight: Confidence scores colors
    confidence_scores = latent_space[:,4]
    confidence_scores_unique = np.unique(confidence_scores).tolist()
    colormap_confidence = matplotlib.cm.get_cmap('plasma_r', len(confidence_scores_unique))
    colors_dict = dict(zip(confidence_scores_unique, colormap_confidence.colors))
    colors_confidence = np.vectorize(colors_dict.get, signature='()->(n)')(confidence_scores)
    #Highlight: Immunodominance scores colors
    immunodominance_scores = latent_space[:,3]
    immunodominance_scores_unique = np.unique(immunodominance_scores)
    immunodominance_scores,immunodominance_scores_unique = VegvisirUtils.replace_nan(immunodominance_scores,immunodominance_scores_unique)
    colormap_immunodominance = matplotlib.cm.get_cmap('plasma_r', len(immunodominance_scores_unique.tolist()))
    colors_dict = dict(zip(immunodominance_scores_unique, colormap_immunodominance.colors))
    colors_immunodominance = np.vectorize(colors_dict.get, signature='()->(n)')(immunodominance_scores)
    #Highlight: Frequency scores per class: https://stackoverflow.com/questions/65927253/linearsegmentedcolormap-to-list
    frequency_class0_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:,0]).tolist()
    colormap_frequency_class0 = matplotlib.cm.get_cmap('BuGn', len(frequency_class0_unique)) #This one is  a LinearSegmentedColor map and works slightly different
    colormap_frequency_class0_array =  np.array([colormap_frequency_class0(i) for i in range(colormap_frequency_class0.N)])
    colors_dict = dict(zip(frequency_class0_unique, colormap_frequency_class0_array))
    colors_frequency_class0 = np.vectorize(colors_dict.get, signature='()->(n)')(predictions_dict["class_binary_prediction_samples_frequencies"][:,0])
    frequency_class1_unique = np.unique(predictions_dict["class_binary_prediction_samples_frequencies"][:,1]).tolist()
    colormap_frequency_class1 = matplotlib.cm.get_cmap('OrRd', len(frequency_class1_unique))
    colormap_frequency_class1_array =  np.array([colormap_frequency_class1(i) for i in range(colormap_frequency_class1.N)])
    colors_dict = dict(zip(frequency_class1_unique, colormap_frequency_class1_array))
    colors_frequency_class1 = np.vectorize(colors_dict.get, signature='()->(n)')(predictions_dict["class_binary_prediction_samples_frequencies"][:,1])
    alpha = 0.7
    fig, [[ax1, ax2, ax3,ax4],[ax5,ax6,ax7,ax8],[ax9,ax10,ax11,ax12],[ax13,ax14,ax15,ax16]] = plt.subplots(4, 4,figsize=(17, 12),gridspec_kw={'width_ratios': [4.5,4.5,4.5,1],'height_ratios': [4,4,4,4]})
    fig.suptitle('UMAP projections',fontsize=20)
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=latent_space[:,2], alpha=alpha,s=30)
    ax1.set_title("True labels",fontsize=20)
    if method == "_single_sample":
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted_binary, alpha=alpha,s=30)
        ax2.set_title("Predicted labels \n (single sample)",fontsize=20)
    else:
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted_binary, alpha=alpha,s=30)
        ax2.set_title("Predicted binary labels \n (samples mode)",fontsize=20)

    ax3.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_confidence, alpha=alpha, s=30)
    ax3.set_title("Confidence scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence),ax=ax3)
    ax5.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_frequency_class0, alpha=alpha, s=30)
    ax5.set_title("Probability class 0 \n (frequency argmax)", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable( norm = Normalize(0,1),cmap=colormap_frequency_class0),ax=ax5)
    ax6.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_frequency_class1, alpha=alpha, s=30)
    ax6.set_title("Probability class 1 \n (frequency argmax)", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable( cmap=colormap_frequency_class1),ax=ax6)
    ax7.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_immunodominance, alpha=alpha, s=30)
    ax7.set_title("Immunodominance scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_immunodominance),ax=ax7)
    if vector_name == "latent_space_z":

        ax9.scatter(umap_proj[:, 0],umap_proj[:, 1], c=colors_clusters, alpha=alpha,s=30)
        ax9.set_title("Coloured by Kmeans cluster")

        ax10.scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors_hydropathy, alpha=alpha, s=30)
        ax10.set_title("Coloured by Hydrophobicity")
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_hydropathy,norm=Normalize(vmin=np.min(hydropathy_scores_unique),vmax=np.max(hydropathy_scores_unique))), ax=ax10)

        ax11.scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors_volume, alpha=alpha, s=30)
        ax11.set_title("Coloured by Peptide volume")
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_volume,norm=Normalize(vmin=np.min(volume_scores_unique),vmax=np.max(volume_scores_unique))), ax=ax11)

        ax13.scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors_side_chain_pka, alpha=alpha, s=30)
        ax13.set_title("Coloured by Side chain pka")
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_volume,norm=Normalize(vmin=np.min(side_chain_pka_scores_unique),vmax=np.max(side_chain_pka_scores_unique))), ax=ax13)

        ax14.scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors_isoelectric, alpha=alpha, s=30)
        ax14.set_title("Coloured by Isoelectric point")
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_isoelectric,norm=Normalize(vmin=np.min(isoelectric_scores_unique),vmax=np.max(isoelectric_scores_unique))), ax=ax14)

        ax15.scatter(umap_proj[:, 0], umap_proj[:, 1], c=colors_aromaticity, alpha=alpha, s=30)
        ax15.set_title("Coloured by aromaticity")
        fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_aromaticity,norm=Normalize(vmin=np.min(aromaticity_scores_unique),vmax=np.max(aromaticity_scores_unique))), ax=ax15)
    else:
        ax9.axis("off")
        ax10.axis("off")
        ax11.axis("off")
        ax13.axis("off")
        ax14.axis("off")
        ax15.axis("off")

    ax4.axis("off")
    ax8.axis("off")
    ax12.axis("off")
    ax16.axis("off")

    fig.suptitle("UMAP of {}".format(vector_name))

    negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
    fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
    plt.legend(handles=[negative_patch,positive_patch], prop={'size': 20},loc= 'center right',bbox_to_anchor=(1.5,2.5),ncol=1)
    plt.savefig("{}/{}/umap_{}_{}".format(results_dir,method,vector_name,sample_mode))
    plt.clf()
    plt.close(fig)

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
    plt.savefig('{}/{}/blosum_cosine.png'.format(storage_folder, args.dataset_name), dpi=600)
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
    plt.savefig("{}/mi_feature_importance".format(results_dir),dpi=600)
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

def plot_precision_recall_curve(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,key_name,stats_name,idx,idx_name):
    """Following https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate."""
    onehot_targets = onehot_labels[idx]
    target_scores = predictions_dict[stats_name][idx]
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(args.num_obs_classes):
        precision[i], recall[i], _ = precision_recall_curve(onehot_targets[:, i], target_scores[:, i])
        average_precision[i] = average_precision_score(onehot_targets[:, i], target_scores[:, i])

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



    # display = PrecisionRecallDisplay(
    #     recall=recall["micro"],
    #     precision=precision["micro"],
    #     average_precision=average_precision["micro"],
    # )
    # display.plot()
    # _ = display.ax_.set_title("Micro-averaged over all classes")

def plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,key_name,stats_name,idx,idx_name):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels = labels[idx]
    onehot_targets = onehot_labels[idx]
    target_scores = predictions_dict[stats_name][idx]
    fig = plt.figure()

    # ROC AUC per class
    for i in range(args.num_obs_classes):
        fpr[i], tpr[i], _ = roc_curve(onehot_targets[:, i], target_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve (AUC_{}: {})'.format(i, roc_auc[i]), c=colors_dict[i])
    # Micro ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(onehot_targets.ravel(), target_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area : {})".format(roc_auc["micro"]),
             linestyle="-.", color="magenta")
    # Macro ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_obs_classes)]))
    fpr["macro"] = all_fpr
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.num_obs_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) #linear interpolation of data points
    tpr["macro"] = mean_tpr / args.num_obs_classes
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve (area : {})".format(roc_auc["macro"]),
             linestyle="-.", color="blue")

    plt.legend(loc='lower right', prop={'size': 15})
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.title("ROC curves")
    plt.savefig("{}/{}/ROC_curves_fold{}_{}".format(results_dir, mode, fold, "{}_{}".format(key_name, idx_name)))
    plt.clf()
    plt.close(fig)

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

    for sample_mode,prob_mode,binary_mode in zip(["samples","single_sample"],["class_probs_predictions_samples_average","class_probs_prediction_single_sample"],["class_binary_predictions_samples_mode","class_binary_prediction_single_sample"]):
        labels = predictions_dict["true_{}".format(sample_mode)]
        # onehot_labels = np.zeros((labels.shape[0],args.num_classes))
        # onehot_labels[np.arange(0,labels.shape[0]),labels.astype(int)] = 1
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
                plot_ROC_curves(labels,onehot_labels,predictions_dict,args,results_dir,mode,fold,sample_mode,prob_mode,idx,idx_name)
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
                ax1.legend(loc='lower right', prop={'size': 8},bbox_to_anchor=(1.3, 0.))
                fig.suptitle("ROC curve. AUC_micro_ovr_average: {}".format(average_micro_auc/args.num_samples),fontsize=12)
                plt.savefig("{}/{}/ROC_curves_PER_SAMPLE_{}".format(results_dir, mode, "{}".format(idx_name)))
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

    for sample_mode in ["single_sample","samples"]:
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
                    ax2.set_title("Attention by Aa type")
                    #Highlight: Aminoacids coloured by functional group (i.e positive, negative ...)
                    sns.heatmap(aminoacids_masked,ax=ax4,cbar=False,cmap=aa_groups_colormap)
                    ax4.set_xticks(np.arange(attention_weights.shape[1] +1) + 0.5,labels=["{}".format(i) for i in range(attention_weights.shape[1] + 1)])
                    ax4.spines['left'].set_visible(False)
                    ax4.yaxis.set_ticklabels([])
                    ax4.set_title("Attention by Aa group")

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

    for sample_mode in ["single_sample","samples"]:
        data_int = summary_dict["data_int_{}".format(sample_mode)]
        data_mask = summary_dict["data_mask_{}".format(sample_mode)]
        data_mask_seq = data_mask[:, 1:,:,0].squeeze(1)
        true_labels = summary_dict["true_{}".format(sample_mode)]
        confidence_scores = summary_dict["confidence_scores_{}".format(sample_mode)]
        idx_all = np.ones_like(confidence_scores).astype(bool)

        idx_highconfidence = (confidence_scores[..., None] > 0.7).any(-1)
        encoder_final_hidden_state = summary_dict["encoder_final_hidden_state_{}".format(sample_mode)]
        decoder_final_hidden_state = summary_dict["decoder_final_hidden_state_{}".format(sample_mode)]

        plot_latent_space(dataset_info,encoder_final_hidden_state, summary_dict, sample_mode, results_dir, method, vector_name="encoder_final_hidden state")
        plot_latent_space(dataset_info,decoder_final_hidden_state, summary_dict, sample_mode, results_dir, method, vector_name="decoder_final_hidden state")

        for data_points,idx_conf in zip(["all","high_confidence"],[idx_all,idx_highconfidence]):
            true_labels = summary_dict["true_{}".format(sample_mode)][idx_conf]
            positives_idx = (true_labels == 1)
            for class_type,idx_class in zip(["positives","negatives"],[positives_idx,~positives_idx]):

                encoder_hidden_states = summary_dict["encoder_hidden_states_{}".format(sample_mode)][idx_conf][idx_class]
                decoder_hidden_states = summary_dict["decoder_hidden_states_{}".format(sample_mode)][idx_conf][idx_class] #TODO: Review the values
                if encoder_hidden_states.size != 0:
                    #Highlight: Compute the cosine similarity measure (distance = 1 - similarity) among the hidden states of the sequence
                    if sample_mode == "single_sample":
                        # Highlight: Encoder
                        encoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs)(
                            delayed(VegvisirUtils.information_shift)(seq,seq_mask,diag_idx_maxlen,dataset_info.seq_max_len) for seq,seq_mask in
                            zip(encoder_hidden_states,data_mask_seq))
                        encoder_information_shift_weights = np.concatenate(encoder_information_shift_weights,axis=0)
                        #Highlight: Decoder
                        decoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs)(
                            delayed(VegvisirUtils.information_shift)(seq,seq_mask,diag_idx_maxlen,dataset_info.seq_max_len) for seq,seq_mask in
                            zip(decoder_hidden_states,data_mask_seq))
                        decoder_information_shift_weights = np.concatenate(decoder_information_shift_weights,axis=0)
                    else:
                        encoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs)(
                            delayed(VegvisirUtils.information_shift_samples)(encoder_hidden_states[:, sample_idx],
                                                                            data_mask_seq, diag_idx_maxlen,dataset_info.seq_max_len) for sample_idx in range(args.num_samples))

                        decoder_information_shift_weights = Parallel(n_jobs=MAX_WORKERs)(
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
                            sns.heatmap(weights, ax=ax1)
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
                            ax2.set_xticks(np.arange(weights.shape[1] + 1) + 0.5,labels=["{}".format(i) for i in range(weights.shape[1] + 1)])
                            ax2.spines['left'].set_visible(False)
                            ax2.yaxis.set_ticklabels([])
                            ax2.set_title("Information shift by Aa type")
                            # Highlight: Aminoacids coloured by functional group (i.e positive, negative ...)
                            sns.heatmap(aminoacids_masked, ax=ax4, cbar=False, cmap=aa_groups_colormap)
                            ax4.set_xticks(np.arange(max_len + 1) + 0.5,
                                           labels=["{}".format(i) for i in range(max_len + 1)])
                            ax4.spines['left'].set_visible(False)
                            ax4.yaxis.set_ticklabels([])
                            ax4.set_title("Information shift by Aa group")

                            ax3.axis("off")
                            ax5.axis("off")
                            ax6.axis("off")

                            legend1 = plt.legend(handles=aa_patches, prop={'size': 8}, loc='center right',
                                                 bbox_to_anchor=(0.9, 0.7), ncol=1)
                            plt.legend(handles=aa_groups_patches, prop={'size': 8}, loc='center right',
                                       bbox_to_anchor=(0.1, 0.5), ncol=1)
                            plt.gca().add_artist(legend1)

                            fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
                            fig.suptitle("{}. Information shift weights: {}, {}, {}".format(nn_name,sample_mode, data_points, class_type))
                            plt.savefig(
                                "{}/{}/{}_information_shift_plots_{}_{}_{}.png".format(results_dir, method, nn_name,sample_mode, data_points, class_type))
                            plt.clf()
                            plt.close(fig)


                else:
                    print("Not data points available for {}/{}/{}/{}".format(sample_mode, data_points,class_type,nn_name))

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