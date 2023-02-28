"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import  matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import umap
import vegvisir.utils as VegvisirUtils
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
plt.style.use('ggplot')
colors_dict = {0: "red", 1: "green"}



def plot_data_information(data, filters_dict, storage_folder, args, name_suffix):
    """"""
    ndata = data.shape[0]
    fig, ax = plt.subplots(nrows=3,ncols=4, figsize=(11, 10))
    num_bins = 50
    #colors_dict = {0: "orange", 1: "blue"}
    labels_dict = {0: "Negative", 1: "Positive"}

    ############LABELS #############
    freq, bins, patches = ax[0][0].hist(data["target"].to_numpy(), bins=2, density=True, edgecolor='white')
    ax[0][0].set_xlabel('Target/Label (0: Non-binder, 1: Binder)')
    ax[0][0].set_title('Histogram of targets/labels \n',fontsize=10)
    ax[0][0].xaxis.set_ticks([0.25, 0.75])
    ax[0][0].set_xticklabels([0, 1])
    # Annotate the bars.
    for color, bar in zip(colors_dict.values(), patches):  # iterate over the bars
        n_data_bin = (bar.get_height() * ndata) / 2
        ax[0][0].annotate(int(n_data_bin),
                          (bar.get_x() + bar.get_width() / 2,
                           bar.get_height()), ha='center', va='center',
                          size=12, xytext=(0, 8),
                          textcoords='offset points')
        bar.set_facecolor(color)

    ############LABELS CORRECTED #############
    freq, bins, patches = ax[0][1].hist(data["target_corrected"].to_numpy(), bins=2, density=True, edgecolor='white')
    ax[0][1].set_xlabel('Target/Label (0: Non-binder, 1: Binder)')
    ax[0][1].set_title('Histogram of re-assigned \n targets/labels',fontsize=10)
    ax[0][1].xaxis.set_ticks([0.25, 0.75])
    ax[0][1].set_xticklabels([0, 1])
    # Annotate the bars.
    for color, bar in zip(colors_dict.values(), patches):  # iterate over the bars
        n_data_bin = (bar.get_height() * ndata) / 2
        ax[0][1].annotate(int(n_data_bin),
                          (bar.get_x() + bar.get_width() / 2,
                           bar.get_height()), ha='center', va='center',
                          size=12, xytext=(0, 8),
                          textcoords='offset points')
        bar.set_facecolor(color)
    ####### Immunodominance scores ###################
    ax[1][0].hist(data["immunodominance_score_scaled"].to_numpy(), num_bins, density=True)
    ax[1][0].set_xlabel('Minmax scaled \n immunodominance score \n (N_+ / Total Nsubjects)',fontsize=10)
    ax[1][0].set_title('Histogram of \n immunodominance scores',fontsize=10)
    ######## Sequence length distribution ####################
    ######Train#############################3
    data_lens_train_negative = data.loc[(data["training"] == True) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_train_positive = data.loc[(data["training"] == True) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    dict_counts = {0: data_lens_train_negative.value_counts(), 1: data_lens_train_positive.value_counts()}
    longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
    position = 0
    positions = []
    for val_i, count_i in dict_counts[longest].items():
        ax[2][0].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.1, edgecolor='white')
        if val_i in dict_counts[shortest].keys():
            count_j = dict_counts[shortest][val_i]
            ax[2][0].bar(position + 0.1, count_j, label=shortest, color=colors_dict[shortest], width=0.1,edgecolor='white')
        positions.append(position + 0.05)
        position += 0.25
    ax[2][0].xaxis.set_ticks(positions)
    ax[2][0].set_xticklabels(dict_counts[longest].keys())
    ax[2][0].set_title("Sequence length distribution of \n  the Train-valid dataset",fontsize=10)
    ###### Test #####################
    data_lens_test_negative = data.loc[(data["training"] == False) & (data["target_corrected"] == 0.), filters_dict["filter_kmers"][2]].str.len()
    data_lens_test_positive = data.loc[(data["training"] == False) & (data["target_corrected"] == 1.), filters_dict["filter_kmers"][2]].str.len()
    dict_counts = {0: data_lens_test_negative.value_counts(), 1: data_lens_test_positive.value_counts()}
    longest, shortest = [(1, 0) if len(dict_counts[1].keys()) > len(dict_counts[0].keys()) else (0, 1)][0]
    position = 0
    positions = []
    for val_i, count_i in dict_counts[longest].items():
        ax[2][1].bar(position, count_i, label=longest, color=colors_dict[longest], width=0.1, edgecolor='white')
        if val_i in dict_counts[shortest].keys():
            count_j = dict_counts[shortest][val_i]
            ax[2][1].bar(position + 0.1, count_j, label=shortest, color=colors_dict[shortest], width=0.1,
                         edgecolor='white')
        positions.append(position + 0.05)
        position += 0.25
    ax[2][1].xaxis.set_ticks(positions)
    ax[2][1].set_xticklabels(dict_counts[longest].keys())
    ax[2][1].set_title("Sequence length distribution of \n  the Test dataset",fontsize=10)
    ############TEST PROPORTIONS #############
    data_partitions = data[["partition", "training", "target_corrected"]]
    test_counts = data_partitions[data_partitions["training"] == False].value_counts("target_corrected")  # returns a dict
    if len(test_counts.keys()) > 1:
        if test_counts[0] > test_counts[1]:
            bar1 = ax[1][1].bar(0, test_counts[0], label="Negative", color=colors_dict[0], width=0.1, edgecolor='white')
            bar1 = bar1.patches[0]
            bar2 = ax[1][1].bar(0, test_counts[1], label="Positive", color=colors_dict[1], width=0.1, edgecolor='white')
            bar2 = bar2.patches[0]
        else:
            bar1 = ax[1][1].bar(0, test_counts[1], label="Positive", color=colors_dict[1], width=0.1, edgecolor='white')
            bar1 = bar1.patches[0]
            bar2 = ax[1][1].bar(0, test_counts[0], label="Negative", color=colors_dict[0], width=0.1, edgecolor='white')
            bar2 = bar2.patches[0]
        ax[1][1].xaxis.set_ticks([0])
        n_data_test = sum([val for key, val in test_counts.items()])
        ax[1][1].annotate("{}({}%)".format(bar1.get_height(), np.round((bar1.get_height() * 100) / n_data_test), 2),
                          (bar1.get_x() + bar1.get_width() / 2,
                           bar1.get_height()), ha='center', va='center',
                          size=12, xytext=(0, 8),
                          textcoords='offset points')
        ax[1][1].annotate("{}({}%)".format(bar2.get_height(), np.round((bar2.get_height() * 100) / n_data_test), 2),
                          (bar2.get_x() + bar2.get_width() / 2,
                           bar2.get_height()), ha='center', va='center',
                          size=12, xytext=(0, 8),
                          textcoords='offset points')
    else:
        key = test_counts.keys()[0]
        bar1 = ax[1][1].bar(key, test_counts[key], label=labels_dict[key], color=colors_dict[key], width=0.1,
                            edgecolor='white')
        bar1 = bar1.patches[0]
        ax[1][1].xaxis.set_ticks([0])
        n_data_test = sum([val for key, val in test_counts.items()])
        ax[1][1].annotate("{}({}%)".format(bar1.get_height(), np.round((bar1.get_height() * 100) / n_data_test), 2),
                          (bar1.get_x() + bar1.get_width() / 2,
                           bar1.get_height()), ha='center', va='center',
                          size=12, xytext=(0, 8),
                          textcoords='offset points')

    ax[1][1].set_xticklabels(["Test proportions"], fontsize=10)
    ax[1][1].set_title('Test dataset \n +/- proportions',fontsize=10)

    ################ TRAIN PARTITION PROPORTIONS###################################
    train_set = data_partitions[data_partitions["training"] == True]
    partitions_groups = [train_set.groupby('partition').get_group(x) for x in train_set.groupby('partition').groups]
    i = 0
    partitions_names = []
    for group in partitions_groups:
        name = group["partition"].iloc[0]
        group_counts = group.value_counts("target_corrected")  # returns a dict
        if len(group_counts.keys()) > 1:
            if group_counts[0] > group_counts[1]:
                bar1 = ax[0][2].bar(i, group_counts[0], label="Negative", color=colors_dict[0], width=0.1, edgecolor='white')
                bar1 = bar1.patches[0]
                bar2 = ax[0][2].bar(i, group_counts[1], label="Positive", color=colors_dict[1], width=0.1)
                bar2 = bar2.patches[0]
            else:
                bar1 = ax[0][2].bar(i, group_counts[1], label="Positive", color=colors_dict[1], width=0.1, edgecolor='white')
                bar1 = bar1.patches[0]
                bar2 = ax[0][2].bar(i, group_counts[0], label="Negative", color=colors_dict[0], width=0.1, edgecolor='white')
                bar2 = bar2.patches[0]

            i += 0.4
            n_data_partition = sum([val for key, val in group_counts.items()])
            ax[0][2].annotate(
                "{}\n({}%)".format(bar1.get_height(), np.round((bar1.get_height() * 100) / n_data_partition), 2),
                (bar1.get_x() + bar1.get_width() / 2,
                 bar1.get_height()), ha='center', va='center',
                size=8, xytext=(0, 8),
                textcoords='offset points')
            ax[0][2].annotate(
                "{}\n({}%)".format(bar2.get_height(), np.round((bar2.get_height() * 100) / n_data_partition), 2),
                (bar2.get_x() + bar2.get_width() / 2,
                 bar2.get_height()), ha='center', va='center',
                size=8, xytext=(0, 8),
                textcoords='offset points')
        else:
            key = group_counts.keys()[0]
            bar1 = ax[0][2].bar(i, group_counts[key], label=labels_dict[key], color=colors_dict[key], width=0.1,edgecolor='white')
            bar1 = bar1.patches[0]
            n_data_partition = sum([val for key, val in group_counts.items()])
            ax[0][2].annotate(
                "{}\n({}%)".format(bar1.get_height(), np.round((bar1.get_height() * 100) / n_data_partition), 2),
                (bar1.get_x() + bar1.get_width() / 2,
                 bar1.get_height()), ha='center', va='center',
                size=10, xytext=(0, 8),
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
    ax[2][3].axis("off")

    legends = [mpatches.Patch(color=color, label='Class {}'.format(label)) for label, color in colors_dict.items()]
    fig.legend(handles=legends, prop={'size': 12}, loc='center right', bbox_to_anchor=(0.9, 0.5))
    fig.tight_layout(pad=0.1)
    plt.savefig("{}/{}/Viruses_histograms_{}".format(storage_folder, args.dataset_name, name_suffix), dpi=500)
    plt.clf()

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
    colors_dict_labels = {0:"mediumaquamarine",1:"orangered"}
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

def plot_heatmap(array, title,file_name):
    plt.figure(figsize=(20, 20))
    sns.heatmap(array, cmap='RdYlGn_r',yticklabels=False,xticklabels=False)
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()

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
    plt.savefig("{}/{}/similarities/{}.png".format(storage_folder,args.dataset_name,file_name))
    plt.clf()

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
        plt.plot(epochs_idx,train_loss, color="dodgerblue",label="train")
        if valid_loss is not None:
            plt.plot(epochs_idx,valid_loss, color="darkorange", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("-ELBO")
        #plt.yscale('log')
        plt.title("Error loss (Train/valid)")
        plt.legend()
        plt.savefig("{}/error_loss_{}fold.png".format(results_dir,fold))
        plt.close()
    plt.clf()

def plot_accuracy(train_accuracies,valid_accuracies,epochs_list,fold,results_dir):
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
        plt.savefig("{}/reconstruction_accuracies_{}fold.png".format(results_dir, fold))
        plt.close()
        plt.clf()
    
    else:
        train_accuracies = np.array(train_accuracies)
        valid_accuracies = np.array(valid_accuracies)
        epochs_idx = np.array(epochs_list)
        train_accuracies = train_accuracies[epochs_idx.astype(int)] #select the same epochs as the vaidation
    
        plt.plot(epochs_idx,train_accuracies, color="deepskyblue",label="train")
        if valid_accuracies is not None:
            plt.plot(epochs_idx,valid_accuracies, color="salmon", label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (number of correct predictions)")
        plt.title("Accuracy (Train/valid)")
        plt.legend()
        plt.savefig("{}/accuracies_{}fold.png".format(results_dir,fold))
        plt.close()
        plt.clf()

def plot_classification_score(train_auc,valid_auc,epochs_list,fold,results_dir,method):
    """Plots the AUC/AUK scores while training
    :param list train_auc: list of accumulated AUC during training
    :param list valid_auc: list of accumulated AUC during validation
    :param str results_dict: path to results directory
    :param str method: AUC or AUK
    """
    epochs_idx = np.array(epochs_list)
    plt.plot(epochs_idx,train_auc, color="deepskyblue",label="train")
    if valid_auc is not None:
        plt.plot(epochs_idx,valid_auc, color="salmon", label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("{}".format(method))
    plt.title("{} (Train/valid)".format(method))
    plt.legend()
    plt.savefig("{}/{}_{}fold.png".format(results_dir,method,fold))
    plt.close()
    plt.clf()

def plot_latent_space(latent_space,predictions,fold,results_dir,method):

    print("Plotting UMAP...")
    reducer = umap.UMAP()
    umap_proj = reducer.fit_transform(latent_space[:, 4:])

    colors_dict_labels = {0:"mediumaquamarine",1:"orangered"}
    colors_true = np.vectorize(colors_dict_labels.get)(latent_space[:,1])
    colors_predicted = np.vectorize(colors_dict_labels.get)(predictions)
    #Highlight: Confidence scores colors
    confidence_scores = latent_space[:,2]
    confidence_scores_unique = np.unique(confidence_scores).tolist()
    colormap_confidence = matplotlib.cm.get_cmap('plasma_r', len(confidence_scores_unique))
    colors_dict = dict(zip(confidence_scores_unique, colormap_confidence.colors))
    colors_confidence = np.vectorize(colors_dict.get, signature='()->(n)')(confidence_scores)
    #Highlight: Immunodominance scores colors
    immunodominance_scores = latent_space[:,3]
    immunodominance_scores_unique = np.unique(immunodominance_scores).tolist()
    colormap_immunodominance = matplotlib.cm.get_cmap('plasma_r', len(immunodominance_scores_unique))
    colors_dict = dict(zip(immunodominance_scores_unique, colormap_immunodominance.colors))
    colors_immunodominance = np.vectorize(colors_dict.get, signature='()->(n)')(immunodominance_scores)


    fig, [[ax1, ax2, ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]] = plt.subplots(3, 3,figsize=(17, 12),gridspec_kw={'width_ratios': [4.5,4.5,1],'height_ratios': [4,4,2]})
    fig.suptitle('UMAP projections',fontsize=20)
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=latent_space[:,2], alpha=1,s=30)
    ax1.set_title("True labels",fontsize=20)
    ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted, label=predictions, alpha=1,s=30)
    ax2.set_title("Predicted labels",fontsize=20)
    ax4.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_confidence, alpha=1, s=30)
    ax4.set_title("Confidence scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_confidence),ax=ax4)
    ax5.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_immunodominance, alpha=1, s=30)
    ax5.set_title("Immunodominance scores", fontsize=20)
    fig.colorbar(plt.cm.ScalarMappable(cmap=colormap_immunodominance),ax=ax5)
    # ax4.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_peptides_identifiers, label=latent_space[:,0], alpha=1, s=30)
    # ax4.set_title("Peptides identifiers", fontsize=20)
    ax3.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")
    ax9.axis("off")
    negative_patch = mpatches.Patch(color=colors_dict_labels[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict_labels[1], label='Class 1')
    #peptides_color_patches = [mpatches.Patch(color='{}'.format(val), label='{}'.format(peptides_labels_dict[key])) if key in peptides_labels_dict.keys() else None for key,val in colors_dict_peptides.items()]
    #peptides_color_patches = [x for x in peptides_color_patches if x is not None]
    #ncol = [1 if len(peptides_color_patches) < 10 else math.ceil((len(peptides_color_patches)/10))][0]
    # xloc = [0.55 if ncol == 1 else 1][0]
    # yloc = [0.5 if ncol == 1 else 0.35][0]
    plt.legend(handles=[negative_patch,positive_patch], prop={'size': 20},loc= 'center right',bbox_to_anchor=(1,0.5),ncol=1)
    plt.savefig("{}/{}/umap_fold{}".format(results_dir,method,fold))
    plt.clf()

def plot_gradients(gradient_norms,results_dir,fold):
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
    ax1.legend(loc= 'center right',bbox_to_anchor=(1.5,0.5),fontsize=11, borderaxespad=0.)
    ax1.set_title('Gradient norms of model parameters')

    plt.savefig("{}/gradients_fold{}".format(results_dir,fold))
    plt.clf()

def plot_ROC_curve(fpr,tpr,roc_auc,auk_score,results_dir,fold):
    plt.title('Receiver Operating Characteristic',fontdict={"size":20})
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f \n '
                                  'AUK = %0.2f' % (roc_auc,auk_score))
    plt.legend(loc='lower right',prop={'size': 15})
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.savefig("{}/ROC_curve_fold{}".format(results_dir,fold))
    plt.clf()

def plot_blosum_cosine(blosum_array,storage_folder,args):
    """

    :param blosum_array:
    :param storage_folder:
    :param args:
    """
    plt.subplots(figsize=(10, 10))
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

def plot_feature_importance_old(feature_dict:dict,max_len:int,features_names:list,results_dir:str) -> None:
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
    row = 0
    col = 0


    if len(feature_dict["Fold_0"]) == max_len:
        labels = ["Pos.{}".format(pos) for pos in list(range(max_len))]
    else:
        labels = ["Pos.{}".format(pos) for pos in list(range(max_len))] + features_names
    colors_dict = dict(zip(labels,colors_list))

    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,squeeze=False) #check this: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
    for fold,features in feature_dict.items(): #ax = fig.add_subplot(2,2,a+1)
        ax[int(row)][int(col)].bar(range(len(features)),features,color=colors_dict.values())
        #ax[int(row)][int(col)].set_xticks(np.arange(len(labels)),labels,rotation=45,fontsize=8)
        ax[int(row)][int(col)].set_title("{}".format(fold))
        col += 1
        if col >= ncols:
            row += 1
            col = 0
            if row >= nrows:
                # if col <= ncols and not len(feature_dict) % 2 == 0:
                #     ax[int(row) -1 ][int(col)].axis("off")
                break

    fig.add_subplot(111) #added to fit the legend
    patches = [mpatches.Patch(color='{}'.format(val), label='{}'.format(key)) for key,val in colors_dict.items()]
    fig.legend(handles=patches, prop={'size': 10},loc= 'center right',bbox_to_anchor=(1.37,0.5))

    fig.tight_layout(pad=3.0)
    fig.suptitle("Feature importance")
    plt.savefig("{}/feature_importance_xgboost".format(results_dir))

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
    plt.bar(range(len(feature_names)), feature_names,color=colors_dict.values())
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, fontsize=8)
    plt.title("Mutual Information feature importance")
    patches = [mpatches.Patch(color='{}'.format(val), label='{}'.format(key)) for key,val in colors_dict.items()]
    # pos = fig.get_position()
    # fig.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    plt.legend(handles=patches, prop={'size': 10},loc= 'center right',bbox_to_anchor=(1.37,0.5))
    plt.savefig("{}/mi_feature_importance".format(results_dir),dpi=600)
    plt.clf()

def plot_confusion_matrix(confusion_matrix,performance_metrics,results_dir,fold):
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
    plt.savefig("{}/confusion_matrix_{}.png".format(results_dir,fold),dpi=100)
    plt.clf()
