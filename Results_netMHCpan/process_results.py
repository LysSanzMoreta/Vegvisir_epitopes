import os,sys
import io
import argparse
from argparse import RawTextHelpFormatter
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mmap
import seaborn as sns
local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("Results_netMHCpan","")

if local_repository: #TODO: The local imports are extremely slow
     sys.path.insert(1, "{}/vegvisir/src".format(script_dir))
     import vegvisir
else:#pip installed module
     import vegvisir

import vegvisir.plots as VegvisirPlots
import vegvisir.similarities as VegvisirSimilarities
import vegvisir.utils as VegvisirUtils

from collections import namedtuple

DatasetInfo = namedtuple("datasetinfo",["corrected_aa_types","seq_max_len"])

def split_string(string,maxlen):
    string = string.split()
    string_len = len(string)
    if string_len == 0 or string_len == 1:
        return None
    elif string[0] == "#" or string[0] == "Protein" or string[0].startswith("HLA-") or string[0] == "Pos":
        return None
    elif string_len < maxlen:
        string += [""]*(maxlen-string_len)
        return string
    else:
        return string

def read_dataframe(folder_path,folder_name):
    headers =  ["Pos","MHC", "Peptide", "Core", "Of", "Gp", "Gl", "Ip", "Il", "Icore","Identity", "Score_EL", "%Rank_EL" "BindLevel"]
    files = os.listdir(folder_path)
    #ignore_str = "-"*123 + "\n"
    maxlen = len(headers)
    alleles_dataframes = []
    for filename in files:
        if ".png" not in filename and ".tsv" not in filename:
            with open("{}/{}".format(folder_path,filename), 'rb', 0) as file:
                s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                lines = s.read().decode("utf-8").split("\n")
                list_of_lines = list(map(lambda string: split_string(string,maxlen),lines))
                list_of_lines = list(filter(lambda v: v is not None, list_of_lines))
                zipped = list(zip(*list_of_lines))
                Pos = pd.Series(zipped[0])
                MHC = pd.Series(zipped[1])
                Peptide = pd.Series(zipped[2])
                Core = pd.Series(zipped[3])
                Of = pd.Series(zipped[4])
                Gp = pd.Series(zipped[5])
                Gl = pd.Series(zipped[6])
                Ip = pd.Series(zipped[7])
                Il = pd.Series(zipped[8])
                Icore = pd.Series(zipped[9])
                Identity = pd.Series(zipped[10])
                Score_EL = pd.Series(zipped[11])
                Rank_EL = pd.Series(zipped[12])

                df = pd.DataFrame({"Pos":Pos,
                                   "MHC":MHC,
                                   "Epitopes":Peptide,
                                   "Core":Core,
                                   "Of":Of,
                                   "Gp":Gp,
                                   "Gl":Gl,
                                   "Ip":Ip,
                                   "Il":Il,
                                   "Icore":Icore,
                                   "Identity":Identity,
                                   "Score_EL":Score_EL,
                                   "%Rank_EL":Rank_EL})



                alleles_dataframes.append(df)

    alleles_dataframes = pd.concat(alleles_dataframes,axis=0)

    alleles_dataframes[["%Rank_EL","Score_EL"]] = alleles_dataframes[["%Rank_EL","Score_EL"]].astype(float)
    n_unique = len(alleles_dataframes["Epitopes"].unique())
    weak_binders = alleles_dataframes[(alleles_dataframes["%Rank_EL"] <= 2.0) & (alleles_dataframes["%Rank_EL"] > 0.5) ]
    strong_binders = alleles_dataframes[alleles_dataframes["%Rank_EL"] <= 0.5 ]

    epitopes_folder_path = "{}/{}/epitopes.tsv".format(folder_path.replace("Results_netMHCpan/","").replace("_immunomodulate","").replace("_generated",""),folder_name)
    epitopes_df = pd.read_csv(epitopes_folder_path, sep="\t")
    epitopes_df.rename(columns={"Epitopes":"Icore"},inplace=True)

    weak_binders_count = weak_binders.groupby('MHC', as_index=False)[["Icore"]].size()
    weak_binders_count["Binder_type"] = "Weak"
    weak_binders_epitopes = weak_binders.groupby('MHC', as_index=False)[["Icore"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0])
    weak_binders_df = pd.concat([weak_binders_epitopes,weak_binders_count.drop("MHC",axis=1)],axis=1)
    weak_binders_df["Icore"] = weak_binders_df["Icore"].str.replace("X","")
    weak_binders_df.drop_duplicates(["Icore"],inplace=True)

    weak_binders_df=weak_binders_df.merge(epitopes_df,on="Icore",how="left")

    strong_binders_count = strong_binders.groupby('MHC', as_index=False)[["Icore"]].size()
    strong_binders_count["Binder_type"] = "Strong"
    strong_binders_epitopes = strong_binders.groupby('MHC', as_index=False)[["Icore"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0])
    strong_binders_df = pd.concat([strong_binders_epitopes,strong_binders_count.drop("MHC",axis=1)],axis=1)
    strong_binders_df["Icore"] = strong_binders_df["Icore"].str.replace("X","")
    strong_binders_df.drop_duplicates(["Icore"],inplace=True)

    strong_binders_df=strong_binders_df.merge(epitopes_df,on="Icore",how="left")

    binders_df = pd.concat([weak_binders_count,strong_binders_count],axis=0)
    binders_df["size"] = ((binders_df["size"]/n_unique)*100).round(2)


    epitopes_binders = pd.concat([weak_binders_df,strong_binders_df],axis=0)
    epitopes_binders["Icore"] = epitopes_binders["Icore"].str.ljust(11, fillchar='#')

    epitopes_binders = epitopes_binders.dropna(subset=["Negative_score","Positive_score"],axis=0)
    epitopes_padded = epitopes_binders["Icore"].tolist()

    positive_sequences = epitopes_binders[epitopes_binders["Positive_score"] >= 0.6]
    positive_sequences_list = positive_sequences["Icore"].tolist()

    negative_sequences = epitopes_binders[epitopes_binders["Negative_score"] >= 0.6]
    negative_sequences_list = negative_sequences["Icore"].tolist()
    #binders_df = binders_df.sort_values(by=['size'],ascending=False) #Highlight: Activate if want to order by counts
    # nrows_empty = len(binders_df["size"])
    # size_with_empty_rows = [0]*nrows_empty*2
    # size_with_empty_rows[::2] = binders_df["size"].values.tolist()
    #
    # mhc_with_empty_rows = [0]*nrows_empty*2
    # mhc_with_empty_rows[::2] = binders_df["MHC"].values.tolist()
    #
    # binder_type_with_empty_rows = [0] * nrows_empty * 2
    # binder_type_with_empty_rows[::2] = binders_df["Binder_type"].values.tolist()
    # binders_df_empty_rows = pd.DataFrame({"size":size_with_empty_rows,"MHC":mhc_with_empty_rows,"Binder_type":binder_type_with_empty_rows})

    fig, ax = plt.subplots(figsize=(18, 18))
    #sns.catplot(y='MHC', x='size', hue='Binder_type', kind='bar', data=binders_df,width=1,dodge=True,ax=ax)
    sns.barplot(y='MHC', x='size', hue='Binder_type', data=binders_df,width=0.8,dodge=True,ax=ax)
    ax.spines[['right', 'top']].set_visible(False)
    #ax.axes.set_ylim(-0.5, n_unique)
    ax.set_xlabel('Percentage of binders', fontsize=15)
    def change_width(ax, new_value):
        for patch in ax.patches:
            current_width = patch.get_width()
            diff = current_width - new_value
            # we change the bar width
            patch.set_width(new_value)
            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    def change_height(ax, new_value):
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar width
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

    #change_width(ax, 1.5)
    #change_height(ax, 1.5)
    plt.xticks(fontsize=20)
    plt.title("MHC-binding count from generated epitopes",fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper right',fontsize=20)
    plt.savefig("{}/barplot_{}.png".format(folder_path,folder_name),dpi=600)

    sequences = positive_sequences_list + negative_sequences_list

    labels = np.array([1]*len(positive_sequences_list) + [0]*len(negative_sequences_list))
    calculate_peptide_features_correlations(sequences, labels, "{}".format(folder_path))

    VegvisirPlots.plot_logos(epitopes_padded,"{}".format(folder_path),"_binders_MHC_all")
    VegvisirPlots.plot_logos(positive_sequences_list,"{}".format(folder_path),"_binders_MHC_positives")
    VegvisirPlots.plot_logos(negative_sequences_list,"{}".format(folder_path),"_binders_MHC_negatives")

    plot_positional_weights(epitopes_padded,11,"ALL","{}".format(folder_path))
    plot_positional_weights(positive_sequences_list,11,"POSITIVES","{}".format(folder_path))
    plot_positional_weights(negative_sequences_list,11,"NEGATIVES","{}".format(folder_path))

def build_arrays(sequences):
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(21, "BLOSUM62",
                                                                               zero_characters=["#"],
                                                                               include_zero_characters=True)
    aa_dict = VegvisirUtils.aminoacid_names_dict(21, zero_characters=["#"])

    sequences = list(map(lambda seq:list(seq),sequences))
    sequences_raw = np.array(sequences)


    sequences_int = np.vectorize(aa_dict.get)(sequences_raw)
    sequences_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(sequences_int)

    sequences_mask = sequences_int.astype(bool)

    return sequences_raw,sequences_int,sequences_blosum,sequences_mask

def plot_positional_weights(sequences,maxlen,subtitle,results_dir):

    sequences_raw, sequences_int, sequences_blosum, sequences_mask = build_arrays(sequences)
    generated_sequences_cosine_similarity = VegvisirSimilarities.cosine_similarity(sequences_blosum,
                                                                                   sequences_blosum,
                                                                                   correlation_matrix=False,
                                                                                   parallel=False)
    batch_size = 100 if sequences_blosum.shape[0] > 100 else sequences_blosum.shape[0]
    positional_weights = VegvisirSimilarities.importance_weight(generated_sequences_cosine_similarity,
                                                                maxlen, sequences_mask,
                                                                batch_size=batch_size, neighbours=1)
    VegvisirPlots.plot_heatmap(positional_weights, "Cosine similarity \n positional weights",
                               "{}/Generated_MHC_bound_positional_weights_{}.png".format(results_dir,subtitle))

def calculate_peptide_features_correlations(sequences,labels,results_dir):
    """"""
    #sequences_raw, sequences_int, sequences_blosum, sequences_mask = build_arrays(sequences)
    seq_max_len = 11

    storage_folder = "/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data"

    features_dict = VegvisirUtils.CalculatePeptideFeatures(seq_max_len, sequences,storage_folder).features_summary()

    spearman_correlations = list(map(lambda feat1, feat2: VegvisirUtils.calculate_correlations(feat1, feat2, method="spearman"),[labels] * len(features_dict.keys()), list(features_dict.values())))
    spearman_correlations = list(zip(*spearman_correlations))
    spearman_coefficients = np.array(spearman_correlations[0])
    spearman_coefficients = np.round(spearman_coefficients, 2)

    # Highlight: Plot spearman coefficients
    fontsize = 15
    with plt.style.context('classic'):
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 12),
                                sharey="all")  # gridspec_kw={'width_ratios': [4.5, 0.5]}

        n_feats = len(features_dict.keys())
        index = np.arange(n_feats)
        positive_idx = np.array(
            spearman_coefficients >= 0)  # divide the coefficients in negative and positive to plot them separately
        right_arr = np.zeros(n_feats)
        left_arr = np.zeros(n_feats)
        right_arr[positive_idx] = spearman_coefficients[positive_idx]
        left_arr[~positive_idx] = spearman_coefficients[~positive_idx]

        ax1.barh(index, left_arr, align="center", color="mediumorchid",
                 zorder=1)  # zorder indicates the plotting order, supposedly
        ax1.barh(index, right_arr, align="center", color="seagreen", zorder=2)

        position_labels = list(range(0, n_feats))
        ax1.axvline(0)

        def clean_labels(label):
            if label == "extintion_coefficient_cystines":
                label = "extintion coefficient \n (cystines)"
            elif label == "extintion_coefficient_cysteines":
                label = "extintion coefficient \n (cysteines)"
            else:
                label = label.replace("_", " ")

            return label

        labels_names = list(map(lambda label: clean_labels(label), list(features_dict.keys())))
        ax1.yaxis.set_ticks(position_labels)
        ax1.set_yticklabels(labels_names, fontsize=fontsize, rotation=0, weight='bold')
        ax1.tick_params(axis="x", labelsize=fontsize)
        # ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
        ax1.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            left=False,
            right=False)  # labels along the bottom edge are off
        plt.subplots_adjust(left=0.28)
        # ax1.margins(y=0.15)
        ax1.spines[['right', 'top', 'left']].set_visible(False)

        fig.suptitle("Correlation coefficients: Features vs Predicted targets", fontsize=fontsize + 8,
                     weight='bold')
        plt.savefig("{}/Generated_target_features_correlations".format(results_dir), dpi=700)

def combine_folder_results(results_dict):
    """"""



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NetMHCpan process results args",formatter_class=RawTextHelpFormatter)

    folder_path = "/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/PLOTS_Vegvisir_viral_dataset9_2023_12_26_21h41min44s738321ms_0epochs_supervised_Icore_0_TESTING"
    #folder_name = "Immunomodulated"
    folder_name = "Generated"
    parser.add_argument('-folder-path',"--folder-path", type=str, nargs='?', default="", help='path to results')
    parser.add_argument('-folder-name',"--folder-name", type=str, nargs='?', default="Generated", help='path to results')
    args = parser.parse_args()
    read_dataframe(folder_path,args.folder_name)