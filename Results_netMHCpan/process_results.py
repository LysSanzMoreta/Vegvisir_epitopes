import os,sys
import io
import argparse
from argparse import RawTextHelpFormatter
from collections import Counter

import matplotlib.pyplot as plt
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
        if ".png" not in filename:
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

    VegvisirPlots.plot_logos(epitopes_padded,"{}".format(folder_path),"_binders_MHC_all")
    VegvisirPlots.plot_logos(positive_sequences_list,"{}".format(folder_path),"_binders_MHC_positives")
    VegvisirPlots.plot_logos(negative_sequences_list,"{}".format(folder_path),"_binders_MHC_negatives")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NetMHCpan process results args",formatter_class=RawTextHelpFormatter)

    #folder_path = "/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING_immunomodulate"
    folder_path = "/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/PLOTS_Vegvisir_viral_dataset9_2023_11_03_22h28min14s922997ms_60epochs_supervised_Icore_blosum_TESTING_generated"
    #folder_name = "Immunomodulated"
    folder_name = "Generated"
    parser.add_argument('-folder-path',"--folder-path", type=str, nargs='?', default="", help='path to results')
    parser.add_argument('-folder-name',"--folder-name", type=str, nargs='?', default="Generated", help='path to results')
    args = parser.parse_args()
    #read_dataframe(args.folder_path)    #TODO: CAHNGE
    read_dataframe(folder_path,folder_name)