import os
import io

import matplotlib.pyplot as plt
import pandas as pd
import mmap
import seaborn as sns

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
def read_dataframe(folder_path):
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
    weak_binders = alleles_dataframes[(alleles_dataframes["%Rank_EL"] < 2.0) & (alleles_dataframes["%Rank_EL"] > 0.5) ]
    #weak_binders.to_csv("{}/weak_binders.tsv".format(folder_path),sep="\t") #find alternative
    strong_binders = alleles_dataframes[alleles_dataframes["%Rank_EL"] < 0.5 ]
    #strong_binders.to_csv("{}/strong_binders.tsv".format(folder_path),sep="\t")

    weak_binders_count = weak_binders.groupby('MHC', as_index=False)[["Icore"]].size()
    weak_binders_count["Binder_type"] = "Weak"

    strong_binders_count = strong_binders.groupby('MHC', as_index=False)[["Icore"]].size()
    strong_binders_count["Binder_type"] = "Strong"
    binders_df = pd.concat([weak_binders_count,strong_binders_count],axis=0)
    binders_df["size"] = ((binders_df["size"]/n_unique)*100).round(2)
    binders_df = binders_df.sort_values(by=['size'],ascending=False)
    print(binders_df)
    fig, ax = plt.subplots(figsize=(18, 18))
    #sns.catplot(y='MHC', x='size', hue='Binder_type', kind='bar', data=binders_df,width=1,dodge=True,ax=ax)
    sns.barplot(y='MHC', x='size', hue='Binder_type', data=binders_df,width=1,dodge=True,ax=ax)

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
    change_height(ax, 1.5)
    plt.xticks(fontsize=20)
    plt.title("MHC-binding count from generated epitopes",fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper right',fontsize=20)
    plt.savefig("{}/barplot.png".format(folder_path),dpi=600)




if __name__ == "__main__":


    #folder_path = "/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/PLOTS_Vegvisir_viral_dataset9_2023_09_12_15h33min42s969996ms_60epochs_supervised_Icore_onehot_TESTING"
    folder_path = "/home/lys/Dropbox/PostDoc/vegvisir/Results_netMHCpan/PLOTS_Vegvisir_viral_dataset9_2023_09_19_11h40min14s189602ms_80epochs_supervised_Icore_blosum_TESTING"
    read_dataframe(folder_path)