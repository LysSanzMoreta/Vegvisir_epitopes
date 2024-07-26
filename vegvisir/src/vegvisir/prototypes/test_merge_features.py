import pandas as pd




storage_folder = "/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data"


def merge_features():
    all_feats_train = pd.read_csv("{}/common_files/dataset_all_features.tsv".format(storage_folder),sep="\s+",index_col=0)
    all_feats_test = pd.read_csv("{}/common_files/dataset_all_features_test.tsv".format(storage_folder),sep="\s+")
    cols = all_feats_test.columns[~all_feats_test.columns.str.contains("Icore")]
    all_feats = all_feats_train.merge(all_feats_test,how="outer",on="Icore")
    for column in cols:
        print(column)
        if f"{column}_x" in all_feats.columns:
            all_feats[column] = all_feats[f"{column}_x"].fillna(all_feats[f"{column}_y"])
            all_feats.drop([f"{column}_x",f"{column}_y"],inplace=True,axis=1)
    #a = all_feats[all_feats["pep_secstruc_turn"].isna()]
    all_feats.to_csv("{}/common_files/dataset_all_features_with_test.tsv".format(storage_folder),sep="\t")

def merge_esmb1():
    esmb1_train = pd.read_csv(f"{storage_folder}/common_files/Epitopes_info_TRAIN_esmb1.tsv",sep="\t")
    esmb1_test = pd.read_csv(f"{storage_folder}/common_files/ESMB1_Embeddings_test.csv",sep=",")

    esmb1_all = pd.concat([esmb1_train,esmb1_test],axis=0)

    esmb1_all.to_csv(f"{storage_folder}/common_files/ESMB1_all.csv",sep=",")


merge_esmb1()