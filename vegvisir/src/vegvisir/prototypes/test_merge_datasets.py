import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score, recall_score

train_z4_path = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_12_26_18h38min16s083132ms_60epochs_supervised_Icore_blosum_TESTING_z4/Train/Epitopes_predictions_Train.tsv"
train_z30_path = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_12_26_19h11min19s422780ms_60epochs_supervised_Icore_60_TESTING_z30/Train/Epitopes_predictions_Train.tsv"

test_z4_path = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_12_26_18h38min16s083132ms_60epochs_supervised_Icore_blosum_TESTING_z4/Test/Epitopes_predictions_Test.tsv"
test_z30_path = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_12_26_19h11min19s422780ms_60epochs_supervised_Icore_60_TESTING_z30/Test/Epitopes_predictions_Test.tsv"


train_z4 = pd.read_csv(train_z4_path,sep="\t")
test_z4 = pd.read_csv(test_z4_path,sep="\t")

train_z30 = pd.read_csv(train_z30_path,sep="\t")
train_z30 = train_z30[["Icore","Latent_vector"]]

test_z30 = pd.read_csv(test_z30_path,sep="\t")
test_z30 = test_z30[["Icore","Latent_vector"]]



train = train_z4.merge(train_z30,how="left",on="Icore",suffixes=('_a', '_b'))
train.drop("Latent_vector_a",axis=1,inplace=True)
train.rename({"Latent_vector_b":"Latent_vector"},inplace=True,axis=1)

test = test_z4.merge(test_z30,how="left",on="Icore",suffixes=('_a', '_b'))
test.drop("Latent_vector_a",axis=1,inplace=True)
test.rename({"Latent_vector_b":"Latent_vector"},inplace=True,axis=1)

train.to_csv("Epitopes_info_TRAIN.tsv",sep="\t")
test.to_csv("Epitopes_info_TEST.tsv",sep="\t")

def test_correctness(results):
    target_scores = results[["Vegvisir_negative_prob", "Vegvisir_positive_prob"]].to_numpy().astype(float)


    targets = np.array(results["Target_corrected"].tolist())
    onehot_labels = np.zeros((targets.shape[0], 2))
    onehot_labels[np.arange(0, targets.shape[0]), targets.astype(int)] = 1
    results["Latent_vector"] = results["Latent_vector"].apply(lambda x: np.array(eval(x), dtype=np.float32))
    latent_space =  results["Latent_vector"].to_numpy()
    latent_space = np.stack(latent_space)

    #Highlight: plot latent representation
    reducer = umap.UMAP()
    umap_proj = reducer.fit_transform(latent_space)
    alpha = 0.7
    size = 5
    colors_dict_labels = {0: "mediumaquamarine", 1: "orangered", 2: "navajowhite"}
    colors_true = np.vectorize(colors_dict_labels.get)(targets)
    fig, ax1 = plt.subplots(1, figsize=(9, 6))
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=targets, alpha=alpha, s=size)
    plt.title("UMAP-2D")
    plt.show()
    #plt.close(fig)
    plt.clf()

    #Highlight: compute ROC- AUC
    tpr=dict()
    fpr=dict()
    roc_auc=dict()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [4.5, 1]})
    # ROC AUC per class
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(onehot_labels[:, i], target_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax1.plot(fpr[i], tpr[i], label='Class {} AUC : {}'.format(i, round(roc_auc[i], 2)), c=colors_dict_labels[i])
    ax2.axis("off")
    fig.legend(bbox_to_anchor=(0.71, 0.35), prop={'size': 15})  # loc='lower right'
    plt.title("ROC-AUC")
    plt.show()
    #plt.close(fig)
    plt.clf()

#test_correctness(train)
test_correctness(test)