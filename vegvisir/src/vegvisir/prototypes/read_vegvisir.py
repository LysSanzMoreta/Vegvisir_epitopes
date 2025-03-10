import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score, recall_score
import torch
import torch.nn.functional as f
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



def test_prob_join(results):
    target_scores = results[["Vegvisir_negative_prob", "Vegvisir_positive_prob"]].to_numpy().astype(float)
    targets = np.array(results["Target_corrected"].tolist())


    #target_scores = torch.randn((4,2)).float()
    #target_scores = np.array([[0.5,0.2],[0.3,0.4],[0.2,0.2]])
    #print(target_scores)
    probs = f.softmax(torch.from_numpy(target_scores),dim=-1)
    fpr, tpr, _ = roc_curve(targets, probs[:,1])
    roc_auc = auc(fpr, tpr)



results_train = pd.read_csv("Epitopes_predictions_Train_fold_0.tsv",sep="\t")
#test_correctness(results_train)
test_prob_join(results_train)

results_test = pd.read_csv("Epitopes_predictions_Test_fold_0.tsv",sep="\t")

test_prob_join(results_test)