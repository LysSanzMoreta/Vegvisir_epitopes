"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.style.use('ggplot')
def plot_heatmap(array, title,file_name):
    plt.figure(figsize=(20, 20))
    sns.heatmap(array, cmap='RdYlGn_r',yticklabels=False,xticklabels=False)
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()
def plot_ELBO(train_loss,valid_loss,epochs_list,fold,results_dir):
    """Plots the model's error loss
    :param list train_elbo: list of accumulated error losses during training
    :param str results_dict: path to results directory
    """
    "train_loss, valid_loss,additional_info.results_dir,epochs_list,fold"
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    epochs_idx = np.array(epochs_list)
    train_loss = train_loss[epochs_idx.astype(int)] #select the same epochs as the vaidation
    if np.isnan(train_loss).any():
        print("Error loss contains nan")
        pass
    else:
        plt.plot(epochs_idx,train_loss, color="dodgerblue",label="train")
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
    """Plots the model's error loss
    :param list train_elbo: list of accumulated error losses during training
    :param str results_dict: path to results directory
    """
    "train_loss, valid_loss,additional_info.results_dir,epochs_list,fold"
    train_accuracies = np.array(train_accuracies)
    valid_accuracies = np.array(valid_accuracies)
    epochs_idx = np.array(epochs_list)
    train_accuracies = train_accuracies[epochs_idx.astype(int)] #select the same epochs as the vaidation

    plt.plot(epochs_idx,train_accuracies, color="deepskyblue",label="train")
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
    plt.plot(epochs_idx,valid_auc, color="salmon", label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("{}".format(method))
    plt.title("{} (Train/valid)".format(method))
    plt.legend()
    plt.savefig("{}/{}_{}fold.png".format(results_dir,method,fold))
    plt.close()
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
    ax1.set_title('Gradient norms during SVI')

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
