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
import umap
import vegvisir.utils as VegvisirUtils
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression

plt.style.use('ggplot')
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
        colormap = matplotlib.cm.get_cmap('plasma', len(unique))
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
    umap_proj = reducer.fit_transform(latent_space[:, 3:]) #First column are TCR-pMHC identifiers,second column are "true" labels
    colors_dict = {0:"mediumaquamarine",1:"orangered"}
    colors_true = np.vectorize(colors_dict.get)(latent_space[:,1])
    colors_predicted = np.vectorize(colors_dict.get)(predictions)
    #colors_peptides_identifiers = np.vectorize(colors_dict_peptides.get)(latent_space[:,0])

    fig, [[ax1, ax2, ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]] = plt.subplots(3, 3,figsize=(17, 12),gridspec_kw={'width_ratios': [4.5,4.5,1],'height_ratios': [4,4,2]})
    fig.suptitle('UMAP projections',fontsize=20)
    ax1.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_true, label=latent_space[:,2], alpha=1,s=30)
    ax1.set_title("True labels",fontsize=20)
    ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_predicted, label=predictions, alpha=1,s=30)
    ax2.set_title("Predicted labels",fontsize=20)
    # ax4.scatter(umap_proj[:, 0], umap_proj[:, 1], color=colors_peptides_identifiers, label=latent_space[:,0], alpha=1, s=30)
    # ax4.set_title("Peptides identifiers", fontsize=20)
    ax3.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")
    ax9.axis("off")
    negative_patch = mpatches.Patch(color=colors_dict[0], label='Class 0')
    positive_patch = mpatches.Patch(color=colors_dict[1], label='Class 1')
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

def plot_feature_importance_old(feature_dict:dict,max_len:int,feature_columns:list,results_dir:str) -> None:
    """
    :rtype: object
    :param feature_dict:
    :param max_len:
    :param feature_columns:
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
        labels = ["Pos.{}".format(pos) for pos in list(range(max_len))] + feature_columns
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

def plot_feature_importance(feature_dict:dict,max_len:int,feature_columns:list,results_dir:str) -> None:
    """
    :rtype: object
    :param feature_dict:
    :param max_len:
    :param feature_columns:
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
        labels = ["Pos.{}".format(pos) for pos in list(range(max_len))] + feature_columns
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

def plot_layers(model,layer_name,results_dir): #TODO: Check
    # Gets the layer object from the model
    for name in layer_name.split('.'):
        layer = getattr(model, name) # Gets the layer object from the model

    # We are only looking at filters for 2D convolutions
    # Takes the weight information
    weights = layer.weight.data.cpu().numpy()
    # The weights have channels_out (filter), channels_in, H, W shape
    n_filters, n_channels, _, _ = weights.shape

    # Builds a figure
    size = (2 * n_channels + 2, 2 * n_filters)
    fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
    axes = np.atleast_2d(axes).reshape(n_filters, n_channels)
    # For each channel_out (filter)
    for i in range(n_filters):
        axs= axes[i,:]
        x= weights[i]
        yhat = None
        y= None
        title = 'Channel' if (i == 0) else None
        layer_name = 'Filter #{}'.format(i),
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape),
                rotation=0, labelpad=40
            )
            xlabel1 = '' if y is None else '\nLabel: {}'.format(y[j])
            xlabel2 = '' if yhat is None else '\nPredicted: {}'.format(yhat[j])
            xlabel = '{}{}'.format(xlabel1, xlabel2)
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )


    for ax in axes.flat:
        ax.label_outer()

    fig.tight_layout()

    plt.savefig("{}/layers.png".format(results_dir),dpi=200)


    return fig

def plot_model_parameters(train_loader, n_layers=5, hidden_units=100, activation_fn=None, use_bn=False, before=True,model=None):
    #(train_loader, n_layers=5, hidden_units=100, activation_fn=None, use_bn=False, before=True,model=None)
    import sys
    sys.path.append('..')
    from stepbystep.v3 import StepByStep

    if model is None:
        n_features = train_loader.dataset.tensors[0].shape[1]
        if activation_fn is None:
            activation_fn = nn.ReLU
        model = build_model(n_layers, n_features, hidden_units, activation_fn, use_bn, before)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    n_layers = len(list(filter(lambda c: c[0][0] == 'h', model.named_children())))

    sbs = StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader)
    sbs.capture_parameters([f'h{i}' for i in range(1, n_layers + 1)])
    sbs.capture_gradients([f'h{i}' for i in range(1, n_layers + 1)])
    sbs.attach_hooks([f'a{i}' for i in range(1, n_layers + 1)])
    sbs.train(1)

    names = [f'h{i}' for i in range(1, n_layers + 1)]

    parameters = [[np.array(sbs._parameters[f'h{i}']['weight']).reshape(-1, ) for i in range(1, n_layers + 1)]]
    parms_data = LayerViolinsData(names=names, values=parameters)

    gradients = [[np.array(sbs._gradients[f'h{i}']['weight']).reshape(-1, ) for i in range(1, n_layers + 1)]]
    gradients_data = LayerViolinsData(names=names, values=gradients)

    activations = [[np.array(sbs.visualization[f'a{i}']).reshape(-1, ) for i in range(1, n_layers + 1)]]
    activations_data = LayerViolinsData(names=names, values=activations)

    return parms_data, gradients_data, activations_data

def plot_confusion_matrix(confusion_matrix,accuracy,results_dir):
    """Plot confusion matrix
    :param pandas dataframe confusion_matrix"""
    confusion_matrix_array = confusion_matrix.to_numpy()
    fig,ax = plt.subplots(figsize=(7,7))
    plt.imshow(confusion_matrix_array,cmap='Pastel1_r')
    for i in range(confusion_matrix_array.shape[0]):
        for j in range(confusion_matrix_array.shape[1]):
              ax.text(j, i, "{:.2f}".format(confusion_matrix_array[i, j]), ha="center", va="center")
    #[[true_negatives,false_positives],[false_negatives,true_positives]]
    plt.xticks([0,1],confusion_matrix.columns)
    plt.yticks([0,1],confusion_matrix.index)
    plt.title("Confusion matrix. Accuracy: {}".format(accuracy))
    plt.savefig("{}/confusion_matrix.png".format(results_dir),dpi=100)
    plt.clf()