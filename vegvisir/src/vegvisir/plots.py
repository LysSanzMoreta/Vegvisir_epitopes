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


def plot_model_parameters(train_loader, n_layers=5, hidden_units=100, activation_fn=None, use_bn=False, before=True,
                  model=None):
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
