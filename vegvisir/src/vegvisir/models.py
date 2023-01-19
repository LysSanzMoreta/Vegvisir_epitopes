"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import numpy as np
import torch.nn as nn
import torch
from collections import defaultdict,namedtuple
from abc import abstractmethod
from vegvisir.model_utils import *
ModelOutput = namedtuple("ModelOutput",["reconstructed_sequences","class_out"])

class VEGVISIRModelClass(nn.Module):
    def __init__(self, model_load):
        super(VEGVISIRModelClass, self).__init__()
        self.aa_types = model_load.aa_types
        self.embedding_dim = 50
        self.hidden_dim = 60
        self.max_len = model_load.max_len
        self.batch_size = model_load.args.batch_size
        self.input_dim = model_load.input_dim
        self.hidden_dim = model_load.args.hidden_dim
        self.device = model_load.args.device
        self.use_cuda = model_load.args.use_cuda
        #self.dropout = model_load.args.dropout
        self.num_classes = model_load.args.num_classes
        self.embedding_dim = model_load.args.embedding_dim
        self.blosum = model_load.blosum
        self.loss_type = model_load.args.loss_func
        if self.use_cuda:
            # calling cuda() here will put all the parameters of
            # the networks into gpu memory
            self.cuda()

        self.gradients_dict = {}
        self.handles_dict = defaultdict(list)
        self.visualization_dict = {}
        self.parameters_dict = {}
        self._parameters = {}
    @abstractmethod
    def forward(self,batch_data,batch_mask):
        raise NotImplementedError
    @abstractmethod
    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, filename,optimizer):
        """Stores the model weight parameters and optimizer status"""
        # Builds dictionary with all elements for resuming training
        checkpoint = {'model_state_dict': self.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename,optimizer=None):
        # Loads dictionary
        checkpoint = torch.load(filename)
        # Restore state for model and optimizer
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train() # resume training

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        """"""
        # Clear any previous values
        self.visualization_dict = {}
        # Creates the dictionary to map layer objects to their names
        modules = list(self.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]} #TODO: why modules[1:]

        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                # Gets the layer name
                name = layer_names[layer]
                # Detaches outputs
                values = outputs.detach().cpu().numpy()
                # Since the hook function may be called multiple times
                # for example, if we make predictions for multiple mini-batches
                # it concatenates the results
                if self.visualization_dict[name] is None:
                    self.visualization_dict[name] = values
                else:
                    self.visualization_dict[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            # If the layer is in our list
            if name in layers_to_hook:
                # Initializes the corresponding key in the dictionary
                self.visualization_dict[name] = None #it will be overwritten
                # Register the forward hook and keep the handle in another dict
                self.handles_dict[name] = layer.register_forward_hook(hook_fn)
        return self.visualization_dict

    def remove_hooks(self):
        # Loops through all hooks and removes them
        for handle in self.handles_dict.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}

    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        def make_log_fn(name, parm_id):
            def log_fn(grad):
                self.gradients_dict[name][parm_id].append(grad.tolist())
                return
            return log_fn

        for name, layer in self.named_modules():
            if name in layers_to_hook:
                self.gradients_dict.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    if p.requires_grad:
                        self.gradients_dict[name].update({parm_id: []})
                        log_fn = make_log_fn(name, parm_id)
                        self.handles_dict[f'{name}.{parm_id}.grad'] = p.register_hook(log_fn)
        return self.gradients_dict

    def capture_parameters(self):

        modules = list(self.named_modules())
        layer_names = {layer: name for name, layer in modules}
        for name, layer in modules:
                self.parameters_dict.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    self._parameters[name].update({parm_id: []})

        def fw_hook_fn(layer, inputs, outputs):
            name = layer_names[layer]
            for parm_id, parameter in layer.named_parameters():
                self._parameters[name][parm_id].append(parameter.tolist())

        self.attach_hooks(layer_names.values(), fw_hook_fn)
        return self.parameters_dict


class VegvisirModel1(VEGVISIRModelClass):
    """
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: MLP
        #self.mlp = MLP(self.aa_types*self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        #Highlight: CNN
        #self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        #self.letnet5 = LetNET5(self.aa_types, self.max_len, self.hidden_dim * 2, self.num_classes, self.device,self.loss_type)
        #Highlight: RNN
        #self.gru_hidden_dim = self.hidden_dim*2
        #self.rnn = RNN_layers(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.device,self.loss_type)
        #self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.c_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #Highlight: Autoencoder
        self.autoencoder = AutoEncoder(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        self.sigmoid = nn.Sigmoid()


    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data[:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: MLP
        #probs = self.mlp(batch_sequences.flatten(1),None)
        #Highlight: CNN
        #probs = self.cnn(batch_sequences.permute(0,2,1),None)
        #logits,probs = self.letnet5(batch_sequences.permute(0, 2, 1), None)
        #Highlight: RNN
        #init_h_0 = self.h_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        #init_c_0 = self.c_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        #probs = self.rnn(batch_sequences,init_h_0,None)
        #Highlight: Autoencoder
        reconstructed_seqs,class_out = self.autoencoder(batch_sequences.permute(0,2,1))

        return ModelOutput(reconstructed_sequences=reconstructed_seqs,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """Weighted loss according to confidence score:
        Notes:
            - https://hal.science/hal-02547012/document
            - https://www.researchgate.net/publication/336069340_A_Comparison_of_Loss_Weighting_Strategies_for_Multi_task_Learning_in_Deep_Neural_Networks
            - Multitask Learning Based on Improved Uncertainty Weighted Loss for Multi-Parameter Meteorological Data Prediction #TODO!!!
            - Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
            """

        predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
        predictions=predictions[torch.arange(0,true_labels.shape[0]),true_labels.long()]
        positives_proportions = true_labels.sum()/true_labels.shape[0]*10 # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        weights_dict = {0:positives_proportions,1:1-positives_proportions} #invert the weights
        class_weights = true_labels.clone()
        class_weights[class_weights==0] = weights_dict[0]
        class_weights[class_weights==1] = weights_dict[1]

        if self.loss_type == "weighted_bce":

            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.weighted_loss(confidence_scores,predictions,None)
            loss = self.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "ae_loss":
            #reconstruction_loss = nn.CosineEmbeddingLoss(reduction='none')(onehot_sequences[:,1],model_outputs.reconstructed_sequences)
            reconstruction_loss = self.argmax_reconstruction_loss(model_outputs.reconstructed_sequences,onehot_sequences[:,1])
            classification_loss = nn.BCELoss(weight=confidence_scores)(predictions,true_labels)
            total_loss = reconstruction_loss + classification_loss.mean()

            return total_loss
        elif self.loss_type == "bceprobs":
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss


    def weighted_loss(self,y_true, y_pred, confidence_scores):
        """E(y_true,y_pred) = -y_true*log(y_pred) -(1 - y_true)*ln(1-y_pred)
        E(y_true,y_pred) = -weights[pos_weight*y_true*log(sigmoid(y_pred)) + (1-y_true)*log(1 -sigmoid(y_pred))]
        Notes:
            - Analyzing BCE: https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
            - https://www.mldawn.com/binary-classification-from-scratch-using-numpy/
            - Weighted BCE= https://github.com/KarimMibrahim/Sample-level-weighted-loss/blob/master/IM-WCE-MSCOCO.ipynb
            - Cross entropy: https://vitalflux.com/cross-entropy-loss-explained-with-python-examples/
            - Logistic regression with BCE: https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
           :param y_true: Probabilities values (min= 0, max= 1)
           NOTE: BCE seems to favour the prediction of class 1 correctly?
           y_true | y_pred  | BCE
           --------------------------------------
              1   |  0.8    | ((-1 * torch.log(0.8)) - ((1.0 - 1) * torch.log(1.0 - 0.8))) = 0.2231
              1   |  0.51   | ((-1 * torch.log(0.51)) - ((1.0 - 1) * torch.log(1.0 - 0.51))) = 0.6733
              0   |  0.8    | ((-0 * torch.log(0.8)) - ((1.0 - 0) * torch.log(1.0 - 0.8))) = 1.6094
              0   |  0.51   | ((-0 * torch.log(0.51)) - ((1.0 - 0) * torch.log(1.0 - 0.51))) = 0.7133


            """
        #beta = (y_true.shape[0] - y_true.sum())/y_true.shape[0] #posible class imbalance corrector
        # clip to prevent NaN's and Inf's
        y_pred = torch.clip(self.sigmoid(y_pred), 1e-7, 1 - 1e-7)
        #loss = ((-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred)))*confidence_scores
        #loss = beta*(-y_true * torch.log(y_pred)) - ((1 - beta)*(1.0 - y_true) * torch.log(1.0 - y_pred))
        loss = (-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred))

        return loss.mean()

    def focal_loss(self,y_true, y_pred, confidence_scores):
        """Notes:
        -https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html"""
        gamma = 0.5 #downweights the easy to classify examples
        y_pred = torch.clip(self.sigmoid(y_pred), 1e-7, 1 - 1e-7)
        ce_loss = (-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred))
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** gamma)*confidence_scores #gamma = 0 ---> focal_loss = ce_loss

        return loss.mean()

    def dice_reconstruction_loss(self,reconstructed_sequences,onehot_sequences,smooth=1):
            """ Sørensen–Dice coefficient . Notes:
            - https://towardsdatascience.com/building-autoencoders-on-sparse-one-hot-encoded-data-53eefdfdbcc7"""
            # comment out if your model contains a sigmoid acitvation
            y_pred = self.sigmoid(reconstructed_sequences)

            # flatten label and prediction tensors
            y_pred = y_pred.view(-1)
            y_true = onehot_sequences.view(-1)

            intersection = (y_pred * y_true).sum()
            dice = (2. * intersection + smooth) /(y_pred.sum() + y_true.sum() + smooth)

            return 1 - dice

    def argmax_reconstruction_loss(self,reconstructed_sequences,onehot_sequences):

        assert reconstructed_sequences.shape == (onehot_sequences.shape[0],self.max_len,self.input_dim)
        idx_true = torch.argmax(onehot_sequences,dim=-1)
        idx_pred = torch.argmax(reconstructed_sequences,dim=-1)
        loss = 1 - (idx_true == idx_pred).float().mean()
        return loss




