"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import numpy as np
import pyro
import torch.nn as nn
import torch
from collections import defaultdict,namedtuple
from abc import abstractmethod
from pyro.nn import PyroModule
import pyro.distributions as dist
from pyro.infer import Trace_ELBO,JitTrace_ELBO
from vegvisir.model_utils import *
from vegvisir.losses import *
ModelOutput = namedtuple("ModelOutput",["reconstructed_sequences","class_out"])
SamplingOutput = namedtuple("SamplingOutput",["latent_z","predicted_labels","immunodominance_scores","reconstructed_sequences"])

class VEGVISIRModelClass(nn.Module):
    def __init__(self, model_load):
        super(VEGVISIRModelClass, self).__init__()
        self.aa_types = model_load.aa_types
        self.seq_max_len = model_load.seq_max_len
        self.max_len = model_load.max_len
        self.batch_size = model_load.args.batch_size
        self.input_dim = model_load.input_dim
        self.hidden_dim = model_load.args.hidden_dim
        self.z_dim = model_load.args.z_dim
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
    Multilayer Perceptron model
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: MLP
        self.mlp = MLP(self.aa_types*self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: MLP
        class_out = self.mlp(batch_sequences.flatten(1),None)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = torch.tensor([1,int(npositives/nnegatives)])
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = torch.tensor([int(nnegatives/npositives),1])
        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss()
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel2a(VEGVISIRModelClass):
    """
    Convolutional networks based models without information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: CNN
        self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        #self.letnet5 = LetNET5(self.aa_types, self.max_len, self.hidden_dim * 2, self.num_classes, self.device,self.loss_type)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: CNN
        class_out = self.cnn(batch_sequences.permute(0,2,1),None)

        #logits,class_out = self.letnet5(batch_sequences.permute(0, 2, 1), None)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = torch.tensor([1,int(npositives/nnegatives)])
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = torch.tensor([int(nnegatives/npositives),1])
        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels,predictions,confidence_scores,self.num_classes)
            return loss
        elif self.loss_type == "bceprobs":

            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel2b(VEGVISIRModelClass):
    """
    Convolutional networks based models without information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.feats_dim = self.max_len - self.seq_max_len
        #Highlight: CNN
        self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        self.fcl3 = FCL3(self.feats_dim,self.hidden_dim*2,self.num_classes,self.device)
        #self.letnet5 = LetNET5(self.aa_types, self.max_len, self.hidden_dim * 2, self.num_classes, self.device,self.loss_type)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        :type batch_data: object
        """
        batch_sequences = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:,1,self.seq_max_len:,0]
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: CNN
        logits_seqs = self.cnn(batch_sequences.permute(0,2,1),None)
        logits_feats = self.fcl3(batch_features)
        # print("sequences")
        # print(logits_seqs)
        # print("features")
        #logits,class_out = self.letnet5(batch_sequences.permute(0, 2, 1), None)
        class_out = logits_seqs + logits_feats
        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = [torch.tensor([1,int(npositives/nnegatives)]) if nnegatives != 0 else torch.tensor([1,1])][0]
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = [torch.tensor([int(nnegatives/npositives),1]) if npositives != 0 else torch.tensor([1,1])][0]

        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels,predictions,confidence_scores,self.num_classes)
            return loss
        elif self.loss_type == "bceprobs":

            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel3a(VEGVISIRModelClass):
    """
    Recurrent Neural Networks models with information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: RNN
        self.gru_hidden_dim = self.hidden_dim*2
        self.rnn = RNN_layers(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.device,self.loss_type)
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.h_0_MODEL_r = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.c_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device) #For LSTM
        self.losses = VegvisirLosses(self.max_len,self.input_dim)


    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: RNN
        init_h_0 = self.h_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        #init_h_0_r = self.h_0_MODEL_r.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        #init_c_0 = self.c_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        class_out = self.rnn(batch_sequences,None,init_h_0)
        #class_out = self.rnn(batch_sequences,None,init_h_0_r)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = torch.tensor([1,int(npositives/nnegatives)])
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = torch.tensor([int(nnegatives/npositives),1])
        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel3b(VEGVISIRModelClass):
    """
    Recurrent Neural Networks models with information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.feats_dim = self.max_len - self.seq_max_len
        #Highlight: RNN
        self.gru_hidden_dim = self.hidden_dim*2
        self.rnn = RNN_layers(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.device,self.loss_type)
        self.fcl3 = FCL3(self.feats_dim,self.hidden_dim*2,self.num_classes,self.device)
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.c_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device) #For LSTM
        self.losses = VegvisirLosses(self.max_len,self.input_dim)


    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        batch_features = batch_data[:,1,self.seq_max_len:,0]

        #Highlight: RNN
        init_h_0 = self.h_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        logits_seqs = self.rnn(batch_sequences,None,init_h_0)
        logits_feats = self.fcl3(batch_features)
        class_out = logits_seqs + logits_feats

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = torch.tensor([1,int(npositives/nnegatives)])
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = torch.tensor([int(nnegatives/npositives),1])
        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel4(VEGVISIRModelClass):
    """
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: Autoencoder
        self.autoencoder = AutoEncoder(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        self.sigmoid = nn.Sigmoid()
        self.losses = VegvisirLosses(self.max_len,self.input_dim)


    def forward(self,batch_data,batch_mask):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: Autoencoder
        reconstructed_seqs,class_out = self.autoencoder(batch_sequences.permute(0,2,1))

        return ModelOutput(reconstructed_sequences=reconstructed_seqs,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return:
        :rtype: object
        """

        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = torch.tensor([1,int(npositives/nnegatives)])
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = torch.tensor([int(nnegatives/npositives),1])
        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "ae_loss":
            #reconstruction_loss = nn.CosineEmbeddingLoss(reduction='none')(onehot_sequences[:,1],model_outputs.reconstructed_sequences)
            reconstruction_loss = self.losses.argmax_reconstruction_loss(model_outputs.reconstructed_sequences,onehot_sequences[:,1])
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions,dim=1).values.squeeze(-1)
                #TODO: torch.round()
            else:
                predictions = predictions.squeeze(-1)
            classification_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)(predictions,true_labels)
            total_loss = reconstruction_loss + classification_loss.mean()

            return total_loss
        else:
            raise ValueError(
                "Error loss: {} not implemented for this model type: {}".format(self.loss_type, self.get_class()))


class VegvisirModel5(VEGVISIRModelClass,PyroModule):
    """
    Bayesian Regression
    -Notes: http://pyro.ai/examples/bayesian_regression.html
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        self.model_rnn = RNN_model(self.aa_types,self.max_len,self.gru_hidden_dim,self.aa_types,self.z_dim,self.device,self.loss_type)
        self.fcl1 = FCL1(self.z_dim,self.hidden_dim,self.num_classes,self.device,self.max_len)
        self.fcl2 = FCL2(self.z_dim,self.hidden_dim,self.num_params,self.device,self.max_len)
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def model(self,batch_data,batch_mask):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
        """
        pyro.module("model_rnn", self.model_rnn)
        pyro.module("model_fcl1", self.fcl1)
        pyro.module("model_fcl2", self.fcl2)

        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        batch_sequences_int = batch_data["int"][:,1].squeeze(1)
        batch_mask = batch_mask[:,1:].squeeze(1)
        batch_mask = batch_mask[:,:,0]
        true_labels = batch_data["blosum"][:,0,0,0]
        immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] == 1.).any(-1)
        with pyro.plate("plate_latent", batch_sequences_blosum.shape[0], dim=-2):
            mean = torch.zeros((batch_sequences_blosum.shape[0], self.z_dim)) + 1
            scale = torch.ones((batch_sequences_blosum.shape[0], self.z_dim))*0.1
            latent_z = pyro.sample("latent_z", dist.Normal(mean, scale))  # [n,z_dim]
            print("latent z model")
            print(latent_z)
            class_logits = self.fcl1(latent_z,None)
            pyro.sample("predictions", dist.Categorical(class_logits), obs=true_labels)
            beta_params = self.fcl2(latent_z, None)
            # beta = pyro.sample("beta",dist.Uniform(0.4,0.6))
            # alpha = pyro.sample("alpha",dist.Uniform(0.5,0.7))
            beta = beta_params[:, 0]
            alpha = beta_params[:, 1]
            #pyro.sample("immunodominance_prediction", dist.Beta(beta, alpha), obs_mask=confidence_mask,obs=immunodominance_scores)  # obs_mask  If provided, events with mask=True will be conditioned on obs and remaining events will be imputed by sampling.
            pyro.sample("immunodominance_prediction", dist.Beta(beta, alpha),obs=immunodominance_scores)  # obs_mask  If provided, events with mask=True will be conditioned on obs and remaining events will be imputed by sampling.

        latent_z = latent_z.repeat(1, self.max_len).reshape(latent_z.shape[0], self.max_len, self.z_dim)
        latent_z[~batch_mask] = 0
        init_h_0 = self.h_0_MODEL.expand(self.model_rnn.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("data_len",batch_sequences_blosum.shape[1],dim=-1):
            with pyro.plate("data", batch_sequences_blosum.shape[0],dim=-2):
                #Highlight: Forward network
                sequences_logits = self.model_rnn(latent_z,init_h_0)
                sequences_logits = self.logsoftmax(sequences_logits)
                pyro.sample("sequences",dist.Categorical(sequences_logits),obs=batch_sequences_int)

        return {"sequences_logits":sequences_logits,
                "beta":beta,
                "alpha":alpha}

    def sample(self,batch_data,batch_mask,guide_estimates):
        """"""
        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        batch_mask = batch_mask[:,1:].squeeze(1)
        batch_mask = batch_mask[:,:,0]
        #true_labels = batch_data[:,0,0,0]
        #immunodominance_scores = batch_data[:,0,0,4]
        #confidence_scores = batch_data[:,0,0,5]
        #confidence_mask = (confidence_scores[..., None] == 1.).any(-1)
        #Highlight: Forward network
        with pyro.plate("plate_latent", batch_sequences_blosum.shape[0], dim=-2):
            mean = torch.zeros((batch_sequences_blosum.shape[0], self.z_dim))
            scale = torch.ones((batch_sequences_blosum.shape[0], self.z_dim))
            latent_z = pyro.sample("latent_z", dist.Normal(mean, scale))  # [n,z_dim]
            class_logits = self.fcl1(latent_z, None)
            predicted_labels=dist.Categorical(class_logits).sample()
            beta_params = self.fcl2(latent_z, None)
            # beta = pyro.sample("beta",dist.Uniform(0.4,0.6))
            # alpha = pyro.sample("alpha",dist.Uniform(0.5,0.7))
            beta = beta_params[:, 0]
            alpha = beta_params[:, 1]
            predicted_immunodominance_scores= dist.Beta(beta, alpha)  # obs_mask  If provided, events with mask=True will be conditioned on obs and remaining events will be imputed by sampling.

        latent_z = latent_z.repeat(1, self.max_len).reshape(latent_z.shape[0], self.max_len, self.z_dim)
        latent_z[~batch_mask] = 0
        init_h_0 = self.h_0_MODEL.expand(self.model_rnn.num_layers * 2, batch_sequences_blosum.shape[0],
                                         self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("data_len", batch_sequences_blosum.shape[1], dim=-1):
            with pyro.plate("data", batch_sequences_blosum.shape[0], dim=-2):
                # Highlight: Forward network
                sequences_logits = self.model_rnn(latent_z, init_h_0)
                sequences_logits = self.logsoftmax(sequences_logits)
                reconstructed_sequences = dist.Categorical(sequences_logits).sample()

        return SamplingOutput(latent_z = latent_z,
                              predicted_labels=predicted_labels,
                              immunodominance_scores=predicted_immunodominance_scores,
                              reconstructed_sequences = reconstructed_sequences)

    def loss(self):
        """
        """
        return Trace_ELBO()
        #return Trace_ELBO_classification(self.max_len,self.input_dim,self.num_classes)


class VegvisirModel6(VEGVISIRModelClass):
    """
    Recurrent Neural Networks models with information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: RNN
        self.nna = NNAlign(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        seq_lens = batch_data["int"][:,1].bool().sum(1)
        #batch_sequences_int = batch_data["int"][:,1]
        #Highlight: NNAlign
        class_out = self.nna(batch_sequences_blosum,seq_lens)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
        nnegatives = true_labels.shape[0] - npositives
        if npositives > nnegatives:
            pos_weights = torch.tensor([1,int(npositives/nnegatives)])
        elif npositives == 0:
            pos_weights = torch.tensor([0.5,1])
        elif nnegatives == 0:
            pos_weights = torch.tensor([1,0.5])
        else:
            pos_weights = torch.tensor([int(nnegatives/npositives),1])
        class_weights = true_labels.clone()
        class_weights[class_weights == 0] = pos_weights[0]
        class_weights[class_weights == 1] = pos_weights[1]
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))


