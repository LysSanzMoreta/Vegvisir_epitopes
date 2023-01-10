"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import torch.nn as nn
import torch
from abc import abstractmethod
from vegvisir.model_utils import *
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
    @abstractmethod
    def forward(self,batch_data,batch_mask):
        raise NotImplementedError
    @abstractmethod
    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name
class VegvisirModel1(VEGVISIRModelClass):
    """
    """

    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #self.mlp = MLP(self.aa_types*self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device)

    def forward(self,batch_data,batch_mask):
        """"""
        batch_sequences = batch_data[:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #probs = self.mlp(batch_sequences.flatten(1),None)
        probs = self.cnn(batch_sequences.permute(0,2,1),None)

        return probs

    def loss(self,confidence_scores,true_labels,logits):
        """Weighted loss according to confidence score"""
        if self.loss_type == "weighted_loss":
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores)
            output = bce_loss(torch.max(logits,dim=1).values, true_labels)
        elif self.loss_type == "bce":
            bce_loss = nn.BCELoss()
            output = bce_loss(torch.max(logits, dim=1).values, true_labels)
        return output