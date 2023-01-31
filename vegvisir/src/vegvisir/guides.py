"""
=======================
2023: Lys Sanz Moreta
Vegvisir
=======================
"""
from pyro.contrib.easyguide import EasyGuide
from pyro.nn import PyroParam
from vegvisir.models import *
from vegvisir.model_utils import *
import torch.nn as nn
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.distributions import constraints
class VEGVISIRGUIDES(EasyGuide):
    def __init__(self,vegvisir_model,ModelLoad, Vegvisir):
        super(VEGVISIRGUIDES, self).__init__(vegvisir_model)
        #self.guide_type = ModelLoad.args.select_guide
        self.Vegvisir = Vegvisir
        self.aa_types = Vegvisir.aa_types
        self.max_len = Vegvisir.max_len
        self.gru_hidden_dim = Vegvisir.gru_hidden_dim
        self.z_dim = Vegvisir.z_dim
        self.device = Vegvisir.device
        self.h_0_GUIDE = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.guide_rnn = RNN_guide(self.aa_types,self.max_len,self.gru_hidden_dim,self.z_dim,self.device)

    def guide(self, batch_data,batch_mask):
        """data_blosum is the data encoded with blosum vectors
        batch_blosum is the weighted average of blosum scores per column alignment for a batch of sequences"""
        pyro.module("guide_rnn", self.guide_rnn)
        #pyro.module("gvae_guide", self)
        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        # true_labels = batch_data[:,0,0,0]
        # immunodominance_scores = batch_data[:,0,0,4]
        # confidence_scores = batch_data[:,0,0,5]
        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        print("init h 0")
        print(init_h_0)
        print("....................................")
        #with pyro.plate("data", batch_sequences.shape[0],dim=-3):
        z_mean, z_std = self.guide_rnn(batch_sequences_blosum,init_h_0)
        print("z mean")
        print(z_mean)
        latent_z = pyro.sample("latent_z", dist.Normal(z_mean, z_std))  # [z_dim,n]

        # class_logits = self.guide_class_logits(latent_z, mask=None)
        # # if self.supervised: #TODO: infer={'is_auxiliary': True} ?????
        # #     pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=batch_data["blosum"][:,0,0,0])
        # # else:
        # pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),infer={'is_auxiliary': True})
        #nodes_logits = self.guide_nodes_logits(latent_z,mask=None)

        return {"latent_z":latent_z,
                "z_mean":z_mean,
                "z_std":z_std}
