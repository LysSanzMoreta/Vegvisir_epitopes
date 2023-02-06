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
    def __init__(self,vegvisir_model,model_load, Vegvisir):
        super(VEGVISIRGUIDES, self).__init__(vegvisir_model)
        #self.guide_type = ModelLoad.args.select_guide
        #self.Vegvisir = Vegvisir
        self.aa_types = model_load.aa_types
        self.max_len = model_load.max_len
        self.seq_max_len = model_load.seq_max_len
        self.hidden_dim = model_load.args.hidden_dim
        self.gru_hidden_dim = self.hidden_dim*2
        self.z_dim = model_load.args.z_dim
        self.device = model_load.args.device
        self.h_0_GUIDE = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.guide_rnn = RNN_guide(self.aa_types,self.seq_max_len,self.gru_hidden_dim,self.z_dim,self.device)

    def guide(self, batch_data,batch_mask):
        """
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        pyro.module("guide_rnn", self.guide_rnn)
        #pyro.module("gvae_guide", self)
        batch_sequences_blosum = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        #self.map_estimate("predictions")
        # true_labels = batch_data[:,0,0,0]
        # immunodominance_scores = batch_data[:,0,0,4]
        # confidence_scores = batch_data[:,0,0,5]
        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("data", batch_sequences_blosum.shape[0],dim=-2):
            z_mean, z_scale = self.guide_rnn(batch_sequences_blosum,init_h_0)
            assert z_mean.shape == (batch_sequences_blosum.shape[0],self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_blosum.shape[0],self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean, z_scale))#,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]

        # class_logits = self.guide_class_logits(latent_z, mask=None)
        # # if self.supervised: #TODO: infer={'is_auxiliary': True} ?????
        # #     pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=batch_data["blosum"][:,0,0,0])
        # # else:
        # pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),infer={'is_auxiliary': True})

        return {"latent_z":latent_z,
                "z_mean":z_mean,
                "z_scale":z_scale}


