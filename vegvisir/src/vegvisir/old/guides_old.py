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
from vegvisir.losses import *
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
        self.semi_supervised = model_load.args.semi_supervised
        self.aa_types = model_load.aa_types
        self.max_len = model_load.max_len
        self.seq_max_len = model_load.seq_max_len
        self.hidden_dim = model_load.args.hidden_dim
        self.gru_hidden_dim = self.hidden_dim*2
        self.z_dim = model_load.args.z_dim
        self.device = model_load.args.device
        self.num_classes = model_load.args.num_classes
        self.feats_dim = self.max_len - self.seq_max_len
        self.input_dim = model_load.input_dim
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        self.h_0_GUIDE = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.guide_rnn = RNN_guide(self.aa_types,self.max_len,self.gru_hidden_dim,self.z_dim,self.device)
        self.guide_rnn = RNN_guide(1,self.max_len,self.gru_hidden_dim,self.z_dim,self.device)
        self.guide_fcl1 = FCL1(self.z_dim, self.hidden_dim, self.num_classes, self.device, self.seq_max_len)
        # if self.seq_max_len != self.max_len:
        #     self.guide_fcl3 = FCL3(self.feats_dim,self.hidden_dim*2,self.num_classes,self.device)

    def guide_a(self, batch_data, batch_mask):
        """
        Amortized inference with only sequences
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        # pyro.module("guide_rnn", self.guide_rnn)
        # #pyro.module("guide_fcl1", self.guide_fcl1)
        pyro.module("vae_guide", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm

        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("data", batch_sequences_norm.shape[0],device=self.device): #dim = -2
            z_mean, z_scale = self.guide_rnn(batch_sequences_norm[:,:,None], init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))  # ,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            logits_class = self.guide_fcl1(latent_z,None)
            class_logits = self.logsoftmax(logits_class)
            # smooth_factor = self.losses.label_smoothing(class_logits, true_labels, confidence_scores, self.num_classes)
            # class_logits = class_logits*smooth_factor
            if self.semi_supervised:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs_mask=confidence_mask,obs=true_labels)
            else:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=true_labels)


        return {"latent_z": latent_z,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "class_predictions":None}
    def guide_b(self, batch_data, batch_mask):
        """
        Amortized inference with only sequences
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        # pyro.module("guide_rnn", self.guide_rnn)
        # #pyro.module("guide_fcl1", self.guide_fcl1)
        pyro.module("vae_guide", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm

        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("data", batch_sequences_norm.shape[0],device=self.device): #dim = -2
            z_mean, z_scale = self.guide_rnn(batch_sequences_norm[:,:,None], init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))  # ,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            logits_class = self.guide_fcl1(latent_z,None)
            class_logits = self.logsoftmax(logits_class)
            # smooth_factor = self.losses.label_smoothing(class_logits, true_labels, confidence_scores, self.num_classes)
            # class_logits = class_logits*smooth_factor
            if self.semi_supervised:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits), obs_mask=confidence_mask,obs=true_labels)
            else:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits), obs=true_labels)


        return {"latent_z": latent_z,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "class_predictions":None}

    def guide_c(self, batch_data,batch_mask):
        """
        Amortized inference with features and sequences
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        # pyro.module("guide_rnn", self.guide_rnn)
        # pyro.module("guide_fcl1", self.guide_fcl1)
        #pyro.module("guide_fcl3", self.guide_fcl3)
        pyro.module("vae_guide", self)
        batch_sequences_blosum = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:, 1, self.seq_max_len:, 0]

        batch_sequences_norm = batch_data["norm"][:, 1, :self.seq_max_len]  # only sequences norm
        batch_sequences_feats = batch_data["norm"][:, 1, self.seq_max_len:]  # only features
        batch_sequences_norm_feats = batch_data["norm"][:, 1]  # both

        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        # mean = batch_sequences_norm.mean(dim=1)
        # mean = mean[:,None].expand(batch_sequences_norm.shape[0],self.z_dim)
        #
        # scale = batch_sequences_norm.std(dim = 1)
        # scale = scale[:,None].expand(batch_sequences_norm.shape[0],self.z_dim)

        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_norm_feats.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_latent", batch_sequences_norm_feats.shape[0],device=self.device):
            z_mean, z_scale = self.guide_rnn(batch_sequences_norm_feats[:,:,None],init_h_0)
            assert z_mean.shape == (batch_sequences_norm_feats.shape[0],self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm_feats.shape[0],self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))#,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            logits_class = self.guide_fcl1(latent_z,None)
            #logits_feats = self.guide_fcl3(batch_features)
            class_logits = self.logsoftmax(logits_class)
            #smooth_factor = self.losses.label_smoothing(class_logits, true_labels, confidence_scores, self.num_classes)
            #class_logits = class_logits*smooth_factor
            if self.semi_supervised:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs_mask=confidence_mask,obs=true_labels)
            else:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=true_labels)

        return {"latent_z":latent_z,
                "z_mean":z_mean,
                "z_scale":z_scale,
                "class_predictions": None}


    def guide_d(self, batch_data,batch_mask):
        """
        Amortized inference with features and sequences
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        # pyro.module("guide_rnn", self.guide_rnn)
        # pyro.module("guide_fcl1", self.guide_fcl1)
        #pyro.module("guide_fcl3", self.guide_fcl3)
        pyro.module("vae_guide", self)
        batch_sequences_blosum = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:, 1, self.seq_max_len:, 0]

        batch_sequences_norm = batch_data["norm"][:, 1, :self.seq_max_len]  # only sequences norm
        batch_sequences_feats = batch_data["norm"][:, 1, self.seq_max_len:]  # only features
        batch_sequences_norm_feats = batch_data["norm"][:, 1]  # both

        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        # mean = batch_sequences_norm.mean(dim=1)
        # mean = mean[:,None].expand(batch_sequences_norm.shape[0],self.z_dim)
        #
        # scale = batch_sequences_norm.std(dim = 1)
        # scale = scale[:,None].expand(batch_sequences_norm.shape[0],self.z_dim)

        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_norm_feats.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_latent", batch_sequences_norm_feats.shape[0],dim=-2,device=self.device):
            z_mean, z_scale = self.guide_rnn(batch_sequences_norm_feats[:,:,None],init_h_0)
            assert z_mean.shape == (batch_sequences_norm_feats.shape[0],self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm_feats.shape[0],self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean, z_scale))#,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            logits_class = self.guide_fcl1(latent_z,None)
            #logits_feats = self.guide_fcl3(batch_features)
            class_logits = self.logsoftmax(logits_class)
            #smooth_factor = self.losses.label_smoothing(class_logits, true_labels, confidence_scores, self.num_classes)
            #class_logits = class_logits*smooth_factor
            if self.semi_supervised:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits), obs_mask=confidence_mask,obs=true_labels)
            else:
                class_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits), obs=true_labels)

        return {"latent_z":latent_z,
                "z_mean":z_mean,
                "z_scale":z_scale,
                "class_predictions": None}


    def guide(self,batch_data,batch_mask):
        if self.seq_max_len == self.max_len:
            return self.guide_a(batch_data,batch_mask)
        else:
            return self.guide_c(batch_data,batch_mask)



