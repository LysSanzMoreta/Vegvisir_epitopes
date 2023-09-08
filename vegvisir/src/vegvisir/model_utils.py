import time

import torch.nn as nn
import torch
from pyro.nn import PyroModule
import  vegvisir
from vegvisir.utils import extract_windows_vectorized
from collections import namedtuple
OutputNN = namedtuple("OutputNN",["output","attn_weights","encoder_hidden_states","decoder_hidden_states","encoder_final_hidden","decoder_final_hidden","init_h_0_decoder"])

class Attention1(nn.Module):
    ''' Scaled Dot-Product Attention as in Attention is all You need, which is based on Dot-product Attention from Luong 2015, and scaled by Vaswani in 2017
    Cross product attention: matrix Q comes from the previous decoder layer, while the key and value matrices K and V come from the encoder
    Notes:
        -https://www.linkedin.com/pulse/explanation-attention-based-encoder-decoder-deep-keshav-bhandari/
        -https://storrs.io/attention/
        -https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
    '''

    def __init__(self,aa_types,hidden_dim,embedding_dim ,device, attn_dropout=0.1):
        super().__init__()
        self.temperature = self.hidden_size ** 0.5 #d_k**0.5, where d_k == hidden_size
        self.dropout = nn.Dropout(attn_dropout)
        self.aa_types = aa_types
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.aa_types,self.embedding_dim)
        self.weight_q = nn.Parameter(torch.randn((self.embedding_dim,self.hidden_dim)),requires_grad=True).to(device)
        self.weight_k = nn.Parameter(torch.randn((self.embedding_dim,self.hidden_dim)),requires_grad=True).to(device)
        self.weight_v = nn.Parameter(torch.randn((self.embedding_dim,self.hidden_dim)),requires_grad=True).to(device)

    def forward(self, input, mask=None):

        input = self.embedding(input)

        q = torch.matmul(input,self.weight_q)
        k = torch.matmul(input,self.weight_k)
        v = torch.matmul(input,self.weight_v)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.nn.softmax(attn, dim=-1)) #attention weights!!!!
        output = torch.matmul(attn, v) #context vector: attention-weighted version of our original query input

        return output, attn

class Attention2(nn.Module):
    """
    Self-attention (scaled dot product) product attention: matrix Q comes from the previous decoder layer, while the key and value matrices K and V come from the encoder

    Notes:
            -https://theaisummer.com/self-attention/
            TODO: https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
                    https://blog.floydhub.com/attention-mechanism/
    """
    def __init__(self, hidden_dim,z_dim,device,attn_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.temperature = self.hidden_dim ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        #self.attention = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.weight_q = nn.Parameter(torch.randn((self.hidden_dim + self.z_dim, self.hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_k = nn.Parameter(torch.randn((self.hidden_dim + self.z_dim, self.hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_v = nn.Parameter(torch.randn((self.hidden_dim + self.z_dim, self.hidden_dim)), requires_grad=True).to(device=self.device)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_hidden_state,decoder_hidden_state, latent_z_seq, mask=None):
        #batch_size = encoder_outputs.shape[0]
        src_len = latent_z_seq.shape[1]
        encoder_hidden_state = torch.sum(encoder_hidden_state,dim=0) #because it is bidirectional
        # repeat encoder/decoder hidden state src_len times
        encoder_hidden_state = encoder_hidden_state.unsqueeze(1).repeat(1, src_len, 1)
        attn_input = torch.cat((encoder_hidden_state, latent_z_seq),dim=2)
        q = torch.matmul(attn_input,self.weight_q)
        k = torch.matmul(attn_input,self.weight_k)
        v = torch.matmul(attn_input,self.weight_v)
        #attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = torch.matmul(q, k.transpose(1, 2))/ self.temperature
        #Highlight: The Z representations and identical per position

        # apply the mask so that the model don't pay the attention to paddings.
        # it's applied before softmax so that the masked values which are very
        # small will be zero'ed out after softmax
        if mask is not None:
            mask=mask[:,:,None].tile(1,1,attn.shape[2])
            attn = attn.masked_fill(mask == 0, -1e9)
        attn =torch.nn.functional.softmax(attn, dim=-1)  # attention weights!!!!
        output = torch.matmul(attn, v)  # context vector: attention-weighted version of our original query input
        return output, attn

class Attention3(nn.Module):
    """
    Self-attention (scaled dot product) product attention: matrix Q comes from the previous decoder layer, while the key and value matrices K and V come from the encoder

    Notes:
            -https://theaisummer.com/self-attention/
            TODO: https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
                    https://blog.floydhub.com/attention-mechanism/
    """
    def __init__(self, hidden_dim,z_dim,device,attn_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.temperature = self.hidden_dim ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        #self.attention = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.weight_q = nn.Parameter(torch.randn((self.hidden_dim, self.hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_k = nn.Parameter(torch.randn((self.hidden_dim, self.hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_v = nn.Parameter(torch.randn((self.z_dim, self.z_dim)), requires_grad=True).to(device=self.device)
        #self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_rnn_out,decoder_rnn_out, latent_z_seq, mask=None):

        q = torch.matmul(encoder_rnn_out,self.weight_q)
        k = torch.matmul(decoder_rnn_out,self.weight_k)
        v = torch.matmul(latent_z_seq,self.weight_v)

        attn = torch.matmul(q, k.transpose(1, 2))/ self.temperature

        # apply the mask so that the model don't pay the attention to paddings.
        # it's applied before softmax so that the masked values which are very
        # small will be zero'ed out after softmax
        if mask is not None:
            mask=mask[:,:,None].tile(1,1,attn.shape[2])
            attn = attn.masked_fill(mask == 0, -1e9)

        attn =torch.nn.functional.softmax(attn, dim=-2)  # attention weights!!!!
        output = torch.matmul(attn, v)  # context vector: attention-weighted version of our original value input
        return output, attn

class Attention4(nn.Module):
    """
    Self-attention (scaled dot product) product attention: matrix Q comes from the previous decoder layer, while the key and value matrices K and V come from the encoder
    Queries. Keys, Values: Source sequence
    RECAP: Cross-attention
        Computes a weighted average of the encoder's hidden states
        Values: Encoder hidden states
        Keys: Encoder hidden states
        Queries. Decoder hidden states
        Source sequence: Input of the encoder
        Last element from source sequence and the final hidden state from the encoder are given to the decoder

    Notes:
        Highlight: https://github.com/VeritasAuthorship/veritas
        https://theaisummer.com/self-attention/
        TODO: https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
                    https://blog.floydhub.com/attention-mechanism/
    """
    def __init__(self, gru_hidden_dim,z_dim,device,attn_dropout=0.1):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.temperature = self.gru_hidden_dim ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        #self.attention = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.weight_q = nn.Parameter(torch.randn((self.gru_hidden_dim, self.gru_hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_k = nn.Parameter(torch.randn((self.gru_hidden_dim, self.gru_hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_v = nn.Parameter(torch.randn((self.z_dim, self.z_dim)), requires_grad=True).to(device=self.device)

    def forward(self, encoder_rnn_hidden_states,encoder_final_hidden_state, latent_z_seq, mask=None,guide_estimates=None):
        src_len = latent_z_seq.shape[1]
        encoder_final_hidden_state = encoder_final_hidden_state.unsqueeze(1).repeat(1, src_len, 1)         # repeat encoder/decoder hidden state src_len times
        k = torch.matmul(encoder_final_hidden_state,self.weight_k)
        q = torch.matmul(encoder_rnn_hidden_states,self.weight_q)
        v = torch.matmul(latent_z_seq,self.weight_v)

        alignment = torch.matmul(q, k.transpose(1, 2))/ self.temperature #alignment scores

        # apply the mask so that the model don't pay the attention to paddings.
        # it's applied before softmax so that the masked values which are very
        # small will be zero'ed out after softmax
        if mask is not None:
            mask=mask[:,:,None].tile(1,1,alignment.shape[2])
            alignment = alignment.masked_fill(mask == 0, -1e9)

        attn =torch.nn.functional.softmax(alignment, dim=-2)  # attention scores or alphas (in literature)
        output = torch.matmul(attn, v)  # Highlight:  context vector: attention-weighted version of our original value input
        return output, attn

class Attention5(nn.Module):
    """
    Self-attention (scaled dot product) product attention: matrix Q comes from the previous decoder layer, while the key and value matrices K and V come from the encoder
    Queries. Keys, Values: Source sequence
    RECAP: Cross-attention
        Computes a weighted average of the encoder's hidden states
        Values: Encoder hidden states
        Keys: Encoder hidden states
        Queries. Decoder hidden states or final hidden state ...
        Source sequence: Input of the encoder
        Last element from source sequence and the final hidden state from the encoder are given to the decoder

    Notes:
        Highlight: https://github.com/VeritasAuthorship/veritas
        https://theaisummer.com/self-attention/
        TODO: https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
                    https://blog.floydhub.com/attention-mechanism/
    """
    def __init__(self, gru_hidden_dim,z_dim,device,attn_dropout=0.1):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.temperature = self.gru_hidden_dim ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        #self.attention = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.weight_q = nn.Parameter(torch.randn((self.gru_hidden_dim, self.gru_hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_k = nn.Parameter(torch.randn((self.gru_hidden_dim, self.gru_hidden_dim)), requires_grad=True).to(device=self.device)
        self.weight_v = nn.Parameter(torch.randn((self.z_dim, self.z_dim)), requires_grad=True).to(device=self.device)

    def forward(self, encoder_hidden_states,decoder_hidden_states,decoder_final_hidden_state,latent_z_seq, mask=None):
        src_len = latent_z_seq.shape[1]
        #decoder_final_hidden_state = decoder_final_hidden_state.unsqueeze(1).repeat(1, src_len, 1)   # repeat decoder final hidden state src_len times

        k = torch.matmul(encoder_hidden_states,self.weight_k)
        q = torch.matmul(decoder_hidden_states,self.weight_q)
        v = torch.matmul(latent_z_seq,self.weight_v)

        alignment = torch.matmul(q, k.transpose(1, 2))/ self.temperature #alignment scores

        # apply the mask so that the model don't pay the attention to paddings.
        # it's applied before softmax so that the masked values which are very
        # small will be zero'ed out after softmax
        if mask is not None:
            mask=mask[:,:,None].tile(1,1,alignment.shape[2])
            alignment = alignment.masked_fill(mask == False, -1e9)

        attn =torch.nn.functional.softmax(alignment, dim=-2)  # attention scores or alphas (in literature)
        output = torch.matmul(attn, v)  # Highlight:  context vector: attention-weighted version of our original value input
        return output, attn

def glorot_init(input_dim, output_dim):
    init_range = torch.sqrt(torch.tensor(6/(input_dim + output_dim)))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return initial

class Init_Hidden(nn.Module):
    def __init__(self,z_dim,max_len,hidden_dim,device):
        super(Init_Hidden, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.fc1 = nn.Linear(self.z_dim,self.hidden_dim)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self,input):
        output = self.fc1(input)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        return output

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_classes,device):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim,bias=True)
        self.fc2 = nn.Linear(self.hidden_dim,self.num_classes,bias=True)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self,input,mask):

        output = self.fc1(input)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        return output

class FCL1(nn.Module):

    def __init__(self,z_dim,max_len,hidden_dim,num_classes,device):
        super(FCL1, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_len = max_len
        self.fc1 = nn.Linear(self.z_dim,self.hidden_dim,bias=True)
        self.fc2 = nn.Linear(self.hidden_dim,self.hidden_dim,bias=True)
        self.fc3 = nn.Linear(self.hidden_dim,self.num_classes,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self,input,mask):

        output = self.fc1(input)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc3(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.logsoftmax(output)
        return output

class FCL2(nn.Module):

    def __init__(self,z_dim,hidden_dim,num_params,device,max_len):
        super(FCL2, self).__init__()
        self.z_dim = z_dim
        self.num_params = num_params
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_len = max_len
        self.fc1 = nn.Linear(self.z_dim,self.hidden_dim,bias=True)
        self.fc2 = nn.Linear(self.hidden_dim,self.num_params,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,input,mask):

        output = self.fc1(input)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        # U, S, VT = torch.linalg.svd(output)
        # output = output @ VT
        output = self.leakyrelu(output)
        output = self.sigmoid(output)
        return output

class FCL3(nn.Module):

    def __init__(self,feats_dim,hidden_dim,num_classes,device):
        super(FCL3, self).__init__()
        self.feats_dim = feats_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.fc1 = nn.Linear(self.feats_dim,self.hidden_dim,bias=True)
        self.fc2 = nn.Linear(self.hidden_dim,self.num_classes,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self,input):

        output = self.fc1(input)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        return output

class FCL4(nn.Module):

    def __init__(self,z_dim,max_len,hidden_dim,num_classes,device):
        super(FCL4, self).__init__()
        self.z_dim = z_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.fc1 = nn.Linear(self.z_dim ,self.hidden_dim,bias=True)
        #self.fc2 = nn.Linear(self.hidden_dim*2,self.hidden_dim,bias=True)
        self.fc2 = nn.Linear(self.hidden_dim,self.num_classes,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        self.relu= nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self,input,mask):
        """
        Notes:
        - https://mane-aajay.medium.com/how-to-calculate-the-svd-from-scratch-with-python-bafcd7fc6945
        -https://towardsdatascience.com/how-to-use-singular-value-decomposition-svd-for-image-classification-in-python-20b1b2ac4990
        - https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15750-s20/www/notebooks/SVD-irises-clustering.html"""

        input = input.flatten(start_dim=1)  # Flattening only has effect if the input latent_space_z
        output = self.fc1(input)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        # Singular-value decomposition
        # U, S, VT = svd(A) #left singular, singular (max var), right singular
        # Data projection = A@VT
        # U,S,VT = torch.linalg.svd(output)
        # output = output@VT
        output = self.fc2(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.relu(output)
        return output

class CNN_FCL(nn.Module):

    def __init__(self,input_dim,hidden_dim,num_parameters,device,max_len):
        super(CNN_FCL, self).__init__()
        self.input_dim = input_dim
        self.num_parameters = num_parameters
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_len = max_len
        self.cnn_out = int(self.max_len/2)
        self.k_size = int(self.max_len + 1 - self.cnn_out)
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.hidden_dim,
                               kernel_size=self.k_size,
                               stride=1,
                               bias=True,
                               padding=0) # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]
        self.avgpool1 = nn.AvgPool1d(kernel_size=self.cnn_out, stride=1,padding=0)

        self.fc1 = nn.Linear(self.hidden_dim,int(self.hidden_dim/2),bias=True)
        self.fc2 = nn.Linear(int(self.hidden_dim/2),self.num_parameters,bias=True)

        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,input,mask):

        output = self.conv1(input)
        output = self.avgpool1(output).squeeze(-1)
        output = self.fc1(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.sigmoid(output)
        return output

class CNN_layers(nn.Module):
    def __init__(self,input_dim,max_len,hidden_dim,num_classes,device,loss_type="elbo"):
        super(CNN_layers, self).__init__()
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.k_size = 4
        self.padding_0 = int((self.k_size-1)/2)
        self.padding_1 = 1
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.hidden_dim,
                               kernel_size=self.k_size,
                               stride=1,
                               bias=True,
                               padding=self.padding_1) # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]

        self.conv1_out= int(((self.max_len + 2*self.conv1.padding[0] - self.conv1.dilation[0]*(self.k_size - 1) -1)/self.conv1.stride[0]) + 1)
        # # self.cnn_out_1 = (self.max_len + 2*int(self.conv1.padding[0])- self.conv1.dilation[0]*(self.k_size - 1) -1) / self.conv1.stride[0] + 1
        self.avgpool1 = nn.AvgPool1d(kernel_size=self.k_size, stride=1,padding=self.padding_1)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=int(self.hidden_dim*2),
                               kernel_size=self.k_size,
                               stride=1,
                               bias=True,
                               padding=self.padding_1) # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]
        self.avgpool2 = nn.AvgPool1d(kernel_size=self.k_size, stride=1,padding=self.padding_1)

        #self.h = int(self.max_len*self.input_dim)
        self.h = int(self.hidden_dim*2)

        self.fc1 = nn.Linear(self.h,int(self.h/2))
        self.fc2 = nn.Linear(int(self.h/2),self.num_classes)
        #self.rbf = RBF(int(self.h/2),self.num_classes,"linear")
        self.dropout= nn.Dropout(p=0) #TODO when using dropout make sure is only used during training (apparently model.eval() does not always work)


    def forward(self, input, mask):
        """https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
        TODO: CNN practices
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        https://cognitivemedium.com/assets/rmnist/Simard.pdf
        https://towardsdatascience.com/a-guide-for-building-convolutional-neural-networks-e4eefd17f4fd
        https://www.jeremyjordan.me/convnet-architectures/
        https://www.upgrad.com/blog/basic-cnn-architecture/#:~:text=LeNet%2D5%20CNN%20Architecture,-In%201998%2C%20the&text=It%20is%20one%20of%20the,resulting%20in%20dimension%20of%2028x28x6.
        TODO: LetNet5
        https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
        https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
        TODO: manifold inspection:
        https://arxiv.org/pdf/1909.11500.pdf
        https://www.stat.cmu.edu/~larry/=sml/Manifolds.pdf
        https://www.mdpi.com/1422-0067/23/14/7775#
        """
        input = input[:,None,:]
        output = self.conv1(self.dropout(input))
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.avgpool1(output)
        output = self.conv2(self.dropout(output))
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.avgpool2(output)
        output = output.permute(0,2,1)[:,-1] #Highlight: Does not seem to matter whether to flatten or take the last output
        #output = self.rbf(self.softmax2(self.dropout(self.fc1(output))))
        output = self.fc2(self.leakyrelu(self.fc1(output)))
        output = self.leakyrelu(output)
        #output = torch.nn.functional.glu(output, dim=1) #divides by 2 the hidden dimensions
        # if mask is not None:
        #     output = output.masked_fill(mask == 0, 1e10)
        return output

class LetNET5(nn.Module):
    """Adaptation of LetNET5.Following:
         - https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
         - Original paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"""
    def __init__(self, input_dim, max_len, embedding_dim, num_classes, device, loss_type):
        super(LetNET5, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.max_len = max_len
        self.ksize = 3
        self.hidden_sizes = [int(self.embedding_dim/2),self.embedding_dim,int(self.embedding_dim*2)]
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_sizes[0], kernel_size=self.ksize, stride=1,padding=int((self.ksize-1)/2)),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=self.ksize),
            nn.Conv1d(in_channels=self.hidden_sizes[0], out_channels=self.hidden_sizes[1], kernel_size=self.ksize, stride=1,padding=int((self.ksize-1)/2)),
            nn.BatchNorm1d(self.hidden_sizes[1]),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=self.ksize),
            nn.Conv1d(in_channels=self.hidden_sizes[1], out_channels=self.hidden_sizes[2], kernel_size=self.ksize, stride=1,padding=int((self.ksize-1)/2)),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_sizes[2], out_features=self.hidden_sizes[1]),
            nn.Tanh(),
            #nn.Linear(in_features=self.hidden_sizes[1], out_features=self.num_classes),
            RBF(in_features=self.hidden_sizes[1],out_features=self.num_classes,basis_func="linear")
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask):
        """
        TODO: CNN practices
        CNN visualizer: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        IMPORTANT : https://towardsdatascience.com/a-guide-for-building-convolutional-neural-networks-e4eefd17f4fd
        https://www.jeremyjordan.me/convnet-architectures/
        https://www.upgrad.com/blog/basic-cnn-architecture/#:~:text=LeNet%2D5%20CNN%20Architecture,-In%201998%2C%20the&text=It%20is%20one%20of%20the,resulting%20in%20dimension%20of%2028x28x6.
        TODO: manifold inspection:
        https://arxiv.org/pdf/1909.11500.pdf
        https://www.stat.cmu.edu/~larry/=sml/Manifolds.pdf
        https://www.mdpi.com/1422-0067/23/14/7775#
        """
        x = self.feature_extractor(input)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = self.softmax(logits) #replaces Euclidean RBF
        return logits, probs

class RNN_layers(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,num_classes,device,loss_type):
        super(RNN_layers, self).__init__()
        self.device = device
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gru_hidden_dim = gru_hidden_dim
        self.loss_type = loss_type
        self.max_len = max_len
        self.num_layers = 1
        # self.rnn1 = nn.GRU(input_size=int(input_dim),
        #                   hidden_size=gru_hidden_dim,
        #                   batch_first=True,
        #                   num_layers=self.num_layers,
        #                   dropout=0.,
        #                   bidirectional=True
        #                   )
        # self.bnn1 = nn.BatchNorm1d(self.max_len).to(self.device)
        self.rnn2 = nn.GRU(input_size=int(input_dim),
                          hidden_size=gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=True
                          )
        self.bnn2 = nn.BatchNorm1d(self.max_len).to(self.device)
        self.leakyrelu = nn.LeakyReLU()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False)
        self.fc3 = nn.Linear(int(self.h/4),self.num_classes,bias=False)

    def forward1(self,input,init_h_0,init_c_0,mask):
        """For LSTM"""
        output, out_hidden = self.rnn(input,(init_h_0,init_c_0))
        output = self.leakyrelu(output)
        forward_out,backward_out = output[:,:,:self.gru_hidden_dim],output[:,:,:self.gru_hidden_dim]
        output = forward_out + backward_out
        output = output[:,-1]

        output = self.leakyrelu(self.fc1(output))
        output = self.leakyrelu(self.fc2(output))
        output = self.leakyrelu(self.fc3(output))
        return output
    def forward2(self,input,init_h_0_f,init_h_0_r):
        "For double GRU"
        #seq_lens = input.bool().sum(1)
        input_reverse = torch.flip(input,(1,))
        output_f, out_hidden = self.rnn1(input,init_h_0_f)
        output_f = self.leakyrelu(output_f)
        if output_f.shape[0] > 0:
            output_f = self.bnn1(output_f)
        forward_out,backward_out = output_f[:,:,:self.gru_hidden_dim],output_f[:,:,:self.gru_hidden_dim]
        output_f = forward_out + backward_out
        output_f = output_f[:,-1]
        #Highlight: Results on reversing the sequences
        output_r, out_hidden = self.rnn2(input_reverse,init_h_0_r)
        output_r = self.leakyrelu(output_r)
        if output_r.shape[0] > 0:
            output_r = self.bnn2(output_r)
        forward_out_r,backward_out_r = output_r[:,:,:self.gru_hidden_dim],output_r[:,:,:self.gru_hidden_dim]
        output_r = forward_out_r + backward_out_r
        output_r = output_r[:,-1]

        output = output_f + output_r
        #output = output_r
        output = self.leakyrelu(self.fc1(output))
        output = self.leakyrelu(self.fc2(output))
        output = self.leakyrelu(self.fc3(output))
        return output
    def forward(self,input,init_h_0_f,init_h_0_r):
        "For GRU with reversed sequences"
        #seq_lens = input.bool().sum(1)
        input_reverse = torch.flip(input,(1,))
        #Highlight: Results on reversing the sequences
        output_r, out_hidden = self.rnn2(input_reverse,init_h_0_r)
        output_r = self.leakyrelu(output_r)
        output_r = self.bnn2(output_r)
        forward_out_r,backward_out_r = output_r[:,:,:self.gru_hidden_dim],output_r[:,:,:self.gru_hidden_dim]
        output_r = forward_out_r + backward_out_r
        output_r = output_r[:,-1]
        output = output_r
        output = self.leakyrelu(self.fc1(output))
        output = self.leakyrelu(self.fc2(output))
        output = self.leakyrelu(self.fc3(output))
        return output

class ReverseSequence(object):
    """Reverses sequences prior to feeding them to torch.nn.utils.rnn.pack_padded_sequence
    [A,T,R,0,0] -> [R,T,A,0,0] """
    def __init__(self,sequences,seqs_lens):
        """
        :param sequences: [N,L,feat_dim]
        :param seqs_lens: [N,]
        """
        self.sequences = sequences
        self.seqs_lens = seqs_lens.int()
    def run(self):
        reversed_list = list(map(lambda seq,seq_len: self.reverse(seq, seq_len), self.sequences,self.seqs_lens))
        return torch.concatenate(reversed_list,dim=0)

    def reverse(self,seq, seq_len):
        seq_reversed = seq.clone()
        seq_reversed[:seq_len] = seq[:seq_len].flip(dims=[0])
        return seq_reversed[None,:]

class RNN_model4(nn.Module):
    """
    Attention-RNN following Attention is all you need "Self attention mechanism"
    Notes:
        -Encoder vector:  Final hidden state produced from the encoder part of the model.It might act as the initial hidden state of the decoder part of the model
        -https://www.kaggle.com/code/kaushal2896/packed-padding-masking-with-attention-rnn-gru"""
    def __init__(self,input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device):
        super(RNN_model4, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.aa_types = aa_types
        #self.embedding = nn.Linear(self.aa_types,self.aa_types)
        self.attention = Attention3(self.gru_hidden_dim,self.z_dim,self.device).to(device=self.device)
        self.num_layers = 1
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=True
                          )
        self.bnn = nn.BatchNorm1d(self.max_len).to(self.device)
        self.softplus = nn.Softplus()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False)
        self.fc3 = nn.Linear(int(self.h/4),self.aa_types,bias=False)
        self.dropout = nn.Dropout(0.2)
        self.init_hidden = Init_Hidden(self.z_dim,max_len,self.gru_hidden_dim,device)

    def forward(self,x,x_lens,init_h_0_decoder,z=None,mask=None,guide_estimates=None):
        """Attention GRU with reversed sequences
        TODO: http://www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/
        https://github.com/adeveloperdiary/DeepLearning_MiniProjects/blob/master/Neural_Machine_Translation/NMT_RNN_with_Attention_train.py
        """

        if isinstance(guide_estimates, dict):
            encoder_rnn_hidden = guide_estimates["rnn_hidden"]
            encoder_rnn_output = guide_estimates["rnn_out"]
        else:
            encoder_rnn_hidden = torch.ones_like(init_h_0_decoder)
            encoder_rnn_output = torch.ones((x.shape[0],x.shape[1],self.gru_hidden_dim))
        x_reverse = ReverseSequence(x,x_lens).run()
        #x_embedded = self.dropout(self.embedding(x_reverse))
        assert not torch.isnan(encoder_rnn_hidden).any(), "found nan in init_h_0"
        assert not torch.isnan(x_reverse).any(), "found nan in x_reverse"
        rnn_input = z
        input_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_input,x_lens.cpu(),batch_first=True,enforce_sorted=False)
        #init_z = self.init_hidden(z[:,0]).expand(self.num_layers * 2, x.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        packed_output, decoder_hidden = self.rnn(input_packed,encoder_rnn_hidden)
        decoder_rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,total_length=self.max_len)
        #rnn_input = torch.cat((x_attn_weighted, x_embedded), dim=2)
        decoder_rnn_output = self.softplus(decoder_rnn_output)
        if decoder_rnn_output.shape[0] > 0:
            decoder_rnn_output = self.bnn(decoder_rnn_output)
        forward_out,backward_out = decoder_rnn_output[:,:,:self.gru_hidden_dim],decoder_rnn_output[:,:,self.gru_hidden_dim:]
        decoder_rnn_output = forward_out + backward_out
        z_attn_weighted, attn_weights = self.attention(encoder_rnn_output,decoder_rnn_output, z, mask)
        assert not torch.isnan(z_attn_weighted).any(), "found nan in z_attn_weighted"

        output = self.softplus(self.fc1(decoder_rnn_output))
        output = self.softplus(self.fc2(output))
        output = self.softplus(self.fc3(output))
        return output,attn_weights

class RNN_model5(nn.Module):
    """
    Attention-RNN following Attention is all you need "Self attention mechanism"
    Notes:
        -Encoder vector:  Final hidden state produced from the encoder part of the model.It might act as the initial hidden state of the decoder part of the model
        -https://www.kaggle.com/code/kaushal2896/packed-padding-masking-with-attention-rnn-gru"""
    def __init__(self,input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device):
        super(RNN_model5, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.aa_types = aa_types
        self.attention = Attention4(self.gru_hidden_dim,self.z_dim,self.device)
        self.num_layers = 1
        self.bidirectional = True
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=self.bidirectional
                          )
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False)
        self.fc3 = nn.Linear(int(self.h/4),self.aa_types,bias=False)
        self.dropout = nn.Dropout(0.2)
        self.init_hidden = Init_Hidden(self.z_dim,max_len,self.gru_hidden_dim,device)

    def forward(self,x,x_lens,init_h_0_decoder,z=None,mask=None,guide_estimates=None):
        """Attention GRU with reversed sequences
        TODO: http://www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/
        https://github.com/adeveloperdiary/DeepLearning_MiniProjects/blob/master/Neural_Machine_Translation/NMT_RNN_with_Attention_train.py
        """

        if isinstance(guide_estimates, dict):
            encoder_final_hidden = guide_estimates["rnn_final_hidden"]
            #encoder_final_hidden_bidirectional = guide_estimates["rnn_final_hidden_bidirectional"]
            encoder_hidden_states = guide_estimates["rnn_hidden_states"]
            encoder_rnn_hidden = guide_estimates["rnn_hidden"]
        else:
            encoder_rnn_hidden = torch.ones_like(init_h_0_decoder)
            encoder_final_hidden =torch.ones((x.shape[0],self.gru_hidden_dim))
            encoder_hidden_states = torch.ones((x.shape[0],x.shape[1],self.gru_hidden_dim))
        #x_reverse = ReverseSequence(x,x_lens).run()
        #x_embedded = self.dropout(self.embedding(x_reverse))
        assert not torch.isnan(encoder_rnn_hidden).any(), "found nan in init_h_0"
        assert not torch.isnan(x).any(), "found nan in x_reverse"
        z_attn_weighted, attn_weights = self.attention(encoder_hidden_states,encoder_final_hidden, z, mask,guide_estimates)
        assert not torch.isnan(z_attn_weighted).any(), "found nan in z_attn_weighted"
        rnn_input = z_attn_weighted
        #rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_input,x_lens.cpu(),batch_first=True,enforce_sorted=False)
        rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_input,x_lens.cpu(),batch_first=True,enforce_sorted=False)
        #init_z = self.init_hidden(z[:,0]).expand(self.num_layers * 2, x.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

        packed_decoder_hidden_states, decoder_final_hidden = self.rnn(rnn_input_packed,encoder_rnn_hidden)
        decoder_hidden_states, seq_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_decoder_hidden_states, batch_first=True,total_length=self.max_len)
        #rnn_input = torch.cat((x_attn_weighted, x_embedded), dim=2)
        decoder_hidden_states = self.softplus(decoder_hidden_states)
        if decoder_hidden_states.shape[0] > 1:
            decoder_hidden_states = self.bnn(decoder_hidden_states)
        forward_hidden_states,backward_hidden_states = decoder_hidden_states[:,:,:self.gru_hidden_dim],decoder_hidden_states[:,:,self.gru_hidden_dim:]
        decoder_hidden_states = forward_hidden_states + backward_hidden_states

        output = self.softplus(self.fc1(decoder_hidden_states))
        output = self.softplus(self.fc2(output))
        output = self.softplus(self.fc3(output))
        outputnn = OutputNN(output=output,
                            attn_weights=attn_weights,
                            encoder_hidden_states=encoder_hidden_states,
                            decoder_hidden_states=decoder_hidden_states,
                            encoder_final_hidden=encoder_final_hidden,
                            decoder_final_hidden=decoder_final_hidden)
        return outputnn

class RNN_model6(nn.Module):
    """
    Attention-RNN following Attention is all you need "Self attention mechanism"
    Notes:
        -https://github.com/dvgodoy/PyTorchStepByStep
        -Encoder vector:  Final hidden state produced from the encoder part of the model.It might act as the initial hidden state of the decoder part of the model
        -https://www.kaggle.com/code/kaushal2896/packed-padding-masking-with-attention-rnn-gru
        TODO: https://becominghuman.ai/visualizing-representations-bd9b62447e38
        """
    def __init__(self,input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device):
        super(RNN_model6, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.aa_types = aa_types
        self.attention = Attention5(self.gru_hidden_dim,self.z_dim,self.device).to(device=self.device)
        self.num_layers = 1
        self.bidirectional = True
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=self.bidirectional
                          ).to(device=self.device)
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.h = self.z_dim + self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False).to(device=self.device)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False).to(device=self.device)
        self.fc3 = nn.Linear(int(self.h/4),self.aa_types,bias=False).to(device=self.device)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x,x_lens,init_h_0_decoder,z=None,mask=None,guide_estimates=None):
        """Attention GRU with reversed sequences
        TODO: http://www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/
        https://github.com/adeveloperdiary/DeepLearning_MiniProjects/blob/master/Neural_Machine_Translation/NMT_RNN_with_Attention_train.py
        """

        if isinstance(guide_estimates, dict):
            encoder_final_hidden = guide_estimates["rnn_final_hidden"].to(device=self.device)
            encoder_hidden_states_bidirectional = guide_estimates["rnn_hidden_states_bidirectional"].to(device=self.device)
            encoder_hidden_states = guide_estimates["rnn_hidden_states"].to(device=self.device)
            encoder_rnn_hidden = guide_estimates["rnn_hidden"].to(device=self.device)
        else:
            #print("NO guide estimates")
            encoder_rnn_hidden = init_h_0_decoder.to(device=self.device) #TODO: pick only one direction? and sort of predict backwards?
            encoder_final_hidden =torch.ones((x.shape[0],self.gru_hidden_dim)).to(device=self.device)
            encoder_hidden_states_bidirectional = torch.ones((x.shape[0],2,self.max_len,self.gru_hidden_dim)).to(device=self.device)
            encoder_hidden_states = torch.ones((x.shape[0],x.shape[1],self.gru_hidden_dim)).to(device=self.device)
        #x_reverse = ReverseSequence(x,x_lens).run()
        #x_embedded = self.dropout(self.embedding(x_reverse))
        assert not torch.isnan(encoder_rnn_hidden).any(), "found nan in init_h_0"
        assert not torch.isnan(z).any(), "found nan in latent space"

        rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(z,x_lens.cpu(),batch_first=True,enforce_sorted=False).to(device=self.device)
        packed_decoder_hidden_states, decoder_final_hidden = self.rnn(rnn_input_packed,encoder_rnn_hidden)
        decoder_hidden_states, seq_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_decoder_hidden_states, batch_first=True,total_length=self.max_len)
        seq_idx = torch.arange(seq_sizes.shape[0])
        decoder_hidden_states = self.softplus(decoder_hidden_states)
        if decoder_hidden_states.shape[0] > 1:
            decoder_hidden_states = self.bnn(decoder_hidden_states)
        forward_hidden_states,backward_hidden_states = decoder_hidden_states[:,:,:self.gru_hidden_dim],decoder_hidden_states[:,:,self.gru_hidden_dim:]
        decoder_hidden_states = forward_hidden_states + backward_hidden_states
        decoder_hidden_states_bidirectional = torch.concatenate([forward_hidden_states[:,None],backward_hidden_states[:,None]],dim=1)
        decoder_final_hidden= decoder_hidden_states[seq_idx,seq_sizes-1]


        z_attn_weighted, attn_weights= self.attention(encoder_hidden_states,decoder_hidden_states,decoder_final_hidden,z,mask=mask)

        c = torch.concatenate([decoder_hidden_states,z_attn_weighted],dim=2)

        output = self.softplus(self.fc1(c))
        output = self.softplus(self.fc2(output))
        output = self.softplus(self.fc3(output))
        outputnn = OutputNN(output=output,
                            attn_weights=attn_weights,
                            encoder_hidden_states=encoder_hidden_states_bidirectional,
                            decoder_hidden_states=decoder_hidden_states_bidirectional,
                            encoder_final_hidden=encoder_final_hidden,
                            decoder_final_hidden=decoder_final_hidden)
        return outputnn

class RNN_model7(nn.Module):
    """
    Attention-RNN following Attention is all you need "Self attention mechanism"
    The encoder hidden states are the input to the decoder, unfortunately this seems to impede learning, and everythin is 100% accuracte from the beginning
    Notes:
        -https://github.com/dvgodoy/PyTorchStepByStep
        -Encoder vector:  Final hidden state produced from the encoder part of the model.It might act as the initial hidden state of the decoder part of the model
        -https://www.kaggle.com/code/kaushal2896/packed-padding-masking-with-attention-rnn-gru
        TODO: https://becominghuman.ai/visualizing-representations-bd9b62447e38
        """
    def __init__(self,input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device):
        super(RNN_model7, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.aa_types = aa_types
        self.attention = Attention5(self.gru_hidden_dim,self.z_dim,self.device).to(device=self.device)
        self.num_layers = 1
        self.bidirectional = True
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0,
                          bidirectional=self.bidirectional
                          ).to(device=self.device)
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.h = self.z_dim + self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False)
        self.fc3 = nn.Linear(int(self.h/4),self.aa_types,bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x,x_lens,init_h_0_decoder,z=None,mask=None,guide_estimates=None):
        """Attention GRU with reversed sequences
        TODO: http://www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/
        https://github.com/adeveloperdiary/DeepLearning_MiniProjects/blob/master/Neural_Machine_Translation/NMT_RNN_with_Attention_train.py
        """

        if isinstance(guide_estimates, dict):
            encoder_final_hidden = guide_estimates["rnn_final_hidden"].to(device=self.device)
            encoder_final_hidden_bidirectional = guide_estimates["rnn_final_hidden_bidirectional"].to(device=self.device)
            encoder_hidden_states_bidirectional = guide_estimates["rnn_hidden_states_bidirectional"].to(device=self.device)
            encoder_hidden_states = guide_estimates["rnn_hidden_states"].to(device=self.device)
            if guide_estimates["rnn_hidden"] is not None:
                encoder_rnn_hidden = guide_estimates["rnn_hidden"].to(device=self.device)
            else:
                encoder_rnn_hidden = init_h_0_decoder.to(device=self.device)

        else: #for epitope generation and model drawing
            encoder_final_hidden =torch.ones((x.shape[0],self.gru_hidden_dim)).to(device=self.device)
            encoder_final_hidden_bidirectional = init_h_0_decoder.to(device=self.device)
            encoder_hidden_states_bidirectional = torch.ones((x.shape[0],2,self.max_len,self.gru_hidden_dim)).to(device=self.device)
            encoder_hidden_states = torch.ones((x.shape[0],x.shape[1],self.gru_hidden_dim)).to(device=self.device)
            encoder_rnn_hidden = init_h_0_decoder.to(device=self.device)

        #x_reverse = ReverseSequence(x,x_lens).run()
        #x_embedded = self.dropout(self.embedding(x_reverse))
        assert not torch.isnan(encoder_rnn_hidden).any(), "found nan in init_h_0"
        assert not torch.isnan(z).any(), "found nan in latent space"

        rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(z,x_lens.cpu(),batch_first=True,enforce_sorted=False).to(device=self.device)
        #rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(z,x_lens,batch_first=True,enforce_sorted=False)

        packed_decoder_hidden_states, decoder_final_hidden = self.rnn(rnn_input_packed,encoder_rnn_hidden) #Highlight: I switched encoder_final_hidden_bidir to encoder_rnn_hidden
        decoder_hidden_states, seq_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_decoder_hidden_states, batch_first=True,total_length=self.max_len)
        seq_idx = torch.arange(seq_sizes.shape[0])
        decoder_hidden_states = self.softplus(decoder_hidden_states)
        decoder_hidden_states = self.bnn(decoder_hidden_states)
        forward_hidden_states,backward_hidden_states = decoder_hidden_states[:,:,:self.gru_hidden_dim],decoder_hidden_states[:,:,self.gru_hidden_dim:]
        decoder_hidden_states = forward_hidden_states + backward_hidden_states
        decoder_hidden_states_bidirectional = torch.concatenate([forward_hidden_states[:,None],backward_hidden_states[:,None]],dim=1)
        decoder_final_hidden= decoder_hidden_states[seq_idx,seq_sizes-1]

        z_attn_weighted, attn_weights= self.attention(encoder_hidden_states,decoder_hidden_states,decoder_final_hidden,z,mask=mask) #Not very important

        c = torch.concatenate([decoder_hidden_states,z_attn_weighted],dim=2)

        output = self.softplus(self.fc1(c))
        output = self.softplus(self.fc2(output))
        output = self.softplus(self.fc3(output))
        outputnn = OutputNN(output=output,
                            attn_weights=attn_weights,
                            encoder_hidden_states=encoder_hidden_states_bidirectional,
                            decoder_hidden_states=decoder_hidden_states_bidirectional,
                            encoder_final_hidden=encoder_final_hidden,
                            decoder_final_hidden=decoder_final_hidden,
                            init_h_0_decoder = encoder_rnn_hidden, #Highlight: Update the init_decoder with the guide estimates values to perform peptide generation later on

                            )



        return outputnn

class RNN_guide1a(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,z_dim,device):
        super(RNN_guide1a, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.num_layers = 1
        self.bidirectional = True
        self.rnn = nn.GRU(input_size=int(input_dim),
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=self.bidirectional
                          )
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2a = nn.Linear(int(self.h/2),self.z_dim,bias=False)
        self.fc2b = nn.Linear(int(self.h/2),self.z_dim,bias=False)


    def forward(self,input,input_lens,init_h_0):
        "Bidirectional GRU"
        #input = torch.flip(input,(1,))
        rnn_hidden_states, rnn_hidden = self.rnn(input,init_h_0)# Highlight: warning : for biRNN the "final" hidden state only contains the final hidden state of the forward network and the first state of the reverse network
        rnn_hidden_states = self.softplus(rnn_hidden_states)
        if rnn_hidden_states.shape[0] > 1:
            rnn_hidden_states = self.bnn(rnn_hidden_states)
        rnn_final_hidden_state_bidirectional = rnn_hidden_states[:,-1]
        forward_out_r,backward_out_r = rnn_hidden_states[:,:,:self.gru_hidden_dim],rnn_hidden_states[:,:,self.gru_hidden_dim:]
        rnn_hidden_states = forward_out_r + backward_out_r
        rnn_final_hidden_state = rnn_hidden_states[:,-1] #final hidden states of both backward and forward networks
        output = self.softplus(self.fc1(rnn_final_hidden_state))
        z_mean = self.fc2a(output)
        z_scale = self.softplus(torch.exp(0.5*self.fc2b(output)))
        return z_mean,z_scale,rnn_hidden_states,rnn_hidden,rnn_final_hidden_state,rnn_final_hidden_state_bidirectional

class RNN_guide1b(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,z_dim,device):
        super(RNN_guide1b, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.num_layers = 1
        self.bidirectional = False
        self.rnn = nn.GRU(input_size=int(input_dim),
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=self.bidirectional
                          ).to(self.device)
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False).to(device=self.device)
        self.fc2a = nn.Linear(int(self.h/2),self.z_dim,bias=False).to(device=self.device)
        self.fc2b = nn.Linear(int(self.h/2),self.z_dim,bias=False).to(device=self.device)


    def forward(self,input,input_lens,init_h_0):
        "Unidirectional GRU with reversed sequences" #TODO: more layers
        raise Warning("Not fixed because it would require too many structural changes")
        input = torch.flip(input,(1,))
        rnn_hidden_states, rnn_hidden = self.rnn(input,init_h_0)# Highlight: warning : for biRNN the "final" hidden state only contains the final hidden state of the forward network and the first state of the reverse network
        rnn_hidden_states = self.softplus(rnn_hidden_states)
        if rnn_hidden_states.shape[0] > 1:
            rnn_hidden_states = self.bnn(rnn_hidden_states)
        rnn_final_hidden_state = rnn_hidden_states[:,-1] #final hidden state
        output = self.softplus(self.fc1(rnn_final_hidden_state))
        z_mean = self.fc2a(output)
        z_scale = self.softplus(torch.exp(0.5*self.fc2b(output)))

        return z_mean,z_scale,rnn_hidden_states,rnn_hidden,rnn_final_hidden_state,None

class RNN_guide2(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,z_dim,device,tensor_type):
        super(RNN_guide2, self).__init__()
        self.device = device
        self.tensor_type = tensor_type
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.num_layers = 1
        self.bidirectional = True
        self.rnn1 = nn.GRU(input_size=int(input_dim),
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0,
                          bidirectional=self.bidirectional
                          ).to(device=self.device,dtype=torch.float64)
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False).to(device=self.device)
        self.fc2a = nn.Linear(int(self.h/2),self.z_dim,bias=False).to(device=self.device)
        self.fc2b = nn.Linear(int(self.h/2),self.z_dim,bias=False).to(device=self.device)


    def forward(self,input,input_lens,init_h_0):
        "Bidirectional GRU with pack and padded sequences"
        #input = ReverseSequence(input,input_lens).run()

        input_packed = torch.nn.utils.rnn.pack_padded_sequence(input,input_lens.cpu(),batch_first=True,enforce_sorted=False).to(device=self.device)

        packed_output, rnn_hidden = self.rnn1(input_packed,init_h_0)
        rnn_hidden_states, seq_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,total_length=self.max_len)
        seq_idx = torch.arange(seq_sizes.shape[0])
        rnn_hidden_states = self.softplus(rnn_hidden_states)
        if rnn_hidden_states.shape[0] > 1:
            rnn_hidden_states = self.bnn(rnn_hidden_states)

        rnn_final_hidden_state_bidirectional = rnn_hidden_states[seq_idx,seq_sizes-1]
        rnn_final_hidden_state_bidirectional = torch.concatenate([rnn_final_hidden_state_bidirectional[:,:self.gru_hidden_dim][None,:],rnn_final_hidden_state_bidirectional[:,self.gru_hidden_dim:][None,:]],dim=0) #Highlight: Equivalent to rnn_hidden, prior to normalization


        forward_out_r,backward_out_r = rnn_hidden_states[:,:,:self.gru_hidden_dim],rnn_hidden_states[:,:,self.gru_hidden_dim:]
        rnn_hidden_states = forward_out_r + backward_out_r
        rnn_hidden_states_bidirectional = torch.concatenate([forward_out_r[:,None],backward_out_r[:,None]],dim=1)
        rnn_final_hidden_state = rnn_hidden_states[seq_idx,seq_sizes-1]
        #rnn_final_hidden_state = rnn_output[:,-1] #Highlight: Do not do this with packed and padded output, not correct
        output = self.softplus(self.fc1(rnn_final_hidden_state))
        z_mean = self.fc2a(output)
        z_scale = self.softplus(torch.exp(0.5*self.fc2b(output)))

        return z_mean,z_scale,rnn_hidden_states,rnn_hidden,rnn_final_hidden_state,rnn_final_hidden_state_bidirectional,rnn_hidden_states_bidirectional

class RNN_classifier(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,num_classes,z_dim,device):
        super(RNN_classifier, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.num_layers = 1
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=True
                          )
        self.bnn = nn.BatchNorm1d(self.max_len).to(device=self.device)
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.h = self.max_len*self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False)
        self.fc3 = nn.Linear(int(self.h/4),self.num_classes,bias=False)

    def forward(self,input,init_h_0):
        "For GRU with reversed sequences"
        #seq_lens = input.bool().sum(1)
        input_reverse = torch.flip(input,(1,))
        #Highlight: Results on reversing the sequences
        output, out_hidden = self.rnn(input_reverse,init_h_0)
        output = self.softplus(output)
        if output.shape[0] > 1:
            output = self.bnn(output)
        forward_out_r,backward_out_r = output[:,:,:self.gru_hidden_dim],output[:,:,:self.gru_hidden_dim]
        output = forward_out_r + backward_out_r
        # output_r = output_r[:,-1]
        # output = output_r
        output = output.flatten(start_dim=1)
        output = self.softplus(self.fc1(output))
        output = self.leakyrelu(output)
        output = self.softplus(self.fc2(output))
        output = self.leakyrelu(output)
        output = self.softplus(self.fc3(output))
        output = self.leakyrelu(output)
        return output

class RBF(nn.Module):
    """
    Funtion from : https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = self.basis_func_choice(basis_func)
        self.reset_parameters()

    def gaussian(self,alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

    def linear(self,alpha):
        phi = alpha
        return phi

    def quadratic(self,alpha):
        phi = alpha.pow(2)
        return phi

    def inverse_quadratic(self,alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        return phi

    def multiquadric(self,alpha):
        phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def inverse_multiquadric(self,alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def spline(self,alpha):
        phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
        return phi

    def poisson_one(self,alpha):
        phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
        return phi

    def poisson_two(self,alpha):
        phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
              * alpha * torch.exp(-alpha)
        return phi

    def matern32(self,alpha):
        phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
        return phi

    def matern52(self,alpha):
        phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
        return phi
    def basis_func_choice(self,name):
        func_dict = {"gaussian":self.gaussian,
                     "linear":self.linear,
                     "quadratic":self.quadratic,
                     "inverse_quadratic":self.inverse_quadratic,
                     "spline":self.spline,
                     "poisson_one":self.poisson_one,
                     "poisson_two":self.poisson_two,
                     "matern32":self.matern32,
                     "matern52":self.matern52}

        return func_dict[name]



    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size) #optimizable parameter
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)

class AutoEncoder(nn.Module):
    def __init__(self,input_dim,max_len,embedding_dim,num_classes,device,loss_type):
        super(AutoEncoder,self).__init__()
        self.input_dim = input_dim
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device
        self.loss_type = loss_type
        self.k_size = 3
        self.padding_0 = int((self.k_size - 1) / 2)
        self.padding_1 = 0
        self.dilation = 1
        self.stride = 1
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=self.input_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.embedding_dim,
                               kernel_size=self.k_size,
                               stride=self.stride,
                               dilation=self.dilation,
                               padding=self.padding_0),
                            nn.BatchNorm1d(self.embedding_dim),
                            nn.LeakyReLU(), #PReLU
                            nn.AvgPool1d(kernel_size=self.k_size, stride=1, padding=int((self.k_size) / 2)),
                            nn.Conv1d(in_channels=self.embedding_dim,# highlight: the input has shape [N,feats-size,max-len]
                                      out_channels=int(self.embedding_dim / 2),
                                      kernel_size=self.k_size,
                                      stride=self.stride,
                                      dilation=self.dilation,
                                      padding=self.padding_0),
                                      #padding=int((self.k_size - 1) / 2)),
                            nn.BatchNorm1d(int(self.embedding_dim / 2)),
                            nn.LeakyReLU(),
                            nn.AvgPool1d(kernel_size=self.k_size, stride=1,padding=int((self.k_size) / 2))
                            )

        """https://towardsdatascience.com/what-are-transposed-convolutions-2d43ac1a0771
        https://medium.com/@santi.pdp/how-pytorch-transposed-convs1d-work-a7adac63c4a5"""
        #self.padding = math.ceil(((self.max_len - 1) * self.stride + self.dilation * (self.k_size - 1) + 1 - self.k_size * 2)/2)
        #self.padding = math.ceil((self.stride * (self.max_len/2) - self.max_len + self.dilation * (self.k_size - 1)-1)/2)
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels = int(self.embedding_dim/2),
                                               out_channels=self.embedding_dim,
                                               kernel_size = self.k_size,
                                               stride =1,
                                               padding=1,#only works because stride = 1, i have not calculated a general formula
                                               output_padding=0),
                                     nn.BatchNorm1d(self.embedding_dim),
                                     # compensates the issues with ReLU handling negative values
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose1d(in_channels = self.embedding_dim,
                                               out_channels=self.input_dim,
                                               kernel_size = self.k_size,
                                               stride =1,padding=1,output_padding=0),
                                     nn.BatchNorm1d(self.input_dim),
                                     nn.LeakyReLU(),
                                     nn.LogSoftmax(dim=-1)
                                     )


        classifier_input_dim = int(self.embedding_dim / 2)*self.max_len
        self.classifier = nn.Sequential(nn.Linear(classifier_input_dim,int(classifier_input_dim/4)),
                                   nn.LeakyReLU(),
                                   nn.Linear(int(classifier_input_dim/4),self.num_classes),
                                   )

    def forward(self,input):
        enc_out = self.encoder(input)
        reconstructed_sequences = self.decoder(enc_out)
        class_output = self.classifier(enc_out.flatten(1))
        return reconstructed_sequences.permute(0,2,1),class_output

class NNAlign(nn.Module):
    """2 step mini batch sampler: https://discuss.pytorch.org/t/custom-batchsampler-for-two-step-mini-batch/20309/2"""
    def __init__(self,input_dim,max_len,hidden_dim,num_classes,device):
        super(NNAlign, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_len = max_len
        self.ksize = 4
        self.weight = nn.Parameter(torch.DoubleTensor(self.input_dim,self.hidden_dim),requires_grad=True).to(device)
        self.bias = nn.Parameter(torch.FloatTensor(self.hidden_dim,), requires_grad=True).to(device)
        #self.bias = torch.nn.init.kaiming_normal_(bias.data, nonlinearity='leaky_relu') #if you we do this we cannot
        self.fc1 = nn.Linear(self.hidden_dim*self.ksize,int(self.hidden_dim/2))
        self.fc2 = nn.Linear(int(self.hidden_dim/2),self.num_classes)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def kmers_windows(self,array, clearing_time_index, max_time, sub_window_size, only_windows=True):
        """
        Creates indexes to extract kmers from a sequence, such as:
             seq =  [A,T,R,P,V,L]
             kmers_idx = [0,1,2,1,2,3,2,3,4,3,4,5]
             seq[kmers_idx] = [A,T,R,T,R,P,R,V,L,P,V,L]
        From https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
        :param int clearing_time_index: Indicates the starting index (0-python idx == 1 clearing_time_index;-1-python idx == 0 clearing_time_index)
        :param max_time: max sequence len
        :param sub_window_size:kmer size
        """
        start = clearing_time_index + 1 - sub_window_size + 1
        sub_windows = (
                start +
                # expand_dims are used to convert a 1D array to 2D array.
                torch.arange(sub_window_size)[None, :] +  # [0,1,2] ---> [[0,1,2]]
                torch.arange(max_time + 1)[None, :].T
        # [0,...,max_len+1] ---expand dim ---> [[[0,...,max_len+1] ]], indicates the
        )  # The first row is the sum of the first row of a + the first element of b, and so on (in the diagonal the result of a[None,:] + b[None,:] is placed (without transposing b). )

        if only_windows:
            return sub_windows
        else:
            return array[:, sub_windows]
    def forward(self,input_blosum,mask):

        overlapping_kmers = self.kmers_windows(input_blosum, 2, self.max_len - self.ksize, self.ksize, only_windows=True)

        input_blosum_kmers = input_blosum[:,overlapping_kmers]
        output = torch.matmul(input_blosum_kmers,self.weight) + self.bias
        output = self.sigmoid(output)
        #mask_kmers = mask[:,overlapping_kmers,0].unsqueeze(-1).expand((output.shape))  #mask2 = output != 0
        #output = (output * mask_kmers).sum(dim=1) / mask_kmers.sum(dim=1)
        output = torch.max(output,dim=1).values #TODO: kmer with highest values (on average??)
        #diff = [list(mask_kmers.size()).index(element) for element in list(mask_kmers.size()) if element not in list(output.size())][0]
        #mask_kmers = [mask_kmers[:,:,0,:].squeeze(2) if diff == 2 else mask_kmers[:,0,:,:].squeeze(1)][0]
        #output = output.mean(1) #Highlight: Mask does not seem to be necessary in the second round, since the "padded kmers" have been excluded on the first average
        output = output.flatten(start_dim=1)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc1(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        output = self.leakyrelu(output)
        return output

class NNAlign2(nn.Module):

    def __init__(self,input_dim,max_len,hidden_dim,num_classes,device):
        super(NNAlign2, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_len = max_len
        self.ksize = 4
        self.weight = nn.Parameter(torch.DoubleTensor(self.input_dim,self.hidden_dim),requires_grad=True).to(device=self.device)
        self.bias = nn.Parameter(torch.FloatTensor(self.hidden_dim,), requires_grad=True).to(device)
        #self.bias = torch.nn.init.kaiming_normal_(bias.data, nonlinearity='leaky_relu') #if you we do this we cannot
        self.fc1 = nn.Linear(self.hidden_dim*self.ksize,int(self.hidden_dim/2))
        self.fc2 = nn.Linear(int(self.hidden_dim/2),self.num_classes)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def kmers_windows(self,array, clearing_time_index, max_time, sub_window_size, only_windows=True):
        """
        Creates indexes to extract kmers from a sequence, such as:
             seq =  [A,T,R,P,V,L]
             kmers_idx = [0,1,2,1,2,3,2,3,4,3,4,5]
             seq[kmers_idx] = [A,T,R,T,R,P,R,V,L,P,V,L]
        From https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
        :param int clearing_time_index: Indicates the starting index (0-python idx == 1 clearing_time_index;-1-python idx == 0 clearing_time_index)
        :param max_time: max sequence len
        :param sub_window_size:kmer size
        """
        start = clearing_time_index + 1 - sub_window_size + 1
        sub_windows = (
                start +
                # expand_dims are used to convert a 1D array to 2D array.
                torch.arange(sub_window_size)[None, :] +  # [0,1,2] ---> [[0,1,2]]
                torch.arange(max_time + 1)[None, :].T
        # [0,...,max_len+1] ---expand dim ---> [[[0,...,max_len+1] ]], indicates the
        )  # The first row is the sum of the first row of a + the first element of b, and so on (in the diagonal the result of a[None,:] + b[None,:] is placed (without transposing b). )

        if only_windows:
            return sub_windows
        else:
            return array[:, sub_windows]
    def forward(self,input_blosum,mask):

        overlapping_kmers = self.kmers_windows(input_blosum, 2, self.max_len - self.ksize, self.ksize, only_windows=True)
        input_blosum_kmers = input_blosum[:,overlapping_kmers]
        output = torch.matmul(input_blosum_kmers,self.weight) + self.bias
        #mask_kmers = mask[:,overlapping_kmers,0].unsqueeze(-1).expand((output.shape))  #mask2 = output != 0
        #output = (output * mask_kmers).sum(dim=1) / mask_kmers.sum(dim=1)
        #diff = [list(mask_kmers.size()).index(element) for element in list(mask_kmers.size()) if element not in list(output.size())][0]
        #mask_kmers = [mask_kmers[:,:,0,:].squeeze(2) if diff == 2 else mask_kmers[:,0,:,:].squeeze(1)][0]
        #output = output.mean(1) #Highlight: Mask does not seem to be necessary in the second round, since the "padded kmers" have been excluded on the first average
        output = output.flatten(start_dim=2)
        if output.shape[0] > 1:
            output = nn.BatchNorm1d(output.size()[1]).to(device=self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc1(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        output = self.leakyrelu(output)
        #output = self.sigmoid(output)
        output = torch.max(output,dim=1).values
        return output

class DistanceMatrixClassifier(nn.Module):
    """
    - Notes:
    https://arxiv.org/pdf/2108.12659.pdf
    https://github.com/MaziarMF/deep-k-means
    """
    def __init__(self,z_dim):
        super(DistanceMatrixClassifier, self).__init__()
        self.z_dim = z_dim

    def forward(self,input):
        return input

