import torch.nn as nn
import torch
from pyro.nn import PyroModule
import  vegvisir
from vegvisir.utils import extract_windows_vectorized

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention as in Attention is all You need
    Notes:
        ''https://storrs.io/attention/ '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.nn.softmax(attn, dim=-1)) #attention weights!!!!
        output = torch.matmul(attn, v)

        return output, attn

def glorot_init(input_dim, output_dim):
    init_range = torch.sqrt(torch.tensor(6/(input_dim + output_dim)))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return initial
class Embedder(nn.Module):
    def __init__(self,aa_types,embedding_dim,device):
        super(Embedder, self).__init__()
        self.aa_types = aa_types
        self.embedding_dim = embedding_dim
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.fc1 = nn.Linear(self.aa_types,self.embedding_dim)
        #self.weight1 = nn.Parameter(glorot_init(self.aa_types, self.embedding_dim), requires_grad=True).to(device)
        self.fc2 = nn.Linear(self.embedding_dim,self.aa_types)
        #self.weight2 = nn.Parameter(glorot_init(self._dim,self.aa_types), requires_grad=True).to(device)


    def forward(self,input,mask):
        #output = torch.matmul(input,self.weight1) #.type(torch.cuda.IntTensor)
        #output = torch.matmul(output,self.weight2)
        output = self.fc2(self.fc1(input))
        output = self.logsoftmax(output)
        #output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        if mask is not None:
            output = output.masked_fill(mask == 0, 1e10) #Highlight: This one does not seem crucial
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc3(output)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
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
        # if input.ndim == 3:
        #     input = input.flatten(start_dim=2)
        #     output = self.fc1(input)
        # else:
        #     input = input.flatten(start_dim=1) #Flattening only has effect if the input latent_space_z
        #     output = self.fc1(input)

        input = input.flatten(start_dim=1)  # Flattening only has effect if the input latent_space_z
        output = self.fc1(input)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        # Singular-value decomposition
        # U, S, VT = svd(A) #left singular, singular (max var), right singular
        # Data projection = A@VT
        # U,S,VT = torch.linalg.svd(output)
        # output = output@VT
        output = self.fc2(output)
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.fc2(output)
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.leakyrelu(output)
        output = self.avgpool1(output)
        output = self.conv2(self.dropout(output))
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
        output_f = self.bnn1(output_f)
        forward_out,backward_out = output_f[:,:,:self.gru_hidden_dim],output_f[:,:,:self.gru_hidden_dim]
        output_f = forward_out + backward_out
        output_f = output_f[:,-1]
        #Highlight: Results on reversing the sequences
        output_r, out_hidden = self.rnn2(input_reverse,init_h_0_r)
        output_r = self.leakyrelu(output_r)
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

class RNN_model(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device):
        super(RNN_model, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.aa_types = aa_types
        self.num_layers = 1
        self.rnn = nn.GRU(input_size=self.z_dim,
                          hidden_size=gru_hidden_dim,
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

    def forward(self,input,init_h_0):
        "For GRU with reversed sequences"
        #seq_lens = input.bool().sum(1)
        input_reverse = torch.flip(input,(1,))
        #Highlight: Results on reversing the sequences
        output, out_hidden = self.rnn(input_reverse,init_h_0)
        output = self.softplus(output)
        output = self.bnn(output)
        forward_out_r,backward_out_r = output[:,:,:self.gru_hidden_dim],output[:,:,:self.gru_hidden_dim]
        output = forward_out_r + backward_out_r
        # output_r = output_r[:,-1]
        # output = output_r
        output = self.softplus(self.fc1(output))
        output = self.softplus(self.fc2(output))
        output = self.softplus(self.fc3(output))
        return output

class RNN_guide(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,z_dim,device):
        super(RNN_guide, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_len = max_len
        self.num_layers = 1
        self.rnn1 = nn.GRU(input_size=int(input_dim),
                          hidden_size=gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=True
                          )
        self.bnn2 = nn.BatchNorm1d(self.max_len).to(self.device)
        self.softplus = nn.Softplus()
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2a = nn.Linear(int(self.h/2),self.z_dim,bias=False)
        self.fc2b = nn.Linear(int(self.h/2),self.z_dim,bias=False)


    def forward(self,input,init_h_0):
        "For GRU with reversed sequences"
        #seq_lens = input.bool().sum(1)
        input_reverse = torch.flip(input,(1,))
        #Highlight: Results on reversing the sequences
        output_r, out_hidden = self.rnn1(input_reverse,init_h_0)
        output_r = self.softplus(output_r)
        output_r = self.bnn2(output_r)
        forward_out_r,backward_out_r = output_r[:,:,:self.gru_hidden_dim],output_r[:,:,:self.gru_hidden_dim]
        output_r = forward_out_r + backward_out_r
        output_r = output_r[:,-1]
        output = output_r
        output = self.softplus(self.fc1(output))
        z_mean = self.fc2a(output)
        z_scale = self.softplus(torch.exp((self.fc2b(output))))
        return z_mean,z_scale

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
        self.bnn = nn.BatchNorm1d(self.max_len).to(self.device)
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
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
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
        #mask_kmers = mask[:,overlapping_kmers,0].unsqueeze(-1).expand((output.shape))  #mask2 = output != 0
        #output = (output * mask_kmers).sum(dim=1) / mask_kmers.sum(dim=1)
        #diff = [list(mask_kmers.size()).index(element) for element in list(mask_kmers.size()) if element not in list(output.size())][0]
        #mask_kmers = [mask_kmers[:,:,0,:].squeeze(2) if diff == 2 else mask_kmers[:,0,:,:].squeeze(1)][0]
        #output = output.mean(1) #Highlight: Mask does not seem to be necessary in the second round, since the "padded kmers" have been excluded on the first average
        output = output.flatten(start_dim=2)
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
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

