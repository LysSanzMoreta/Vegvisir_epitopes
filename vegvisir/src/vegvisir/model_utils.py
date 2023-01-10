import torch.nn as nn
import torch
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
    def __init__(self,input_dim,embedding_dim,num_classes,device):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        #self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        #self.fc1 = nn.Linear(self.input_dim,self.embedding_dim)
        self.weight1 = nn.Parameter(glorot_init(self.input_dim, self.embedding_dim), requires_grad=True).to(device)
        #self.fc2 = nn.Linear(self.embedding_dim,self.num_classes)
        self.weight2 = nn.Parameter(glorot_init(self.embedding_dim,self.num_classes), requires_grad=True).to(device)

    def forward(self,input,mask):
        output = torch.matmul(input,self.weight1) #.type(torch.cuda.IntTensor)
        output = torch.matmul(output,self.weight2)
        #output = self.fc1(input)
        #output = self.fc2(output)
        output = self.sigmoid(output)
        #output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        if mask is not None:
            output = output.masked_fill(mask == 0, 1e10) #Highlight: This one does not seem crucial
        return output

class CNN_layers(nn.Module):
    def __init__(self,input_dim,max_len,embedding_dim,num_classes,device):
        super(CNN_layers, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.k_size = 3
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.embedding_dim,
                               kernel_size=self.k_size,
                               stride=1,
                               bias=False,
                               padding=int((self.k_size-1)/2)) # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]
        self.maxpool1 = nn.MaxPool1d(kernel_size=self.k_size, stride=1,padding=int((self.k_size-1)/2))
        self.conv2 = nn.Conv1d(in_channels=self.embedding_dim,  # highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.embedding_dim*2,
                               kernel_size=self.k_size,
                               stride=1,
                               bias=False,
                               padding=int((self.k_size - 1) / 2))  # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]
        self.maxpool2 = nn.MaxPool1d(kernel_size=self.k_size, stride=1, padding=int((self.k_size - 1) / 2))
        self.conv3 = nn.Conv1d(in_channels=self.embedding_dim*2,  # highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.embedding_dim,
                               kernel_size=self.k_size,
                               stride=1,
                               bias=False,
                               padding=int((self.k_size - 1) / 2))  # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]
        self.maxpool3 = nn.MaxPool1d(kernel_size=self.k_size, stride=1, padding=int((self.k_size - 1) / 2))
        #self.weight1 = nn.Parameter(glorot_init(self.embedding_dim/2, self.num_classes), requires_grad=True).to(device)
        #self.weight1 = nn.Parameter(torch.rand(int((self.embedding_dim/2)*self.max_len), int((self.embedding_dim/8)*self.max_len)), requires_grad=True).to(device)
        #self.weight2 = nn.Parameter(torch.rand(int((self.embedding_dim/8)*self.max_len), self.num_classes), requires_grad=True).to(device)
        self.fc1 = nn.Linear(int(self.embedding_dim*self.max_len),int((self.embedding_dim*self.max_len)/2),bias =False)
        self.fc2 = nn.Linear(int((self.embedding_dim*self.max_len)/2),self.num_classes,bias =False)

        self.dropout= nn.Dropout(p=0.2)

    def forward1(self,input,mask):
        """https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb"""

        output = self.conv1(self.dropout(input))
        #output = self.relu(output)
        output = torch.nn.functional.glu(output, dim=1) #divides by 2 the hidden dimensions
        output = output.permute(0,2,1).flatten(1) #Highlight: i do the flatten here so that the convolution can keep the sequence structure? otherwise it would be [N,max_len*input_dim,1]
        output = torch.matmul(output,self.weight1)
        output = torch.matmul(output,self.weight2)
        output = self.sigmoid(output)
        #output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        if mask is not None:
            output = output.masked_fill(mask == 0, 1e10) #Highlight: This one does not seem crucial
        return output

    def forward(self, input, mask):
        """https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb"""

        output = self.relu(self.conv1(self.dropout(input))) #TODO : probably Dropout + 1 conv also works (dropout just seems to delay the problem). Bring back GLU. Implement GRU/LSTM as well. Bring back glorot init just in case.
        #TODO: weight loss penalization seems too severe to data points that have 0.9 ---Think again and read documentation
        output = self.maxpool1(output)
        output = self.relu(self.conv2(self.dropout(output)))
        output = self.maxpool2(output)
        output = self.relu(self.conv3(self.dropout(output)))
        output = self.maxpool3(output)
        output = output.permute(0, 2, 1).flatten(1)  # Highlight: i do the flatten here so that the convolution can keep the sequence structure? otherwise it would be [N,max_len*input_dim,1]
        #output = torch.matmul(output, self.weight1)
        #output = torch.matmul(output, self.weight2)
        output = self.fc2(self.fc1(output))
        #TODO: GLU activation after fc2? (change output dim)
        output = self.sigmoid(output)
        # output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        if mask is not None:
            output = output.masked_fill(mask == 0, 1e10)  # Highlight: This one does not seem crucial
        return output