import torch.nn as nn
from torch import Tensor, true_divide
import torch.nn.functional as F
import torch, math
import time
"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""

class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self, seq_length=500, n_hidden = 1000, feature_size=1,num_layers=3,dropout=0,batch_first=True):
        super(Transformer, self).__init__()
        self.seq_length = seq_length
        self.mha = nn.MultiheadAttention(embed_dim=feature_size, num_heads=1, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.linear1 = nn.Linear(seq_length, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src, eps):
        src = src.view(src.size()[0],src.size()[1],1)
        #print(src[:,0,0], src.size())
        x = self.mha(src, src, src, need_weights=False)[0]
        #output = self.transformer_encoder(src)
        #print(output[:,0,0],output.size())
        x = torch.squeeze(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x, torch.zeros(x.size()), torch.zeros(x.size())

class Temporal_Conv_Transformer(nn.Module):
    def __init__(self, seq_length=500, n_hidden = 1000, feature_size=1,num_layers=3, out_channels=3, dropout=0,batch_first=True):
        super().__init__()
        self.seq_length = seq_length
        self.cns = out_channels
        self.knl = 3
        self.dil_convs = nn.ModuleList([])
        for dil in range(feature_size):
            self.dil_convs.append(nn.Conv1d(1,self.cns,self.knl,1,0,int(pow(2,dil))))
        self.reduced_length = self.seq_length-pow(2,(feature_size-1))*(self.knl-1)
        self.pos_enc = PositionalEncoding(d_model = 1+feature_size*self.cns)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1+feature_size*self.cns, nhead=1, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.linear1 = nn.Linear(self.reduced_length*(1+feature_size*self.cns), n_hidden)
        self.linear2 = nn.Linear(n_hidden, 100)
        self.linear3 = nn.Linear(100, 2)
        self.act = nn.Tanh()
        #self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, src, vol, eps):
        src = torch.unsqueeze(src,1)
        x = src
        for dd, dil_conv in enumerate(self.dil_convs):
            z = dil_conv(src)
            x = torch.cat((x[:,:,-self.reduced_length:],z[:,:,-self.reduced_length:]),dim=1)
        x = torch.swapaxes(x,1,2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear3(x)
        y = eps*torch.abs(x[:,1])+x[:,0]
        #return x.squeeze(dim=-1), torch.zeros(x.size()), torch.zeros(x.size())
        return y, x[:,0], x[:,1]

class Temporal_Conv_Transformer_Vol(nn.Module):
    def __init__(self, seq_length=500, n_hidden = 1000, feature_size=1,num_layers=3, out_channels=3, kernel_size=3, dropout=0,batch_first=True):
        super().__init__()
        self.seq_length = seq_length
        self.cns = out_channels
        self.knl = kernel_size
        self.d_model =  1+1+feature_size*self.cns
        self.reduced_length = self.seq_length-pow(2,(feature_size-1))*(self.knl-1)
        self.dil_convs = nn.ModuleList([])
        for dil in range(feature_size):
            self.dil_convs.append(nn.Conv1d(1,self.cns,self.knl,1,0,int(pow(2,dil))))
        self.pos_enc = PositionalEncoding(d_model =self.d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=1, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.linear1 = nn.Linear(self.reduced_length*self.d_model, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 100)
        self.linear3 = nn.Linear(100, 2)
        self.act = nn.Tanh()
        #self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, src, vol, eps):
        src = torch.unsqueeze(src,1)
        vol = torch.unsqueeze(vol,1)
        x = src
        for dd, dil_conv in enumerate(self.dil_convs):
            z = dil_conv(src)
            x = torch.cat((x[:,:,-self.reduced_length:],z[:,:,-self.reduced_length:]),dim=1)
        x = torch.cat((x,vol[:,:,-self.reduced_length:]), dim=1)
        x = torch.swapaxes(x,1,2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear3(x)
        y = eps*torch.abs(x[:,1])+x[:,0]
        #return x.squeeze(dim=-1), torch.zeros(x.size()), torch.zeros(x.size())
        return y, x[:,0], x[:,1]

class Temporal_Conv_Transformer_Discrim(nn.Module):
    def __init__(self, seq_length=500, n_hidden = 1000, feature_size=1,num_layers=3, out_channels=3, dropout=0, batch_first=True):
        super().__init__()
        self.seq_length = seq_length
        self.cns = out_channels
        self.knl = 3
        self.dil_convs = nn.ModuleList([])
        for dil in range(feature_size):
            self.dil_convs.append(nn.Conv1d(1,self.cns,self.knl,1,0,int(pow(2,dil))))
        self.reduced_length = self.seq_length-pow(2,(feature_size-1))*(self.knl-1)
        self.pos_enc = PositionalEncoding(d_model = 1+feature_size*self.cns)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1+feature_size*self.cns, nhead=1, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.linear1 = nn.Linear(self.reduced_length*(1+feature_size*self.cns), n_hidden)
        self.linear2 = nn.Linear(n_hidden, 100)
        self.linear3 = nn.Linear(100, 2)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, eps):
        src = torch.unsqueeze(src,1)
        x = src
        for dd, dil_conv in enumerate(self.dil_convs):
            z = dil_conv(src)
            x = torch.cat((x[:,:,-self.reduced_length:],z[:,:,-self.reduced_length:]),dim=1)
        x = torch.swapaxes(x,1,2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear3(x)
        y = self.sigmoid(eps*torch.abs(x[:,1])+x[:,0])
        #return x.squeeze(dim=-1), torch.zeros(x.size()), torch.zeros(x.size())
        return y, x[:,0], x[:,1]

class Conv(nn.Module):
    def __init__(self,seq_length = 500, device="cpu"):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 3, 1, 1)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(3, 9, 3, 1, 1)
        self.fc1 = nn.Linear(9*seq_length, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x, eps):
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, torch.zeros(x.size()), torch.zeros(x.size())
        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model%2==1:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
