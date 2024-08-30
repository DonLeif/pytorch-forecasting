import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import rnn

from typing import Dict, List, Tuple, Callable

from pytorch_forecasting.models.nn.embeddings import MultiEmbedding

import math
import os


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# class Out(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.c_fc1    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
#         self.c_fc2  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
#         self.heads = nn.ModuleList([
#             nn.Sequential(nn.Linear(config.n_embd,config.n_embd),nn.Dropout(config.dropout),nn.GELU(),nn.Linear(config.n_embd,1))
#             for _ in range(config.outputs)])
    

#         self.dropout = nn.Dropout(config.dropout)

#     def forward(self, x):
#         x = self.c_fc1(x)
#         x = F.gelu(x)
#         x = self.c_fc2(x)
#         x = F.gelu(x)

#         x = [head(x) for head in self.heads]
#         x = torch.cat(x,dim=-1)
#         return x
    
# class Out(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.lstm    = nn.LSTM(config.n_embd,config.n_embd, bias=config.bias)
#         self.fc = nn.Linear(config.n_embd, config.outputs, bias=config.bias)
    

#     def forward(self, x):
#         x,_ = self.lstm(x)
#         x = F.gelu(x)
#         x = self.fc(x)

#         return x

class TimeSeriesConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(TimeSeriesConvBlock, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, groups=in_channels, bias=False)

    def forward(self, x):
        # Input shape: (batch_size, time_steps, channels)
        # Change shape to: (batch_size, channels, time_steps)
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv1d(x)
        
        # Change shape back to: (batch_size, time_steps, channels)
        x = x.transpose(1, 2)
        return x



# class Out(nn.Module):

#     def __init__(self, config):
#         super().__init__()

#         if config.losstype != 'distribution':

#             self.out = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
#             nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
#             self.c_fc3  = nn.Linear(config.n_embd, config.outputs, bias=config.bias)
#             self.dropout = nn.Dropout(config.dropout)

#     def forward(self, x):
#         x = self.c_fc1(x)
#         x = F.gelu(x)
#         x = self.c_fc2(x)
#         x = F.gelu(x)
#         x = self.c_fc3(x)
#         return x
    
    
class Out(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_fc3 = nn.Linear(config.n_embd, config.outputs, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.c_fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.c_fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.c_fc3.weight, mode='fan_in', nonlinearity='leaky_relu')

        if self.config.bias:
            nn.init.zeros_(self.c_fc1.bias)
            nn.init.zeros_(self.c_fc2.bias)
            nn.init.zeros_(self.c_fc3.bias)

    def forward(self, x):
        x = self.c_fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.c_fc2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.c_fc3(x)
        return x

import torch.nn.functional as F
from torch import nn

class Out(nn.Module):
    def __init__(self, config,i=None):
        super().__init__()
        self.config = config
        if i is not None:
            self.outputs= config.outputs[i]
        else:
            self.outputs = config.outputs

        if config.losstype != 'distribution':
            self.layers = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                nn.Dropout(config.dropout),
                # nn.Linear(4 * config.n_embd, 4*  config.n_embd, bias=config.bias),
                # nn.GELU(),
                # nn.Dropout(config.dropout),
                nn.Linear(4 * config.n_embd, self.outputs, bias=config.bias),
            )
        else:
            self.p1 = nn.Sequential(nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
                                nn.GELU(),
                                nn.Dropout(config.dropout),
                                nn.Linear(config.n_embd, self.outputs, bias=config.bias),
                                )
            self.p2 = nn.Sequential(nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
                                nn.GELU(),
                                nn.Dropout(config.dropout),
                                nn.Linear(config.n_embd, self.outputs, bias=config.bias),
                                nn.ReLU(),
                                )

        self.reset_parameters()


    def forward(self, x):
        if self.config.losstype != 'distribution':
            return self.layers(x)
        else:
            return torch.cat([self.p1(x), self.p2(x)], dim=-1)


    def reset_parameters(self):
        if self.config.losstype == 'distribution':
            self.layers = self.p1 + self.p2
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if self.config.bias:
                    nn.init.zeros_(layer.bias)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x

class TimeEmbedding(nn.Module):
    def __init__(self, config):
        super(TimeEmbedding, self).__init__()
        self.embedding_size = config.n_embd
        self.projection_one = nn.Linear(1, self.embedding_size//4 )
        # self.projection_two = nn.Linear(config.enc_len, self.embedding_size )
        self.frequency = nn.Parameter(torch.Tensor(self.embedding_size//4))
        self.projection_out = nn.Linear(self.embedding_size//4, self.embedding_size)
        nn.init.uniform_(self.frequency, 0, 1)

    def forward(self, x):
        # x.shape: (batch_size, time_steps, 1)
        projection_one = self.projection_one(x)  # shape: (batch_size, time_steps, embedding_size // 2)
        # projection_two = self.projection_two(x.squeeze())
        # projection_two = projection_two.reshape(x.shape[0],1,self.embedding_size)
        
        #time_embedding = torch.sin(2 * math.pi * projection_one * projection_two)
        time_embedding = torch.sin(2 * math.pi * projection_one * self.frequency)
        time_embedding = self.projection_out(time_embedding)
        #time_embedding = torch.cat([sinus_transform, projection_two], dim=-1)  # shape: (batch_size, time_steps, embedding_size)

        return time_embedding
    

import torch
import torch.nn as nn
import math

class TimeEmbedding2(torch.nn.Module):
    def __init__(self, d_model, pos_len=10000):
        super().__init__()
        self.d_model = d_model
        self.pos_len = pos_len

        # Create a positional encoding matrix
        pos_enc = torch.zeros(pos_len, d_model)
        position = torch.arange(0, pos_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):

        # Split relative and absolute time indices
        rel_time, abs_time = x[:, :, 0], x[:, :, 1]

        # Convert Z-normalized time indices to approximate the original time indices
        rel_time = ((rel_time * self.pos_len / 2) + (self.pos_len / 2)).long()
        abs_time = ((abs_time * self.pos_len / 2) + (self.pos_len / 2)).long()

        # Get the time embeddings for relative and absolute time indices
        rel_emb = self.pos_enc[rel_time, :]
        abs_emb = self.pos_enc[abs_time, :]

        # Combine the relative and absolute embeddings using addition
        time_emb = rel_emb + abs_emb

        return time_emb
    
    
# class TimeEmbedding(nn.Module):
#     def __init__(self, config):
#         super(TimeEmbedding, self).__init__()
#         self.embedding_size = config.n_embd
#         self.projection_one = nn.Linear(1, self.embedding_size//1 )
#         # self.projection_two = nn.Linear(config.enc_len, self.embedding_size )
#         self.frequency = nn.Parameter(torch.Tensor(self.embedding_size//1))
        
#         self.fc1 = nn.Linear(self.embedding_size, self.embedding_size)
        
#         self.projection_out = nn.Linear(self.embedding_size//1, self.embedding_size)
#         nn.init.uniform_(self.frequency, 0, 1)
        
#         self.norm = LayerNorm(config.n_embd, bias=config.bias)

#     def forward(self, x):
#         # x.shape: (batch_size, time_steps, 1)
#         projection_one = self.projection_one(x)  # shape: (batch_size, time_steps, embedding_size // 2)
#         # projection_two = self.projection_two(x.squeeze())
#         # projection_two = projection_two.reshape(x.shape[0],1,self.embedding_size)
        
#         #time_embedding = torch.sin(2 * math.pi * projection_one * projection_two)
#         x = torch.sin(2 * math.pi * projection_one * self.frequency)
#         time_embedding = self.fc1(x)
#         time_embedding = F.gelu(time_embedding)
#         time_embedding = self.projection_out(time_embedding)
#         x = x + time_embedding
#         x = self.norm(x)

#         #time_embedding = torch.cat([sinus_transform, projection_two], dim=-1)  # shape: (batch_size, time_steps, embedding_size)

#         return x


class CrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn_q  = nn.Linear(config.n_embd, 1 * config.n_embd, bias=config.bias)
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence

    def forward(self,q,kv):

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_attn_q(q)
        k ,v  = self.c_attn_kv(kv).split(self.n_embd, dim=2)

        B, T, C = kv.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

        
class CausalSelfAttention(nn.Module):

    def __init__(self, config,causal=False,reverse=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = causal
        self.reverse = reverse
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(200, 200))
                                        .view(1, 1, 200, 200))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal and not self.reverse:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            elif self.causal and self.reverse:
                reverse_bias = torch.flip(self.bias, dims=(2, 3))
                att = att.masked_fill(reverse_bias[:,:,:T,:T] == 0, float('-inf'))
               
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class Block(nn.Module):

    def __init__(self, config,causal=False,reverse=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config,causal,reverse)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CrossBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CrossAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, q,kv):
        x = q + self.attn(self.ln_1(q),self.ln_2(kv))
        x = x + self.mlp(self.ln_3(x))
        return x

class RealEmbedding(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.linears1 = nn.ModuleList([nn.Linear(1, c_out) for i in range(c_in)])
        self.linears2 = nn.ModuleList([nn.Linear(c_out, c_out) for i in range(c_in)])

    def forward(self, x):
        b,t,c = x.shape
        x = x.reshape(b*t,c)      
        out = [F.gelu(linear(x[...,i].unsqueeze(-1))) for i,linear in enumerate(self.linears1)]
        out = [linear(x_i) for linear, x_i in zip(self.linears2, out)]
        out = torch.cat(out, dim=-1)
        out = out.reshape(b,t,-1) 
        
        return out
    
class RealEmbedding(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.linears1 = nn.ModuleList([nn.Linear(1, c_out) for i in range(c_in)])

    def forward(self, x):
        b,t,c = x.shape
        x = x.reshape(b*t,c)      
        out = [linear(x[...,i].unsqueeze(-1)) for i,linear in enumerate(self.linears1)]
        out = torch.cat(out, dim=-1)
        out = out.reshape(b,t,-1) 
        
        return out
    

    
    
class ConvEmbedding(nn.Module):
    def __init__(self,config,kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(1 , config.n_real_embd-1, kernel_size=kernel_size, padding=0)
        
    def forward(self, x):
        x_org = x
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = x.permute(0,2,1)
        # print(x.shape)
        # print(x[:,:self.kernel_size,:].mean(dim=1).shape)
        # print(torch.ones(x.shape[0],self.kernel_size-1,x.shape[2]).shape)
        pad = torch.ones(x.shape[0],self.kernel_size-1,x.shape[2]).to(x.device) * x[:,:self.kernel_size,:].mean(dim=1,keepdim=True)
        x = torch.cat([pad,x],dim=1)
        x = torch.cat([x_org,x],dim=-1)
        return x 
        
class MultiConvEmbedding(nn.Module):
    def __init__(self,config,features):
        super().__init__()
        self.conv_embs = nn.ModuleList([ConvEmbedding(config) for _ in range(features)])
        
    def forward(self,x):
        b,t,c = x.shape
        out = [conv_emb(x[...,i].unsqueeze(-1)) for i,conv_emb in enumerate(self.conv_embs)]
        out = torch.cat(out,dim=-1)
        return out 
    
class PosBlock(nn.Module):
    
    def __init__(self,config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.n_ * 2)
        self.linear1 = nn.Linear(config.n_ * 2, config.n_)
        self.linear2 = nn.Linear(config.n_, config.n_real_embd)

        self.proj_abs = nn.Linear(1, config.n_)
        

    def forward(self, abs_pos,rel_pos):
        assert abs_pos.size(-1) == 1

        abs_pos = self.proj_abs(abs_pos) 
        # add noise
        abs_pos = abs_pos + torch.randn_like(abs_pos) * 0.0
        x = torch.cat([abs_pos,rel_pos],dim=-1)
        x = self.layer_norm(x)
        x = self.linear1(F.gelu(x))
        x = self.linear2(x)
        return x