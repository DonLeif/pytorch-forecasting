
from copy import copy
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from dataclasses import dataclass
import torch.nn.functional as F
import math

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss,DistributionLoss


class Projection(nn.Module):

    def __init__(self,in_size,out_size,bias=True):
        super().__init__()
        self.c_fc = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self,in_size,out_size,factor=2, dropout=0.3, bias=False):
        super().__init__()

        self.w1 = nn.Linear(in_size,out_size*factor, bias=bias)
        self.w2 = nn.Linear(out_size*factor,out_size, bias=bias)
        self.w3 = nn.Linear(in_size,out_size*factor,bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# class MLP(nn.Module):

#     def __init__(self,in_size,out_size,factor=2,dropout=0.3,bias=True):
#         super().__init__()
#         self.c_fc    = nn.Linear(in_size, factor * out_size, bias=bias)
#         self.c_proj  = nn.Linear(factor * out_size, out_size, bias=bias)
#         self.dropout = dropout
#         if self.dropout:
#             self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.c_fc(x)
#         x = F.gelu(x)
#         x = self.c_proj(x)
#         if self.dropout:
#             x = self.dropout(x)
#         return x
    
    
class Trend(nn.Module):
    """multi head attention trend block where each head is for one quantile"""

    def __init__(self,cov_emb,trend_emb,factor=2,output_size=1,n_head=2, bias=False,dropout=0.3):
        super().__init__()

        self.trend_emb = trend_emb
        self.output_size = output_size


        self.q_proj = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        self.k_proj = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        self.v_proj = TemporalConvolutionalLayer(cov_emb, dropout=dropout)

        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)    # also for each feature maybe?
        self.proj_trend = nn.Linear(cov_emb, output_size, bias=bias)
        self.proj_trend_emb = nn.Linear(cov_emb, cov_emb, bias=bias)



    def forward(self, trend_q, trend_kv, trend_normed):

        q = self.q_proj(trend_q)
        k = self.k_proj(trend_kv)
        v = self.v_proj(trend_kv)

        B, T, C = k.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = v.size()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head output_size side by side
        y = y + self.mlp(self.ln_att(y))

        trend = self.proj_trend(y)
        trend_emb = self.proj_trend_emb(y)

        return trend,trend_emb

    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
    

class TemporalConvolutionalLayer(nn.Module):
    def __init__(self, cov_emb,factor=2, dropout=0.3):
        super(TemporalConvolutionalLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels=cov_emb, out_channels=cov_emb*factor,
                              kernel_size=3, padding=2, dilation=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.proj_back = nn.Linear(cov_emb*factor, cov_emb)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, cov_emb, seq_len]
        x = self.conv(x)
        x = x[:, :, :-2] 
        x = self.activation(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.proj_back(x)
        
        return x
    

class Attention(nn.Module):

    def __init__(self,cov_emb,n_head,dropout,bias):
        super().__init__()

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q  = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        self.c_attn_k = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        self.c_attn_v = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        # output projection
        self.c_proj = nn.Linear(cov_emb, cov_emb, bias=bias)
        # regularization

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.mlp = MLP(cov_emb,cov_emb)
        self.ln_att = LayerNorm(cov_emb,bias=bias)

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self,q_in,k,v):

        q = self.c_attn_q(q_in)
        k = self.c_attn_k(k)
        v = self.c_attn_v(v) 

        B, T, C = k.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = v.size()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=self.dropout, is_causal=False)


        y = y.transpose(1,2).contiguous().view(B,T,C)  

        x = q_in + self.resid_dropout(self.c_proj(y))
        x  = x +  self.mlp(self.ln_att(x))
        return x

    
class MultiAttention(nn.Module):
    def __init__(self,causal,cov_emb:int = 32,bias:bool = False,n_head:int = 2,factor:int = 1,dropout:float = 0.3,att_proj='linear'):
        super().__init__()

        self.causal = causal

        if att_proj == 'linear':
            self.c_attn_q  = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb,bias=bias) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:nn.Linear(cov_emb,cov_emb,bias=bias) for feature in self.causal})

        elif att_proj == 'mlp':
            self.c_attn_q  = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature: MLP(cov_emb, cov_emb, bias=bias,factor=factor,dropout=dropout) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})

        elif att_proj == 'cnn':
            self.c_attn_q  = nn.ModuleDict({feature:TemporalConvolutionalLayer(cov_emb, dropout=dropout) for feature in self.causal})
            self.c_attn_k = nn.ModuleDict({feature:TemporalConvolutionalLayer(cov_emb, dropout=dropout) for feature in self.causal})
            self.c_attn_v = nn.ModuleDict({feature:TemporalConvolutionalLayer(cov_emb, dropout=dropout) for feature in self.causal})

        self.c_proj = nn.ModuleDict({feature:nn.Linear(cov_emb, cov_emb, bias=bias) for feature in self.causal})

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        # self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)    # also for each feature maybe?
        # one mlp per feature
        self.mlp = nn.ModuleDict({feature:MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias) for feature in self.causal})
        self.ln_post = LayerNorm(cov_emb,bias=bias)

    def forward(self,causal_decoder,past_kv):

        queries = {feature: self.c_attn_q[feature](causal_decoder[feature]) for feature in self.causal}
        queries = torch.stack(tuple(queries.values()),dim=1)

        keys = {feature: self.c_attn_k[feature](past_kv) for feature in self.causal}
        keys = torch.stack(tuple(keys.values()),dim=1)

        values = {feature: self.c_attn_v[feature](past_kv) for feature in self.causal}
        values = torch.stack(tuple(values.values()),dim=1)

        B, c, T, C = keys.size()
        keys = keys.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)   # going from B,c,T,nh,Ch to B,c,nh,T,ch

        B, c, T, C = values.size()
        values = values.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) 

        B, c, T, C = queries.size()
        queries = queries.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)

        y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout, is_causal=False)

        # output is B,c,nh,T,ch
        # after transpose is    B,c,T,nh,ch
        # whereas i had it      B,nh,c,T,ch....whatever came out of this after viewing...
        # after view is B,c,T,C but previously was a weird mixture of time, channels, and covariates 

        y = y.transpose(2, 3).contiguous().view(B,c, T, C)   # neural nets fail silently...erroniously had transposed 1 and 2

        for i,feature in enumerate(self.causal):
            causal_decoder[feature] = causal_decoder[feature] + self.resid_dropout(self.c_proj[feature](y[:,i])) 
            # causal_decoder[feature] = causal_decoder[feature] + self.mlp[feature](self.ln_att(causal_decoder[feature]))
            causal_decoder[feature] = self.ln_post(self.mlp[feature](self.ln_att(causal_decoder[feature])))
        return causal_decoder


class Crossformer(BaseModelWithCovariates):
    def __init__(
        self,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 4,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        logging_metrics: nn.ModuleList = None,
        trend_emb: int = 32,
        cov_emb: int = 32,
        cov_heads: int = 2,
        dropout: float = 0.1,
        factor: int = 2,
        bias: bool = False,
        trend_query: List[str] = [],
        base = [],
        causal = [],
        attention = True,
        att_proj = 'linear',
        rescale = True,
        **kwargs,
    ):
       
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])

        embedding_sizes = {k:(v[0],cov_emb) for k,v in embedding_sizes.items()}

        self.save_hyperparameters()

        # store loss function separately as it is a module
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)


        output_size = self.hparams.output_size
        static_cats = self.hparams.static_categoricals
        cats_dec = self.hparams.time_varying_categoricals_decoder
        trend_emb = self.hparams.trend_emb
        cov_emb = self.hparams.cov_emb
        cov_heads = self.hparams.cov_heads
        dropout = self.hparams.dropout
        bias = self.hparams.bias

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.cov_emb
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.cov_emb)
                for name in self.reals
            }
        )


        self.kv_block = Attention(cov_emb,cov_heads,dropout,bias)
        # self.lstm = LSTM(input_size = cov_emb,hidden_size=cov_emb)
        n_att = 2
        self.att_blocks_cross = nn.ModuleList([Attention(cov_emb,cov_heads,dropout,bias) for _ in range(n_att)])
        # self.attn = Attention(cov_emb,cov_heads,dropout,bias)
        # self.attn2 = Attention(cov_emb,cov_heads,dropout,bias)
        self.att_blocks_out = nn.ModuleList([Attention(cov_emb,cov_heads,dropout,bias) for _ in range(n_att)])
        self.mlp = MLP(cov_emb,cov_emb)


        self.proj_target = nn.Linear(cov_emb,output_size)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        # DEFINING SHAPES
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        target_scale = x["target_scale"]
        max_encoder_length = int(encoder_lengths.max())

        # DEFINING FEATURES
        reals =    self.hparams.x_reals
        cats = self.hparams.x_categoricals
        static_cats = self.hparams.static_categoricals
        static_reals = self.hparams.static_reals
        reals_enc = self.hparams.time_varying_reals_encoder
        cats_enc = self.hparams.time_varying_categoricals_encoder
        reals_dec =  self.hparams.time_varying_reals_decoder
        cats_dec = self.hparams.time_varying_categoricals_decoder
        target = [self.hparams.x_reals[-1]]

        ### ASSEMBLING INPUTS AS DICTIONARIES AND SCALING
        # RAW TENSORS
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)          # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)       # concatenate in time dimension

        x_cat_encoded = self.input_embeddings(x_cat)
        x_cont_encoded = {feature: self.prescalers[feature](x_cont[..., idx].unsqueeze(-1))
                        for idx, feature in enumerate(reals)}
        
        x_encoded = {**x_cat_encoded,**x_cont_encoded}

        query = self.dtf(x_encoded,reals_dec+cats_dec+static_reals+static_cats,stack=True).sum(axis=2)[:,max_encoder_length:,:]
        keys_values = self.dtf(x_encoded,cats+reals,stack=True).sum(axis=2)[:,:max_encoder_length]

        keys_values = self.kv_block(keys_values,keys_values,keys_values)

        for block in self.att_blocks_cross:
            query = block(query,keys_values,keys_values)
        
        for block in self.att_blocks_out:
            query = block(query,query,query)

        x = self.mlp(query)
        x = self.proj_target(x)



        if self.hparams.rescale:
            prediction = self.transform_output(x, target_scale=target_scale)
        else:
            prediction = x

        return self.to_network_output(
                    prediction= prediction)

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        **kwargs,
    ):
        
        default_kwargs = dict(  cov_emb = 32,
                trend_emb = 32,
                cov_heads = 2,
                dropout  = 0.3,
                factor = 2,
                bias = False,
                loss = QuantileLoss(),
                attention = True,
                att_proj = 'linear',
                trend_query  = ['time_idx'])
        
        kwargs = {**default_kwargs,**kwargs}

        if isinstance(kwargs['loss'],DistributionLoss):
             kwargs['output_size'] = len(kwargs['loss'].distribution_arguments)

        elif isinstance(kwargs['loss'],QuantileLoss):
            kwargs['output_size'] = len(kwargs['loss'].quantiles)

        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))
        # create class and return
        return super().from_dataset(
            dataset, **new_kwargs
        )
    

    def rescale_on(self):
        self.hparams.rescale = True

    def rescale_off(self):
        self.hparams.rescale = False

    def get_emb_size(self,features):
        cat_features = [feature for feature in features if feature in self.input_embeddings.keys()]
        real_features = [feature for feature in features if feature in self.prescalers]
        cats = sum([self.input_embeddings[feature].weight.shape[-1] for feature in cat_features])
        reals = sum([self.prescalers[feature].weight.shape[0] for feature in real_features])
        return cats+reals

    def dtf(self,dict,keys,stack=False):
        if not stack:
            return torch.cat([dict[key] for key in keys],dim=-1)
        else:
            return torch.stack([dict[key] for key in keys],dim=-2)
