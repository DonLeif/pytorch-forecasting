
from copy import copy
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from dataclasses import dataclass
import torch.nn.functional as F
import math


class Projection(nn.Module):

    def __init__(self,in_size,out_size,bias=True):
        super().__init__()
        self.c_fc = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        return x

class MLP(nn.Module):

    def __init__(self,in_size,out_size,factor=2,dropout=0.3,bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(in_size, factor * out_size, bias=bias)
        self.c_proj  = nn.Linear(factor * out_size, out_size, bias=bias)
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
    
# class MLP(nn.Module):
#     def __init__(self, in_size, out_size, intermediates=1, factor=2, dropout=0.3, bias=True, activation=nn.GELU):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(in_size, factor * out_size, bias=bias))
#         self.layers.append(nn.LayerNorm(factor * out_size))
#         self.layers.append(activation())
#         self.layers.append(nn.Dropout(dropout))
#         for _ in range(intermediates):
#             self.layers.append(nn.Linear(factor * out_size, factor * out_size, bias=bias))
#             self.layers.append(nn.LayerNorm(factor * out_size))
#             self.layers.append(activation())
#             self.layers.append(nn.Dropout(dropout))
#         self.layers.append(nn.Linear(factor * out_size, out_size, bias=bias))

#         self.skip = intermediates > 0 and in_size == out_size

#     def forward(self, x):
#         out = x
#         for i, layer in enumerate(self.layers):
#             if self.skip and i % 2 == 0 and i > 0:
#                 out = out + layer(out)
#             else:
#                 out = layer(out)
#         return out if not self.skip else out + x

class Trend(nn.Module):

    def __init__(self,q_in,k_in,trend_emb,factor=2,time_emb=False):
        super().__init__()

        self.query_proj = MLP(q_in,trend_emb,factor)
        self.key = MLP(k_in,trend_emb,factor)
        self.time_emb = time_emb
        if self.time_emb:
            self.time_emb_matrix = torch.nn.Embedding(50, trend_emb)

    def forward(self,trend_query,trend_key,trend_values):

        trend_query = self.query_proj(trend_query)
        trend_key = self.key(trend_key)
        if self.time_emb:
            trend_query += self.time_emb_matrix(torch.arange(trend_query.size(1),device=trend_query.device))
            trend_key += self.time_emb_matrix(torch.arange(trend_key.size(1),device=trend_key.device))

        attention = (trend_query @ trend_key.transpose(-2, -1)) * (1.0 / math.sqrt(trend_key.size(-1)))
        attention = F.softmax(attention,dim=-1)

        x_trend = torch.einsum('btk,bkd->btd',attention,trend_values)
        return x_trend,attention
    

class Trend(nn.Module):
    def __init__(self, q_in, k_in, trend_emb, factor=2, time_emb=False,bias=False):
        super().__init__()

        self.ln_q = LayerNorm(q_in, bias=bias)
        self.ln_k = LayerNorm(k_in, bias=bias)
        self.query_proj = MLP(q_in, trend_emb, factor)
        self.key_proj = MLP(k_in, trend_emb, factor)

        self.time_emb = time_emb
        if time_emb:
            self.time_emb_matrix = torch.nn.Parameter(torch.randn(time_emb, trend_emb)/10)

    def forward(self, trend_query, trend_key, trend_values):
        trend_query = self.ln_q(trend_query)
        trend_key = self.ln_k(trend_key)
        trend_query = self.query_proj(trend_query)
        trend_key = self.key_proj(trend_key)

        if self.time_emb:
            trend_query += self.time_emb_matrix[-trend_query.size(1):]
            trend_key += self.time_emb_matrix[:trend_key.size(1)]

        attention = (trend_query @ trend_key.transpose(-2, -1)) / math.sqrt(trend_key.size(-1))
        attention = F.softmax(attention, dim=-1)

        x_trend = attention @ trend_values
        return x_trend, attention
    

class CovPreprocess(nn.Module):

    def __init__(self,q_in,kv_in1,kv_in2,cov_emb,factor=2):
        super().__init__()
        self.query_proj = MLP(q_in,cov_emb,factor)
        self.kv1 = MLP(kv_in1,cov_emb,factor)
        self.kv2 = MLP(kv_in2,cov_emb,factor)

    def forward(self,cov_query,cov_kv1,cov_kv2):
        cov_query = self.query_proj(cov_query)
        cov_kv1 = self.kv1(cov_kv1)
        cov_kv2 = self.kv2(cov_kv2)

        cov_kv = torch.cat([cov_kv1,cov_kv2],dim=1)

        return cov_query,cov_kv
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):

    def __init__(self,cov_emb,n_head,dropout,bias,time_emb):
        super().__init__()

        self.time_emb = time_emb
        if time_emb:
            self.time_emb_matrix = torch.nn.Parameter(torch.randn(time_emb, cov_emb)/10)

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q  = nn.Linear(cov_emb, 1 * cov_emb, bias=bias)
        self.c_attn_k = nn.Linear(cov_emb, 1 * cov_emb, bias=bias)
        self.c_attn_v = nn.Linear(cov_emb, 1 * cov_emb, bias=bias)
        # output projection
        self.c_proj = nn.Linear(cov_emb, cov_emb, bias=bias)
        # regularization

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0

    def forward(self,q,k,v):

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        dec_len = q.size(1)
        enc_len = k.size(1)

        if self.time_emb:
            q += self.time_emb_matrix[-dec_len:]
            k += self.time_emb_matrix[:enc_len]
            v += self.time_emb_matrix[:enc_len]

        q = self.c_attn_q(q)
        k = self.c_attn_k(k)
        v = self.c_attn_v(v)  # is this even necessary

        B, T, C = k.size()
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = v.size()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        B, T, C = q.size()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
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
    
class Block(nn.Module):

    def __init__(self,cov_emb,n_head=2,dropout=0.3,factor=2,bias=True,time_emb=False):
        super().__init__()
        assert cov_emb % n_head == 0, "n_embd must be divisible by n_head"
        self.ln_1 = LayerNorm(cov_emb, bias=bias)
        self.ln_2 = LayerNorm(cov_emb, bias=bias)
        self.ln_3 = LayerNorm(cov_emb, bias=bias)
        self.ln_4 = LayerNorm(cov_emb, bias=bias)
        self.attn = Attention(cov_emb,n_head,dropout,bias,time_emb)
        self.mlp  = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)


    def forward(self, q,kv):
        q = self.ln_1(q)
        kv = self.ln_2(kv)
        x = self.attn(q,kv,kv) + q
        x = x + self.mlp(self.ln_3(x))
        return x

class NAM_TS(BaseModelWithCovariates):
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
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,
        time_emb: bool = False,
        trend_emb: int = 16,
        max_cat_emb: int = 8,
        # trend_heads: int = 4,
        cov_emb: int = 32,
        cov_heads: int = 4,
        dropout: float = 0.1,
        factor: int = 2,
        bias: bool = False,
        trend_query: List[str] = [],
        exclude_cat: List[str] = [],
        exclude_cont: List[str] = [],
        pre_blocks = 0,
        **kwargs,
    ):
       
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            loss = QuantileLoss()
        self.save_hyperparameters()

        # store loss function separately as it is a module
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        try:
            outputs = len(self.loss.quantiles)
        except:
            outputs = 1
    
        print("outputs", outputs)

        reals = self.hparams.x_reals
        cats = self.hparams.x_categoricals
        static_cats = self.hparams.static_categoricals
        reals_enc = self.hparams.time_varying_reals_encoder
        cats_enc = self.hparams.time_varying_categoricals_encoder
        reals_dec =  self.hparams.time_varying_reals_decoder
        cats_dec = self.hparams.time_varying_categoricals_decoder
        wday = 'weekday' in cats_dec
        if wday:
            cats_dec.remove('weekday')
            cats_dec.insert(0, 'weekday')
        target = reals[-1]

        trend_emb = self.hparams.trend_emb
        # trend_heads = self.hparams.trend_heads
        cov_emb = self.hparams.cov_emb
        cov_heads = self.hparams.cov_heads
        dropout = self.hparams.dropout
        bias = self.hparams.bias
        trend_query = self.hparams.trend_query
        exclude_cat = self.hparams.exclude_cat
        exclude_cont = self.hparams.exclude_cont

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.max_cat_emb
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.reals
            }
        )

        self.trend_query = static_cats + static_reals + trend_query
        self.exclude_cat = exclude_cat
        self.exclude_cont = exclude_cont

        self.trend_block = Trend(q_in = self.get_emb_size(self.trend_query),
                                 k_in = self.get_emb_size(cats+reals),
                                 trend_emb = trend_emb,factor = factor,time_emb=time_emb)
        
        self.cov_preprocess = CovPreprocess(q_in = self.get_emb_size(cats_dec+reals_dec+static_cats),
                                            kv_in1 = self.get_emb_size(cats+reals),
                                            kv_in2 = self.get_emb_size(cats_dec+reals_dec+static_cats+static_reals),
                                            cov_emb = cov_emb,
                                            factor = factor)
        

        self.pre_blocks = nn.ModuleList([Block(cov_emb=cov_emb,n_head=cov_heads,dropout=dropout,bias=bias,time_emb=time_emb,factor=factor) for _ in range(pre_blocks)])
        
        self.cov_block = Block(cov_emb=cov_emb,n_head=cov_heads,dropout=dropout,bias=bias,time_emb=time_emb,factor=factor)

        self.cov_mlp = MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias)

        self.n_classes = [self.hparams.embedding_sizes[feature][0] for feature in cats_dec if feature not in self.exclude_cat]
        self.n_classes_sum = sum(self.n_classes)
        self.n_classes_sum_sparse = self.n_classes_sum - len(self.n_classes) + 1

        n_reals_dec = len([r for r in reals_dec if r not in self.exclude_cont])
        self.cov_proj = nn.Linear(cov_emb,self.n_classes_sum_sparse + n_reals_dec)  # + n reals
        

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        
        default_kwargs = dict(  cov_emb = 32,
                trend_emb = 16,
                max_cat_emb = 8,
                time_emb=False,
                cov_heads = 2,
                dropout  = 0.3,
                factor = 2,
                pre_blocks = 0,
                bias = False,
                trend_query  = ['time_idx','weekofyear'], 
                exclude_cat  = ['weekofyear'],
                exclude_cont = ['time_idx','relative_time_idx'])
        
        kwargs = {**default_kwargs,**kwargs}
        

        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))
        # create class and return
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

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
        reals = self.hparams.x_reals
        cats = self.hparams.x_categoricals
        static_cats = self.hparams.static_categoricals
        static_reals = self.hparams.static_reals
        reals_enc = self.hparams.time_varying_reals_encoder
        cats_enc = self.hparams.time_varying_categoricals_encoder
        reals_dec =  self.hparams.time_varying_reals_decoder
        cats_dec = self.hparams.time_varying_categoricals_decoder
        wday = 'weekday' in cats_dec
        if wday:
            cats_dec.remove('weekday')
            cats_dec.insert(0, 'weekday')
        target = [reals[-1]]

        ### ASSEMBLING INPUTS AS DICTIONARIES AND SCALING
        # RAW TENSORS
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)          # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)       # concatenate in time dimension

        # RAW DICTIONARIES
        x_cat_normal = {name: x_cat[..., i].unsqueeze(-1) for i, name in enumerate(cats)}
        x_cont_normal = {name: x_cont[..., i].unsqueeze(-1) for i, name in enumerate(reals)}
        x_normal = {**x_cat_normal,**x_cont_normal}

        # SCALED DICTIONARIES
        x_cat_encoded = self.input_embeddings(x_cat)
        x_cont_encoded = {feature: self.prescalers[feature](x_cont[..., idx].unsqueeze(-1))
                        for idx, feature in enumerate(reals)}
        x_encoded = {**x_cat_encoded,**x_cont_encoded}

        # TREND BLOCK
        trend_query = self.dtf(x_encoded,self.trend_query)[:,max_encoder_length:,:]
        trend_key = self.dtf(x_encoded,cats+reals)[:,:max_encoder_length,:]
        trend_values = self.dtf(x_normal,target)[:,:max_encoder_length,:]
        x_trend,attention = self.trend_block(trend_query,trend_key,trend_values)

        # COVARIATES Q,K,V
        cov_query = self.dtf(x_encoded,cats_dec+reals_dec+static_cats+static_reals)[:,max_encoder_length:,:]
        cov_kv_enc = self.dtf(x_encoded,cats+reals)[:,:max_encoder_length,:]
        cov_kv_dec = self.dtf(x_encoded,cats_dec+reals_dec+static_cats+static_reals)[:,max_encoder_length:,:]

        # COVARIATES PREPROCESSING
        cov_query,cov_kv = self.cov_preprocess(cov_query,cov_kv_enc,cov_kv_dec)

        # KEY VALUE ATTENTION PREPROCESSING
        for block in self.pre_blocks:
            cov_kv = block(cov_kv,cov_kv)

        # COVARIATES ATTENTION
        x_cov = self.cov_block(cov_query,cov_kv)
        x_cov = self.cov_mlp(x_cov)
        x_cov = self.cov_proj(x_cov)

        cats_dec2 = [c for c in cats_dec if c not in self.exclude_cat]

        wday = 'weekday' in cats_dec2 ### place weekday at the beginning because we will want ot keep all categories
        if wday:
            cats_dec2.remove('weekday')
            cats_dec2.insert(0, 'weekday')  

        cat_values = self.dtf(x_normal,cats_dec2)[:,max_encoder_length:,:]
        cat_one_hot = self.one_hot_encode(cat_values,self.n_classes,not_sparse=sum([wday]))  # <- do not leave out first category for the first sum([...]) features
        x_cat_values = cat_one_hot * x_cov[:,:,:self.n_classes_sum_sparse]

        cat_dict = x_cat_values.split([i-1 if c!= 'weekday' else i for i,c in zip(self.n_classes,cats_dec2)],dim=-1)
        cat_dict = [x.sum(axis=-1,keepdim=True) for x in cat_dict]
        cat_dict = {feature:x for feature,x in zip(cats_dec2,cat_dict)}  # <<<<

        reals_dec2 = [r for r in reals_dec if r not in self.exclude_cont]
        if len(reals_dec2) > 0:
            x_cov_cont = self.dtf(x_normal,reals_dec2)[:,max_encoder_length:,:]
            x_cov_coef = x_cov[:,:,self.n_classes_sum_sparse:]
            x_cont_values = x_cov_coef * x_cov_cont

        output = x_trend.clone()
        output += x_cat_values.sum(axis=-1,keepdim=True)
        if len(reals_dec2) > 0:
            output += x_cont_values.sum(axis=-1,keepdim=True)

        
        prediction = self.transform_output(output, target_scale=target_scale)
        #also scale t
        x_trend = self.transform_output(x_trend, target_scale=target_scale)
        cat_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in cat_dict.items()}
        if len(reals_dec2) > 0:
            cont_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in zip(reals_dec2,x_cont_values.split(1,dim=-1))}
        else:
            cont_dict = {}

        return self.to_network_output(
                    prediction= prediction,
                    x_trend=x_trend,
                    attention=attention,
                    **cat_dict,
                    **cont_dict)


    def one_hot_encode(self,labels, num_classes,not_sparse=0):
        batch_size, timesteps, num_class_features = labels.shape
        one_hot_tensor = []
        for i in range(num_class_features):
            class_feature_tensor = labels[:, :, i]
            one_hot_class_feature = torch.nn.functional.one_hot(class_feature_tensor, num_classes[i])
            if i >= not_sparse:
                one_hot_class_feature = one_hot_class_feature[:, :,1:]
            one_hot_tensor.append(one_hot_class_feature)
        one_hot_tensor = torch.cat(one_hot_tensor, dim=-1)
        return one_hot_tensor
    
    def get_emb_size(self,features):
        cat_features = [feature for feature in features if feature in self.input_embeddings.keys()]
        real_features = [feature for feature in features if feature in self.prescalers]
        cats = sum([self.input_embeddings[feature].weight.shape[-1] for feature in cat_features])
        reals = sum([self.prescalers[feature].weight.shape[0] for feature in real_features])
        return cats+reals

    def dtf(self,dict,keys):
        return torch.cat([dict[key] for key in keys],dim=-1)

    