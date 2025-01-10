
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

    """Just a linear projection with no activation"""

    def __init__(self,in_size,out_size,bias=True):
        super().__init__()
        self.c_fc = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        x = self.c_fc(x)
        return x
    

class MLP(nn.Module):

    """Simple feedforward network with GELU activation and dropout"""

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
    

class Smoothing_GRU(nn.Module):

    """A double exponential smoothing -esque layer that parameterizes smoothing parameters with a GRU"""


    def __init__(self,cov_emb,dropout,enc_len, dec_len,factor=1,damped=True):
        super().__init__()

        self.gru = nn.GRU(input_size=cov_emb,hidden_size=cov_emb*factor,dropout=0,num_layers=1,bias=True,batch_first=True,bidirectional=False)
        self.param_projection = nn.Linear(cov_emb*factor,3 if damped else 2)
        # self.trend_proj = nn.Linear(1,cov_emb)

        self.len_enc = enc_len
        self.len_dec = dec_len
        self.damped = damped

    def forward(self, x, level_past):
        b, t, s = x.size()
        assert level_past.dim() == 2, "level_past must be 2-dimensional (batch_size x len_enc)"

        enc_range = torch.arange(self.len_enc,device=x.device).flip(0)

        _,h = self.gru(x)
        x = h.permute(1,0,2)[:,-1,:]
        x = self.param_projection(x)
        x = torch.sigmoid(x)
        alpha = x[:,0].unsqueeze(1)
        beta = x[:,1].unsqueeze(1)
        phi = x[:,2].unsqueeze(1) if self.damped else None

        weights = (alpha * (1 - alpha) ** enc_range)
        weights = weights / weights.sum(1, keepdim=True)

        output_level = (weights.unsqueeze(1) @ level_past.unsqueeze(-1)).repeat(1,self.len_dec,1)

        # Trend calculation
        target_diff = level_past.diff(dim=1)
        
        weights = (beta * (1 - beta) ** enc_range[1:])
        weights /= weights.sum(1, keepdim=True)
        output_trend = (weights.unsqueeze(1) @ target_diff.unsqueeze(-1)).repeat(1, self.len_dec, 1)

        # Damped trend calculation
        if self.damped:
            output_trend *= phi.unsqueeze(-1) ** torch.arange(self.len_dec,device=x.device).unsqueeze(0).unsqueeze(-1)

        # Cumulative sum to generate the trend over time
        output_trend = output_trend.cumsum(dim=1)
        
        # Combine level and trend
        output_tensor = output_level + output_trend

        return output_tensor, (alpha,beta,phi)

    
class LayerNorm(nn.Module):

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
        x = x.transpose(-1,-2).contiguous()

        reshape = False
        if x.dim() == 4:
            reshape=True
            b, f, c, t = x.size()
            x = x.view(b*f,c,t)

        x = self.conv(x)
        x = x[..., :-2] 
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(-1,-2)
        x = self.proj_back(x)
        if reshape:
            x = x.view(b,f,t,c)

        return x

class MultiAttention(nn.Module):
    def __init__(self,causal,cov_emb:int = 32,bias:bool = False,n_head:int = 2,factor:int = 1,dropout:float = 0.3,att_proj='linear'):
        super().__init__()

        self.causal = causal

        self.c_attn_q  = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        self.c_attn_k = TemporalConvolutionalLayer(cov_emb, dropout=dropout)
        self.c_attn_v = TemporalConvolutionalLayer(cov_emb, dropout=dropout)

        self.c_proj = nn.Linear(cov_emb, cov_emb, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = cov_emb
        self.dropout = dropout

        self.ln_att = LayerNorm(cov_emb, bias=bias)
        self.mlp = MLP(cov_emb, cov_emb,factor=factor,dropout=dropout, bias=bias)

    def forward(self,causal_decoder,past_kv):

        queries_stacked = torch.stack(tuple(causal_decoder.values()),dim=1)
        queries = self.c_attn_q(queries_stacked)

        keys = self.c_attn_k(past_kv).unsqueeze(1)
        values = self.c_attn_v(past_kv).unsqueeze(1)

        B, c, T, C = keys.size()
        keys = keys.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)   # going from B,c,T,nh,Ch to B,c,nh,T,ch

        B, c, T, C = values.size()
        values = values.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3) 

        B, c, T, C = queries.size()
        queries = queries.view(B, c, T, self.n_head, C // self.n_head).transpose(2, 3)

        y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(2, 3).contiguous().view(B, c, T, C) 

        output  = queries_stacked + self.resid_dropout(self.c_proj(y))
        output = output + self.mlp(self.ln_att(output))
        for i,feature in enumerate(self.causal):
            causal_decoder[feature] = output[:,i]

        return causal_decoder



class HNAM(BaseModelWithCovariates):
    def __init__(
        self,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,
        max_encoder_length: int = 10,
        max_prediction_length: int = 10,
        target = [],
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
        cov_emb: int = 32,
        cov_heads: int = 2,
        dropout: float = 0.1,
        factor: int = 2,
        bias: bool = False,
        clean_past = True,
        base = [],
        causal = [],
        one_hot_not_minus_one = [],
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

        self.causal_lns = nn.ModuleDict({feature:LayerNorm(cov_emb, bias=bias) for feature in self.hparams.causal})
        self.past_ln = LayerNorm(cov_emb,bias=bias)
        self.post_lns = nn.ModuleDict({feature:LayerNorm(cov_emb,bias=bias) for feature in self.hparams.causal})


        self.trend_block = Smoothing_GRU(
                                cov_emb = cov_emb,
                                dropout = dropout,
                                enc_len = self.hparams.max_encoder_length,
                                dec_len = self.hparams.max_prediction_length,
                                factor=1,
                                damped=True
        )

        if self.hparams.attention:
            self.multi_attention = MultiAttention(causal = self.hparams.causal,cov_emb=cov_emb,bias=bias,n_head=cov_heads,factor=1,dropout=dropout,att_proj=self.hparams.att_proj)

        self.causal_mlps_post = nn.ModuleDict({feature:MLP(cov_emb,cov_emb,factor=factor,dropout=dropout,bias=bias) for feature in self.hparams.causal})
        self.causal_projections = nn.ModuleDict({feature:Projection(cov_emb,(self.hparams['embedding_sizes'].get(feature,(2,None))[0]-int(feature not in self.hparams.one_hot_not_minus_one))*output_size,bias=True) for feature in self.hparams.causal})

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        # DEFINING SHAPES and HPARAMS
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        target_scale = x["target_scale"]
        max_encoder_length = int(encoder_lengths.max())
        max_decoder_length = int(decoder_lengths.max())
        batch_size = x["encoder_lengths"].size(0)

        # DEFINING FEATURES
        reals =    self.hparams.x_reals
        cats = self.hparams.x_categoricals
        static_cats = self.hparams.static_categoricals
        static_reals = self.hparams.static_reals
        reals_enc = self.hparams.time_varying_reals_encoder
        cats_enc = self.hparams.time_varying_categoricals_encoder
        reals_dec =  self.hparams.time_varying_reals_decoder
        cats_dec = self.hparams.time_varying_categoricals_decoder

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


        # All we know about the past
        past_key_values = self.dtf(x_encoded,cats+reals,stack=True).sum(axis=2)[:,:max_encoder_length,:]
        past_key_values = self.past_ln(past_key_values)  # dim 3


        clean_past = self.hparams.clean_past
        dec_slice = slice(max_encoder_length,None)
        enc_slice = slice(0,max_encoder_length)
        all_slice = slice(0,None)
        if clean_past:
            causal_slice = all_slice
            limit_slice = dec_slice
            cov_effects = max_encoder_length + max_decoder_length
        else:
            causal_slice = dec_slice
            limit_slice = all_slice
            cov_effects = max_decoder_length



        # FINDING COVARIATE COEFFICIENTS
        # assemble tensors that only 'know' the cov it self and those lower in the interaction hierarchy + base information
        causal_decoder = {}
        for i,feature in enumerate(self.hparams.causal):
            # preparing queries for attention
            causal_decoder[feature] = self.dtf(x_encoded,self.hparams.causal[:i+1]+self.hparams.base,stack=True).sum(axis=2)[:,causal_slice,:]
            causal_decoder[feature] = self.causal_lns[feature](causal_decoder[feature])

        # MULTI ATTENTION BLOCK
        if self.hparams.attention:
            causal_decoder = self.multi_attention(causal_decoder,past_key_values)

        # PROJECT BACK ACCOUNTING FOR NO OF CLASSES AND QUANTILES
        for feature in self.hparams.causal:
            causal_decoder[feature] = self.causal_mlps_post[feature](self.post_lns[feature](causal_decoder[feature]))
            causal_decoder[feature] = self.causal_projections[feature](causal_decoder[feature])  
            causal_decoder[feature] = torch.stack(causal_decoder[feature].chunk(self.hparams.output_size,dim=-1),dim=1)
    
        cov_effects = torch.zeros(batch_size,cov_effects,self.hparams.output_size).to(x_encoded[cats[0]].device)

        cats_dec = [c for c in self.hparams.causal if c in cats_dec]
        cat_effects = {}
        for c in cats_dec:
            n_classes = self.hparams['embedding_sizes'][c][0]
            one_hot_cat = torch.nn.functional.one_hot(x_normal[c][:,causal_slice,:],n_classes) # apply one hot encoding to the interger encoded categoricals acc to no classes
            one_hot_cat = one_hot_cat.transpose(1,2)                         # unsqueeze is already done in one hot 
            one_hot_cat = one_hot_cat[:,:,:,int(c not in self.hparams.one_hot_not_minus_one):]  # sparse one hot if not excluded


            cat_effect =  causal_decoder[c] * one_hot_cat                    # multiplication with one hot -> only one effect is left
            cat_effects[c] = cat_effect.sum(dim=-1).transpose(1,2)           # back to batch x time x output_size(quantiles)

            #experimental
            if c in self.hparams.one_hot_not_minus_one:
                cat_effects[c] = cat_effects[c] - cat_effects[c].mean(dim=1,keepdim=True)

            cov_effects += cat_effects[c]                


        reals_dec = [c for c in self.hparams.causal if c in reals_dec]
        real_effects = {}
        for c in reals_dec:
            real_normal = x_normal[c][:,causal_slice,:]
            real_effects[c] = causal_decoder[c].squeeze(-1).transpose(1,2) * real_normal
            cov_effects += real_effects[c]

        # LEVEL CALCULATIONS

        level_past = self.dtf(x_normal,self.hparams.target,stack=True).sum(axis=2)[:,:max_encoder_length,0]  # dim2

        if clean_past:
            level_past = level_past - cov_effects[:,enc_slice,(math.ceil(self.hparams.output_size/2)-1)]


        level_pred, smoothing_params = self.trend_block(past_key_values,level_past)
        level_past = level_past.unsqueeze(-1)

        # OUTPUT

        output = level_pred + cov_effects[:,limit_slice,:]

        if self.hparams.rescale:
            prediction = self.transform_output(output, target_scale=target_scale)
            level_pred = self.transform_output(level_pred, target_scale=target_scale)
            level_past = self.transform_output(level_past, target_scale=target_scale)
            cat_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in cat_effects.items()}
            if len(real_effects) > 0:
                cont_dict = {feature:o * target_scale[:,1].view(o.size(0),1,1) for feature,o in real_effects.items()}
            else:
                cont_dict = {}
        else:
            prediction = output
            cat_dict = cat_effects
            if len(real_effects) > 0:
                cont_dict = real_effects
            else:
                cont_dict = {}


        return self.to_network_output(
                    prediction= prediction,
                    level_past = level_past,
                    level_pred=level_pred,
                    alpha = smoothing_params[0],
                    beta = smoothing_params[1],
                    phi = smoothing_params[2],
                    **cat_dict,
                    **cont_dict)

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
                cov_heads = 2,
                dropout  = 0.3,
                factor = 2,
                bias = False,
                loss = RMSE(),
                attention = True,
                clean_past = True,
                att_proj = 'cnn')
        
        kwargs = {**default_kwargs,**kwargs}
        kwargs['output_size'] = len(kwargs['loss'].quantiles)

        
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs['target'] = [dataset.target] if isinstance(dataset.target, str) else dataset.target
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs["max_prediction_length"] = dataset.max_prediction_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, RMSE()))
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