from pytorch_forecasting.models import BaseModelWithCovariates
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss,DistributionLoss
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .submodules import *

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

def get_config():
    return {'n_embd': 128,'max_cat_embd':16,'n_real_embd':4,
          'n_head':4,'dropout':0.20,'bias':True,
          'n_cross':1}

class Timeformer(BaseModelWithCovariates):

    def __init__(
        self,
        config_dict: Dict,
        x_reals: List[str],
        n_reals: int,
        x_cats: List[str],
        n_cats: int,
        nfuk_reals: int,
        nfuk_cats: int,
        enc_len: int,
        dec_len: int,
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        embedding_paddings: List[str],
        categorical_groups: Dict[str, List[str]],
        learning_rate: float = 1e-3,
        limit_decoder: int = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs):

        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])

        self.save_hyperparameters(ignore=['loss','logging_metrics'])
        super().__init__(**kwargs,logging_metrics=logging_metrics)

        self.pred_mode = False
        
        @dataclass
        class Config:
            n_embd: int
            max_cat_embd: int
            n_real_embd: int
            n_head: int
            n_cross: int
            dropout: float
            bias: bool
            enc_len: int
            dec_len: int
            outputs: int
            losstype: str
            time: dict
            limit_decoder: int

        self.config = Config(**self.hparams.config_dict)


        # how large is the dimensionality of all categoricals
        if self.hparams.x_cats:
            _,e =zip(*self.hparams.embedding_sizes.values())
            self.cat_dims = [min(self.config.max_cat_embd,i) for i in list(e)]
            self.dim_emb_cats_enc = sum(self.cat_dims)
            self.dim_emb_cats_dec = sum(self.cat_dims[-self.hparams.nfuk_cats:])
        else:
            self.dim_emb_cats_enc = 0
            self.dim_emb_cats_dec = 0

        # how large is the dimensionality of all reals
        self.dim_emb_reals_enc =  self.hparams.n_reals * self.config.n_real_embd
        self.dim_emb_reals_dec = (self.hparams.n_reals - self.hparams.nfuk_reals) * self.config.n_real_embd

        enc_proj = self.dim_emb_cats_enc + self.dim_emb_reals_enc
        dec_proj = self.dim_emb_cats_dec + self.dim_emb_reals_dec
        
        # EMBED CATEGORICALS
        if self.hparams.x_cats:
            self.embedder_cat = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.config.max_cat_embd,
            )

        # EMBED REALS
        self.embedder_real_enc_only = RealEmbedding(self.hparams.nfuk_reals,self.config.n_real_embd)
        self.embedder_real_dec_and_enc = RealEmbedding(self.hparams.n_reals-self.hparams.nfuk_reals,self.config.n_real_embd)
        
        # TIME PROJECT
        self.time_project_rel = nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
                                          nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
                                          nn.Linear(self.config.n_embd, self.config.n_embd))
        
        self.time_project_abs = nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
                                        nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
                                        nn.Linear(self.config.n_embd, self.config.n_embd))


        # ENC PROJECT
        self.enc_project = nn.Linear(enc_proj, self.config.n_embd)

        # DEC PROJECT
        self.dec_project = nn.Linear(dec_proj, self.config.n_embd)

        self.prenorm_enc = nn.LayerNorm(self.config.n_embd)
        self.prenorm_dec = nn.LayerNorm(self.config.n_embd)

        # SELF ATTENTION - understand the past in relationship to itself
        self.block1 = Block(self.config,causal=False,reverse=False)
        
        self.blocks_cross = nn.ModuleList([CrossBlock(self.config) for _ in range(self.config.n_cross)])

        # self.block_sa1 = Block(self.config,causal=False,reverse=False)
        if self.config.limit_decoder is None:
            self.block_sa1 = Block(self.config)
        else:
            self.block_sa1 = CrossBlock(self.config)
        
        # self.block_sa2 = Block(self.config,causal=True)

        # OUT BLOCK # find distributional output
        


###### added
        if self.n_targets > 1:  # if to run with multiple targets
            self.block_out = nn.ModuleList(
                [Out(self.config,i=i) for i in range(self.n_targets)]
                )
        else:
            self.block_out = Out(self.config)
###### added end


    def forward(self, x):
        
        target_scale = x["target_scale"]

        if self.config.time['relative'] is not None:
            time_enc = x['encoder_cont'][:,:,self.config.time['relative']].unsqueeze(-1)
            time_dec = x['decoder_cont'][:,:,self.config.time['relative']].unsqueeze(-1)
            time_dec_rel = self.time_project_rel(time_dec)
            time_enc_rel = self.time_project_rel(time_enc)
        if self.config.time['absolute'] is not None:
            abstime_enc = x['encoder_cont'][:,:,self.config.time['absolute']].unsqueeze(-1)
            abstime_dec = x['decoder_cont'][:,:,self.config.time['absolute']].unsqueeze(-1)
            time_dec_abs = self.time_project_abs(abstime_dec)
            time_enc_abs = self.time_project_abs(abstime_enc)



    
        nfuk_reals = self.hparams.nfuk_reals
        nfuk_cats = self.hparams.nfuk_cats

        enc_len = self.hparams.enc_len

        # time_enc,time_dec = (time_enc + 1) * enc_len, (time_dec + 1) * enc_len


        if self.hparams.x_cats:
            x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  
            x_cat = self.embedder_cat(x_cat)
            x_cat = torch.cat(tuple(x_cat.values()),dim=-1)

        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  
        
        only_past_known_reals = x_cont[:,:enc_len,-nfuk_reals:]   
        only_past_known_reals = self.embedder_real_enc_only(only_past_known_reals)

        # embed case of not known future reals, tho we always have time idx
        fully_known_reals = x_cont[:,:,:-nfuk_reals]  
        fully_known_reals = self.embedder_real_dec_and_enc(fully_known_reals)

        if self.hparams.x_cats:
            x_cat_enc = x_cat[:,:enc_len,:] 
            x_cat_dec = x_cat[:,enc_len:,sum(self.cat_dims[:-nfuk_cats]):]

        x_cont_enc = fully_known_reals[:,:enc_len,:]
        x_cont_enc = torch.cat([x_cont_enc,only_past_known_reals],dim=-1)
        x_cont_dec = fully_known_reals[:,enc_len:,:]
        
        if self.hparams.x_cats:
            x_enc = torch.cat([x_cat_enc,x_cont_enc],dim=-1)
            x_dec = torch.cat([x_cat_dec,x_cont_dec],dim=-1)
        else:
            x_enc = x_cont_enc
            x_dec = x_cont_dec
            
        x_dec = self.dec_project(x_dec)
        x_enc = self.enc_project(x_enc)

        if self.config.time['relative'] is not None:
            x_enc += time_enc_rel
            x_dec += time_dec_rel

        if self.config.time['absolute'] is not None:
            x_enc += time_enc_abs
            x_dec += time_dec_abs

        x_dec = self.prenorm_dec(x_dec)
        x_enc = self.prenorm_enc(x_enc)
        
        x_enc = self.block1(x_enc)

        x = x_dec

        space = torch.logspace(-1,0,self.config.n_cross) * self.hparams.dec_len
        space = space.int().to(x.device)
        for i,block in enumerate(self.blocks_cross):
            if self.config.n_cross > 1:
                x_enc_i = x_enc[:,-space[i]:,:]
            else:
                x_enc_i = x_enc
            x = block(x,x_enc_i)


        if self.config.limit_decoder is None:
            x = self.block_sa1(x)
        else:
            x = self.block_sa1(x[:,:self.config.limit_decoder,:],x)

        if self.n_targets > 1:  # if to use multi-target architecture
            output = [block_out(x) for block_out in self.block_out]
        else:
            output = self.block_out(x)


        if not self.pred_mode:
            pred = output
        else:
            pred = self.transform_output(output, target_scale=target_scale)

        return self.to_network_output(
            prediction=pred,
            raw = output, target_scale = target_scale)
        
    def predict_df(self,dataloader):
        predictions,idx = self.predict(dataloader,mode='quantiles',return_index=True)
        quantiles = len(self.loss.quantiles)
        horizon = self.hparams.dec_len
        preds = pd.DataFrame(np.tile(idx.values,quantiles*horizon).reshape(len(idx)*quantiles*horizon,idx.shape[-1]))
        preds.columns = idx.columns
        preds['horizon'] = np.tile(np.repeat(np.arange(1,horizon+1),quantiles),len(preds) // (quantiles * horizon)).flatten()
        preds['quantile'] = np.tile([self.loss.quantiles],len(preds)//quantiles).flatten()
        preds['yhat'] = predictions.cpu().flatten()
        preds['pred_idx'] = preds['time_idx'] + preds['horizon'] - 1
        return preds
    
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet,config={'n_embd': 32,
                                                            'max_cat_embd': 16,
                                                            'n_real_embd': 4,
                                                            'n_head': 4,
                                                            'dropout': 0.2,
                                                            'bias': True,
                                                            'n_cross': 1,
                                                            'limit_decoder':None},
                     **kwargs):
        
        config['enc_len'] = dataset.max_encoder_length
        config['dec_len'] = dataset.max_prediction_length


        if 'limit_decoder' in kwargs:
            config['limit_decoder'] = kwargs['limit_decoder']

        if 'loss' in kwargs:
            loss = kwargs['loss']
            try:
                if isinstance(loss,MultiLoss):
                    config['outputs'] = [len(i) for i in loss.quantiles]
                else:
                    config['outputs'] = len(loss.quantiles)
                config['losstype'] = 'quantiles'
            except:
                config['outputs'] = 1
                config['losstype'] = 'point'

            if isinstance(loss,DistributionLoss):
                config['outputs'] = len(loss.distribution_arguments)
                config['losstype'] = 'distribution'

        else:
            config['outputs'] = 1


        
        x_reals = dataset.reals
        time = {'absolute':None,'relative':None}
        
        if 'time_idx' in x_reals:
            time['absolute'] = x_reals.index('time_idx')

        if 'relative_time_idx' in x_reals:
            time['relative'] = x_reals.index('relative_time_idx')
            
        config['time'] = time
            
        new_kwargs = {
            "config_dict": config,
            "x_reals": x_reals,
            "n_reals": len(x_reals),
            "x_cats": dataset.categoricals,
            "n_cats": len(dataset.categoricals),
            "nfuk_reals": len(dataset.time_varying_unknown_reals),
            "nfuk_cats": len(dataset.time_varying_unknown_categoricals),
            "enc_len": dataset.max_encoder_length,
            "dec_len": dataset.max_prediction_length,
        }
        new_kwargs.update(kwargs)  
        return super().from_dataset(dataset, **new_kwargs)