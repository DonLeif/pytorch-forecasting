from pytorch_forecasting.models import BaseModelWithCovariates
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.metrics import DistributionLoss
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
        **kwargs):

        self.save_hyperparameters(ignore=['loss','logging_metrics'])
        super().__init__(**kwargs)
        
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
        
        # self.embedder_real_enc_only = MultiConvEmbedding(self.config,self.hparams.nfuk_reals)
        # self.embedder_real_dec_and_enc = MultiConvEmbedding(self.config,self.hparams.n_reals-self.hparams.nfuk_reals)
    
        # TIME PROJECT
        self.time_project1 = nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
                                          nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
                                          nn.Linear(self.config.n_embd, self.config.n_embd))
        
        self.time_project2 = nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
                                          nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
                                          nn.Linear(self.config.n_embd, self.config.n_embd))
        
        # self.time_project3 = nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
        #                                   nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
        #                                   nn.Linear(self.config.n_embd, self.config.n_embd))
        
        # self.time_project4 = nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
        #                                   nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
        #                                   nn.Linear(self.config.n_embd, self.config.n_embd))
        
        


        # self.time_embedding = nn.Sequential(nn.Linear(2, self.config.n_embd),nn.GELU(),
        #                                   nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
        #                                   nn.Linear(self.config.n_embd, self.config.n_embd))


        # self.time_embedding_abs = TimeEmbedding(self.config)
        # self.time_embedding_rel = TimeEmbedding(self.config)

        # self.time_embedding = TimeEmbedding2(self.config.n_embd)


        # self.conv = TimeSeriesConvBlock(self.config.n_embd,10)

        # ENC PROJECT
        self.enc_project = nn.Linear(enc_proj, self.config.n_embd)

        # DEC PROJECT
        self.dec_project = nn.Linear(dec_proj, self.config.n_embd)

        # SELF ATTENTION - understand the past in relationship to itself
        self.block1 = Block(self.config)

        # CROSS ATTENTION - inform the future with the past
        #self.block_cross = CrossBlock(self.config)
        #self.block_cross2 = CrossBlock(self.config)
        #if self.config.n_cross > 1:
        self.time_cross = nn.ModuleList([nn.Sequential(nn.Linear(1, self.config.n_embd),nn.GELU(),
                                        nn.Linear(self.config.n_embd, self.config.n_embd),nn.GELU(),
                                        nn.Linear(self.config.n_embd, self.config.n_embd)) for _ in range(self.config.n_cross)])
        
        self.blocks_cross = nn.ModuleList([CrossBlock(self.config) for _ in range(self.config.n_cross)])


        self.block_sa1 = Block(self.config,causal=False,reverse=False)
        # CAUSAL SA # roll out the future cumulatively
        self.block_sa2 = Block(self.config,causal=True,reverse=False)
        # self.block_sa3 = Block(self.config,causal=True)

        # OUT BLOCK # find distributional output
        self.block_out = Out(self.config)


    def forward(self, x):
        
        target_scale = x["target_scale"]

        if self.config.time['relative'] is not None:
            time_enc = x['encoder_cont'][:,:,self.config.time['relative']].unsqueeze(-1)
            time_dec = x['decoder_cont'][:,:,self.config.time['relative']].unsqueeze(-1)
        if self.config.time['absolute'] is not None:
            abstime_enc = x['encoder_cont'][:,:,self.config.time['absolute']].unsqueeze(-1)
            abstime_dec = x['decoder_cont'][:,:,self.config.time['absolute']].unsqueeze(-1)

        time_dec1 = self.time_project1(time_dec)
        time_enc1 = self.time_project1(time_enc)
        time_dec2 = self.time_project2(time_dec) # torch.linspace(0,1,self.hparams.dec_len,device=time_dec.device).reshape(1,-1,1)
        time_dec3 = self.time_project3(time_dec)



    
        nfuk_reals = self.hparams.nfuk_reals
        nfuk_cats = self.hparams.nfuk_cats

        enc_len = self.hparams.enc_len

        # time_enc,time_dec = (time_enc + 1) * enc_len, (time_dec + 1) * enc_len


        if self.hparams.x_cats:
            x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  
            x_cat = self.embedder_cat(x_cat)
            x_cat = torch.cat(tuple(x_cat.values()),dim=-1)

        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  
        
        # if self.config.time['absolute'] is not None:
        #     r = self.config.time['absolute']
        #     x_cont = torch.cat((x_cont[..., :r], x_cont[..., r+1:]), dim=-1)
        
        only_past_known_reals = x_cont[:,:enc_len,-nfuk_reals:]   
        only_past_known_reals = self.embedder_real_enc_only(only_past_known_reals)

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




        # if self.config.time['relative'] is not None:
        #     x_dec += self.time_project(time_dec)
        #     x_enc += self.time_project(time_enc)

        # if self.config.time['absolute'] is not None:
        #     x_dec += self.time_embedding(abstime_dec)
        #     x_enc += self.time_embedding(abstime_enc)

        # timedec = torch.cat((time_dec,abstime_dec),dim=-1)
        # x_dec += self.time_embedding(timedec)

        # timeenc = torch.cat((time_enc,abstime_enc),dim=-1)
        # x_enc += self.time_embedding(timeenc)

        # if self.config.time['relative'] is not None:
        #     x_dec += self.time_embedding_rel(time_dec)
        #     x_enc += self.time_embedding_rel(time_enc)

        # if self.config.time['absolute'] is not None:
        #     x_dec += self.time_embedding_abs(abstime_dec)
        #     x_enc += self.time_embedding_abs(abstime_enc)

        
        x = x_dec
        # x = self.block_cross(x_dec,x_enc)

        b,t,c = x_dec.shape
        #splits = t // self.config.n_cross
        for i,block in enumerate(self.blocks_cross):
            x_enc += time_enc1
            x += time_dec1
            x = block(x,x_enc)

        x += time_dec2
        x = self.block_sa1(x)
        x += time_dec3
        x = self.block_sa2(x)

        
        output = self.block_out(x)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=target_scale),
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
                                                            'n_cross': 1},
                     **kwargs):
        
        config['enc_len'] = dataset.max_encoder_length
        config['dec_len'] = dataset.max_prediction_length
        if 'loss' in kwargs:
            loss = kwargs['loss']
            try:
                config['outputs'] = len(loss.quantiles)
                config['losstype'] = 'quantiles'
            except:
                config['outputs'] = 1

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