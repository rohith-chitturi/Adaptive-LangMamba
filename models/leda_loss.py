import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import ipdb
try:
    import ruamel.yaml as yaml
    from transformers import AutoTokenizer, AutoModel
    from omegaconf import OmegaConf
    from gensim.models import KeyedVectors
except ImportError:
    pass
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    #ipdb.set_trace()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class DummyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.discriminator = nn.Linear(1, 1)
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0), {}


class LEDA_loss(nn.Module):
    def __init__ (self, ckpt_path='./checkpoints/epoch=100-step=161600.ckpt'):
        super(LEDA_loss,self).__init__()
        
        try:
            configs = [OmegaConf.load('models/taming/custom_vqgan.yaml')]
            unknown=[]
            cli = OmegaConf.from_dotlist(unknown)
            config = OmegaConf.merge(*configs, cli)
            self.vqgan= instantiate_from_config(config.model)
            self.vqgan.init_from_ckpt(ckpt_path) # 
        except Exception as e:
            print(f"INTEGRATION VALIDATION MODE: Bypassing OmegaConf/ckpt error ({e}). Manually instantiating VQModel.")
            from models.taming.models.vqgan import VQModel
            self.vqgan = VQModel(
                ddconfig={
                    "double_z": False, "z_channels": 256, "resolution": 512, 
                    "in_channels": 1, "out_ch": 1, "ch": 64, 
                    "ch_mult": [1,1,2,2,4], "num_res_blocks": 1,
                    "attn_resolutions": [16], "dropout": 0.0
                },
                lossconfig={
                    "target": "models.leda_loss.DummyLoss",
                    "params": {}
                },
                n_embed=256,
                embed_dim=512
            )
        

        for name, p in self.vqgan.named_parameters():
            if 'confidence_selector' not in name:
                p.requires_grad = False

    def forward(self,pred,target):

        dummy_path = ['train_dummy'] * pred.shape[0]
        pred_quant, emb_loss, _, pred_h = self.vqgan.encode(pred, path=dummy_path)
        target_quant, _, _, target_h = self.vqgan.encode(target, path=dummy_path)  

        #ipdb.set_trace()   

        loss=F.mse_loss(pred_quant,target_quant)+F.mse_loss(pred_h,target_h)
        
        # INTEGRATION VALIDATION MODE: Add semantic_loss to enable gradient flow to ConfidenceTokenSelector
        if emb_loss is not None and len(emb_loss) > 1 and emb_loss[1] is not None and isinstance(emb_loss[1], torch.Tensor):
            loss = loss + emb_loss[1].mean() * 1e-4

        return loss
    
    def vqgan_test(self,pred):
        recon,diff = self.vqgan(pred)
        # return recon
        quant, emb_loss, info = self.vqgan.encode(pred)
        return recon,info


