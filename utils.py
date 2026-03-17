import os
from glob import glob

import torch
from safetensors import safe_open

def load_weights(model, path: str):
    """copy original weights from safetensor to the model we have"""
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                _weight_name = weight_name.replace("model.", "")
                randn_weight = model.get_parameter(_weight_name)
                true_weight = f.get_tensor(weight_name)

                with torch.no_grad():
                    randn_weight.copy_(true_weight)
            
            # do for the final weight tied layer seperately
            randn_weight = model.get_parameter("out_head.weight")
            true_weight = f.get_tensor("model.embed_tokens.weight")

            with torch.no_grad():
                randn_weight.copy_(true_weight)