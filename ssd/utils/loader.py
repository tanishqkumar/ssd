import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_eagle_model(model: nn.Module, path: str, packed_modules_mapping: dict):
    """Load EAGLE3 draft model weights from pytorch_model.bin"""
    print(f"[load_model] Detected EAGLE3 draft model, loading from pytorch_model.bin")
    bin_file = os.path.join(path, "pytorch_model.bin")
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Expected pytorch_model.bin at {bin_file} for EAGLE3 draft model")
    
    state_dict = torch.load(bin_file, map_location="cpu")
    
    # Load d2t and t2d dictionaries
    if hasattr(model, 'd2t') and 'd2t' in state_dict:
        d2t_tensor = state_dict['d2t']
        model.d2t = {i: int(d2t_tensor[i].item()) for i in range(len(d2t_tensor))}
        model.d2t_tensor = d2t_tensor.to('cuda').long()  # keep as tensor for fast indexing
        print(f"[load_model] Loaded d2t dictionary with {len(model.d2t)} entries")
    
    if hasattr(model, 't2d') and 't2d' in state_dict:
        t2d_tensor = state_dict['t2d']
        model.t2d = {i: int(t2d_tensor[i].item()) for i in range(len(t2d_tensor))}
        model.t2d_tensor = t2d_tensor.to('cuda').long()  # keep as tensor for fast indexing
        print(f"[load_model] Loaded t2d dictionary with {len(model.t2d)} entries")
    
    # Load model weights
    for weight_name, loaded_weight in tqdm(state_dict.items(), desc="Loading EAGLE3 weights"):
        # Skip the dictionary tensors as we've already processed them
        if weight_name in ['d2t', 't2d']:
            continue
        
        # Check if this weight should use packed module loading
        is_packed = False
        for k, (v, shard_id) in packed_modules_mapping.items():
            if k in weight_name:
                # Replace the module name but keep the .weight suffix
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                is_packed = True
                break
        
        if is_packed:
            continue
            
        # Map EAGLE3 weight names to our architecture for unpacked weights
        if weight_name == 'midlayer.hidden_norm.weight':
            # Special handling for conditioning feature layernorm
            param_name = 'model.layer.conditioning_feature_ln.weight'
        elif weight_name.startswith('midlayer.'):
            # The single layer is stored in model.layer (not layers list) in our implementation
            param_name = weight_name.replace('midlayer.', 'model.layer.')
        elif weight_name == 'norm.weight':
            # norm.weight -> final_norm.weight
            param_name = 'final_norm.weight'
        else:
            # fc.weight and lm_head.weight stay the same
            param_name = weight_name
        
        # Load the parameter
        param = model.get_parameter(param_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def load_safetensors_model(model: nn.Module, path: str, packed_modules_mapping: dict):
    """Load model weights from safetensors files"""
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    for file in tqdm(safetensor_files, desc="Loading model files"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_model(model: nn.Module, path: str):
    print(f"[load_model] loading model from {path}")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Check if this is an EAGLE3 draft model
    is_eagle = 'eagle' in path.lower()
    
    if is_eagle:
        load_eagle_model(model, path, packed_modules_mapping)
    else:
        load_safetensors_model(model, path, packed_modules_mapping)

    print(f"[load_model] finished loading model from {path}")