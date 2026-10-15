"""Persistent deployment-aligned BF16 forward for the original HYPIR LoRA.

FP32 LoRA masters/Adam/state_dict names stay unchanged. Every layer invocation
rebuilds the deployment BF16 A/B -> delta -> merged weight differentiably.
Backward through BF16 casts is a straight-through surrogate, not an exact
derivative of discrete rounding. No teacher-output loss or image cache.
"""
import hashlib
import inspect
import types
from pathlib import Path

import torch
from peft.tuners.lora.layer import Linear, Conv2d

PEFT_LAYER_SHA='39e2f6908c9f3faf4d2de2b6ecfdf44c96d2cae67b8155ed1b51bc67fb25b734'
ADAPTER='default'
EXPECTED_MODULES=257
EXPECTED_MASTER_TENSORS=514
EXPECTED_MASTER_ELEMENTS=259474432


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok,message):
    if not ok:raise RuntimeError(message)


def _guard(layer,name):
    need(type(layer) in (Linear,Conv2d),f'{name}: only recorded PEFT Linear/Conv2d supported')
    need(list(layer.active_adapters)==[ADAPTER] and not layer.disable_adapters and not layer.merged,
         f'{name}: adapter must be active default, enabled and unmerged')
    need(list(layer.lora_A)==list(layer.lora_B)==[ADAPTER],f'{name}: unexpected adapter set')
    need(not layer.lora_variant and layer.use_dora[ADAPTER] is False and layer.lora_bias[ADAPTER] is False,
         f'{name}: variants/DoRA/LoRA bias are unsupported')
    need(layer.r[ADAPTER]==layer.lora_alpha[ADAPTER]==256 and layer.scaling[ADAPTER]==1.0,
         f'{name}: rank256/alpha256/scale1 required; no RSLoRA')
    need(isinstance(layer.lora_dropout[ADAPTER],torch.nn.Identity),f'{name}: dropout must be original Identity')
    base=layer.get_base_layer();a=layer.lora_A[ADAPTER];b=layer.lora_B[ADAPTER]
    need(base.weight.device.type=='cuda' and base.weight.dtype==torch.bfloat16 and not base.weight.requires_grad,
         f'{name}: frozen original CUDA BF16 base required')
    need(base.bias is None or (base.bias.dtype==torch.bfloat16 and not base.bias.requires_grad),f'{name}: frozen BF16 bias required')
    need(a.bias is None and b.bias is None,f'{name}: no adapter biases allowed')
    for p in (a.weight,b.weight):
        need(p.dtype==torch.float32 and p.device==base.weight.device and p.requires_grad,
             f'{name}: exact CUDA FP32 trainable masters required')
    if type(layer) is Linear:
        need(type(base) is torch.nn.Linear and not layer.fan_in_fan_out,f'{name}: original nn.Linear layout required')
    else:
        need(type(base) is torch.nn.Conv2d and base.groups==1,f'{name}: original non-grouped Conv2d required')


def deployment_delta_weight(layer):
    """CUDA get_delta_weight sequence after deployment A/B load casts.

    The installed PEFT CPU BF16 path deliberately uses FP32; it is not the CUDA
    deployment rule and is rejected by _guard. No .data/Parameter replacement.
    """
    # Caller AMP must not silently change these deployment fusion operations.
    with torch.autocast(device_type='cuda',enabled=False):
        a=layer.lora_A[ADAPTER].weight.to(torch.bfloat16)
        b=layer.lora_B[ADAPTER].weight.to(torch.bfloat16)
        if type(layer) is Linear:
            delta=(b @ a)*layer.scaling[ADAPTER]
        elif layer.get_base_layer().weight.shape[2:4]==(1,1):
            delta=(b.squeeze(3).squeeze(2) @ a.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)*layer.scaling[ADAPTER]
        else:
            # Exact PEFT Conv2d recipe, not a flattened-matmul approximation.
            delta=layer.conv_fn(a.transpose(0,1),b).transpose(0,1)*layer.scaling[ADAPTER]
        return delta


def deployment_merged_weight(layer):
    with torch.autocast(device_type='cuda',enabled=False):
        delta=deployment_delta_weight(layer)
        # Same BF16 base + delta cast/add as PEFT safe_merge. Fresh tensor each
        # invocation; base is neither mutated nor repeatedly merged/unmerged.
        return layer.get_base_layer().weight.detach()+delta.to(torch.bfloat16)


def _fused_forward(self,x,*args,**kwargs):
    need(not args and not kwargs,'Deployment-aligned HYPIR layers accept only the original tensor argument')
    _guard(self,self._deployment_layer_name)
    self._deployment_forward_calls+=1
    weight=deployment_merged_weight(self);base=self.get_base_layer()
    if type(self) is Linear:return torch.nn.functional.linear(x,weight,base.bias)
    return base._conv_forward(x,weight,base.bias)


def installed_layers(model):
    return [(name,m) for name,m in model.named_modules() if getattr(m,'_deployment_forward_installed',False)]


def install_deployment_fused_forward(model):
    """Install once AFTER restoring exact FP32 LoRA, before any training call.

    Persistent per-layer forwarding remains installed during Diffusers gradient
    checkpoint recomputation. State_dict/optimizer keys and Parameter identities
    are unchanged. The function refuses already installed/mixed/unrecorded LoRA.
    """
    source=Path(inspect.getfile(Linear)).resolve()
    need(sha(source)==PEFT_LAYER_SHA,'Installed PEFT fusion implementation changed')
    layers=[(n,m) for n,m in model.named_modules() if hasattr(m,'lora_A')]
    need(len(layers)==EXPECTED_MODULES,f'Expected257 LoRA layers, found{len(layers)}')
    expected={};records=[]
    for name,layer in layers:
        need(not getattr(layer,'_deployment_forward_installed',False),'Refusing duplicate installation')
        _guard(layer,name)
        for suffix,p in [('lora_A.default.weight',layer.lora_A[ADAPTER].weight),('lora_B.default.weight',layer.lora_B[ADAPTER].weight)]:
            need(bool(torch.isfinite(p).all()),f'{name}: nonfinite master')
            expected[name+'.'+suffix]=p
        records.append(dict(name=name,kind=type(layer).__name__,base_shape=list(layer.get_base_layer().weight.shape),base_dtype=str(layer.get_base_layer().weight.dtype)))
    actual={n:p for n,p in model.named_parameters() if p.requires_grad}
    need(set(actual)==set(expected) and all(actual[n] is p for n,p in expected.items()),'Trainable Parameter set is not the original514 LoRA masters')
    need(len(expected)==EXPECTED_MASTER_TENSORS and sum(p.numel() for p in expected.values())==EXPECTED_MASTER_ELEMENTS,'LoRA master size changed')
    before=list(model.state_dict());parameter_ids={n:id(p) for n,p in model.named_parameters()}
    for name,layer in layers:
        layer._deployment_original_forward=layer.forward
        layer._deployment_layer_name=name;layer._deployment_forward_calls=0
        layer._deployment_forward_installed=True
        layer.forward=types.MethodType(_fused_forward,layer)
    need(before==list(model.state_dict()) and parameter_ids=={n:id(p) for n,p in model.named_parameters()},'Installation altered serialization/Parameter identity')
    return dict(schema='hypir_deployment_fused_lora_v1',module_count=len(layers),master_tensors=len(expected),master_elements=sum(p.numel() for p in expected.values()),master_dtype='torch.float32',base_dtype='torch.bfloat16',forward='A/B BF16 casts -> original CUDA PEFT delta -> BF16 base merge -> original Linear/Conv2d',backward='autograd through BF16 casts; STE surrogate for rounding',persistent_during_checkpoint_recomputation=True,temporary_parameter_swap=False,base_mutated=False,state_dict_keys_and_parameter_identity_preserved=True,teacher_output_loss=False,peft_layer_source=str(source),peft_layer_sha256=sha(source),module_source_sha256=sha(__file__),layers=records)
