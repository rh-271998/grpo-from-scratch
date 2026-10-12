from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import math

class LoRALinear(nn.Module):

    def __init__(self, base, r=16, alpha=16):
        super().__init__()

        self.base = base
        inF, outF = base.in_features, base.out_features

        self.A = nn.Parameter(torch.empty(r, inF))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.B = nn.Parameter(torch.zeros(outF, r))

        self.scaler = alpha/r

        self.enabled = True

    def forward(self, x):
        
        out = self.base(x)

        if self.enabled:
            out = out + self.scaler * ((x@self.A.T)@self.B.T)
        return out

modelName = "Qwen/Qwen2.5-0.5B-Instruct"

def get_lora_model(modelName):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        torch_dtype=torch.float32,
    )

    # get q_proj and v_proj layers
    loraLayerNames = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    loraLayers = []
    for name, module in model.named_modules():
        for layerName in loraLayerNames:
            if layerName in name:
                loraLayers.append(name)

    for p in model.parameters():
        p.requires_grad_(False)

    for base in loraLayers:

        parentPath, _, childName = base.rpartition('.')
        parent = model.get_submodule(parentPath)
        original = getattr(parent, childName)
        setattr(parent, childName, LoRALinear(original))


    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"targets: {len(loraLayers)}  trainable: {trainable:,}  total: {total:,}  ({100*trainable/total:.3f}%)")

    return model
