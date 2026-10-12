import json
from verifier import build_prompt
from transformers import AutoTokenizer
from loraLinear import get_lora_model
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
import time

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    

modelName = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(modelName)

model = get_lora_model(modelName)

with open('../data/sft_data.json', 'r') as f:
    sftData = json.load(f)

maxLen = 256

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
padId = tokenizer.pad_token_id

allInput, allMask, allLabel = [], [], []

for k in sftData:

    nums = sftData[k]['numbers']
    target = sftData[k]['target']
    expr = sftData[k]['expr']
    trace = sftData[k]['trace']

    prompt = build_prompt(nums, target, tokenizer)
    reasoning = f"<think>{trace}</think><answer>{expr}</answer><|im_end|>"

    promptToken = tokenizer(prompt, add_special_tokens=False)
    promptIds = promptToken['input_ids']
    promptAttnMask = promptToken['attention_mask']

    reasoningToken = tokenizer(reasoning, add_special_tokens=False)
    reasoningIds = reasoningToken['input_ids']
    reasoningAttnMask = reasoningToken['attention_mask']
    
    inputIds = promptIds + reasoningIds

    labels = [-100] * len(promptIds) + reasoningIds

    attnMask = [1]*len(inputIds)

    padLen = maxLen - len(inputIds)

    assert padLen >=0, "sequence longer than maxLen"

    attnMask += [0] * padLen
    labels += [-100] * padLen
    inputIds += [padId] * padLen

    allInput.append(inputIds)
    allMask.append(attnMask)
    allLabel.append(labels)

inputT = torch.tensor(allInput)
maskT = torch.tensor(allMask)
labelT = torch.tensor(allLabel)

dataset = TensorDataset(inputT, maskT, labelT)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model.to(device)
model.train()
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

for epoch in range(3):
    for step, (inputId, attn, label) in enumerate(loader):
        
        t = time.time()
        
        inputId, attn, label = inputId.to(device), attn.to(device), label.to(device)

        loss = model(input_ids=inputId, attention_mask=attn, labels=label).loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%20 == 0:
            print(f"ep: {epoch}, step: {step}, loss: {loss.item():.4f}, time: {time.time()-t}")

adapters = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
torch.save(adapters, "../result/sft_lora_trace1.pt")
