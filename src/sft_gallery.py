import json
from transformers import AutoTokenizer
from loraLinear import get_lora_model
import torch
from verifier import build_prompt, compute_reward

torch.manual_seed(0)

modelName = "Qwen/Qwen2.5-0.5B-Instruct"
model = get_lora_model(modelName)

adapterPath = '../result/sft_lora_trace1.pt'

loraWeights = torch.load(adapterPath, map_location='cpu')

missingKeys, unexpectedKeys = model.load_state_dict(loraWeights, strict=False)
print(f'base model weights unchanged: {len(missingKeys)}')
print(f'unexpected keys: {unexpectedKeys}')

model.to('mps')

with open('../data/eval_instances.json', 'r') as f:
    evalData = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(modelName)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
padId = tokenizer.pad_token_id

d = {}

for k in evalData:
    nums = evalData[k]['numbers']
    target = evalData[k]['target']

    prompt = build_prompt(nums, target, tokenizer)

    promptToken = tokenizer([prompt], add_special_tokens=False, return_tensors='pt').to(model.device)
    
    promptIds = promptToken['input_ids']
    
    generatedTokens = model.generate(
        **promptToken,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2,
    )
    
    generatedTokens = generatedTokens[:, len(promptIds[0]):]
    response = tokenizer.batch_decode(generatedTokens, skip_special_tokens=True)[0]

    reward = compute_reward(response, nums, target)

    d[k] = {'numbers': nums,
            'target': target,
            'response': response,
            'reward': reward}

with open('../result/sft_gallery_trace1.json', 'w') as f:
    json.dump(d, f, indent=4)