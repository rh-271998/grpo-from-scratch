from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from verifier import compute_reward, build_prompt
import json

torch.manual_seed(0)

modelName = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(modelName)

model = AutoModelForCausalLM.from_pretrained(
    modelName,
    dtype=torch.float32,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model.to(device)

with open('../data/eval_100.json') as f:
    evalData = json.load(f)

d = {}
for k in evalData:
    prompt = build_prompt(evalData[k]['numbers'], evalData[k]['target'], tokenizer)

    tokens = tokenizer([prompt], return_tensors='pt').to(model.device)
    tokenIds = tokens['input_ids']
    attnMask = tokens['attention_mask']
    
    generatedTokens = model.generate(
        **tokens,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )
   
    generatedTokens = generatedTokens[:, len(tokenIds[0]):]

    response = tokenizer.batch_decode(generatedTokens, skip_special_tokens=True)[0]
    # print(response)
    reward = compute_reward(response, evalData[k]['numbers'], evalData[k]['target'])
    # print(reward)
    
    d[k] = {'numbers': evalData[k]['numbers'],
            'target': evalData[k]['target'],
            'response': response,
            'reward': reward}

with open('../result/before_gallery_100.json', 'w', encoding="utf-8") as f:
    json.dump(d, f, indent=4)