from transformers import AutoTokenizer
from loraLinear import get_lora_model
from verifier import compute_reward, build_prompt
import torch
import json

torch.manual_seed(0)

with open('../data/eval_instances.json', 'r') as f:
    evalData = json.load(f)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

modelName = "Qwen/Qwen2.5-0.5B-Instruct"
model = get_lora_model(modelName)

tokenizer = AutoTokenizer.from_pretrained(modelName)

# for ckpt in range(0, 301, 10):

grpoWeights = torch.load(f"../result/grpo_ckpt_runE/240.pt.pt", map_location='cpu')
model.load_state_dict(grpoWeights, strict=False)

model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
padId = tokenizer.pad_token_id


samplesPerInstance = 8
d = {}

corr = 0
gen = 0

indiv = 0
correctInstArr = []

for inst in evalData:
    
    d[inst] = {}
    
    nums = evalData[inst]['numbers']
    target = evalData[inst]['target']

    prompt = build_prompt(nums, target, tokenizer)
    
    promptToken = tokenizer([prompt], add_special_tokens=False, return_tensors='pt').to(model.device)
    promptIds = promptToken['input_ids']

    rewardArr = []

    totalCorrect = 0
    totalGen = 0

    for s in range(samplesPerInstance):
        
        generatedTokens = model.generate(
            **promptToken,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )

        generatedTokens = generatedTokens[:, len(promptIds[0]):]
        response = tokenizer.batch_decode(generatedTokens, skip_special_tokens=True)[0]
        reward = compute_reward(response, nums, target)

        if reward >= 1.0:
            totalCorrect += 1
        
        totalGen += 1

        d[inst][s] = {'numbers': nums,
                    'target': target,
                    'response': response,
                    'reward': reward}

        rewardArr.append(reward)
    
    d[inst]['rewardMean'] = sum(rewardArr)/len(rewardArr)
    d[inst]['solveRate'] = totalCorrect / totalGen

    corr += totalCorrect
    gen += totalGen

    if totalCorrect > 0:
        indiv += 1
        correctInstArr.append(inst)

print(f"ckpt: ../result/grpo_eval_runE, instance correct: {indiv}/{len(evalData)}({correctInstArr}), total correct: {corr}/{gen}")

with open(f'../result/sft_lora_trace1_eval100.json', 'w') as f:
    json.dump(d, f, indent=4)
