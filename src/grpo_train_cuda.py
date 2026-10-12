import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer
import torch
from verifier import compute_reward, build_prompt
from sampleGenerator import generate_instance
from loraLinear import get_lora_model, LoRALinear
from torch.optim import AdamW
import numpy as np
import time
import os
import random

torch.manual_seed(0)

def set_adapters(model, on):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = on

modelName = "Qwen/Qwen2.5-0.5B-Instruct"
# 1. Explicitly move model to CUDA for Kaggle GPU
model = get_lora_model(modelName).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(modelName)

# 2. Updated path pointing to working directory
adapterPath = '../result/sft_lora_trace1.pt'

loraWeights = torch.load(adapterPath, map_location='cpu')

missingKeys, unexpectedKeys = model.load_state_dict(loraWeights, strict=False)

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)

epochs = 301
P = 4
G = 4
maxNewTokens = 256
Beta = 0.02
eps = 0.2
ckptEvery = 10

evalSet = [([4,1,4,19], 27),
           ([7,7,7,19], 26),
           ([10,14,1,9], 33),
           ([9,1,10,8], 82),
           ([2,15,2,14], 1)]

rewardHist = []
klHist = []
lossHist = []

for e in range(epochs):
    
    t = time.time()

    d = {}

    numSamples = 0
    while numSamples < P:

        start = 1
        end = 20
        numsOrig = random.choices(range(start, end + 1), k=4)
        signsOrig = random.choices(range(0, 4), k=3)

        nums, target, expr, trace = generate_instance(numsOrig, signsOrig)

        if nums != None:
            inVal = False
            
            for k in evalSet:
                if sorted(k[0]) == sorted(nums) and k[1] == target:
                    print('nums in val data. skipping...')
                    inVal = True
                    break
            
            if not inVal:
                numSamples += 1

                prompt = build_prompt(nums, target, tokenizer)
                d[numSamples] = {}
                d[numSamples]['prompt'] = prompt
                
                # Explicitly map inputs to cuda
                promptToken = tokenizer([prompt], add_special_tokens=False, return_tensors='pt').to('cuda')

                promptIds = promptToken['input_ids']
                promptMask = promptToken['attention_mask']

                d[numSamples]['promptLen'] = promptIds.shape[1]

                genToken = model.generate(input_ids = promptIds,
                                          attention_mask = promptMask,
                                          max_new_tokens=maxNewTokens,
                                          num_return_sequences=G,
                                          do_sample=True,
                                          temperature=1.0,
                                          top_k=0,
                                          top_p=1.0)
                
                d[numSamples]['genToken'] = genToken

                with torch.no_grad():

                    logits = model(genToken).logits
                    # predLogits = logits[:, :-1, :]
                    # targets = genToken[:, 1:]

                    sl = logits[:, :-1, :]
                    gathered = sl.gather(-1, genToken[:, 1:].unsqueeze(-1)).squeeze(-1)   # [G, L-1] — the token's logit
                    tokenLogp = gathered - torch.logsumexp(sl, dim=-1)              # [G, L-1] — minus vocab logsumexp

                    respLogp = tokenLogp[:, promptIds.shape[1]-1:]
                    d[numSamples]['respLogp']= respLogp

                    gen = genToken[:, promptIds.shape[1]:]
                    isEos = (gen == tokenizer.eos_token_id)
                    mask = (isEos.long().cumsum(dim=1) - isEos.long()) == 0
                    d[numSamples]['mask'] = mask

                    response = tokenizer.batch_decode(gen, skip_special_tokens=True)
                    d[numSamples]['response'] = response

                    rewardArr = []
                    for res in response:
                        rewardArr.append(compute_reward(res, nums, target))
                    d[numSamples]['reward'] = rewardArr

                rewards = torch.tensor(d[numSamples]['reward'], device='cuda')
                adv = (rewards - rewards.mean())/(rewards.std()+1e-8)
                d[numSamples]['adv'] = adv

    torch.cuda.empty_cache()

    policySnapshot = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(loraWeights, strict=False)

    # set_adapters(model, False)

    with torch.no_grad():
        for i in d:
            genToken = d[i]['genToken']
            logits = model(genToken).logits
            # logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            # refLogp = logp.gather(-1, genToken[:, 1:].unsqueeze(-1)).squeeze(-1)

            sl = logits[:, :-1, :]
            gathered = sl.gather(-1, genToken[:, 1:].unsqueeze(-1)).squeeze(-1)   # [G, L-1] — the token's logit
            refLogp = gathered - torch.logsumexp(sl, dim=-1)              # [G, L-1] — minus vocab logsumexp

            d[i]['refLogp'] = refLogp[:, d[i]['promptLen']-1:]

    torch.cuda.empty_cache()

    model.load_state_dict(policySnapshot, strict=False)
    # set_adapters(model, True)

    lossArr = []

    for i in d:
        oldLogp = d[i]['respLogp']
        refLogp = d[i]['refLogp']
        genToken = d[i]['genToken']
        mask = d[i]['mask']
        
        logits = model(genToken).logits
        sl = logits[:, :-1, :]
        gathered = sl.gather(-1, genToken[:, 1:].unsqueeze(-1)).squeeze(-1)   # [G, L-1] — the token's logit
        newLogp = gathered - torch.logsumexp(sl, dim=-1)              # [G, L-1] — minus vocab logsumexp
        newLogp = newLogp[:, d[i]['promptLen']-1:]

        ratio = torch.exp(newLogp - oldLogp)
        adv = d[i]['adv'][:, None]
        pg = -torch.min(ratio*adv, torch.clamp(ratio, 1-eps, 1+eps)*adv)
        kl = torch.exp(refLogp - newLogp) - (refLogp - newLogp) - 1
        perToken = pg + Beta * kl

        loss = (perToken * mask.float()).sum() / mask.sum()
        lossArr.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avgReward = sum(sum(d[i]['reward']) for i in d) / (P*G)
    avgKL = torch.stack([((torch.exp(d[i]['refLogp']-d[i]['respLogp']) - (d[i]['refLogp']-d[i]['respLogp']) - 1) * d[i]['mask']).sum()/d[i]['mask'].sum() for i in d]).mean()
    avgLoss = sum(lossArr)/len(lossArr)

    solve_rate = sum(1 for i in d for r in d[i]['reward'] if r >= 1.0) / (P*G)   # fraction correct
    grp_std    = sum(float(torch.tensor(d[i]['reward']).std()) for i in d) / P   # avg within-group spread

    rewardHist.append(float(avgReward))
    klHist.append(float(avgKL))
    lossHist.append(float(avgLoss))

    if e%10 == 0:
        print(f"*** iter: {e}  loss: {avgLoss:.4f}, reward: {avgReward:.3f}  kl: {avgKL:.4f}, solve_rate: {solve_rate}, grp_std: {grp_std} ***")

    if e % ckptEvery == 0:
        os.makedirs('../result/grpo_ckpt_runE', exist_ok=True)
        adapters = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(adapters, f'../result/grpo_ckpt_runE/{e}.pt')
        print(f'saved checkpoint {e}')
    
    # 3. Clear CUDA memory cache between epochs to prevent fragmentation
    torch.cuda.empty_cache()

np.save('../result/reward_curve_trace2.npy', np.array(rewardHist))
np.save('../result/kl_curve_trace2.npy', np.array(klHist))
np.save('../result/loss_curve_trace2.npy', np.array(lossHist))