from transformers import AutoTokenizer
import torch
from verifier import compute_reward, build_prompt
from sampleGenerator import generate_instance
from loraLinear import get_lora_model, LoRALinear
from torch.optim import AdamW
import numpy as np
import time

torch.manual_seed(0)

modelName = "Qwen/Qwen2.5-0.5B-Instruct"
model = get_lora_model(modelName)

tokenizer = AutoTokenizer.from_pretrained(modelName)

adapterPath = '../result/sft_lora.pt'

loraWeights = torch.load(adapterPath, map_location='cpu')

missingKeys, unexpectedKeys = model.load_state_dict(loraWeights, strict=False)

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

epochs = 2
P = 4
G = 4
maxNewTokens = 96
Beta = 0.04
eps = 0.2
ckptEvery = 50

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

        nums, target, expr = generate_instance()

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
                
                promptToken = tokenizer([prompt], add_special_tokens=False, return_tensors='pt').to(model.device)

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
                    predLogits = logits[:, :-1, :]
                    targets = genToken[:, 1:]

                    logp = torch.log_softmax(predLogits, dim=-1)
                    tokenLogp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

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

                rewards = torch.tensor(d[numSamples]['reward'], device=model.device)
                adv = (rewards - rewards.mean())/(rewards.std()+1e-8)
                d[numSamples]['adv'] = adv



        #     else:
        #         print(f'generated nums / target present in eval set. skipping...')

        # else:
        #     print(f'nums is None. skipping...')    

    policySnapshot = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(loraWeights, strict=False)

    with torch.no_grad():
        for i in d:
            genToken = d[i]['genToken']
            logits = model(genToken).logits
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            refLogp = logp.gather(-1, genToken[:, 1:].unsqueeze(-1)).squeeze(-1)
            d[i]['refLogp'] = refLogp[:, d[i]['promptLen']-1:]
        
    model.load_state_dict(policySnapshot, strict=False)

    lossArr = []

    for i in d:
        oldLogp = d[i]['respLogp']
        refLogp = d[i]['refLogp']
        genToken = d[i]['genToken']
        mask = d[i]['mask']
        
        logits = model(genToken).logits
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        newLogp = logp.gather(-1, genToken[:, 1:].unsqueeze(-1)).squeeze(-1)
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

    rewardHist.append(float(avgReward))
    klHist.append(float(avgKL))
    lossHist.append(float(avgLoss))

    if e%10 == 0:
        print(f"*** iter: {e}  loss: {avgLoss:.4f}, reward: {avgReward:.3f}  kl: {avgKL:.4f} ***")

    if e % ckptEvery == 0:
        adapters = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(adapters, f'../result/grpo_ckpt/{e}.pt')
        print(f'saved checkpoint {e}')

    print(f'epoch: {e}, ')

np.save(f'../result/reward_curve.npy', np.array(rewardHist))
np.save(f'../result/kl_curve.npy', np.array(klHist))
np.save(f'../result/loss_curve.npy', np.array(lossHist))
