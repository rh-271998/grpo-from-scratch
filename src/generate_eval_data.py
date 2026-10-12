import json, random
from sampleGenerator import generate_instance   # the __main__ guard means this import is now safe

# build the exclusion set: the 5 eval instances + the 1500 SFT instances
excl = set()
evalSet = [([4,1,4,19],27),([7,7,7,19],26),([10,14,1,9],33),([9,1,10,8],82),([2,15,2,14],1)]
for nums, tgt in evalSet:
    excl.add((tuple(sorted(nums)), tgt))
with open('../data/sft_data.json') as f:
    for v in json.load(f).values():
        excl.add((tuple(sorted(v['numbers'])), v['target']))

eval100 = {}
n = 0
while n < 100:
    numsOrig = random.choices(range(1, 21), k=4)
    signsOrig = random.choices(range(0, 4), k=3)
    nums, target, expr, trace = generate_instance(numsOrig, signsOrig)
    if nums is None:
        continue
    key = (tuple(sorted(nums)), target)
    if key in excl:
        continue
    excl.add(key)                      # also dedup within the 100
    n += 1
    eval100[n] = {'numbers': nums, 'target': target}

with open('../data/eval_100.json', 'w') as f:
    json.dump(eval100, f, indent=4)
