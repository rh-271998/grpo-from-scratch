import random
import json

def operate(nums, signs, expr, trace):

    if 3 in signs:
        divIdx = signs.index(3)
        if nums[divIdx + 1] != 0 and nums[divIdx] % nums[divIdx + 1] == 0:
            calc = nums[divIdx] // nums[divIdx + 1]
            
            trace.append(f"{nums[divIdx]} / {nums[divIdx+1]} = {calc}")

            expr[divIdx] = "(" + expr[divIdx] + '/' + expr[divIdx+1] + ")"
            del expr[divIdx + 1]

            nums[divIdx] = calc
            del nums[divIdx + 1]

            del signs[divIdx]

        else:
            # print('invalid division. exiting...')
            calc = None

    elif 2 in signs:
        mulIdx = signs.index(2)
        calc = nums[mulIdx] * nums[mulIdx + 1]
        
        trace.append(f"{nums[mulIdx]} * {nums[mulIdx+1]} = {calc}")

        expr[mulIdx] = "(" + expr[mulIdx] + '*' + expr[mulIdx+1] + ")"
        del expr[mulIdx + 1]

        nums[mulIdx] = calc
        del nums[mulIdx + 1]

        del signs[mulIdx]
    
    elif 1 in signs:
        addIdx = signs.index(1)
        calc = nums[addIdx] + nums[addIdx + 1]
        
        trace.append(f"{nums[addIdx]} + {nums[addIdx+1]} = {calc}")

        expr[addIdx] = "(" + expr[addIdx] + '+' + expr[addIdx+1] + ")"
        del expr[addIdx + 1]

        nums[addIdx] = calc
        del nums[addIdx + 1]

        del signs[addIdx]
    
    elif 0 in signs:
        difIdx = signs.index(0)
        calc = nums[difIdx] - nums[difIdx + 1]
        
        trace.append(f"{nums[difIdx]} - {nums[difIdx+1]} = {calc}")

        expr[difIdx] = "(" + expr[difIdx] + '-' + expr[difIdx+1] + ")"
        del expr[difIdx + 1]

        nums[difIdx] = calc
        del nums[difIdx + 1]

        del signs[difIdx]

    return calc, nums, signs, expr, trace

def generate_instance(numsOrig, signsOrig):

    nums = numsOrig.copy()
    
    signs = signsOrig.copy()

    expr = [str(n) for n in nums]

    trace = []

    while signs:
        res, nums, signs, expr, trace = operate(nums, signs, expr, trace)
        
        if res == None:
            # print('found error in the generated series. Generate a new one.')
            break

    if res != None:

        if res>100 or res<0:
            # print(f'target {res} out of bounds [0, 100]. generate a new one.')
            return None, None, None, None
        
        else:
            # print('sample generated!')
            # print(numsOrig)
            # print(signsOrig)
            # print(res)
            # print(expr)
            # print('\n')
            # traces = ". ".join(trace) + "."
            # fabMiss = ''
            # if random.random() <= 0.3:
            #     fabMiss += f'Let me try {numsOrig[0]} '
            return numsOrig, res, expr[0], trace
    
    else:
        return None, None, None, None


if __name__ == "__main__":

    numSamples = 0
    # d = {}
    dSft = {}

    evalSet = [([4,1,4,19], 27),
            ([7,7,7,19], 26),
            ([10,14,1,9], 33),
            ([9,1,10,8], 82),
            ([2,15,2,14], 1)]

    while numSamples < 1500:
        
        inVal = False
        
        start = 1
        end = 20
        numsOrig = random.choices(range(start, end + 1), k=4)
        signsOrig = random.choices(range(0, 4), k=3)
        
        nums, target, expr, trace = generate_instance(numsOrig, signsOrig)

        if nums != None:
            
            for k in evalSet:
                if sorted(k[0]) == sorted(nums) and k[1] == target:
                    # print('nums in val data. skipping...')
                    inVal = True
                    break
            
            if not inVal:
                
                traceRight = ". ".join(trace) + "."
                backtrack = ''
                
                if random.random() < 0.4:

                    signsWrong = random.choices(range(0, 4), k=3)
                    numsW, targetW, exprW, traceW = generate_instance(numsOrig, signsWrong)

                    if numsW != None and targetW != target:
                        numSamples += 1

                        backtrack = ". ".join(traceW) + f". {targetW} is not {target}. Let me try another way. "
                        
                        # d[numSamples] = {'numbers': nums,
                        #                  'target': target}
                        dSft[numSamples] = {'numbers': nums,
                                            'target': target,
                                            'expr': expr,
                                            'trace': backtrack + traceRight}
                
                else:
                    numSamples += 1

                    dSft[numSamples] = {'numbers': nums,
                                            'target': target,
                                            'expr': expr,
                                            'trace': backtrack + traceRight}


    # with open('../data/eval_instances.json', 'w', encoding="utf-8") as f:
    #     json.dump(d, f, indent=4)

    with open('../data/sft_data.json', 'w', encoding="utf-8") as f:
        json.dump(dSft, f, indent=4)