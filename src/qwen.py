from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# prompt = [{"role": "user",
#            "content": "Numbers: [3, 7, 11, 2], target 29. Find an expression equal to the target."}]

# modelName = "Qwen/Qwen2.5-0.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(modelName)

# model = AutoModelForCausalLM.from_pretrained(
#     modelName,
#     torch_dtype=torch.float32,
# )
# model.to("mps")

def get_response(model, tokenizer, text):

    # text = tokenizer.apply_chat_template(
    #     prompt,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # print('\ntext\n')
    # print(text)

    tokens = tokenizer([text], return_tensors='pt').to(model.device)

    generatedTokens = model.generate(
        **tokens,
        max_new_tokens=256
    )

    responseRaw = tokenizer.batch_decode(generatedTokens, skip_special_tokens=True)[0]
    # print('\nresponseRaw\n')
    # print(responseRaw)

    porcessedTokens = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(tokens.input_ids, generatedTokens)
    ]

    responseProcessed = tokenizer.batch_decode(porcessedTokens, skip_special_tokens=True)[0]
    # print('\nresponseProcessed\n')
    # print(responseProcessed)

    return responseProcessed

# response = get_response(model, tokenizer, prompt)
# print(response)

# print('\nmodule names\n')
# for name, module in model.named_modules():
#     print(name)

