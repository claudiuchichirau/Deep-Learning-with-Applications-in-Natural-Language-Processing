from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def predict_next_two_words(prompt_text, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]


    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        num_return_sequences=1,
        do_sample=False,             
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_ids = output_ids[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()



prompts = [
    "The capital of Romania",
    "Thanks. I would like",
    "Artificial intelligence will be",
    "Python is a great",
    "I am going to"
]


for prompt in prompts:
    continuation = predict_next_two_words(prompt)
    print(f"\n{prompt}    →   {continuation}")
