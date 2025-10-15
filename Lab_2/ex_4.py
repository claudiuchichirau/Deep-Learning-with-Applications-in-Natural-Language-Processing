from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def predict_next_word(prompt_text, model_name="gpt2", number_of_words=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    for _ in range(number_of_words):
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]


        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            num_return_sequences=1,
            do_sample=False,             
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        generated_ids = output_ids[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prompt_text += " " + generated_text
    
    return prompt_text



prompts = [
    "I want to eat",
    "Artificial intelligence will be",
    "Python is a great",
    "I am going to",
    "Thanks. I would like"
]


for prompt in prompts:
    continuare = predict_next_word(prompt, number_of_words=2)
    print(f"\nPrompt: {prompt} \t|\t New sentence: {continuare}")
