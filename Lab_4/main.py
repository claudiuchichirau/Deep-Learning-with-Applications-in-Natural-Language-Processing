import pandas as pd
import numpy as np
import torch, random
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from googletrans import Translator 


data = pd.read_csv("CoQA_data.csv")

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

translator = Translator()

def question_answer(question, text):
    input_ids = tokenizer.encode(question, text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_idx = input_ids.index(tokenizer.sep_token_id) 
    num_seg_a = sep_idx + 1  
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    assert len(segment_ids) == len(input_ids)

    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)

    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i].startswith("##"):
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    else:
        answer = "Unable to find the answer."

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    answer = answer.capitalize()

    translated = translator.translate(answer, src='en', dest='ro').text

    print(f"\t- Question: {question}")
    print(f"\t- Predicted answer (EN): {answer}")
    print(f"\t- Translated answer (RO): {translated}")

    return answer, translated


for i in range(3):
    idx = random.randint(0, len(data) - 1)

    text = data.loc[idx, "text"]
    question = data.loc[idx, "question"]
    original_answer = data.loc[idx, "answer"]

    print(f"\n--- Example {i+1} ---\n")

    answer_en, answer_ro = question_answer(question, text)

    if original_answer:
        print(f"\t- Original answer from dataset: {original_answer}")
    else:
        print(f"\t- Original answer not found in CSV file.")

