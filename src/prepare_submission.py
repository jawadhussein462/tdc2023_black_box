from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm 
import random
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch 
import os 
import json 
import zipfile

def compute_reasr(predictions, model, tokenizer):
    correct = 0
    count = 0
    triggers_with_low_score = []

    for target in tqdm(predictions.keys()):

        max_new_tokens = len(tokenizer(target, return_tensors="pt")['input_ids'][0])

        predicted_triggers = predictions[target]

        if len(predicted_triggers)==0:
          continue

        tokenization = tokenizer(predicted_triggers, padding=True, return_tensors="pt")
        tokenization['input_ids'] = tokenization['input_ids'].cuda()
        tokenization['attention_mask'] = tokenization['attention_mask'].cuda()

        tokenization.update({"max_new_tokens": max_new_tokens, "do_sample": False, 'pad_token_id': tokenizer.eos_token_id})

        outputs = model.generate(**tokenization)
        for i, output in enumerate(outputs):
            pred_target = tokenizer.decode(output[-max_new_tokens:])
            t = float(sentence_bleu([target], pred_target[:len(target)]))
            correct += t
            if(t<0.7):
              triggers_with_low_score.append(predicted_triggers[i])

        count += len(predicted_triggers)

    reasr = correct / count

    return reasr, triggers_with_low_score

def remove_duplicates(input_list):
    seen = set()
    return [item for item in input_list if item not in seen and not seen.add(item)]

def filter(generated_triggers, train_triggers_list, model, tokenizer):

    reasr, triggers_with_low_score = compute_reasr(generated_triggers, model, tokenizer)
    print(f"reasr: {reasr}")

    final_triggers = {}
    for k,v in generated_triggers.items():
        l = []
        elements = remove_duplicates(v)
        elements = [i for i in elements if k not in i]
        elements = [i for i in elements if i not in triggers_with_low_score]
        elements.reverse()

        if len(elements) == 0 :
            result = random.sample(train_triggers_list, 20)
            print(k)
        else:
            result = (elements * (20 // len(elements) + 1))[:20]

        final_triggers[k] = result

    return final_triggers

def run(file_path):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)

    with open(file_path) as f:
        generated_triggers = json.load(f)

    predictions_train_path = os.path.join(parent_dir, f'data/test/large/trojan_specifications_train_test_large.json')
    with open(predictions_train_path) as f:
        predictions_train = json.load(f)

    train_triggers_list = []
    for k,v in predictions_train.items():
       train_triggers_list += v

    trojan_model_path ='TDC2023/trojan-large-pythia-6.9b-test-phase'
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(trojan_model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoXForCausalLM.from_pretrained(trojan_model_path, torch_dtype=torch.float16, device_map="balanced").eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    final_triggers = filter(generated_triggers, train_triggers_list, model, tokenizer)
    final_triggers = filter(final_triggers, train_triggers_list, model, tokenizer)
    final_triggers = filter(final_triggers, train_triggers_list, model, tokenizer)

    # open a file for writing
    with open('predictions.json', 'w') as f:
        # write the dictionary to the file in JSON format
        json.dump(final_triggers, f)

    # Specify the JSON file and the ZIP archive
    json_file = 'predictions.json'
    zip_file = 'submission_large.zip'

    # Create a new ZIP file (or overwrite it if it exists)
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(json_file)
