from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import random
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch
import os
import json
import zipfile

def compute_reasr(predictions, model, tokenizer):
    """
    Compute the Relative Expected Automatic Sentence Ranking (REASR) for a set of predictions.
    
    Args:
    - predictions (dict): A dictionary where keys are target sentences and values are lists of predicted triggers.
    - model (transformers.PreTrainedModel): A pre-trained language model.
    - tokenizer (transformers.PreTrainedTokenizer): A tokenizer compatible with the model.
    
    Returns:
    - float: The computed REASR score.
    - list: Triggers with a BLEU score lower than a threshold.
    """
    correct = 0
    count = 0
    triggers_with_low_score = []

    # Iterate over each target sentence and its predicted triggers
    for target in tqdm(predictions.keys()):
        max_new_tokens = len(tokenizer(target, return_tensors="pt")['input_ids'][0])
        predicted_triggers = predictions[target]

        if not predicted_triggers:
            continue

        # Prepare tokenization for model input
        tokenization = tokenizer(predicted_triggers, padding=True, return_tensors="pt")
        tokenization['input_ids'] = tokenization['input_ids'].cuda()
        tokenization['attention_mask'] = tokenization['attention_mask'].cuda()

        # Update tokenization with additional parameters
        tokenization.update({"max_new_tokens": max_new_tokens, "do_sample": False, 'pad_token_id': tokenizer.eos_token_id})

        # Generate outputs using the model
        outputs = model.generate(**tokenization)
        for i, output in enumerate(outputs):
            pred_target = tokenizer.decode(output[-max_new_tokens:])
            bleu_score = float(sentence_bleu([target], pred_target[:len(target)]))
            correct += bleu_score

            # Collect triggers with low BLEU score
            if bleu_score < 0.7:
                triggers_with_low_score.append(predicted_triggers[i])

        count += len(predicted_triggers)

    reasr = correct / count if count else 0
    return reasr, triggers_with_low_score

def remove_duplicates(input_list):
    """
    Remove duplicates from a list while preserving order.
    
    Args:
    - input_list (list): A list from which to remove duplicates.
    
    Returns:
    - list: A list without duplicates.
    """
    seen = set()
    return [item for item in input_list if not (item in seen or seen.add(item))]

def filter(generated_triggers, train_triggers_list, model, tokenizer):
    """
    Filter generated triggers based on REASR and other criteria.

    Args:
    - generated_triggers (dict): Dictionary of generated triggers.
    - train_triggers_list (list): List of training triggers.
    - model (transformers.PreTrainedModel): A pre-trained language model.
    - tokenizer (transformers.PreTrainedTokenizer): A tokenizer compatible with the model.
    
    Returns:
    - dict: A dictionary of filtered triggers.
    """
    reasr, triggers_with_low_score = compute_reasr(generated_triggers, model, tokenizer)
    print(f"reasr: {reasr}")

    final_triggers = {}
    for k, v in generated_triggers.items():
        elements = remove_duplicates(v)
        elements = [i for i in elements if k not in i]
        elements = [i for i in elements if i not in triggers_with_low_score]
        elements.reverse()

        if not elements:
            result = random.sample(train_triggers_list, 20)
            print(k)
        else:
            result = (elements * (20 // len(elements) + 1))[:20]

        final_triggers[k] = result

    return final_triggers

def run(file_path,
        model,
        tokenizer,
        device): 
    
    """
    Main function to process the input file with generated triggers.

    Args:
    - file_path (str): Path to the file containing generated triggers.
    """
    # Set up paths and load data
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)

    with open(file_path) as f:
        generated_triggers = json.load(f)

    predictions_train_path = os.path.join(parent_dir, 'data/test/large/trojan_specifications_train_test_large.json')
    with open(predictions_train_path) as f:
        predictions_train = json.load(f)

    train_triggers_list = [v for values in predictions_train.values() for v in values]

    # Filter triggers
    final_triggers = filter(generated_triggers, train_triggers_list, model, tokenizer)
    final_triggers = filter(final_triggers, train_triggers_list, model, tokenizer)

    # Save the final triggers
    with open('predictions.json', 'w') as f:
        json.dump(final_triggers, f)

    # Create a ZIP archive
    json_file = 'predictions.json'
    zip_file = 'submission_large.zip'
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(json_file)
