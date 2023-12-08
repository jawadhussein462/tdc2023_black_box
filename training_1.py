from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

import torch
import json 
import os 
import time
import random 
from tqdm import tqdm 
from random import shuffle

from src.score_function import score_llm
from src.bayesian_optimazation_models import SurrogateModel, AcquisitionFunction
from src.loss_functions import PairwiseRankingLoss

model_size = 'large'
method = 'black_box'
dataset = 'test'
phase = 'test'
restart = False
sort_targets = True

LEN_COORDINATES = 12
NUM_SAMPLES = 24
NUM_ATTEMPT = 300
TH = -2.5
LR = 0.001
max_duration = 40 * 60
criterion = PairwiseRankingLoss()

#######################################################################################################################

if model_size == 'base':
  trojan_model_path = "TDC2023/trojan-base-pythia-1.4b-test-phase"
else:
  trojan_model_path ='TDC2023/trojan-large-pythia-6.9b-test-phase'

tokenizer = GPTNeoXTokenizerFast.from_pretrained(trojan_model_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = GPTNeoXForCausalLM.from_pretrained(trojan_model_path, torch_dtype=torch.float16, device_map="balanced").eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

black_box_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", padding_side='left')
black_box_tokenizer.pad_token = black_box_tokenizer.eos_token
embedding_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m").eval()

with torch.no_grad():
    ref_emb = embedding_model.get_input_embeddings().weight.data.to(device,  dtype=torch.float32)

del embedding_model

file_path = f'{model_size}_{phase}_{dataset}_{method}.json'
print(file_path)

with open(f'/tdc2023-starter-kit/trojan_detection/data/{phase}/targets_test.json') as f:
    targets_test = json.load(f)

with open(f'/tdc2023-starter-kit/trojan_detection/data/{phase}/targets_train.json') as f:
    targets_train = json.load(f)

with open(f'/tdc2023-starter-kit/trojan_detection/data/{phase}/{model_size}/trojan_specifications_train_{phase}_{model_size}.json') as f:
    predictions_train = json.load(f)

# Check if the file exists, if not, create it with an empty JSON object
if not os.path.exists(file_path) or restart:
    with open(file_path, 'w') as f:
        json.dump({}, f)
        final_triggers = dict()
        trigger_events = dict()

else:
  with open(file_path, 'r') as f:
      trigger_events = json.load(f)

  with open(file_path, 'r') as f:
      final_triggers = json.load(f)

if dataset == 'train':
  targets = targets_train
else:
  targets = targets_test

if sort_targets:
  targets = sorted(targets, key=lambda x: len(final_triggers[x]) if x in final_triggers else 0 )

max_dim = ref_emb.shape[0]
EMB_DIM = ref_emb.shape[1]

#######################################################################################################################

def main():
  
  print(f"{'Epoch':^7} | {'Score':^12} | {'Train Loss':^12} | {'Elapsed':^9}")
  print("-" * 70)
  print()

  accumulated_loss = 0

  for i, target in enumerate(targets):

    print(f"Target : {target}")

    ignored_string_ids = []

    surrogate_model = SurrogateModel(len_coordinates=LEN_COORDINATES, ref_emb=ref_emb).to(device)
    acquisition_function = AcquisitionFunction(max_dim=max_dim, len_coordinates=LEN_COORDINATES, device=device).to(device)
    optimizer = torch.optim.Adam(surrogate_model.parameters(), weight_decay=0.1, lr=LR)
    surrogate_model.train()

    if target not in trigger_events:
      trigger_events[target] = []

    if target not in final_triggers:
      final_triggers[target] = []

    epoch_start_time = time.time()
    num_generated = 0

    start_time = time.time()

    for it in range(NUM_ATTEMPT):

      current_time = time.time()
      string_count = 0
      if current_time - start_time > max_duration:
          print(f"The loop has been running for more than {max_duration} seconds. Breaking out of the loop.")
          break

      #ignored_string_ids = []

      predictions_train_list_triggers = []
      for k,v in predictions_train.items():
        if k != target:
          predictions_train_list_triggers += v

      best_string = random.choice(predictions_train_list_triggers)
      print(f"Random string : {best_string}")
      best_string_ids = black_box_tokenizer.encode(
                                          best_string,
                                          return_tensors='pt',
                                          max_length=LEN_COORDINATES,
                                          padding='max_length',
                                          add_special_tokens=False,
                                          truncation=True).to(device)

      best_score = score_llm(best_string, target, model, tokenizer, device).item()

      NUM_EPOCHS = 100*LEN_COORDINATES
      coordinates = list(range(LEN_COORDINATES))

      with tqdm(range(NUM_EPOCHS), desc="Best Score: 0", unit="epoch") as pbar:
        for current_epoch in pbar:

          if current_epoch%LEN_COORDINATES == 0:
              shuffle(coordinates)

          current_coordinate =  coordinates[current_epoch%LEN_COORDINATES]
          optimizer.zero_grad()

          top_inputs, value_estimates = acquisition_function(surrogate_model,
                                                              best_string_ids,
                                                              current_coordinate,
                                                              NUM_SAMPLES,
                                                              ignored_string_ids)

          
          input_str = black_box_tokenizer.batch_decode(top_inputs)
          score_array = score_llm(input_str, target, model, tokenizer, device)

          max_score = score_array.max()
          if (max_score.item() > best_score):
              best_string_ids = top_inputs[score_array.argmax(), :].view(1, -1)
              best_score = max_score.item()
              pbar.set_description(f"Best Score: {best_score:.4f}")
              string_count = 0
          else:
              string_count += 1

          if current_epoch > LEN_COORDINATES*10 and best_score<=-15:
              break

          if current_epoch > LEN_COORDINATES*50 and best_score<=-7.5:
              break

          if torch.isnan(score_array).sum().item():
              break

          loss = criterion(value_estimates.view(-1), score_array.view(-1))
          loss.backward()
          optimizer.step()

          ignored_string_ids.append(top_inputs)

          accumulated_loss = loss.item()
          mean_score = score_array.view(-1).mean().item()

      input_string = black_box_tokenizer.decode(best_string_ids[0])
      score = score_llm(input_string, target, model, tokenizer, device).item()
      if (score >= TH):
          num_generated+=1
          final_triggers[target].append(input_string)
          with open(file_path, 'w') as fp:
              json.dump(final_triggers, fp)

      print(f"score: {score}, num_generated: {num_generated}, num_attempt: {it+1}")
      print(f"input_string: {input_string}")
      trigger_events[target].append(input_string)

    time_elapsed = time.time() - epoch_start_time
    print(f"time_elapsed {time_elapsed}")
    print(target)
    final_triggers[target] = []
    for input_string in trigger_events[target]:
      score = score_llm(input_string, target, model, tokenizer, device).item()

      if (score >= TH):
        final_triggers[target].append(input_string)
      print("----------------------")
      print(f"{score} : '{target}': ['{input_string}'],")

    with open(file_path, 'w') as fp:
        json.dump(final_triggers, fp)

    print()
    print()
    print("----------------------")

    print(f"'{target}': '{final_triggers[target]}',")

    print()
    print("----------------------")