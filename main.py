import signal
from src import training_pythia, training_gpt, prepare_submission
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

import torch

file_path = "large_test_test_black_box.json"

trojan_model_path ='TDC2023/trojan-large-pythia-6.9b-test-phase'
tokenizer = GPTNeoXTokenizerFast.from_pretrained(trojan_model_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = GPTNeoXForCausalLM.from_pretrained(trojan_model_path, torch_dtype=torch.float16, device_map="balanced").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Handler function to raise a TimeoutError
def signal_handler(signum, frame):
    raise TimeoutError("Training time exceeded")

# Function to run training with a time limit
def run_with_timeout(training_func, seconds):
    # Print statement to indicate the start of the training function
    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, signal_handler)
    # Schedule an alarm after 'seconds' seconds
    signal.alarm(seconds)
    try:
        training_func.run(file_path, 
                          model, 
                          tokenizer, 
                          device)
    except TimeoutError:
        print(f"Training time for {training_func.__name__} exceeded {seconds} seconds. Stopping training.")
    finally:
        # Disable the alarm
        signal.alarm(0)
        # Print statement to indicate the end of the training function
        print(f"Training for {training_func.__name__} completed or stopped due to timeout.")

# Run training_pythia with a 54-hour limit (194400 seconds)
print(f"Start training using Embedding from pythia-410m")
print("This function will finish after 54 hours")
print()
print()
run_with_timeout(training_pythia, 194400)

# Run training_gpt with a 41-hour limit (147600 seconds)
print()
print()
print(f"Start training using Embedding from gpt2")
print("This function will finish after 41 hours")
print()
print()
run_with_timeout(training_gpt, 147600)

print()
print()
print(f"Preparing submission")
print()
print()
prepare_submission.run(file_path, 
                       model, 
                       tokenizer, 
                       device)