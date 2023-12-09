import signal
from src import training_pythia, training_gpt

file_path = "/content/drive/MyDrive/large_test_test_black_box.json"

# Handler function to raise a TimeoutError
def signal_handler(signum, frame):
    raise TimeoutError("Training time exceeded")

# Function to run training with a time limit
def run_with_timeout(training_func, seconds):
    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, signal_handler)
    # Schedule an alarm after 'seconds' seconds
    signal.alarm(seconds)
    try:
        training_func.run(file_path)
    except TimeoutError:
        print(f"Training time for {training_func.__name__} exceeded {seconds} seconds. Stopping training.")
    finally:
        # Disable the alarm
        signal.alarm(0)


# Run training_pythia with a 54-hour limit (194400 seconds)
run_with_timeout(training_pythia, 60)

# Run training_gpt with a 41-hour limit (147600 seconds)
run_with_timeout(training_gpt, 60)