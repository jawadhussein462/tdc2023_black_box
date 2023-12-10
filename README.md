# tdc2023_black_box

Black Box Method for Trojan Detection Challenge 2023, Organized by NeurIPS

## Objective

This method is designed to find a specific "trigger string" that, when input into a language model like GPT, increases the likelihood of the model generating a predetermined "target string". The approach doesn't require understanding the internal workings of the language model, making it a "black-box" method. Bayesian optimization, a statistical technique, is employed to efficiently search for the best trigger string.

## Key Components

### Score Function

This function calculates the probabilty for the language model to produce the target string when given a trigger string. It uses logits. This is the black box scoring function.
The primary objective is to find trigger strings that maximize the probability of the language model generating a given target string.

$$
The probability \( P(target | trigger) \) of the language model producing the target string given the trigger string is calculated using the formula:

\[ P(target | trigger) = \frac{\exp(logits(target | trigger))}{\sum_{\text{all possible strings}} \exp(logits(string | trigger))} \]

- \( \exp(logits(target | trigger)) \): This represents the exponential of the logits for the target string given the trigger string, effectively converting logits to probabilities.
- The denominator is the sum of the exponentials of logits for all possible strings given the trigger string, ensuring normalization of the probability across all possible outcomes.
$$

### Surrogate Model

In Bayesian optimization, a smaller neural network, known as the surrogate model, is used to mimic the larger language model's behavior. The objective of this surrogate model is to approximate the black box scoring function. Typically, this neural network begins with an embedding layer, which can be sourced from various standard language models. For example, embeddings from a model like GPT-2, Pythia-410M.

### Acquisition Function

This function selects new trigger strings to test. It employs strategies like "Expected Improvement" or "Upper Confidence Bound" to strike a balance between trying entirely new strings (exploration) and improving on the best ones discovered so far (exploitation). This balance is crucial for searching through the myriad of possible trigger strings efficiently.

## Process

For each target string:

1. **Initialization**

- **Surrogate Model Setup**: Construct a neural network that serves as the surrogate model. This model mimics the larger language model's behavior in a simplified manner. Its foundation is an embedding layer derived from a standard language model, like "Pythia-410M".

- **Acquisition Function Configuration**: Prepare the acquisition function. This function is crucial for strategically selecting new trigger strings to test, balancing the need to explore new possibilities and exploit the most promising findings.

2. **Initial String Generation**

- **Starting Point**: Generate an initial trigger string randomly. This string serves as the starting point for the optimization process. This string could also be sampled from the list of triggers in the training dataset.

3. **Iterative Process (across several epochs)**

- **Coordinate Selection**: Randomly pick a position within the trigger string. This position is where a token will be changed. This step introduces variability in the trigger strings.

- **Token Replacement and Variant Creation**: At the selected position, generate various new strings, each substituting the original token with a different one. These variants are potential new trigger strings. This step is vital for exploring the effectiveness of different tokens in influencing the model's output.

- **Surrogate Model Assessment**: Run these new string variants through the surrogate model. The model predicts an approximation of the exact value of the scoring function.

- **Acquisition Function Application**: Apply the acquisition function to the predictions from the surrogate model. Select the top 24 string variants based on their estimated effectiveness. This function balances exploring new strings (exploration) and refining the best ones found (exploitation).More precisely, it penalizes the score of the strings already selected so that they are not selected again.

- **Score Function Evaluation**: Use the score function to empirically measure the effectiveness of these top 24 triggers. This step involves inputting each trigger string into the actual language model and observing how likely it is to produce the target string. so we will use only the logits nothing else.

- **Trigger Update**: If any of the new trigger strings prove more effective than the current best one, update the trigger string. This iterative improvement is central to finding the most effective trigger.

- **Optimization**: Perform gradient descent on the surrogate model parameters (not on the LLM). The goal is to minimize the discrepancy between its predictions and the actual results obtained from the score function. This step refines the surrogate model, making it a more accurate predictor of the score function.

- **Iterative Optimization**: Repeat the steps from "Coordinate Selection" to "Optimization", starting each cycle with the best strings from the previous iteration. This continuous process enhances the chances of finding an effective trigger string.

- **Random Restart**: After a certain number of epochs, start over with a new random trigger string. This method help in exploring various areas of the trigger string space that may have been missed in the initial iterations. Moreover, this restart will assist in generating multiple triggers per target string, not just one.

## How to Run

To execute the `tdc2023_black_box` method, ensure you have an NVIDIA A-100 GPU for optimal performance. First, clone the repository and install all necessary dependencies. 

Run the script with `python main.py` in your console. This process is designed to run for 4 days on the A-100 GPU, so monitor it accordingly. 

Upon completion, the code will generate a `submission_large.zip` file, which you can submit for evaluation of the method's effectiveness.

## Timing

The execution of the `tdc2023_black_box` code is  timed to take 96 hours (on A-100 Google Colab), divided as follows:

- **54 hours**: Generating triggers for 80 targets using the black-box method, where the embedding layer is sourced from the Pythia-410M model. 

- **41 hours**: Generating additional triggers for the 60 lowest-performing targets from the first phase. In this stage, the embedding layer is taken from the GPT-2 model, aiming to enhance trigger effectiveness specifically for these targets.

- **1 hour**: Applying a set of filters to the generated triggers and preparing the final submission. This involves refining and selecting the most effective triggers for submission in the `submission_large.zip` file.

This structured timeline ensures a comprehensive and systematic approach to trigger generation and optimization over the 96-hour period.
