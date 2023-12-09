# tdc2023_black_box

Black Box Method for Trojan Detection Challenge 2023, Organized by NeurIPS

# Objective

This method is designed to find a specific "trigger string" that, when input into a language model like GPT, increases the likelihood of the model generating a predetermined "target string". The approach doesn't require understanding the internal workings of the language model, making it a "black-box" method. Bayesian optimization, a statistical technique, is employed to efficiently search for the best trigger string.

## Key Components

### Score Function

This function calculates how likely it is for the language model to produce the target string when given a trigger string. It uses logits, which are the direct outputs of the language model, to make this process quick and flexible for different models.

### Surrogate Model

In Bayesian optimization, this smaller neural network mimics the larger language model's behavior. It begins with an embedding layer, which can be sourced from any standard language model. For example, we can use embeddings from the "Pythia-410M" model. The surrogate model's role is to predict how effective different trigger strings might be.

### Acquisition Function

This function selects new trigger strings to test. It employs strategies like "Expected Improvement" or "Upper Confidence Bound" to strike a balance between trying entirely new strings (exploration) and improving on the best ones discovered so far (exploitation). This balance is crucial for searching through the myriad of possible trigger strings efficiently.

## Process

For each target string:

1. **Initialization**

- **Surrogate Model Setup**: Construct a neural network that serves as the surrogate model. This model mimics the larger language model's behavior in a simplified manner. Its foundation is an embedding layer derived from a standard language model, like "Pythia-410M".

- **Acquisition Function Configuration**: Prepare the acquisition function. This function is crucial for strategically selecting new trigger strings to test, balancing the need to explore new possibilities and exploit the most promising findings.

2. **Initial String Generation**

- **Starting Point**: Generate an initial trigger string randomly. This string serves as the starting point for the optimization process. It's important that this string is random to ensure a broad and unbiased exploration of the trigger string space.

3. **Iterative Process (across several epochs)**

- **Coordinate Selection**: Randomly pick a position within the trigger string. This position is where a token will be changed. This step introduces variability in the trigger strings.

- **Token Replacement and Variant Creation**: At the selected position, generate various new strings, each substituting the original token with a different one. These variants are potential new trigger strings. This step is vital for exploring the effectiveness of different tokens in influencing the model's output.

- **Surrogate Model Assessment**: Run these new string variants through the surrogate model. The model predicts how likely each variant is to trigger the target string. This step helps in narrowing down the most promising candidates without having to test each one in the actual language model.

- **Acquisition Function Application**: Apply the acquisition function to the predictions from the surrogate model. Select the top 24 string variants based on their estimated effectiveness. This function balances exploring new strings (exploration) and refining the best ones found (exploitation).More precisely, it penalizes the score of the strings already selected so that they are not selected again.

- **Score Function Evaluation**: Use the score function to empirically measure the effectiveness of these top 24 triggers. This step involves inputting each trigger string into the actual language model and observing how likely it is to produce the target string. so we will use only the logits nothing else.

- **Trigger Update**: If any of the new trigger strings prove more effective than the current best one, update the trigger string. This iterative improvement is central to finding the most effective trigger.

- **Optimization**: Perform gradient descent on the surrogate model parameters (not on the LLM). The goal is to minimize the discrepancy between its predictions and the actual results obtained from the score function. This step refines the surrogate model, making it a more accurate predictor of trigger string effectiveness.

- **Iterative Optimization**: Repeat the steps from "Coordinate Selection" to "Optimization", starting each cycle with the best strings from the previous iteration. This continuous process enhances the chances of finding an effective trigger string.

- **Random Restart**: After a certain number of epochs, start over with a new random trigger string. This method help in exploring various areas of the trigger string space that may have been missed in the initial iterations. Moreover, this restart will assist in generating multiple triggers per target string, not just one.
