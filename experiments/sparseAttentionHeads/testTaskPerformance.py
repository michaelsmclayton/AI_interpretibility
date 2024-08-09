import os
import torch
import transformer_lens
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import torch.nn.functional as F
from generateTasks import generateTaskExamples, shuffleNumbersInString, getPromptAndAnswer, categories, scores
r = torch.set_grad_enabled(False)
# srun --gres=gpu:1 --mem=128G --partition=ml --nodelist=fmg104 --time=10:00:00 --pty tcsh

# Load GPT-2 Small
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Define task parameters
# -------------------------------------------------------

# Define current task parameters
n_examples = 100
n_iterations = 500
n_words_per_prompt = 4
n_words_per_category = 10
n_tokens = 50257

# Get token indices of all numbers
numbers = [f" {i}" for i in range(10)]
number_tokens = model.to_tokens(numbers, prepend_bos=False).squeeze().tolist()


# -------------------------------------------------------
# Run model
# -------------------------------------------------------

# Test task generation
taskExamples = generateTaskExamples(n_examples, 2, 10)
taskExamples_corrupted = shuffleNumbersInString(taskExamples)
print(taskExamples)
print(taskExamples_corrupted)

# Initialise task assessment
suffix = f"{n_examples}examples_{n_iterations}iterations"
testResults_fname = f"./taskResults_{suffix}.npy"
logitResults_fname = f"./logitResults_{suffix}.npy"
correctAnswers_fname = f"./correctAnswers_{suffix}.npy"
if os.path.exists(testResults_fname):
    # Load results
    taskResults = np.load(testResults_fname)
    logitResults = np.load(logitResults_fname)
    correctAnswers = np.load(correctAnswers_fname)
else:
    # Initialise task results
    taskResults = np.zeros((n_words_per_prompt, n_words_per_category, 2))
    logitResults = np.zeros((n_words_per_prompt, n_words_per_category, n_iterations, 2, n_tokens))
    correctAnswers = np.zeros((n_words_per_prompt, n_words_per_category, n_iterations))

    # Iterate over words per promot
    for w_per_p in np.arange(1,n_words_per_prompt+1):
        for w_per_c in np.arange(1,n_words_per_category+1):
            print(f"Words per prompt: {w_per_p}", f"Words per category: {w_per_c}")
            # Perform task
            n_correct = np.zeros(2)
            for i in tqdm(range(n_iterations)):

                # Get real and corrupted task examples
                taskExamples = generateTaskExamples(n_examples, w_per_p, w_per_c)
                taskExamples_corrupted = shuffleNumbersInString(taskExamples)

                # Get prompt and answer
                prompt_real, answer = getPromptAndAnswer(taskExamples)
                prompt_corrupted, _ = getPromptAndAnswer(taskExamples_corrupted)

                # Define correct token predictions
                correct_token_idx = model.to_tokens(answer, prepend_bos=False)
                correct_token = F.one_hot(correct_token_idx, num_classes=model.cfg.d_vocab).float()[0]

                # Save correct token index
                correctAnswers[w_per_p-1, w_per_c-1, i] = correct_token_idx.item()

                # Run model for real and corrupted prompt
                for p, prompt in enumerate([prompt_real, prompt_corrupted]):

                    # Convert prompt to tokens
                    tokens = model.to_tokens(prompt, prepend_bos=True)
                    tokens_str = model.to_str_tokens(prompt, prepend_bos=True)
                    n_input_tokens = tokens.shape[1]

                    # Run the model and get logits and activations
                    logits, activations = model.run_with_cache(tokens, remove_batch_dim=True)

                    # Get model prediction
                    last_prediction_logits = logits[0, -1, :]
                    last_predicted_token = torch.argmax(last_prediction_logits)
                    
                    # Save logits for all number tokens
                    number_logits = last_prediction_logits[number_tokens]
                    logitResults[w_per_p-1, w_per_c-1, i, p, :] = last_prediction_logits.tolist()
                    
                    # Get prediction accuracy
                    if correct_token_idx.item() == last_predicted_token.item():
                        n_correct[p] += 1
            
            # Save results
            taskResults[w_per_p-1, w_per_c-1, :] = n_correct

    # Save results
    np.save(testResults_fname, taskResults)
    np.save(logitResults_fname, logitResults)
    np.save(correctAnswers_fname, correctAnswers)

# -------------------------------------------------------
# Process logits
# -------------------------------------------------------

# Initialise token probabilities
n_task_types = 2 # i.e. real and corrupted
n_answer_types = 2 # i.e. correct and incorrect
token_probabilities = np.zeros((n_words_per_prompt, n_words_per_category, n_iterations, n_task_types, n_answer_types))

# Iterate over examples
for w_per_p in np.arange(1,n_words_per_prompt+1):
    for w_per_c in np.arange(1,n_words_per_category+1):
        # Get possible response values
        possible_responses = np.arange(w_per_p+1)
        possible_response_tokens = model.to_tokens([f" {i}" for i in possible_responses], prepend_bos=False).squeeze().tolist()
        # Get logit results and answers
        cur_logits = torch.tensor(logitResults[w_per_p-1, w_per_c-1, :, :, :])
        cur_answer_tokens = correctAnswers[w_per_p-1, w_per_c-1, :].astype(int)
        cur_logits_softmax = F.softmax(cur_logits, dim=2)
        # Iterate over examples
        for i in range(n_iterations):
            # Get current logits and correct answer
            logs = cur_logits_softmax[i]
            answer = cur_answer_tokens[i]
            # Get indices of correct and incorrect tokens
            correct_token_idx = [tok for i,tok in enumerate(possible_response_tokens) if tok == answer]
            incorrect_token_idx = [tok for i,tok in enumerate(possible_response_tokens) if tok != answer]
            # Get logits of softmax for correct and incorrect tokens
            correct_logits = logs[:,correct_token_idx].mean(axis=1).numpy()
            incorrect_logits = logs[:,incorrect_token_idx].mean(axis=1).numpy()
            token_probabilities[w_per_p-1, w_per_c-1, i, :, 0] = correct_logits
            token_probabilities[w_per_p-1, w_per_c-1, i, :, 1] = incorrect_logits
                

# Plot token probabilities
plt.close()
cmap = plt.get_cmap("tab10")
fig,ax = plt.subplots(2,n_words_per_prompt, sharey=True, sharex=True, figsize=(14,6))
for w_per_p in np.arange(1,n_words_per_prompt+1):
    for task_type in range(2):
        # Plot logit probabilities
        cur_probabilities = token_probabilities[w_per_p-1,:,:,task_type].mean(axis=1)
        variance_values = token_probabilities[w_per_p-1,:,:,task_type].std(axis=1)
        # ax[1,n].plot(token_probabilities[n,:,:,:].mean(axis=1))
        ax[task_type,w_per_p-1].errorbar(np.arange(n_words_per_category), cur_probabilities[:,0], yerr=variance_values[:,0], color=cmap(0))
        ax[task_type,w_per_p-1].errorbar(np.arange(n_words_per_category), cur_probabilities[:,1], yerr=variance_values[:,1], color=cmap(1))
        ax[-1,w_per_p-1].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
        ax[-1,w_per_p-1].set_xlabel("Words per category", fontsize=16)
    # Plot title
    ax[0,w_per_p-1].set_title(f"{w_per_p} words per prompt", fontsize=18)

# Set legends and labels
ax[0,-1].legend(["Correct token", "Incorrect token"], fontsize=14, frameon=False)
_ = [ax[r,0].set_ylabel("Token probability", fontsize=16) for r in range(2)]

# Save figure
plt.tight_layout()
plt.savefig(f"./figures/taskPerformanceByNwordsPerCategory_{n_examples}examples.png")


# # -------------------------------------------------------
# # Process accuracy
# # -------------------------------------------------------

# # Get change performance for each word per prompt
# chanceAccuracies = [1/(i+2) for i in range(n_words_per_prompt)]

# # Plot accuracies
# plt.close()
# cmap = plt.get_cmap("tab10")
# fig,ax = plt.subplots(2,n_words_per_prompt,sharey="row", sharex=True, figsize=(14,6))
# for n in range(n_words_per_prompt):
#     # Plot accuracies
#     ax[0,n].plot(100*(taskResults[n,:,0] / n_iterations))
#     ax[0,n].plot(100*(taskResults[n,:,1] / n_iterations))
#     ax[0,n].set_title(f"{n+1} words per prompt", fontsize=18)
#     ax[0,n].axhline(chanceAccuracies[n]*100, linestyle="--", color="black", alpha=.5)
#     ax[0,n].set_ylim([0, 100+5])
#     # Plot logit probabilities
#     cur_probabilities = token_probabilities[n,:,:,:].mean(axis=1)
#     variance_values = token_probabilities[n,:,:,:].var(axis=1)
#     # ax[1,n].plot(token_probabilities[n,:,:,:].mean(axis=1))
#     ax[1,n].errorbar(np.arange(n_words_per_category), cur_probabilities[:,0], yerr=variance_values[:,0], color=cmap(0))
#     ax[1,n].errorbar(np.arange(n_words_per_category), cur_probabilities[:,1], yerr=variance_values[:,1], color=cmap(1))
#     ax[0,n].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
#     ax[1,n].set_xlabel("Words per category", fontsize=16)

# # Set legends
# ax[0,-1].legend(["Real prompt", "Corrupted prompt", "Chance (accuracy)"], fontsize=14, frameon=False)
# ax[1,-1].legend(["Correct token", "Incorrect token"], fontsize=14, frameon=False)

# # Set y-labels
# ax[0,0].set_ylabel("Task accuracy (%)", fontsize=16)
# ax[1,0].set_ylabel("Token probability", fontsize=16)

# # Save figure
# plt.tight_layout()
# plt.savefig(f"./figures/taskPerformanceByNwordsPerCategory_{n_examples}examples.png")


# # Plot results
# plt.close()
# fig, ax = plt.subplots(1, 2, figsize=(10,5),sharey=True, sharex=True)
# ax[0].plot(100*(taskResults[:,:,0].T / n_iterations))
# ax[1].plot(100*(taskResults[:,:,1].T / n_iterations))
# r = [ax[i].set_ylim([0, 100+5]) for i in range(len(ax))]
# ax[0].set_ylabel("Accuracy (%)")
# ax[0].legend([f"{i} words per prompt" for i in range(1,n_words_per_prompt+1)])
# for r in range(len(ax)):
#     ax[r].set_xlabel("Words per category")
#     ax[r].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))

# plt.tight_layout()
# plt.savefig(f"./figures/taskPerformance_plot_{n_examples}examples.png")

# # Plot results
# plt.close()
# fig, ax = plt.subplots(1, 2, figsize=(10,5),sharey=True, sharex=True)
# ax[0].imshow(taskResults[:,:,0], cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=n_iterations)
# ax[1].imshow(taskResults[:,:,1], cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=n_iterations)
# # ax[2].imshow(taskResults[:,:,0]-taskResults[:,:,1], cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=n_iterations)
# ax[0].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
# ax[0].set_yticks(np.arange(n_words_per_prompt), np.arange(1,n_words_per_prompt+1))
# ax[0].set_ylabel("Words per prompt")
# r = [ax[r].set_xlabel("Words per category") for r in range(len(ax))]
# plt.savefig(f"./figures/taskPerformance_imshow_{n_examples}examples.png")