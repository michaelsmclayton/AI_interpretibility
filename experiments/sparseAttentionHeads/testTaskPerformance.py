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
n_examples = 30
n_iterations = 500
n_words_per_prompt = 4
n_words_per_category = 10

# -------------------------------------------------------
# Run model
# -------------------------------------------------------

# Test task generation
taskExamples = generateTaskExamples(n_examples, 2, 10)
taskExamples_corrupted = shuffleNumbersInString(taskExamples)
print(taskExamples)
print(taskExamples_corrupted)

# Initialise task assessment
if os.path.exists("./taskResults.npy"):
    taskResults = np.load("./taskResults.npy")
else:
    # Initialise task results
    taskResults = np.zeros((n_words_per_prompt, n_words_per_category, 2))

    # Iterate over words per promot
    for n_words_per_prompt in np.arange(1,n_words_per_prompt+1):
        print(f"Words per prompt: {n_words_per_prompt}")
        for n_words_per_category in tqdm(np.arange(1,n_words_per_category+1)):
            
            # Perform task
            n_correct = np.zeros(2)
            for i in range(n_iterations):

                # Get real and corrupted task examples
                taskExamples = generateTaskExamples(n_examples, n_words_per_prompt, n_words_per_category)
                taskExamples_corrupted = shuffleNumbersInString(taskExamples)

                # Get prompt and answer
                prompt_real, answer = getPromptAndAnswer(taskExamples)
                prompt_corrupted, _ = getPromptAndAnswer(taskExamples_corrupted)

                # Define correct token predictions
                correct_token_idx = model.to_tokens(answer, prepend_bos=False)
                correct_token = F.one_hot(correct_token_idx, num_classes=model.cfg.d_vocab).float()[0]

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
                    
                    # Get prediction accuracy
                    if correct_token_idx.item() == last_predicted_token.item():
                        n_correct[p] += 1
            
            # Save results
            taskResults[n_words_per_prompt-1, n_words_per_category-1, :] = n_correct

    # Save results
    np.save("./taskResults.npy", taskResults)

# -------------------------------------------------------
# Process results
# -------------------------------------------------------

# Get change performance for each word per prompt
chanceAccuracies = [1/(i+2) for i in range(n_words_per_prompt)]

# Plot results
plt.close()
fig,ax = plt.subplots(1,n_words_per_prompt,sharey=True, sharex=True, figsize=(10,4))
for n in range(n_words_per_prompt):
    ax[n].plot(100*(taskResults[n,:,0] / n_iterations))
    ax[n].plot(100*(taskResults[n,:,1] / n_iterations))
    ax[n].axhline(chanceAccuracies[n]*100, linestyle="--", color="black", alpha=.5)
    ax[n].set_title(f"{n+1} words per prompt")
    ax[n].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
    ax[n].set_xlabel("Words per category", fontsize=12)

ax[0].set_ylabel("Accuracy (%)", fontsize=12)
plt.tight_layout()
plt.savefig("./figures/taskPerformanceByNwordsPerCategory.png")

# Plot results
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10,5),sharey=True, sharex=True)
ax[0].plot(100*(taskResults[:,:,0].T / n_iterations))
ax[1].plot(100*(taskResults[:,:,1].T / n_iterations))
r = [ax[i].set_ylim([0, 100+5]) for i in range(len(ax))]
ax[0].set_ylabel("Accuracy (%)")
ax[0].legend([f"{i} words per prompt" for i in range(1,n_words_per_prompt+1)])
for r in range(len(ax)):
    ax[r].set_xlabel("Words per category")
    ax[r].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))

plt.tight_layout()
plt.savefig("./figures/taskPerformance_plot.png")

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
# plt.savefig("./figures/taskPerformance_imshow.png")