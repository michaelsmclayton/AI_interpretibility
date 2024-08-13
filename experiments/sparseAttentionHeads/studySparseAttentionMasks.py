import os
import torch
import matplotlib.pyplot as plt
import transformer_lens
from tqdm import tqdm
from scipy.stats import linregress
import numpy as np
from sparseAttentionHeads import runModelWithHooks
# srun --gres=gpu:1 --mem=128G --partition=ml --nodelist=fmg104 --time=10:00:00 --pty tcsh

# Load sparse attention masks
n_words_per_prompt = 4
n_words_per_category = 10
n_epochs = 100
n_examples = 5
n_heads = 12
n_layers = 12
attention_masks = np.zeros([n_words_per_prompt, n_words_per_category, n_layers, n_heads])
for w_per_p in range(1,n_words_per_prompt+1):
    for w_per_c in range(1,n_words_per_category+1):
        # Load current sparse attention mask
        cur_mask = torch.load(f"./savedSparsityMasks/{w_per_p}_{w_per_c}_best_attention_mask_{n_epochs}epochs_{n_examples}examples.pt")
        attention_masks[w_per_p-1, w_per_c-1] = cur_mask

# -------------------------------------------------------
# Assess task performance
# -------------------------------------------------------

# Define iterations per batch
n_iterations_per_batch = 25
n_iterations = 500
n_tokens = 50257

# Load GPT-2 Small
print("Loading model...")
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get token indices of all numbers
numbers = [f" {i}" for i in range(10)]
number_tokens = model.to_tokens(numbers, prepend_bos=False).squeeze().tolist()

# Define path save names
suffix = f"{n_examples}examples_{n_iterations}iterations"
logitResults_fname = f"./savedSparsityMasks/logitResults_{suffix}.npy"
correctAnswers_fname = f"./savedSparsityMasks/correctAnswers_{suffix}.npy"

# Load results if they exist
if os.path.exists(logitResults_fname):
    logitResults = np.load(logitResults_fname)
    correctAnswers = np.load(correctAnswers_fname)
else:

    # Initialise task results
    taskResults = np.zeros((n_words_per_prompt, n_words_per_category, 2))
    logitResults = np.zeros((n_words_per_prompt, n_words_per_category, n_iterations, n_tokens))
    correctAnswers = np.zeros((n_words_per_prompt, n_words_per_category, n_iterations))

    # Iterate over words per prompt
    for w_per_p in range(1,n_words_per_prompt+1):
        for w_per_c in range(1,n_words_per_category+1):
        
            # Get current attention mask
            print(f"Gettting attention mask ({w_per_p},{w_per_c})..")
            cur_attention_mask = attention_masks[w_per_p-1, w_per_c-1]

            # Iterate over batches
            print(f"Running model..")
            for i in tqdm(range(0, n_iterations, n_iterations_per_batch)):
                
                # Run model with hooks
                valid_logits_hk, valid_logits, correct_tokens = runModelWithHooks(model, cur_attention_mask, w_per_p, w_per_c, n_iterations_per_batch, n_layers, device)
                
                # Get softmax of logits
                valid_logits_hk_softmax = torch.nn.functional.softmax(valid_logits_hk, dim=-1)
                
                # Get model prediction
                last_prediction_logits = valid_logits_hk_softmax[:, -1, :]
                
                # Save logits for all number tokens
                logitResults[w_per_p-1, w_per_c-1, i:i+n_iterations_per_batch, :] = last_prediction_logits.tolist()
                
                # Save correct answers
                correctAnswers[w_per_p-1, w_per_c-1, i:i+n_iterations_per_batch] = correct_tokens.tolist()
                
                # Remove data from CUDA
                if device == "cuda":
                    del valid_logits_hk, valid_logits_hk_softmax, last_prediction_logits
                    torch.cuda.empty_cache()

        # Save results
        np.save(logitResults_fname, logitResults)
        np.save(correctAnswers_fname, correctAnswers)

asdkjl

# -------------------------------------------------------
# Study token probabilities
# -------------------------------------------------------

# Initialise token probabilities
n_answer_types = 2 # i.e. correct and incorrect
token_probabilities = np.zeros((n_words_per_prompt, n_words_per_category, n_iterations, n_answer_types))

# Iterate over examples
for w_per_p in np.arange(1,n_words_per_prompt+1):
    for w_per_c in np.arange(1,n_words_per_category+1):
        # Get possible response values
        possible_responses = np.arange(w_per_p+1)
        possible_response_tokens = model.to_tokens([f" {i}" for i in possible_responses], prepend_bos=False).squeeze().tolist()
        # Get logit results and answers
        cur_logits = torch.tensor(logitResults[w_per_p-1, w_per_c-1, :, :])
        cur_answer_tokens = correctAnswers[w_per_p-1, w_per_c-1, :].astype(int)
        # Iterate over examples
        for i in range(n_iterations):
            # Get current logits and correct answer
            logs = cur_logits[i]
            answer = cur_answer_tokens[i]
            # Get indices of correct and incorrect tokens
            correct_token_idx = [tok for i,tok in enumerate(possible_response_tokens) if tok == answer]
            incorrect_token_idx = [tok for i,tok in enumerate(possible_response_tokens) if tok != answer]
            # Get logits of softmax for correct and incorrect tokens
            correct_logits = logs[correct_token_idx].mean(axis=0).numpy()
            incorrect_logits = logs[incorrect_token_idx].mean(axis=0).numpy()
            token_probabilities[w_per_p-1, w_per_c-1, i, 0] = correct_logits
            token_probabilities[w_per_p-1, w_per_c-1, i, 1] = incorrect_logits

# Plot performance
plt.close()
fig,ax = plt.subplots(1,n_words_per_prompt, figsize=(12, 6), sharex=True, sharey=True)
for w_per_p in range(1,n_words_per_prompt+1):
    ax[w_per_p-1].plot(token_probabilities[w_per_p-1, :, :, 0].mean(axis=1), color="blue", alpha=0.5)
    ax[w_per_p-1].plot(token_probabilities[w_per_p-1, :, :, 1].mean(axis=1), color="orange", alpha=0.5)
    ax[w_per_p-1].set_ylim([0, 1])

plt.savefig("tmp.png")

asdklj

# Get loss
error = torch.nn.MSELoss()(valid_logits, valid_logits_hk)

asdjkl

# Get attention mask sparsity by task types
sparsity_by_task = np.linalg.norm(attention_masks, ord=1, axis=(2,3))
plt.close()
plt.plot(sparsity_by_task.T, alpha=0.5)
plt.plot(sparsity_by_task.mean(axis=0), color="black", linewidth=3)
plt.savefig("./figures/sparsity_by_task.png")

# Plot attention masks
plt.close()
fig,ax = plt.subplots(n_words_per_prompt, n_words_per_category, figsize=(12, 6), sharex=True, sharey=True)
vmin, vmax = attention_masks.min(), attention_masks.max()
for w_per_p in range(1,n_words_per_prompt+1):
    for w_per_c in range(1,n_words_per_category+1):
        cur_mask = attention_masks[w_per_p-1, w_per_c-1]
        ax[w_per_p-1, w_per_c-1].imshow(cur_mask, aspect="auto", origin="lower", interpolation="none", cmap="Greys", vmin=vmin, vmax=vmax)
    ax[w_per_p-1, 0].set_ylabel(f"{w_per_p}")

_ = [ax[0, w_per_c-1].set_title(f"{w_per_c}") for w_per_c in range(1,n_words_per_category+1)]
plt.tight_layout()
plt.savefig("./figures/attention_masks.png")

# Plot mean attention masks (over words per category)
plt.close()
mean_attention_masks_over_category = attention_masks.mean(axis=1)
vmin, vmax = mean_attention_masks_over_category.min(), mean_attention_masks_over_category.max()
fig,ax = plt.subplots(2, n_words_per_prompt, figsize=(12, 6), sharex=True, sharey=True)
for w_per_p in range(1,n_words_per_prompt+1):
    cur_mask = mean_attention_masks_over_category[w_per_p-1]
    ax[0,w_per_p-1].imshow(cur_mask, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax[0,w_per_p-1].set_title(f"{w_per_p} - {np.linalg.norm(cur_mask, ord=1):.2f}")

ax[1,0].imshow(mean_attention_masks_over_category[1]-mean_attention_masks_over_category[0], aspect="auto", origin="lower")#, vmin=vmin, vmax=vmax)
ax[1,1].imshow(mean_attention_masks_over_category[2]-mean_attention_masks_over_category[1], aspect="auto", origin="lower")#, vmin=vmin, vmax=vmax)
ax[1,2].imshow(mean_attention_masks_over_category[2]-mean_attention_masks_over_category[0], aspect="auto", origin="lower")#, vmin=vmin, vmax=vmax)
plt.tight_layout()
plt.savefig("./figures/attention_masks_mean.png")

# # Plot mean attention masks (over words per category)
# plt.close()
# mean_attention_masks_over_category = attention_masks.mean(axis=1)
# vmin, vmax = mean_attention_masks_over_category.min(), mean_attention_masks_over_category.max()
# fig,ax = plt.subplots(2, n_words_per_prompt, figsize=(12, 6), sharex=True, sharey=True)
# for w_per_p in range(1,n_words_per_prompt+1):
#     cur_mask = mean_attention_masks_over_category[w_per_p-1]
#     ax[0,w_per_p-1].imshow(cur_mask, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
#     ax[0,w_per_p-1].set_title(f"{w_per_p} - {np.linalg.norm(cur_mask, ord=1):.2f}")

# ax[1,0].imshow(mean_attention_masks_over_category[1]-mean_attention_masks_over_category[0], aspect="auto", origin="lower")#, vmin=vmin, vmax=vmax)
# ax[1,1].imshow(mean_attention_masks_over_category[2]-mean_attention_masks_over_category[1], aspect="auto", origin="lower")#, vmin=vmin, vmax=vmax)
# ax[1,2].imshow(mean_attention_masks_over_category[2]-mean_attention_masks_over_category[0], aspect="auto", origin="lower")#, vmin=vmin, vmax=vmax)
# plt.tight_layout()
# plt.savefig("./figures/attention_masks_mean.png")