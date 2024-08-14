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
        cur_mask = np.clip(cur_mask, 0, 1)
        attention_masks[w_per_p-1, w_per_c-1] = cur_mask

# -------------------------------------------------------
# Study attention masks
# -------------------------------------------------------

# Get attention mask sparsity by task types
sparsity_by_task = np.linalg.norm(attention_masks, ord=1, axis=(2,3))

# Plot distribution of mask values
attention_masks_flat = attention_masks.flatten()
plt.close()
plt.hist(attention_masks_flat, bins=100)
plt.savefig("./figures/attention_masks_distribution.png")

# Run regression
X = np.array([np.arange(n_words_per_category) for i in range(n_words_per_prompt)]).flatten()
Y = sparsity_by_task.flatten()
r = linregress(X, Y)

# Plot sparsity by task
plt.close()
Xs = np.arange(n_words_per_category)
Ys = r.slope*Xs + r.intercept
fig = plt.figure(figsize=(12, 6))
plt.plot(sparsity_by_task.T, alpha=0.5, linewidth=3)
plt.plot(sparsity_by_task.mean(axis=0), color="black", linewidth=5)
plt.plot(Xs, Ys, color="gray", linestyle="--", linewidth=3)
plt.legend([f"{i+1}" for i in range(n_words_per_prompt)], fontsize=12, title="Words per line", title_fontsize=16, frameon=False)
plt.xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1), fontsize=14)
plt.ylabel("Attention mask sparsity", fontsize=16)
plt.xlabel("Words per category", fontsize=16)
plt.tight_layout()
plt.savefig("./figures/sparsity_by_number_of_words_per_category.png")

# Plot sparsity by task
plt.close()
fig = plt.figure(figsize=(12, 6))
plt.imshow(sparsity_by_task, aspect="auto", origin="upper")
plt.yticks(np.arange(n_words_per_prompt), np.arange(1,n_words_per_prompt+1))
plt.xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
plt.ylabel("Words per line", fontsize=18, fontweight="bold")
plt.xlabel("Words per category", fontsize=18, fontweight="bold")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/sparsity_by_task.png")

# Plot attention masks
plt.close()
fig,ax = plt.subplots(n_words_per_prompt, n_words_per_category, figsize=(12, 6), sharex=True, sharey=True)
vmin, vmax = attention_masks.min(), attention_masks.max()
for w_per_p in range(1,n_words_per_prompt+1):
    for w_per_c in range(1,n_words_per_category+1):
        cur_mask = attention_masks[w_per_p-1, w_per_c-1]
        ax[w_per_p-1, w_per_c-1].imshow(cur_mask, aspect="auto", origin="upper", interpolation="none", cmap="Greys", vmin=vmin, vmax=vmax)
        ax[w_per_p-1, 0].set_ylabel("Layer index")
        ax[-1, w_per_c-1].set_xlabel("Head index")
        ax[w_per_p-1, w_per_c-1].set_yticks(np.arange(n_layers)[1::2], np.arange(n_layers+1)[1::2], fontsize=8)
        ax[w_per_p-1, w_per_c-1].set_xticks(np.arange(n_layers)[1::2], np.arange(n_layers+1)[1::2], fontsize=8)

_ = [ax[0, w_per_c-1].set_title(f"{w_per_c}", fontsize=18) for w_per_c in range(1,n_words_per_category+1)]
# ax[2, -1].set_xlabel("Head index")
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

# -------------------------------------------------------
# Find heads associated with words per category
# -------------------------------------------------------

# Flatten attention masks to get value for each task type
attention_masks_flat = attention_masks.reshape(n_words_per_prompt, n_words_per_category, n_layers*n_heads)

# Get mean over n_words_per_category
attention_masks_flat_mean = attention_masks_flat.mean(axis=0)

# Fit linear regression to each head
hit_heads = []; names = []
for i in range(n_layers*n_heads):
    cur_d = attention_masks_flat_mean[:, i]
    r = linregress(np.arange(n_words_per_category), cur_d)
    if r.pvalue < (.01):#/(n_layers*n_heads)):
        print(f"Layer {i//n_heads}, Head {i%n_heads} - R^2: {r.rvalue**2:.2f}, p-value: {r.pvalue:.2f}")
        hit_heads.append([i//n_heads, i%n_heads])
        names.append(f"H{i//n_heads}.{i%n_heads}")

# View specific heads
plt.close()
fig = plt.figure(figsize=(12, 5))
for h in hit_heads:
    layer, head = h
    cur_d = attention_masks[:, :, layer, head].mean(axis=0)
    plt.plot(cur_d, linewidth=3)

plt.xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
plt.ylabel("Attention mask value", fontsize=16)
plt.xlabel("Words per category", fontsize=16)
plt.legend(names, fontsize=12, title="Layer/Head", title_fontsize=16, frameon=False)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("./figures/attentionHeadsAssociatedWithWordsPerCategory.png")

# -------------------------------------------------------
# Find heads associated with 1 vs 2 and more words per prompt
# -------------------------------------------------------

one_word_mean = attention_masks[0].mean(axis=0)
multi_word_mean = attention_masks[1:].mean(axis=(0,1))

plt.close()
fig,ax = plt.subplots(1,3, figsize=(12, 6))
ax[0].imshow(one_word_mean, aspect="auto", origin="upper")
ax[1].imshow(multi_word_mean, aspect="auto", origin="upper")
ax[2].imshow(multi_word_mean-one_word_mean, aspect="auto", origin="upper")
_ = [ax[i].set_yticks(np.arange(n_layers), np.arange(1,n_layers+1), fontsize=8) for i in range(3)]
_ = [ax[i].set_xticks(np.arange(n_heads), np.arange(1,n_heads+1), fontsize=8) for i in range(3)]
plt.tight_layout()
plt.savefig("tmp.png")

# -------------------------------------------------------
# Look at specific heads
# -------------------------------------------------------

layer_indices = [11, 9, 8]
head_indices = [3, 9, 7]

plt.close()
fig,ax = plt.subplots(1,len(layer_indices), figsize=(12, 4))
for i in range(len(layer_indices)):
    layer_idx = layer_indices[i]
    head_idx = head_indices[i]
    cur_attention_masks = attention_masks[:, :, layer_idx-1, head_idx-1]
    for w_per_p in range(1,n_words_per_prompt+1):
        ax[i].boxplot(cur_attention_masks[w_per_p-1], positions=[w_per_p], widths=0.5)
    ax[i].set_title(f"Layer {layer_idx-1}, Head {head_idx-1}", fontsize=16)
    ax[i].set_xlabel("Words per line", fontsize=16)

ax[0].set_ylabel("Attention mask value", fontsize=16)
plt.tight_layout()
plt.savefig("./figures/attentionHeadsAssociatedWithWordsPerPrompt.png")

asdkj

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

# Get change performance for each word per prompt
chanceAccuracies = [1/(i+2) for i in range(n_words_per_prompt)]

# Plot token probabilities
plt.close()
cmap = plt.get_cmap("tab10")
fig,ax = plt.subplots(1, n_words_per_prompt, sharey=True, sharex=True, figsize=(12,3))
for w_per_p in np.arange(1,n_words_per_prompt+1):
    # Plot logit probabilities
    cur_probabilities = token_probabilities[w_per_p-1,:,:].mean(axis=1)
    # Get standard error
    variance_values = token_probabilities[w_per_p-1,:,:].std(axis=1) / np.sqrt(n_iterations)
    # ax[1,n].plot(token_probabilities[n,:,:,:].mean(axis=1))
    ax[w_per_p-1].errorbar(np.arange(n_words_per_category), cur_probabilities[:,0], yerr=variance_values[:,0], color=cmap(0), capsize=3)
    ax[w_per_p-1].errorbar(np.arange(n_words_per_category), cur_probabilities[:,1], yerr=variance_values[:,1], color=cmap(1), capsize=3)
    ax[w_per_p-1].axhline(chanceAccuracies[w_per_p-1], linestyle="--", color="black", alpha=.25)
    ax[w_per_p-1].set_xticks(np.arange(n_words_per_category), np.arange(1,n_words_per_category+1))
    ax[w_per_p-1].set_xlabel("Words per category", fontsize=16)
    # Plot title
    suffix = "word" if w_per_p == 1 else "words"
    ax[w_per_p-1].set_title(f"{w_per_p} {suffix} per line", fontsize=18)

# Set legends and labels
ax[-1].legend(["Chance", "Correct token", "Incorrect token"], fontsize=14, frameon=False)
_ = [ax[0].set_ylabel("Token probability\n" + ("(Valid task)","(Corrupted task)")[r], fontsize=16) for r in range(2)]

# Save figure
plt.tight_layout()
plt.savefig(f"./figures/attentionMasked_taskPerformanceByNwordsPerCategory_100examples.png")





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