import torch
import transformer_lens
# import circuitsvis as cv
import numpy as np
import transformer_lens.utils as utils
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import matplotlib.pylab as plt
import torch.nn.functional as F
# r = torch.set_grad_enabled(False)
from functools import partial
# srun --gres=gpu:1 --mem=128G --partition=ml --nodelist=fmg104 --time=10:00:00 --pty tcsh
'''
Advantages of sparse attention heads:
- Doesn't rely on a pruning threshold (as in ABC)
- Gives each attention head a scalar value of importance, rather than each edge being present / absent

'''
# Load GPT-2 Small
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define prompt
prompt = '''July lizard: 0
red cherry: 3
red lion: 2
April fish: 0
April frog: 0
gray bird: 2
red snake: 2
September apple: 1
January bird: 0
blue pear: 3
yellow frog: 2
gray grape: 3
January cat: 0
October pear: 1
gray strawberry: 3
gray cat: 2
June pineapple: 1
red snake: 2
March cherry: 1
purple frog: 2
green wolf: 2
blue snake: 2
green banana: 3
March apple: 1
June banana: 1
June frog: 0
March fish: 0
green pineapple: 3
January apple: 1
October lizard:'''

# Convert prompt to tokens
tokens = model.to_tokens(prompt, prepend_bos=True)
tokens_str = model.to_str_tokens(prompt, prepend_bos=True)
n_input_tokens = tokens.shape[1]

# Define correct token predictions
correct_token_idx = model.to_tokens(" 0", prepend_bos=False)
correct_token = F.one_hot(correct_token_idx, num_classes=model.cfg.d_vocab).float()[0]

# Run the model and get logits and activations
logits, activations = model.run_with_cache(tokens, remove_batch_dim=True)

# Initialise attention mask
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
attention_mask = 1 * torch.ones(n_layers, n_heads)

# Define loss function
def getLoss(logit, attention_mask):
    '''Calculate L1 norm of attention mask'''
    accuracy_loss = torch.nn.functional.cross_entropy(logit, correct_token)
    sparsity_loss = torch.norm(nonlinearity(attention_mask), p=1)
    return accuracy_loss*100, sparsity_loss

# Non-linearities
# nonlinearity = lambda x : torch.sigmoid(x)
nonlinearity = lambda x : x # this may not be neccessary

# We define a head ablation hook
def head_ablation_hook(value, hook, layer_idx):
    for head_index in range(n_heads):
        value[:, :, head_index, :] *= nonlinearity(torch.abs(attention_mask[layer_idx, head_index]))
    return value

# Reset model hooks
model.reset_hooks()

# Generate hooks across all layers
for layer_idx in range(n_layers):
    temp_hook_fn = partial(head_ablation_hook, layer_idx=layer_idx)
    cur_activation_name = utils.get_act_name("v", layer_idx, "attn")
    model.add_hook(cur_activation_name, temp_hook_fn)

# -------------------------------------------------------
# Train model
# -------------------------------------------------------

# Define optimizer
attention_mask.requires_grad = True
optimizer = torch.optim.Adam([attention_mask], lr=0.01)

# Define number of epochs
n_epochs = 1000

# Print loss every n epochs
n_print = 10

# Train model
losses = []
for i in range(n_epochs):
    # Run the model and get logits and activations
    logits, activations = model.run_with_cache(tokens, remove_batch_dim=True)
    # Get loss
    accuracy_loss, sparsity_loss = getLoss(logits[:,-1], attention_mask)
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss = accuracy_loss + sparsity_loss
    loss.backward()
    optimizer.step()
    losses.append([accuracy_loss.item(), sparsity_loss.item()])
    # Print loss
    if i % n_print == 0:
        print(f"Epoch {i+1}/{n_epochs}, Accuracy Loss: {accuracy_loss.item():.3f}, Sparsity Loss: {sparsity_loss.item():.3f}, Predicted token = {torch.argmax(logits[:,-1]).item()}")

# Save attention mask
torch.save(attention_mask, "attention_mask.pt")

# -------------------------------------------------------
# Get remaining attention head patterns
# -------------------------------------------------------

# Get remaining attention head indices
flat_attention_mask = np.abs(attention_mask.detach().flatten())
flat_attention_mask_round = np.round(flat_attention_mask, 1)
attention_heads_to_keep = np.where(np.isclose(flat_attention_mask_round, 0.0)!=True)[0]
n_attention_heads_to_keep = len(attention_heads_to_keep)
print(f"Number of attention heads to keep: {len(attention_heads_to_keep)}/{flat_attention_mask_round.shape[0]}")

# Get layer indices of remaining attention heads
layer_indices = attention_heads_to_keep // n_heads
head_indices = attention_heads_to_keep % n_heads
remaining_attention_heads = np.array([layer_indices, head_indices])

# Get head importance scores
head_importance_scores = np.round(flat_attention_mask[attention_heads_to_keep].numpy(), 2)

# Get attention patterns
attention_patterns = []
for layer_index, head_index in remaining_attention_heads.T:
    cur_pattern = activations["pattern", layer_index, "attn"]
    attention_patterns.append(cur_pattern[head_index].cpu().numpy())

# Plot attention patterns
nRows = 3
nCols = np.ceil(n_attention_heads_to_keep / nRows).astype(int)
head_importance_scores_str = [f"{head_importance_scores[i]:.2f}" for i in range(n_attention_heads_to_keep)]
plt.close()
fig, ax = plt.subplots(nRows, nCols, figsize=(12, 10), sharex=True, sharey=True)
ax = ax.flatten()
for i, cur_pattern in enumerate(attention_patterns):
    ax[i].imshow(cur_pattern, aspect="auto", origin="upper")
    ax[i].set_title(f"Layer {layer_indices[i]}\n({head_importance_scores_str[i]}), Head {head_indices[i]}")

plt.tight_layout()
plt.savefig("attention_patterns.png", dpi=500)

# -------------------------------------------------------
# Plot results
# -------------------------------------------------------

# Plot attention mask
plt.figure()
plt.imshow(np.abs(attention_mask.detach().numpy()), aspect="auto", origin="lower")
plt.colorbar()
plt.ylabel("Layer")
plt.xlabel("Head")
plt.savefig("attention_mask.png")

# Plot attention mask distribution
percentage_attention_heads_removed = n_attenton_heads_removed / flat_attention_mask_round.shape[0] * 100
print(f"Percentage of attention heads removed: {percentage_attention_heads_removed:.2f}%")

plt.figure()
flat_attention_mask_positive = flat_attention_mask_round[flat_attention_mask_round > 0]
plt.hist(flat_attention_mask_positive.detach().numpy(), bins=100)
plt.savefig("attention_mask_dist.png")