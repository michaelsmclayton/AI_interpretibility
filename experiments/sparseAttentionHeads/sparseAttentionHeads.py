import os
import torch
import transformer_lens
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import torch.nn.functional as F
import transformer_lens.utils as utils
# from sae_lens import SAE
from functools import partial
from generateTasks import generateTaskExamples, shuffleNumbersInString, getPromptAndAnswer
# r = torch.set_grad_enabled(False)
# srun --gres=gpu:1 --mem=128G --partition=ml --nodelist=fmg104 --time=10:00:00 --pty tcsh
torch.autograd.set_detect_anomaly(True)

# Define function to get task
def getTask(n_examples, w_per_p, w_per_c, model):
    valid_examples = []
    corrupted_examples = []
    correct_token_indices = []
    for i in range(n_examples):
        # Get task examples
        t = generateTaskExamples(n_examples, w_per_p, w_per_c)
        t_corrupted = shuffleNumbersInString(t)
        # Get prompt and answer
        prompt_real, answer = getPromptAndAnswer(t)
        prompt_corrupted, _ = getPromptAndAnswer(t_corrupted)
        # Define correct token predictions
        correct_token_idx = model.to_tokens(answer, prepend_bos=False)
        # Save results
        valid_examples.append(prompt_real)
        corrupted_examples.append(prompt_corrupted)
        correct_token_indices.append(correct_token_idx)
    return valid_examples, corrupted_examples, correct_token_indices

# Define loss function
def getLogitDifference(cur_logits, prev_logits):
    mse_error = torch.nn.MSELoss()
    return mse_error(cur_logits, prev_logits)

# We define a head ablation hook
def head_ablation_hook(value, hook, layer_idx, corrupt_activations, attention_head_mask, n_heads=12):
    new_value = value.clone()
    for head_index in range(n_heads):
        # Get current attention mask value (0 - 1)
        cur_attention_mask = attention_head_mask[layer_idx, head_index]
        cur_attention_v = value[:, :, head_index, :]
        corrupt_attention_v = corrupt_activations[:, :, head_index, :]
        valid_activation_v = cur_attention_v * cur_attention_mask
        corrupt_activation_v = corrupt_attention_v * (1-cur_attention_mask)
        new_value[:, :, head_index, :] = valid_activation_v + corrupt_activation_v
    return new_value

def runModelWithHooks(model, attention_head_mask, n_w_p, n_w_c, n_iterations_per_batch, n_layers, device):
    # Reset all hooks
    model.reset_hooks()
    
    # Get task examples
    valid, corrupt, correct_tokens = getTask(n_iterations_per_batch, n_w_p, n_w_c, model)
    correct_tokens = torch.tensor(correct_tokens).to(device)
    
    # Run model on corrupted prompts
    corrupt_tokens = model.to_tokens(corrupt, prepend_bos=True, padding_side="left")
    corrupt_logits, corrupt_activations = model.run_with_cache(corrupt_tokens, remove_batch_dim=False)

    # Run model on valid prompts
    valid_tokens = model.to_tokens(valid, prepend_bos=True, padding_side="left")
    valid_logits, _ = model.run_with_cache(valid_tokens, remove_batch_dim=False)

    # Generate hooks across all layers
    for layer_idx in range(n_layers):
        temp_hook_fn = partial(head_ablation_hook, layer_idx=layer_idx, corrupt_activations=corrupt_activations[f"blocks.{layer_idx}.attn.hook_v"], attention_head_mask=attention_head_mask)
        cur_activation_name = utils.get_act_name("v", layer_idx, "attn")
        model.add_hook(cur_activation_name, temp_hook_fn)
    
    # Run model on valid prompts (with hooks)
    valid_tokens = model.to_tokens(valid, prepend_bos=True, padding_side="left")
    valid_logits_hk, valid_activations = model.run_with_cache(valid_tokens, remove_batch_dim=False)
    
    # Return
    return valid_logits_hk, valid_logits, correct_tokens

# -------------------------------------------------------
# Train model
# -------------------------------------------------------

if __name__=="__main__":

    # Load GPT-2 Small
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define task parameters
    n_iterations_per_batch = 5
    n_words_per_prompt = 4
    n_words_per_category = 10
    n_tokens = 50257
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    # Get token indices of all numbers
    numbers = [f" {i}" for i in range(10)]
    number_tokens = model.to_tokens(numbers, prepend_bos=False).squeeze().tolist()

    # Training parameters
    n_epochs = 100
    n_print = 1
    logit_difference_loss_weight = 1
    sparsity_loss_weight = 1

    # Define n_words_per_prompt and n_words_per_category
    for n_w_p in range(1,n_words_per_prompt+1):
        for n_w_c in range(1,n_words_per_category+1):
            print(f"Training for {n_w_p} words per prompt and {n_w_c} words per category")
            
            # Check if attention mask already exists
            if os.path.exists(f"./savedSparsityMasks/{n_w_p}_{n_w_c}_attention_mask_{n_epochs}epochs_{n_iterations_per_batch}examples.pt"):
                continue

            # Initialise attention masks
            attention_head_mask = 1 * torch.ones(model.cfg.n_layers, model.cfg.n_heads).to(device)

            # Define optimizer
            attention_head_mask.requires_grad = True
            optimizer = torch.optim.Adam([attention_head_mask], lr=0.05)

            # Define best loss
            best_loss = np.inf
            best_attention_head_mask = None

            # Train model
            losses = []
            for i in range(n_epochs):
                
                # Clip attention head mask values between 0 and 1
                attention_head_mask.data = torch.clamp(attention_head_mask, 0, 1)
            
                # Run model with hooks
                valid_logits_hk, valid_logits, correct_tokens = runModelWithHooks(model, attention_head_mask, n_w_p, n_w_c, n_iterations_per_batch, n_layers, device)
                
                # Get loss
                logit_difference_loss = getLogitDifference(valid_logits_hk[:,-1,:], valid_logits[:,-1,:]) * logit_difference_loss_weight
                    
                # Get task accuracy
                answers = torch.argmax(valid_logits_hk[:,-1,:], dim=-1)
                accuracy_loss = torch.mean((answers == correct_tokens).float())
                
                # Get sparsity loss
                sparsity_loss = torch.norm(attention_head_mask, p=1)
                sparsity_loss = sparsity_loss / np.prod(attention_head_mask.shape)
                sparsity_loss *= sparsity_loss_weight

                # Get total loss and backpropagate    
                optimizer.zero_grad()
                loss = logit_difference_loss + sparsity_loss
                loss.backward()
                optimizer.step()
                losses.append([logit_difference_loss.item(), sparsity_loss.item()])
                
                # Save current sparse attention head mask (if loss is better)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_attention_head_mask = attention_head_mask.detach().clone()
                
                # Print loss
                if i % n_print == 0:
                    print(f"Epoch {i}: Logit difference loss: {logit_difference_loss.item()}, Sparsity Loss: {sparsity_loss.item()}, (Accuracy = {accuracy_loss.item()})")

            # Get remaining attention head patterns
            attention_head_mask_np = attention_head_mask.detach().cpu().numpy()
            best_attention_head_mask_np = best_attention_head_mask.detach().cpu().numpy()
            
            # Save attention masks
            torch.save(attention_head_mask, f"./savedSparsityMasks/{n_w_p}_{n_w_c}_attention_mask_{n_epochs}epochs_{n_iterations_per_batch}examples.pt")
            torch.save(best_attention_head_mask_np, f"./savedSparsityMasks/{n_w_p}_{n_w_c}_best_attention_mask_{n_epochs}epochs_{n_iterations_per_batch}examples.pt")