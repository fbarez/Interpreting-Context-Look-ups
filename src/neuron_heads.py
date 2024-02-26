import torch
from fancy_einsum import einsum
from tqdm import tqdm

def head_attribution_over_all_data(model, data, device, neurons, batch_size=8):

    head_attribution_dict = {}

    for neuron in tqdm(neurons):
        # Load and Truncate Prompts
        trunc_prompts, prompts_metadata = data.load_truncated_prompts(model, neuron)

        num_batches = len(trunc_prompts) // batch_size + (1 if len(trunc_prompts) % batch_size != 0 else 0)

        head_results = {}
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(trunc_prompts))
            current_batch = trunc_prompts[start_idx:end_idx]

            # Run head attribution
            tokens = model.to_tokens(current_batch, prepend_bos=True).to(device=device)
            original_logits, cache = model.run_with_cache(tokens, )

            # Prepare prompts by heads
            head_attribution = _get_head_attribution(model, cache, tokens, neuron)
            _, top_heads = torch.topk(head_attribution, k=3, dim=-1)
            top_heads_list = top_heads.tolist()

            for i, prompt in enumerate(current_batch):
                head_results[prompt] = top_heads_list[i]

        head_attribution_dict[str(neuron)] = head_results

    return head_attribution_dict

def _get_head_attribution(model, cache, tokens, neuron):
    # Get prompt lengths
    pad_token = bos_token = 50256 # This is true for GPT-2
    prompt_lengths = (torch.logical_and(tokens != pad_token, tokens != bos_token)).sum(dim=-1)

    # Get the correct last-seq for each prompt (since they are padded, the last seq position differs for each prompt)
    head_output = cache.stack_head_results()
    expanded_index_tensor = prompt_lengths.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(head_output.shape[0], head_output.shape[1], 1, head_output.shape[-1])
    head_output_last_seq = torch.gather(head_output, 2, expanded_index_tensor).squeeze(2)

    # Get dot product of neuron with each head's output
    layer, no = neuron
    neuron_w_in = model.W_in[layer, :, no]
    head_attribution = einsum("head batch weight, weight->batch head", head_output_last_seq, neuron_w_in)
    head_attribution = head_attribution[:, :layer*model.cfg.n_heads] # Filter for only heads before the neuron
    return head_attribution # Shape [batch, head]