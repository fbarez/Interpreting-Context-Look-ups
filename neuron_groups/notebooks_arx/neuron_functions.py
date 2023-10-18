import torch
from fancy_einsum import einsum

def get_head_attribution(model, cache, tokens, neuron):
    """
    Args:
        cache: ActivationCache,
        tokens:Tensor
        neuron:Tuple
    Returns:
        head_attribution:Tensor of shape [batch, head]
    """
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