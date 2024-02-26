from fancy_einsum import einsum
import torch
from tqdm import tqdm


def find_neurons(model, layers, num_neurons_per_layer, rejected_neurons):
    results = {}
    rejected_neurons_with_tokens = []

    # Handle layers if they are negative
    num_layers = model.cfg.n_layers
    layers = [layer if layer >= 0 else num_layers + layer for layer in layers]

    for layer_num in layers:
        n_layer_neurons = model.W_out[layer_num, :, :]
        unembedding = model.W_U
        dot_product = einsum("neuron embed, embed token -> neuron token", n_layer_neurons, unembedding)
        
        values, indices = torch.max(dot_product, dim=-1) # Get the highest congruence with any given token for each neuron
        top_values, top_indices = torch.topk(values, indices.shape[0])

        neurons_found = 0
        
        for i in range(unembedding.shape[1]):
            # Check if the token starts with a space, and if it is longer than 1 character (excl. the space)
            str_token = model.to_string(indices[top_indices][i])
            if len(str_token) <= 2 or str_token[0] != " ":
                continue

            # Check if the neuron is manually rejected (e.g. if it's a glitch token neuron)
            neuron_index = top_indices[i].item()
            if (layer_num, neuron_index) in rejected_neurons:
                rejected_neurons_with_tokens.append((layer_num, neuron_index, str_token))
                continue


            results[(layer_num, neuron_index)] = {'token': str_token, 'congruence': top_values[i].item()}

            neurons_found += 1
            if neurons_found >= num_neurons_per_layer:
                break

    return results, rejected_neurons_with_tokens

def _cache_to_tuples(cache):
    """Converts the model's cache to a list of tuples, i.e. [(max_value, max_index), ...]"""
    new_cache = {}
    for key in cache.keys():
        x = torch.max(cache[key], dim=1)
        y = list(x)
        y = [y[0].tolist(), y[1].tolist()]
        y = list(zip(*y))
        new_cache[key] = y # y is a list of tuples, i.e. [(max_value, max_index), ...]
    return new_cache

def get_neuron_max_acts(model, dataset_text_list, neurons_list, batch_size, device):
    batched_texts = [dataset_text_list[i: i+batch_size] for i in range(0, len(dataset_text_list), batch_size)]

    neuron_max_acts = {neuron: [] for neuron in neurons_list}

    for texts in tqdm(batched_texts):
        model.reset_hooks()

        cache = {}

        def return_caching_hook(neuron):
            layer, neuron_index = neuron
            def caching_hook(act, hook):
                cache[(layer, neuron_index)] = act[:, :, neuron_index] # act shape is (batch_size, seq_len, neuron_index)
            return caching_hook
        
        hooks = list(((f"blocks.{layer}.mlp.hook_post", return_caching_hook((layer, index))) for layer, index in neurons_list))

        model.run_with_hooks(
            model.to_tokens(texts).to(device),
            fwd_hooks=hooks,
        )
        cache = _cache_to_tuples(cache)

        for key in cache.keys():
            neuron_max_acts[key].extend(cache[key])

    return neuron_max_acts

def find_truncated_texts(model, neuron_max_acts, dataset, device, num_samples=20, act_ratio=0.8):
    neuron_20_examples = {}

    for neuron_str in tqdm(neuron_max_acts):
        # neuron_str is '(layer, neuron_index)'. Make it a tuple
        neuron = tuple([int(x) for x in neuron_str[1:-1].split(", ")])
        neuron_acts = neuron_max_acts[neuron_str]
        # Get the top 20 examples
        neuron_acts = [x + [i] for i, x in enumerate(neuron_acts)] # Add the index of the example in the dataset
        sorted_acts = sorted(neuron_acts, key=lambda x: x[0], reverse=True) # Sort by activation
        top_examples_indices = [x[2] for x in sorted_acts[:num_samples]] # Get the indices of the top examples
        top_examples_pos = [x[1] for x in sorted_acts[:num_samples]] # Get the token positions of the top examples
        top_examples_acts = [x[0] for x in sorted_acts[:num_samples]] # Get the activations of the top examples

        data = sorted_acts[:num_samples]
        data = [{'og_act': v[0], 'token_pos': v[1], 'dataset_index': v[2]} for v in data]

        neuron_20_examples[neuron_str] = data

        examples = []
        for index in top_examples_indices:
            examples.append(dataset[index]['text'])

        for example_index, example in tqdm(enumerate(examples)):
            # Tokenize the example
            example_tokens = model.to_tokens(example, prepend_bos=False).to(device)
            # Truncate the example to the right length
            example_tokens = example_tokens[:, :top_examples_pos[example_index]]
            # print(example_tokens.shape)

            layer, neuron_index = neuron
            neurons = [neuron]

            cache = []
            def return_caching_hook(neuron):
                layer, neuron_index = neuron
                def caching_hook(act, hook):
                    cache.append(act[:, -1, neuron_index]) # act shape is (batch_size, seq_len, neuron_index)
                return caching_hook

            original_act = neuron_20_examples[neuron_str][example_index]['og_act']

            hooks = list(((f"blocks.{layer}.mlp.hook_post", return_caching_hook((layer, index))) for layer, index in neurons))

            # Do binary search
            start = 0
            end = example_tokens.shape[1]-1
            mid = (start + end) // 2

            if start == mid: # Some weird edge case where the example is only 2 tokens long (Every once... "in")
                neuron_20_examples[neuron_str][example_index]['start_pos'] = mid
                neuron_20_examples[neuron_str][example_index]['trunc_act'] = original_act
                continue
            
            while start != mid:
                cache = []
                new_tokens = example_tokens[:, mid:]
                model.run_with_hooks(
                    new_tokens,
                    fwd_hooks=hooks,
                )
                if cache[-1].item() > act_ratio * original_act:
                    start = mid
                else:
                    end = mid
                mid = (start + end) // 2
            neuron_20_examples[neuron_str][example_index]['start_pos'] = mid
            # and add the activation
            neuron_20_examples[neuron_str][example_index]['trunc_act'] = cache[-1].item()

    return neuron_20_examples