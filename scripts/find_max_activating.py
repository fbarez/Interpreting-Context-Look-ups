from transformer_lens import HookedTransformer
import torch

import json
import time
import os
import pickle

from src.utils import tuple_str_to_tuple
from src.neuron_texts import get_neuron_max_acts

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Find max activating text for a list of neurons")
    parser.add_argument("filename", help="Name of the file to load the neurons from")
    parser.add_argument("--mode", help="Mode: train or test", choices=['train', 'test'], default='train')
    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)


    args = parser.parse_args()
    filename = args.filename
    batch_size = args.batch_size
    train = True if args.mode == 'train' else False
    # filename = "2024-02-13_04-33-18_pythia-1.4b"

    # Load jsons from ../experiment_data/1_next_token_neuron
    with open(f'./experiment_data/1_next_token_neurons/{filename}.json') as f:
        neurons_data = json.load(f)

    # Parameters
    parameters = neurons_data['parameters']
    model_name = parameters['model_name']
    neurons_list = [tuple_str_to_tuple(x) for x in neurons_data['neurons'].keys()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {model_name} on {device}")
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        # refactor_factored_attn_matrices=True,
        device=device,
    )
    print("Model loaded!")

    # Load dataset
    if train:
        if not os.path.exists('./text_list_dict_train.pkl'):
            from datasets import load_dataset
            dataset = load_dataset("NeelNanda/pile-10k", split="train")
            dataset_text_list = [x['text'] for x in dataset]
                

            with open('./text_list_dict_train.pkl', 'wb') as f:
                pickle.dump(dataset_text_list, f)
        else:
            with open('./text_list_dict_train.pkl', 'rb') as f:
                dataset_text_list = pickle.load(f)
    else:
        with open('./text_list_dict_test.pkl', 'rb') as f:
            dataset_text_list = pickle.load(f)

    neuron_max_acts = get_neuron_max_acts(
        model=model,
        dataset_text_list=dataset_text_list,
        neurons_list=neurons_list,
        batch_size=batch_size,
        device=device,
    )

    # Save neuron_max_acts
    output = {
        'parameters': parameters,
        'neuron_max_acts': {str(key): value for key, value in neuron_max_acts.items()},
        'prior_filename': filename,
    }

    # Save json to ../experiment_data/2_max_activating_texts
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(int(time.time())))
    test_string = "_test" if not train else ""
    new_filename = f"{timestamp}_{model_name}{test_string}.json"

    if not os.path.exists('./experiment_data/2_max_activating_texts'):
        os.makedirs('./experiment_data/2_max_activating_texts')
    with open(f'./experiment_data/2_max_activating_texts/{new_filename}', 'w') as f:
        json.dump(output, f)

# Do if name == main and args
if __name__ == "__main__":
    main()
