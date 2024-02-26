import json
from datasets import load_dataset
from dataclasses import dataclass, field

class ActivatingDataset:
    def __init__(self, trunc_data_dict, dataset=None):
        if dataset is None:
            self.data = load_dataset("NeelNanda/pile-10k", split="train")
        else:
            self.data = dataset

        self.markers = trunc_data_dict

        # Convert keys to tuple if they are string
        def _neuron_str_to_tuple(string):
            if type(string) == str:
                return tuple([int(x) for x in string[1:-1].split(", ")])
            else:
                return string

        self.markers = {_neuron_str_to_tuple(key): value for key, value in self.markers.items()}

    def remove_prompts_longer_than(self, length=100):
        for neuron in self.markers:
            self.markers[neuron] = [prompt for prompt in self.markers[neuron] if prompt['token_pos']-prompt['start_pos'] < length]

    def count_average_prompt_length(self):
        for neuron in self.markers:
            lengths = []
            for prompt in self.markers[neuron]:
                print(prompt)
                prompt_length = prompt['token_pos']-prompt['start_pos']
                lengths.append(prompt_length)
            average_length = sum(lengths)/len(lengths)
            print(f"Average prompt length for neuron {neuron}: {average_length}")

    def load_truncated_prompts(self, model, neuron, replace_newlines_with_spaces=True):

        prompt_indices = [prompt['dataset_index'] for prompt in self.markers[neuron]]
        prompts = [self.data[prompt_index]['text'] for prompt_index in prompt_indices]
        prompts_metadata = [self.data[prompt_index]['meta']['pile_set_name'] for prompt_index in prompt_indices]

        prompts_tokens = [model.to_tokens(prompt, prepend_bos=False) for prompt in prompts]
        prompts_tokens_trunc = []
        for i, prompt_tokens in enumerate(prompts_tokens):
            start, end = self.markers[neuron][i]['start_pos'], self.markers[neuron][i]['token_pos']
            trunc_prompt_token = prompt_tokens[:, start:end]

            prompts_tokens_trunc.append(trunc_prompt_token)


        trunc_prompts = [model.to_string(prompt_tokens)[0] for prompt_tokens in prompts_tokens_trunc]
        if replace_newlines_with_spaces:
            trunc_prompts = [prompt.replace("\n", " ") for prompt in trunc_prompts]
        return trunc_prompts, prompts_metadata
    

@dataclass
class ExplanationPromptGen:
    '''In a zero-shot explanation prompt, we have
    context - attn_head - instruction - active_eg - inactive_eg - explanation_prompt

    In a few-shot explanation prompt, we have
    context - ATTN_N - instruction - active_eg - inactive_eg - EXPLANATION_FULL

    The difference between for few-shot is that we have the ATTN_N demarcator ("Attention Head N\n") and the EXPLANATION_FULL sections.
    '''
    token: str
    positive_examples: list
    negative_examples: list
    context: str = "We are studying attention heads in a transformer architecture neural network. Each attention head looks for some particular thing in a short document.\n"
    attn_head: str = field(init=False) # f"This attention head in particular helps to predict that the next token is {self.token}, but it is only active in some documents and not others."
    instruction: str = "Look at the documents and explain what makes the attention head active, taking into consideration the inactive examples.\n"
    active_eg: str = field(init=False) # Examples where the attention head is active:\n * Example 1\n * Example 2\n...
    inactive_eg: str = field(init=False) # Examples where the attention head is inactive:\n * Example 1\n * Example 2\n...
    # explanation_prompt: str = "Explanation: This attention head is active when the document"
    explanation_prompt: str = "In two to three sentences, suggest when the attention head is active."
    shots_file_path: str = None

    def __post_init__(self):
        self.attn_head = self._prepare_attn_head_str(self.token)
        self.active_eg, self.inactive_eg = self._prepare_example_list_to_str(self.positive_examples, self.negative_examples)

    def get_prompt(self, shots=0):
        if shots == 0:
            return self.context + self.attn_head + self.instruction + self.active_eg + self.inactive_eg + self.explanation_prompt
        else:
            shots_string = self.prepare_shots(shots)
            return self.context + shots_string + self.attn_head + self.instruction + self.active_eg + self.inactive_eg + self.explanation_prompt
        
    def prepare_shots(self, num_shots):
        shots_string = ""
        shots_data = ExplanationShots(self.shots_file_path)
        for i in range(num_shots):
            token, active_examples, inactive_examples, explanation = shots_data[i]
            shots_string += f"Attention Head {i+1}:\n"
            shots_string += self._prepare_attn_head_str(token)
            shots_string += self.instruction
            active_eg, inactive_eg = self._prepare_example_list_to_str(active_examples, inactive_examples)
            shots_string += active_eg + inactive_eg
            shots_string += self.explanation_prompt + explanation + "\n"
        shots_string += f"Attention Head {num_shots+1}:\n"

        return shots_string
   
    def _prepare_attn_head_str(self, token):
        return f'This attention head in particular helps to predict that the next token is "{token}", but it is only active in some documents and not others. '

    def _prepare_example_list_to_str(self, positive_examples, negative_examples):
        newline = "\n"
        positive_str = f'Examples where the attention head is active: """\n*{f"{newline}*".join(positive_examples)}\n"""\n'
        negative_str = f'Examples where the attention head is inactive: """\n*{f"{newline}*".join(negative_examples)}\n"""\n'
        return positive_str, negative_str

    
@dataclass
class ClassifyPromptGen:
    example: str
    token: str
    explanation_str: str
    context: str = "We are studying attention heads in a transformer architecture neural network. Each attention head looks for some particular thing in a short document.\n"
    neuron_str: str = field(init=False) # 'The attention head being studied helps to predict that the next token is "{token}", but it is only active in some documents and not others.\n'
    explanation: str = field(init=False) # "In particular, this attention head is active when the document " + explanation_str + "\n"
    instruction: str = "\nGiven the following set of documents, use the explanation sort them into two groups based on whether the attention head is active or not.\n"
    formatting: str = "Output the following format: \nExamples where the attention head is active:\n1. <example_1>\n2. <example_2>\n...\n\Examples where the attention head is inactive:\n1. <example_1>\n2. <example_2>\n...\n"

    question: str = "Is the given example an active example? (Yes/No)\n"
    # pream: str = "The example is active when the example "
    pream: str = ""



    def __post_init__(self):
        self.context = "We are studying attention heads in a transformer architecture neural network. Each attention head looks for some particular thing in a short document.\n"
        self.neuron_str = f'The attention head being studied helps to predict that the next token is "{self.token}", but it is only active in some documents and not others.\n'
        # self.explanation = "In particular, this attention head is active when the document " + self.explanation_str + "\n"
        self.explanation = self.explanation_str + "\n"

    def get_str_from_examples(self):

        # return examples_str
        return 'Example: """\n' + self.example + '\n"""'

    def get_prompt(self):
        examples_str = self.get_str_from_examples()

        # return self.context + self.neuron_str + self.explanation + self.instruction + self.formatting + examples_str
        # return self.question + self.pream + self.explanation_str.replace("\n", " ") + "\n" + examples_str + "\nAnswer: "
        return self.explanation_str.replace("\n", " ") + "\n" + self.question + "\n" + examples_str + "\nAnswer: "

        # Essentially:
        # We have ten documents, sort them into two groups based on the attention head.
        # Required Format: 
        # Examples where the attention head is active:
        # 1. <example_1>
        # 2. <example_2>
        # ...
        # Examples where the attention head is inactive:
        # 1. <example_1>
        # 2. <example_2>
        # ...
