import json
from datasets import load_dataset
from dataclasses import dataclass, field

class ActivatingDataset:
    def __init__(self, json_file, dataset=None):
        if dataset is None:
            self.data = load_dataset("NeelNanda/pile-10k", split="train")
        else:
            self.data = dataset
        with open(json_file, "r") as f:
            self.markers = json.load(f)

        # Convert keys to tuple if they are string
        def _neuron_str_to_tuple(string):
            if type(string) == str:
                return tuple([int(x) for x in string[1:-1].split(", ")])
            else:
                return string

        self.markers = {_neuron_str_to_tuple(key): value for key, value in self.markers.items()}

    def remove_prompts_longer_than(self, length=100):
        for neuron in self.markers:
            self.markers[neuron] = [prompt for prompt in self.markers[neuron] if prompt['end']-prompt['start'] < length]

    def load_truncated_prompts(self, model, neuron, replace_newlines_with_spaces=True):

        prompt_indices = [prompt['index'] for prompt in self.markers[neuron]]
        prompts = [self.data[prompt_index]['text'] for prompt_index in prompt_indices]

        prompts_tokens = [model.to_tokens(prompt, prepend_bos=False) for prompt in prompts]
        prompts_tokens_trunc = []
        for i, prompt_tokens in enumerate(prompts_tokens):
            start, end = self.markers[neuron][i]['start'], self.markers[neuron][i]['end']
            trunc_prompt_token = prompt_tokens[:, start:end]

            prompts_tokens_trunc.append(trunc_prompt_token)


        trunc_prompts = [model.to_string(prompt_tokens)[0] for prompt_tokens in prompts_tokens_trunc]
        if replace_newlines_with_spaces:
            trunc_prompts = [prompt.replace("\n", " ") for prompt in trunc_prompts]
        return trunc_prompts
    
class ExplanationPrompt:
    def __init__(self, neuron, trunc_prompts, top_heads, neuron_to_token, shots_dict):
        # neuron is a tuple of length 2, (layer, head)
        # trunc_prompts is a list of strings, each string is a truncated prompt
        # top_heads is a list of lists, where each inner list is a list of top 3 attention heads that are active for that prompt
        # neuron_to_token is a dictionary mapping neurons to tokens
        # context is a string that is prepended to the prompt

        CONTEXT = "We are studying attention heads in a transformer architecture neural network. Each attention head looks for some particular thing in a short document.\n"
        EXPLANATION = "Explanation: This attention head is active when the document"
        self.neuron = neuron
        self.trunc_prompts = trunc_prompts
        self.neuron_to_token = neuron_to_token
        self.top_heads = top_heads
        self.context = CONTEXT
        self.explanation_prompt = EXPLANATION
        self.shots_dict = shots_dict

    def get_prompt(self, head, shots=0):
        positive_examples, negative_examples = self._get_positive_and_negative_examples(head, self.trunc_prompts, self.top_heads)
        example_string = self.prepare_examples(positive_examples, negative_examples, token=self.neuron_to_token[self.neuron])

        if shots == 0:
            shots_string = ""
        else:
            shots_string = self.prepare_shots(shots)

        return self.context + shots_string + example_string + self.explanation_prompt
        # This gives
        # CONTEXT
        # <the main string>

    def prepare_shots(self, num_shots):
        # Shots Dict is {token: [[positive_examples], [negative_examples], explanation]}
        shots_string = ""
        for i, (token, (positive_examples, negative_examples, explanation)) in enumerate(self.shots_dict.items()):
            shots_string += f"\nAttention Head {i+1}\n"
            if i >= num_shots:
                break
            shots_string += self.prepare_one_shot_example(token, positive_examples, negative_examples, explanation)

        shots_string += f"\nAttention Head {num_shots+1}\n"
        
        return shots_string

    def prepare_one_shot_example(self, token, positive_examples, negative_examples, explanation):
        example_string = self.prepare_examples(positive_examples, negative_examples, token=token)
        return example_string + explanation + "\n"
        # This gives 
        # <the main string>
        # <the explanation string>


    def _get_positive_and_negative_examples(self, head, trunc_prompts, top_heads):
        """This prepares the positive and negative examples for a given attention head
        Positive and negative examples are a list of strings, where each string is a truncated prompt"""
        positive_examples = []
        negative_examples = []
        for i, example in enumerate(trunc_prompts):
            if head in top_heads[i]:
                positive_examples.append(trunc_prompts[i])
            else:
                negative_examples.append(trunc_prompts[i])
        return positive_examples, negative_examples

    def prepare_examples(self, positive_examples, negative_examples, token):
        """This prepares the positive and negative examples for a given attention head.
        It also adds a string that explains the token that is going to be predicted next.
        """
        newline = "\n"
        if token is not None:
            neuron_string = f'This attention head in particular helps to predict that the next token is "{token}", but it is only active in some documents and not others. Look at the documents and explain what makes the attention head active, taking into consideration the inactive examples.\n'
        else:
            neuron_string = ""
        example_string = \
            f'Examples where the attention head is active: """\n*{f"{newline}*".join(positive_examples)}\n"""\nExamples where the attention head is inactive: """\n*{f"{newline}*".join(negative_examples)}\n"""\n'
        return neuron_string + example_string

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
    explanation_prompt: str = "Explanation: This attention head is active when the document"
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
    
class ExplanationShots:
    def __init__(self, file_path):
        self.shots_dict = json.loads(open(file_path).read())

    def __getitem__(self):
        pass

class GenerationPrompt:
    def __init__(self, explanation_str, token):
        self.context = "We are studying attention heads in a transformer architecture neural network. Each attention head looks for some particular thing in a short document.\n"
        self.neuron_str = f'The attention head being studied helps to predict that the next token is "{token}", but it is only active in some documents and not others.\n'
        self.explanation = "In particular, this attention head is active when the document " + explanation_str + "\n"
        self.explanation_str = explanation_str
        self.token = token

    def get_prompt(self, num_examples=10, generate_negative=False):
        response_str = ""
        response_str += self.context
        response_str += self.neuron_str
        response_str += 'Explanation: """\n' + self.explanation + '"""\n'
        response_str += f"With this explanation, generate {num_examples} examples of documents that activate this attention head"
        if generate_negative:
            response_str += f", and {num_examples} examples of documents that do not activate this attention head.\n"
        else:
            response_str += ".\n"
        response_str += f'The documents should be 2 sentences long, and the token "{self.token}" should appear in the second sentence.\n'
        response_str += f"\nDesired Format:\nExamples where the attention head is active:\n1. <example_1>\n2. <example_2>\n...\n\Examples where the attention head is inactive:\n1. <example_1>\n2. <example_2>\n...\n"
        return response_str
    

@dataclass
class IterationPromptGen:
    active_examples_generated: list
    inactive_examples_generated: list
    active_examples_corrected: list
    inactive_examples_corrected: list
    token: str
    explanation: str

    prob_context: str = "The following solutions are the output of a Bayesian reasoner which is optimized to explain the function of attention heads in a neural network using limited evidence. Each attention head looks for some particular thing in a short passage.\n"
    attn_context: str = field(init=False) # 'The attention head being studied helps to predict that the next token is "{token}", but it is only active in some documents and not others.\n'
    current_explanation: str = "The current explanation is: This attention head is active when the document "
    original_category: str = "With the explanation, the reasoner categorises the following examples of documents that activate this attention head:\n"
    new_evidence: str = "\nThe reasoner receives the following new evidence. The examples were ran through the model to determine whether the attention head was active or not. Here are the correct categories for each example:\n"
    revision_str: str = "\nIn light of the new evidence, the reasoner revises the current explanation to: This attention head is active when the document <explanation>"

    def __post_init__(self):
        self.attn_context = f'The reasoner is trying to revise the explanation for an attention head that helps the model to predict that the next token is "{self.token}".\n'

    def get_str_from_categories(self, active_examples, inactive_examples):
        newline = "\n"
        active_str = f'Examples where the attention head is active: """\n*{f"{newline}*".join(active_examples)}\n"""\n'
        inactive_str = f'Examples where the attention head is inactive: """\n*{f"{newline}*".join(inactive_examples)}\n"""\n'
        return active_str, inactive_str
    
    def get_prompt(self):
        active_str, inactive_str = self.get_str_from_categories(self.active_examples_generated, self.inactive_examples_generated)
        active_str_corrected, inactive_str_corrected = self.get_str_from_categories(self.active_examples_corrected, self.inactive_examples_corrected)
        return self.prob_context + self.attn_context + self.current_explanation + "<explanation>" + self.explanation + "</explanation>" "\n" + self.original_category + active_str + inactive_str + self.new_evidence + active_str_corrected + inactive_str_corrected + self.revision_str