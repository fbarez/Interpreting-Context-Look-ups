from src.datahandlers import ExplanationPromptGen, ClassifyPromptGen

def generate_classify_prompt_dict(nh_to_exp, neuron_to_prompts, neuron_to_token, max_per_nh=10):
    gpt_4_prompts_dict = {}

    # for neuron in neurons:
    for nh in nh_to_exp.keys():
        neuron = (nh[0], nh[1])
        token = neuron_to_token[neuron]
        trunc_prompts = neuron_to_prompts[neuron]
        trunc_prompts = [trunc_prompt + token for trunc_prompt in trunc_prompts]
        explanation=nh_to_exp[nh]

        classify_prompts = {}

        for i, trunc_prompt in enumerate(trunc_prompts):
            if i>=max_per_nh:
                break
            prompt_gen = ClassifyPromptGen(
                example=trunc_prompt,
                token=token,
                explanation_str=explanation
            ) # This wants an example, a token, and an explanation str
            prompt = prompt_gen.get_prompt()
            classify_prompts[prompt] = trunc_prompt

        gpt_4_prompts_dict[nh] = classify_prompts

    return gpt_4_prompts_dict

# Later on, I'll need to have classify prompt -> nh. But classify prompts aren't unique

def generate_explanation_prompt_dict(head_attribution_dict, neurons, neuron_to_token):
    gpt_4_prompts_dict = {}
    nh_to_pos_neg_examples = {}

    for neuron in neurons:
        if neuron != (31, 364):
            continue
    # if True: 
    #     neuron = neurons[3]
        token = neuron_to_token[neuron]
        trunc_prompts = list(head_attribution_dict[neuron].keys())
        trunc_prompts = [trunc_prompt + token for trunc_prompt in trunc_prompts] # Add back the token -- experimential
        top_heads = list(head_attribution_dict[neuron].values())

        num_prompts = len(top_heads)
        print(num_prompts)
        from collections import Counter
        flattened_heads = [x for y in top_heads for x in y]
        head_count = Counter(flattened_heads)
        PERCENT_TO_KEEP = 0.25
        print(head_count)
        relevant_heads = [x for x in head_count if head_count[x] > PERCENT_TO_KEEP * num_prompts]
        print(relevant_heads)
        return
        for head in relevant_heads:

            positive_examples = []
            negative_examples = []
            for i, example in enumerate(trunc_prompts):
                if head in top_heads[i]:
                    positive_examples.append(trunc_prompts[i])
                else:
                    negative_examples.append(trunc_prompts[i])

            # prompt_gen = ExplanationPrompt(neuron, trunc_prompts, top_heads, neuron_to_token, shots_dict)
            prompt_gen = ExplanationPromptGen(token, positive_examples, negative_examples)
            prompt = prompt_gen.get_prompt(shots=0)
            gpt_4_prompts_dict[(*neuron, head)] = prompt
            nh_to_pos_neg_examples[(*neuron, head)] = (positive_examples, negative_examples)

    return gpt_4_prompts_dict, nh_to_pos_neg_examples

