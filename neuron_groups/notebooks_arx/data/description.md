# Relevant Files
0. neuron_finder_results.json has results of the top 10 neurons over the last 5 layers of GPT-2 Large.
1. neurons_max_act.json has results of the max-activating dataset over Neel's 10k subset, neuron_max_acts_test.json is the same but for the first 10k entries of The Pile's test set
2. neuron_20_examples_2.json, neuron_20_examples_test.json has the top 20 examples trimmed to 80% activation
3. head_attribution_dict.json, head_attribution_dict_test.json has the head attributions per example for each neuron
4. categorised_prompts_1.pkl is nh_to_pos_neg_examples, for the train set
5. head_explanation_1_prompts.json, head_explanation_1.jsonl, head_explanation_1_results.jsonl, head_explanation_1_nh_to_exp.json are the head explanations (nh_to_exp is just the formatted results)
6. text_list_test, text_list_dict_test.pkl was a hacky way to convert the Pile's 10k test prompts into a format suitable for ActivatingDataset
7. eg_classify_1, eg_classify_1_results.jsonl are the GPT-4 prompts to classify the test set
8. nh_to_results_test.json is nh -> {'tp':5, 'fp', ...}
9. nh_to_egs_results.json is nh -> {'tp': [eg1, eg2, ...]...}
10. nh_to_egs_test.json is nh -> ['examples'...]
11. nh_neurons...noabs.pkl and nh_token_prob...noabs.pkl are the ablation results