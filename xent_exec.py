import os, pathlib, regex, sys

import torch, torch.nn as nn, torch.nn.functional as F
import transformers

##############
# HOW TO RUN #
##############

# Have xent codes in homedir/xent-codes/
# Have a huggingface (pytorch) model in homedir/model-name (e.g. gpt2-xl) (or set the hf_dir_name to None below)

###############
# CONFIG VARS #
###############

default_model_name = 'gpt2-xl'
hf_dir_name = 'hf' # the hf dir path in the home folder. Set to None to download models from huggingface directly
verbose = True
xent_code_dir_name = 'xent-in'
xent_out_dir_name= 'xent-out'

####################
# XENT LANG PARAMS #
####################

xent_file_ext = '.xent'
xent_yield = ':>:>'
xent_top_sep = '::::'
xent_args_sep = ':::'
xent_list_sep = '::'

############
# ENV VARS #
############

home_dir_path = os.path.expanduser('~')
xent_code_dir_path = os.path.join(home_dir_path, xent_code_dir_name)
xent_out_dir_path = os.path.join(home_dir_path, xent_out_dir_name)

#######################
# AUXILIARY FUNCTIONS #
#######################

log = print if verbose else (lambda *args: None) # does nothing if verbose is false

def strip_strings(strings): return [string.strip() for string in strings]
def split_and_strip(string, sep): return strip_strings(string.split(sep))
def ensure_dirs_exist(dirs): return [pathlib.Path(d).mkdir(parents=True, exist_ok=True) for d in dirs]
def xcd_str(xent_code_data): return f"{xent_code_data['xsfn']} [part {xent_code_data['xci']}]" 

#####################
# LOADING FUNCTIONS #
#####################

def load_xent_source_dict():
	xent_source_file_names = sorted([xsfn for xsfn in os.listdir(xent_code_dir_path) if xsfn.endswith(xent_file_ext)]) # just keep the *.xent files
	xent_source_file_paths = [os.path.join(xent_code_dir_path, xsfn) for xsfn in xent_source_file_names]
	xent_sources = [pathlib.Path(xsfp).read_text() for xsfp in xent_source_file_paths]
	xent_source_dict = { xsfn:xs for (xsfn, xs) in zip(xent_source_file_names, xent_sources) }
	return xent_source_dict

def load_model_and_tokenizer(model_name):
	hf_dir_path = os.path.join(home_dir_path, hf_dir_name) if hf_dir_name is not None else None
	model_path = os.path.join(hf_dir_path, model_name) if hf_dir_path is not None else model_name # [will download from huggingface in case]
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
	model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
	return (model, tokenizer)

# (model, tokenizer) = load_model_and_tokenizer()
xent_source_dict = load_xent_source_dict()

#########################
# TRANSFORMER FUNCTIONS #
#########################

(default_model, default_tokenizer) = (None, None) # loaded in preload

# str -> [1, len]
def tokenize(str, tokenizer=None): 
	if tokenizer is None: tokenizer = default_tokenizer
	return tokenizer(str, return_tensors='pt').input_ids

def flat_cat(tensors, tot_dims=2):
	flat_tensors = [tensor.view(-1) for tensor in tensors] # [b_1, l_1], [b_2, l_2], ..., [b_k, l_k] (or higher-dim tensors)
	flat_cat_res = torch.cat(flat_tensors) # [L := b_1*l_1 + ... + b_k*l_k]
	for i in range(tot_dims - 1): flat_cat_res = flat_cat_res.unsqueeze(0) # adds dimensions
	return flat_cat_res # [1, ..., 1, L] # at the end: len(shape) = tot_dims


def detokenize(tokens, *, tokenizer=None, separator=' '): # [len] -> str or 
	if tokenizer is None: tokenizer = default_tokenizer
	if len(tokens.shape) == 2: tokens = tokens.squeeze(0) # if [b, t] => [t]
	t_strings = [tokenizer.decode(t) for t in tokens.tolist()]
	return separator.join(t_strings)

# [1, l], [1, rl], [1, bl] -> [1, ..., 1, l-1] (...: tot_dims-1 times) # r=red, b=blue
def xent_diff_prefix(tokens, r_prefix_tokens, b_prefix_tokens, *, model=None, tot_dims=2):
	if model is None: model = default_model
	r_tokens = flat_cat([r_prefix_tokens, tokens]) # [1, rl+l]
	b_tokens = flat_cat([b_prefix_tokens, tokens]) # [1, bl+l]
	r_logits = model(r_tokens).logits # [1, rl+l, vs]
	b_logits = model(b_tokens).logits # [1, bl+1, vs]
	r_relevant_logits = r_logits[:, r_prefix_tokens.shape[-1]:-1].squeeze(0) # [l-1, vs]
	b_relevant_logits = b_logits[:, b_prefix_tokens.shape[-1]:-1].squeeze(0) # [l-1, vs]
	relevant_tokens = tokens[0, 1:] # [l-1]
	r_xents = F.cross_entropy(r_relevant_logits, relevant_tokens, reduction='none') # [l-1]
	b_xents = F.cross_entropy(b_relevant_logits, relevant_tokens, reduction='none') # [l-1]
	xent_diffs = r_xents - b_xents # [l-1]
	for _ in range(tot_dims - 1): xent_diffs = xent_diffs.unsqueeze(0) # adds 'void' batch dimensions
	return xent_diffs

# [1, l] -> [l-1]
def xent_token_prob_entropies(tokens, *, model=None, tot_dims=2):
	""" returns the probabilities of the last l-1 tokens (the first one cannot be given an entropy, as it is not given by the model)"""
	if model is None: model = default_model
	logits = model(tokens).logits # [1, l, vs]
	relevant_logits = logits[:, :-1, :] # [1, l-1, vs]
	relevant_probs = relevant_logits.softmax(dim=-1) # [1, l-1, vs]
	relevant_entropies = -(relevant_probs * relevant_probs.log()).sum(dim=-1).squeeze(0) # [l-1]
	for _ in range(tot_dims - 1): relevant_entropies.unsqueeze(0)
	return relevant_entropies

def xent_token_xents(tokens, *, model=None, tot_dims=2):
	""" returns the tokens' cross-entropies (for the last l-1 tokens, as there is no prediction of the first one) """
	if model is None: model = default_model
	logits = model(tokens).logits # [1, l, vs]
	relevant_logits = logits[0, :-1, :] # [l-1, vs]
	relevant_tokens = tokens[0, 1:] # [l-1]
	return F.cross_entropy(relevant_logits, relevant_tokens, reduction='none') # [l-1]

##################
# XENT CODE EXEC #
##################

def exec_xent_source(*, xent_source, xsfn): 
	""" Generates a synthetic data piece based on the code in xent_source """
	xent_codes = split_and_strip(xent_source, xent_yield)
	xc_outputs = [exec_xent_code(xent_code=xc, xent_code_data=dict(xsfn=xsfn, xci=xci)) for (xci, xc) in enumerate(xent_codes)]
	xent_out_data = (xent_yield + '\n').join(xc_outputs)
	xent_out_file_name = xsfn.removesuffix(xent_file_ext) + '.out' + xent_file_ext
	xent_out_file_path = os.path.join(xent_out_dir_path, xent_out_file_name) 
	pathlib.Path(xent_out_file_path).write_text(xent_out_data)

def exec_xent_code(*, xent_code, xent_code_data):
	""" Executes a xent code section, i.e. a program (there can be several sections in a xent source file) """
	xent_sections = split_and_strip(xent_code, xent_top_sep) # top_sep: ::::
	if len(xent_sections) == 2: 
		return exec_xent_instruct(xent_data=xent_sections[0], xent_instruct=xent_sections[1], xent_code_data=xent_code_data)
	else: 
		return xent_code 

def exec_xent_instruct(*, xent_data, xent_instruct, xent_code_data): # xent_code contains everything... just useful as a reference
	""" returns the xent_data followed by xent_instruct followed by the output """
	global xent_funct_dict
	xisplit = split_and_strip(xent_instruct, xent_args_sep) # args_sep: :::
	(xent_funct_name, xent_args) = (xisplit[0], xisplit[1:])
	if xent_funct_name in xent_funct_dict: output = xent_funct_dict[xent_funct_name](xent_args, xent_data, xent_code_data) 
	else: output = f"function {xent_funct_name} not implemented"
	print(f"\n{xcd_str(xent_code_data)}:\n\t", output, "\n")
	return f"{xent_data}\n{xent_top_sep}\n{xent_instruct}\n{xent_yield}\n{output}" 

# performs a greedy key search {?} should probably be done by regex
def get_arg_text(xent_arg, xent_data): 
	xent_pointer_keys = split_and_strip(xent_arg, xent_list_sep)
	if len(xent_pointer_keys) <= 1: return xent_arg
	# otherwise, we have a xent arg of the form { beg_key } :: { end_key }
	(beg_key, end_key) = (xent_pointer_keys[0], xent_pointer_keys[1])
	beg_index = xent_data.rfind(beg_key) 
	if beg_index == -1: beg_index = 0 # if the key is not found 
	end_index = xent_data.find(end_key, beg_index) 
	if end_index >= 0: end_index += len(end_key) # the end_key is also inclusive
	else: end_index = len(xent_data) # we just finish at the end of the text
	return xent_data[beg_index:end_index]


##################################
# XENT INSTRUCT [FUNCTIONS] CODE #
##################################

def xent_diff_prefix_ranked(xent_args, xent_data, xent_code_data): 
	""" xent_data :::: diff-prefix-ranked ::: text ::: r_prefix ::: b_prefix :>:> xent_top_b_tokens ::: xent_top_r_tokens """
	xent_arg_texts = [get_arg_text(xent_arg, xent_data) for xent_arg in xent_args]
	if (n_args := len(xent_arg_texts)) < 3: return log(f"{xcd_str(xent_code_data)}: {n_args=} while the minimum value is 3 (text, prefix_1, prefix_2)")
	(text, r_prefix, b_prefix) = (xent_arg_texts[0], xent_arg_texts[1], xent_arg_texts[2])

	log(f"Should compute the xent diff with \n\nred_prefix: \n\t{r_prefix} \n\nblue_prefix: \n\t{b_prefix}\n\ntext: \n\t{text}")

	(tokens, r_prefix_tokens, b_prefix_tokens) = [tokenize(_) for _ in [text, r_prefix, b_prefix]] # [1, l] [1, rl], [1, bl]
	xent_diffs = xent_diff_prefix(tokens, r_prefix_tokens, b_prefix_tokens, tot_dims=1) # [l - 1] (first token removed)
	relevant_tokens = tokens.squeeze(0)[1:] # [k_top_k] (we get rid of the first element because the first element does not have a xent_diff

	k_top_k = min(6, xent_diffs.shape[0]) # {?} hard-coded number
	xent_top_r_indices = (-xent_diffs).topk(k_top_k).indices # [k_top_k] 
	xent_top_b_indices = (+xent_diffs).topk(k_top_k).indices  # [k_top_k]

	xent_top_r_tokens = relevant_tokens[xent_top_r_indices] # [k_top_k] 
	xent_top_b_tokens = relevant_tokens[xent_top_b_indices] # [k_top_k] 

	r_output = detokenize(xent_top_r_tokens, separator=f' {xent_list_sep} ')
	b_output = detokenize(xent_top_b_tokens, separator=f' {xent_list_sep} ')

	output = r_output + f' {xent_args_sep} ' + b_output
	return output

def xent_token_prob_entropy_ranked(xent_args, xent_data, xent_code_data):
	""" xent_data :::: token-prob-entropy-ranked ::: text :>:> min_entropy_indices ::: max_entropy_indices """
	log('Compute the min and max token probability entropies')
	
	xent_arg_texts = [get_arg_text(xent_arg, xent_data) for xent_arg in xent_args]
	if (n_args := len(xent_arg_texts)) < 1: return log(f"{xcd_str(xent_code_data)}: {n_args=} while the minimum value is 1 (text)")
	text = xent_arg_texts[0]
	tokens = tokenize(text) # [1, l]
	if tokens.shape[1] < 2: return log(f"Text has only {tokens.shape[1]} token: cannot compute entropies")
	token_entropies = xent_token_prob_entropies(tokens, tot_dims=1) # [l - 1]
	k_top_k = min(6, token_entropies.shape[0]) # {?} hard-coded
	relevant_tokens = tokens[:, 1:].squeeze(0) # [l - 1] # we don't say anything about the first token
	min_entropy_indices = (-token_entropies).topk(k_top_k).indices # [l - 1]
	max_entropy_indices = (+token_entropies).topk(k_top_k).indices # [l - 1]
	min_entropy_tokens = relevant_tokens[min_entropy_indices] # [l - 1]
	max_entropy_tokens = relevant_tokens[max_entropy_indices] # [k_top_k]
	min_output = detokenize(min_entropy_tokens, separator=f' {xent_list_sep} ')
	max_output = detokenize(max_entropy_tokens, separator=f' {xent_list_sep} ')
	output =  f'{min_output} {xent_args_sep} {max_output}' 
	return output

def xent_token_xents_ranked(xent_args, xent_data, xent_code_data):
	""" xent_data :::: token-prob-entropy-ranked ::: text :>:> min_entropy_indices ::: max_entropy_indices """
	log('Compute the min and max token empirical cross-entropies')

	xent_arg_texts = [get_arg_text(xent_arg, xent_data) for xent_arg in xent_args]
	if (n_args := len(xent_arg_texts)) < 1: return log(f"{xcd_str(xent_code_data)}: {n_args=} while the minimum value is 1 (text)")
	text = xent_arg_texts[0]
	tokens = tokenize(text) # [1, l]
	if tokens.shape[1] < 2: return log(f"Text has only {tokens.shape[1]} token: cannot compute entropies")
	token_xents = xent_token_xents(tokens, tot_dims=1) # [l - 1]
	relevant_tokens = tokens[0, 1:] # [l - 1]
	k_top_k = min(6, token_xents.shape[0]) # {?} hard-coded
	min_xent_indices = (-token_xents).topk(k_top_k).indices # [k_top_k]
	max_xent_indices = (+token_xents).topk(k_top_k).indices # [k_top_k]
	min_xent_tokens = relevant_tokens[min_xent_indices] # [k_top_k]
	max_xent_tokens = relevant_tokens[max_xent_indices] # [k_top_k]
	min_output = detokenize(min_xent_tokens, separator=f' {xent_list_sep} ')
	max_output = detokenize(max_xent_tokens, separator=f' {xent_list_sep} ')

	output = f'{min_output} {xent_args_sep} {max_output}'

	return output
 

# The prototype of the functions should be (xent_args, xent_data, xent_source_name)
xent_funct_dict = {	'diff-prefix-ranked': xent_diff_prefix_ranked, 
					'token-prob-entropy-ranked': xent_token_prob_entropy_ranked,
					'token-xents-ranked': xent_token_xents_ranked } 

################
# PRELOAD CODE #
################

ensure_dirs_exist([xent_code_dir_path, xent_out_dir_path])
(default_model, default_tokenizer) = load_model_and_tokenizer(default_model_name)


#############
# MAIN CODE #
#############

for (xsfn, xent_source) in xent_source_dict.items(): exec_xent_source(xent_source=xent_source, xsfn=xsfn)


