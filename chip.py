import copy
import re

from modules import shared, chat

from modules.text_generation import get_encoded_length, get_max_prompt_length
from modules.extensions import apply_extensions
from modules.chat import get_generation_prompt

import torch
import random

from modules.exllamav2 import Exllamav2Model

from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.model import _torch_device, ExLlamaV2
from exllamav2.compat import safe_move_tensor
from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
)

from jinja2.sandbox import ImmutableSandboxedEnvironment
jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
from functools import partial

from extensions.brainhackingchip.settings_classes import HackingchipSettings

# Override functions to inject hackingchip behavior into model loaders. These functions need to be kept up to date with oobabooga's exllamav2

def hijack_generate_with_streaming(self, prompt, state):
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = state['temperature']
    settings.top_k = state['top_k']
    settings.top_p = state['top_p']
    settings.min_p = state['min_p']
    settings.tfs = state['tfs']
    settings.typical = state['typical_p']
    settings.mirostat = state['mirostat_mode'] == 2
    settings.mirostat_tau = state['mirostat_tau']
    settings.mirostat_eta = state['mirostat_eta']
    settings.token_repetition_penalty = state['repetition_penalty']
    settings.token_repetition_range = -1 if state['repetition_penalty_range'] <= 0 else state['repetition_penalty_range']
    if state['ban_eos_token']:
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            settings.disallow_tokens(self.tokenizer, to_ban)
            
    hackingchip = self.generator.model.hackingchip if hasattr(self.generator.model, 'hackingchip') else None
    if hackingchip:
        if hackingchip.prompts.batch_size != self.cache.batch_size: # the hackingchip tends to have extra batches, so it's time to prepare for that
            # I'm not correctly deleting the existing cache, but it gets removed from VRAM somehow anyway
            
            if shared.args.cache_8bit:
                self.cache = ExLlamaV2Cache_8bit(self.model, hackingchip.prompts.batch_size)
            else:
                self.cache = ExLlamaV2Cache(self.model, hackingchip.prompts.batch_size)

            self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)

    ids = self.tokenizer.encode(hackingchip.prompts.batch_prompts if hasattr(hackingchip.prompts, 'batch_prompts') else prompt, add_bos=state['add_bos_token'], encode_special_tokens=True)
    ids = ids[:, -get_max_prompt_length(state):]

    if state['auto_max_new_tokens']:
        max_new_tokens = state['truncation_length'] - ids.shape[-1]
    else:
        max_new_tokens = state['max_new_tokens']

    self.generator.begin_stream(ids, settings, loras=self.loras)
    
    # If generator gets reconstructed above, my attempt to overwrite this function from earlier will be reset. So I'm setting the function here.
    self.generator._gen_single_token = hijack_gen_single_token.__get__(shared.model.generator, ExLlamaV2StreamingGenerator)

    decoded_text = ''
    for i in range(max_new_tokens):
        chunk, eos, _ = self.generator.stream()
        if eos or shared.stop_everything:
            if hasattr(self.generator.model, 'hackingchip'): del self.generator.model.hackingchip # remove hackingchip after use, just in case
            # TODO: I realized I probably should return the functions back to normal too, have to store and retrieve to do so
            break

        decoded_text += chunk
        yield decoded_text
        
def hijack_gen_single_token(self, gen_settings, prefix_token = None):
    hackingchip = self.model.hackingchip if hasattr(self.model, 'hackingchip') else None
    
    batch_token = None

    if self.draft_model is None:

        logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, loras = self.active_loras).float().cpu()
        
        if hackingchip: # the hackingchip starts with extra batches that then are not sampled, so it's time to fix that
            logits = logits[0].unsqueeze(0)
            samplerids = self.sequence_ids[0].unsqueeze(0)
        else:
            samplerids = self.sequence_ids
        
        token, _, eos = ExLlamaV2Sampler.sample(logits, gen_settings, samplerids, random.random(), self.tokenizer, prefix_token)
        
        # only sampled one token, but need to expand out to the ids size to put it in all other batches, can't replace token because that will break stream
        batch_token = token.expand(self.sequence_ids.size(0), -1)

    else:

        token, eos = self._gen_single_token_speculative(gen_settings, prefix_token)

    self.sequence_ids = torch.cat([self.sequence_ids, batch_token if batch_token is not None else token], dim = 1)
    gen_settings.feed_filters(token)
    return token, eos

@torch.inference_mode()
def hijack_forward(self,
                input_ids,
                cache = None,
                input_mask = None,
                preprocess_only = False,
                last_id_only = False,
                loras = None,
                return_last_state = False,
                position_offsets = None):

    batch_size, seq_len = input_ids.shape
    past_len = 0
    if cache is not None:
        if isinstance(cache, ExLlamaV2CacheBase):
            past_len = cache.current_seq_len
        else:
            pl = [c.current_seq_len for c in cache]
            past_len = torch.tensor(pl, dtype = torch.int)
            past_len = (past_len, past_len)

    # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

    x = input_ids
    prev_device = None
    attn_mask = None
    last_state = None
    
    hackingchip = self.hackingchip if hasattr(self, 'hackingchip') else None

    for idx, module in enumerate(self.modules):

        device = _torch_device(module.device_idx)

        # Build attention mask

        if device != prev_device and device != "cpu":

            prev_device = device
            attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, input_mask, device)
            if isinstance(past_len, tuple): past_len = (safe_move_tensor(past_len[0], device), past_len[1])
            if position_offsets is not None: position_offsets = safe_move_tensor(position_offsets, device)

        # Onward

        if idx == self.head_layer_idx:
            if last_id_only and return_last_state:
                x = x.narrow(-2, -1, 1)
                last_state = x
            elif last_id_only:
                x = x.narrow(-2, -1, 1)
            elif return_last_state:
                last_state = x.narrow(-2, -1, 1)

        x = safe_move_tensor(x, device)
        x = module.forward(x, cache = cache, attn_mask = attn_mask, past_len = past_len, loras = loras, position_offsets = position_offsets)
        
        if hackingchip and hackingchip.settings.on and hackingchip.prompts.numneg > 0 and hackingchip.settings.layer_settings[idx] != None:
            settings = hackingchip.settings.layer_settings[idx]
            
            x_neg_orig = x[hackingchip.prompts.numpos:hackingchip.prompts.numpos+hackingchip.prompts.numneg, :, :]
            x_neg_steering = torch.mean(x_neg_orig, dim=0, keepdim=False) # probably not the best way to handle this but oh well
            x_neg_steering = settings.weight * (x_neg_steering - x[0])

            # It's important to steer all of the vectors, or else the difference artificially accumulates and accelerates.
            x -= x_neg_steering
        
        if preprocess_only and idx == self.last_kv_layer_idx:
            x = None
            break

    # Advance cache

    if cache is not None:
        if isinstance(cache, list):
            for c in cache: c.current_seq_len += seq_len
        else:
            cache.current_seq_len += seq_len

    # Set padding logits to -inf

    if x is not None:
        head_padding = self.modules[-1].padding
        if head_padding > 0:
            x[:, :, -head_padding:] = -65504.

    return x, last_state

# Here is the actual construction and injection of the hackingchip into the model

class Hackingchip:
    def __init__(self, settings, prompts):
        self.settings = settings
        self.prompts = prompts
        
class HackingchipPrompts:
    def __init__(self, prompts, numpos, numneg):
        self.batch_prompts = prompts
        self.numpos = numpos
        self.numneg = numneg
        self.batch_size = numpos + numneg
        
def gen_full_prompt(user_settings, user_input, state, **kwargs):
    if shared.model != None and isinstance(shared.model, Exllamav2Model): # hackingchippable
        shared.model.generate_with_streaming = hijack_generate_with_streaming.__get__(shared.model, Exllamav2Model)
        # shared.model.generator._gen_single_token = hijack_gen_single_token.__get__(shared.model.generator, ExLlamaV2StreamingGenerator)
        # The above line isn't working here, had to relocate it
        shared.model.generator.model._forward = hijack_forward.__get__(shared.model.generator.model, ExLlamaV2)
        
        last_kv_layer = shared.model.generator.model.last_kv_layer_idx
        head_layer = shared.model.generator.model.head_layer_idx
        
        settings = user_settings.brainhackingchip_settings(HackingchipSettings(head_layer), last_kv_layer, head_layer) # prepare hackingchip settings
        baseprompt, prompts = gen_full_prompt2(user_input, state, **kwargs) # prepare hackingchip prompts
        
        hackingchip = Hackingchip(settings, prompts)
        shared.model.generator.model.hackingchip = hackingchip # hackingchip installed
        
        if hackingchip.settings.output_prompts:
            print("Hackingchip prompts:")
            for prompt in hackingchip.prompts.batch_prompts:
                print(prompt)
        
        return baseprompt
    else:
        # Should I warn the user that they aren't able to use hackingchip with their current model loader? Or would that be annoying?
        print("Unsupported model loader: Brain-Hacking Chip won't work with it")
        return chat.generate_chat_prompt(user_input, state, **kwargs)
            
            
            
# I wrote the below code awhile ago as a quick hack to get things working then shoddily reworked it multiple times, please forgive me for my sins

# The code below prepares the prompts using the [[POSITIVE]] and [[NEGATIVE]] tags
    
def gen_full_prompt2(user_input, state, **kwargs): # modifies hackingchip and state
    # global last_prompt
    # global last_chip
    
    prompt = None
    
    # custom prompt generation stuff could go here

    # if prompt is not None and kwargs.get('_continue', False):
    #     prompt = last_prompt
    #     hackingchip = last_chip
        
    numpos = 1
    numneg = 0
                
    if prompt is None:
        positive_context, negative_context, positive_context_extras, negative_context_extras = process_context(state['context'])
        positive_context_instruct, negative_context_instruct, positive_context_instruct_extras, negative_context_instruct_extras = process_context(state['custom_system_message'])
        
        if len(negative_context) > 0 and len(negative_context_instruct) == 0:
            negative_context_instruct = positive_context_instruct
        
        positive_extras = {}
        negative_extras = {}
        
        for name, text in positive_context_extras.items():
            if name not in positive_extras:
                positive_extras[name] = ExtraInfo()
                positive_extras[name].inst = positive_context_instruct
            positive_extras[name].char = text
            
        for name, text in positive_context_instruct_extras.items():
            if name not in positive_extras:
                positive_extras[name] = ExtraInfo()
                positive_extras[name].char = positive_context
            positive_extras[name].inst = text
            
        for name, text in negative_context_extras.items():
            if name not in negative_extras:
                negative_extras[name] = ExtraInfo()
                negative_extras[name].inst = negative_context_instruct
            negative_extras[name].char = text
            
        for name, text in negative_context_instruct_extras.items():
            if name not in negative_extras:
                negative_extras[name] = ExtraInfo()
                negative_extras[name].char = negative_context
            negative_extras[name].inst = text
            
        state['context'] = positive_context
        state['custom_system_message'] = positive_context_instruct
        posprompt = generate_chat_prompt(user_input, state, **kwargs)
        prompt = [posprompt]

        if positive_extras:
            for name, extras in positive_extras.items():
                state['context'] = extras.char
                state['custom_system_message'] = extras.inst
                prompt.append(generate_chat_prompt(user_input, state, **kwargs))
                numpos += 1

        if negative_extras:
            for name, extras in negative_extras.items():
                state['context'] = extras.char
                state['custom_system_message'] = extras.inst
                prompt.append(generate_chat_prompt(user_input, state, **kwargs))
                numneg += 1
        
        if len(negative_context) + len(negative_context_instruct) > 0:
            state['context'] = negative_context
            state['custom_system_message'] = negative_context_instruct
            prompt.append(generate_chat_prompt(user_input, state, **kwargs))
            numneg += 1
            
        state['context'] = positive_context
        state['custom_system_message'] = positive_context_instruct
        
        prompt_info = HackingchipPrompts(prompt, numpos, numneg)
        
        # TODO: load the default negative cfg here in state for convenience
        
        prompt = posprompt # for compatibility
    else:
        positive, negative, positive_extras, negative_extras = process_context(prompt)

        prompt = [positive]
        
        if positive_extras:
            prompt.append(positive_extras)
            numpos += positive_extras.len()

        if len(negative) > 0:
            prompt.append(negative)
            numneg += 1
            
        if negative_extras:
            prompt.append(negative_extras)
            numneg += negative_extras.len()

        prompt_info = HackingchipPrompts(prompt, numpos, numneg)
        
        prompt = positive
        
    # If the user hits continue, this is how I can continue without breaking things... although I haven't completed this currently
    # Actually, this is deprecated now, commented out all related code and will probably remove
    # last_prompt = prompt
    # last_chip = copy.deepcopy(hackingchip)
    
    # TODO <|nochat|> may be bad for continue as is, because it may also exclude the AI's output so far
    # I'm sure it can be fixed if it's busted, need to look into it
        
    for oneprompt in prompt:
        if '<|nochat|>' in oneprompt:
            oneprompt = oneprompt.replace('<|nochat|>', '').replace('<|chat|>', '')
            
    return prompt, prompt_info
            
class ExtraInfo:
    def __init__(self):
        self.char = ''
        self.inst = ''
        
def process_context(context):
    pattern = r"(?:\[\[(?P<name>[^\]]+)\]\]\n)?(?P<text>(?:(?!\n\[\[[^\]]+\]\]\n).)*)"
    
    if not re.match(r'^\[\[(.*?)\]\]\n', context): context = "[[SHARED]]\n" + context
    
    matches = re.finditer(pattern, context, re.DOTALL)
    
    regions = {}
    
    positive = ''
    negative = ''
    
    positive_extras = {}
    negative_extras = {}

    for match in matches:
        if match.group("name") is not None:
            name = match.group("name").upper().strip()
            text = match.group("text") # don't strip
            
            if name.startswith('POSITIVE') and name != 'POSITIVE':
                positive_extras[name] = text
            if name.startswith('NEGATIVE') and name != 'NEGATIVE':
                negative_extras[name] = text
            
            regions[name] = text
        
    if 'POSITIVE' in regions:
        positive = regions['POSITIVE']
    elif positive_extras:
        name, text = positive_extras.popitem()
        positive = text
    else:
        if 'SHARED' in regions:
            positive = regions['SHARED']
        else:
            positive = context # maybe not good?
    
    if 'NEGATIVE' in regions:
        negative = regions['NEGATIVE']
    elif negative_extras:
        name, text = negative_extras.popitem()
        negative = text
    
    for name, text in regions.items():
        positive = positive.replace('{{' + name + '}}', text)
        negative = negative.replace('{{' + name + '}}', text)
        for name2, text2 in positive_extras.items():
            positive_extras[name2] = text2.replace('{{' + name + '}}', text)
        for name2, text2 in negative_extras.items():
            negative_extras[name2] = text2.replace('{{' + name + '}}', text)
       
    return positive, negative, positive_extras, negative_extras
        
# Just copying the entirety of generate_chat_prompt so I can put <|nochat|> support in it

def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history = kwargs.get('history', state['history'])['internal']

    # Templates
    chat_template = jinja_env.from_string(state['chat_template_str'])
    instruction_template = jinja_env.from_string(state['instruction_template_str'])
    chat_renderer = partial(chat_template.render, add_generation_prompt=False, name1=state['name1'], name2=state['name2'])
    instruct_renderer = partial(instruction_template.render, add_generation_prompt=False)

    messages = []

    if state['mode'] == 'instruct':
        renderer = instruct_renderer
        if state['custom_system_message'].strip() != '':
            messages.append({"role": "system", "content": state['custom_system_message']})
    else:
        renderer = chat_renderer
        if state['context'].strip() != '':
            messages.append({"role": "system", "content": state['context']})

    insert_pos = len(messages)
    for user_msg, assistant_msg in reversed(history):
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip()

        if assistant_msg:
            messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            messages.insert(insert_pos, {"role": "user", "content": user_msg})

    user_input = user_input.strip()
    if user_input and not impersonate and not _continue:
        messages.append({"role": "user", "content": user_input})

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

            command = state['chat-instruct_command']
            command = command.replace('<|character|>', state['name2'] if not impersonate else state['name1'])
            command = command.replace('<|prompt|>', prompt)

            if _continue:
                prefix = get_generation_prompt(renderer, impersonate=impersonate, strip_trailing_spaces=False)[0]
                prefix += messages[-1]["content"]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)
                    
            if '<|nochat|>' not in user_input:
                outer_messages.append({"role": "user", "content": command})
                outer_messages.append({"role": "assistant", "content": prefix})

            prompt = instruction_template.render(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            prompt = prompt[:-len(suffix)]

        else:
            if _continue:
                suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                prompt = prompt[:-len(suffix)]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if state['mode'] == 'chat' and not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

                prompt += prefix

        return prompt

    prompt = make_prompt(messages)

    # Handle truncation
    max_length = get_max_prompt_length(state)
    while len(messages) > 0 and get_encoded_length(prompt) > max_length:
        # Try to save the system message
        if len(messages) > 1 and messages[0]['role'] == 'system':
            messages.pop(1)
        else:
            messages.pop(0)

        prompt = make_prompt(messages)

    if also_return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt
                    