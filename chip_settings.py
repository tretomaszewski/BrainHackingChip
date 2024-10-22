import torch
from extensions.BrainHackingChip.settings_classes import LayerSettings, AttnSettings, AttnQSettings, AttnKSettings, AttnVSettings

def brainhackingchip_settings(chip, last_kv_layer, head_layer):
    """
    Layer count is head_layer + 1
    layer_settings indices range from 0 to head_layer
    layer_settings[last_kv_layer - 1] is the last layer that will influence the preprocessed cache (because of the order this is done)
    layer_settings[head_layer] is the last index and will output logits rather than a hidden state
    
    How the CFG works:
    1. The difference between the tensors for negative prompt - positive prompt is found
    2. That difference is multiplied by the weight in LayerSettings
    3. That difference is subtracted from the tensors (steering positive away from negative by their difference)
    The larger the weight, the more intense of an effect
    
    
    
    There is also support for a custom CFG function instead of the default above, by setting cfg_func in any layer you want to override
    LayerSettings(cfg_func=YourFuncionHere)
    The function will be called with these parameters: new_tensors = cfg_func(tensor, settings, hackingchip)
      Tensor will be the full block of positive and negative tensors
      Settings will be the respective settings object (layer or QKV settings)
      Hackingchip is included for extra info, such as hackingchip.prompt.numpos, hackingchip.prompt.numneg, and hackingchip.prompt.negend
    You must return the new full tensors (same shape as input tensors) after your modification
    
    For an example of a custom cfg_func, see cfg_repulsor below
    """
    
    # Repels positive from negative, up to a distance determined by negative's magnitude * settings.weight
    def cfg_repulsor(tensor, settings, hackingchip):
            # Get negative tensor
            tensor_neg_orig = tensor[hackingchip.prompts.numpos:hackingchip.prompts.negend, :, :]
            tensor_neg_orig = torch.mean(tensor_neg_orig, dim=0, keepdim=False)
            
            # Find the strength of the repulsion, the difference between pos and neg, and the magnitude of that difference
            strength = torch.norm(tensor_neg_orig).item() * settings.weight
            difference = tensor[0] - tensor_neg_orig
            diff_strength = torch.norm(difference).item()
            
            # If the repulsion strength is higher than the distance between pos and neg, repel pos so it's at that distance away
            # If pos is already repulsion strength distance away or more, then don't modify it (although this could be changed to pull it back toward neg)
            repulsion = ((strength / diff_strength) - 1) * difference if strength > diff_strength else torch.zeros_like(difference)
            
            tensor[0] += repulsion # There's no accelerating issues with this cfg func, so only need to modify positive
            return tensor
        
    # Set a cfg_func for my thought CFG example
    thought_cfg_func = None
    
    # The weight of the CFG for thoughts in this example, change this value to change how strongly thoughts are affected by negative prompts
    # The amount of weight to use seems to depend on how many layers you are putting it on
    # It seems like once you accumulate 0.5 weight among all layers or more, things can get weird. The default puts 0.2 weight into two different layers.
    thought_weight = 0.2
    
    chip.layer_settings[last_kv_layer - 1] = LayerSettings(weight=thought_weight, cfg_func=thought_cfg_func)
    chip.layer_settings[last_kv_layer + 1] = LayerSettings(weight=thought_weight, cfg_func=thought_cfg_func)
    
    

    # The head_layer is the final layer that outputs values for all of the logits (every possible token)
    # This will do "normal" CFG that directly affects token probabilities
    # This can help the thought CFG's impact get amplified enough to change the output reliably, but also tends to make things crazier
    # Also, if the LLM is cutting off output prematurely, this may help with that
    
    # Currently commented out below, but can be enabled
    
    # logits_weight = 0.1
    # chip.layer_settings[head_layer] = LayerSettings(weight=logits_weight)
    

    
    # Extra options:
    
    # The On/Off button for the hackingchip, you can turn it off easily for debugging
    chip.on = True # If this isn't set, the default is True
    
    # All the generated prompts will be printed to the console
    chip.output_prompts = False # If this isn't set, the default is False
    
    # The output of each extra prompt will be printed to the console after generation completes
    chip.output_extra_samples = False # If this isn't set, the default is False
    
    
    
    
    
    
    # Extra spicy experimental part! CFG on Q, K, and/or V on every attn layer
    # You may want to comment out any other layer settings when testing this, cause who knows how that will interact with this
    
    # chip.attn_layers contains an array storing the layer index of each attention layer
    # Their index in attn_layers is their attention index, so this can be used to convert from attention layer index to layer index
    
    # From my preliminary testing so far, attn_weight 0.1 applied to every attention layer seems be at least somewhat performing proper CFG!
    # Any combination of Q, K, and/or V has some kind of visible negative steering effect, although each combination is likely different
    # Try just Q on all layers with 0.2 weight! Still doesn't seem as good as my default settings above though.
    # Q, K, V seem a lot more forgiving with weights, it takes much higher weights to break the output
    
    attn_weight = 0.2
    
    # You can do custom cfg_func with Q, K, V as well! It's the exact same function signature (so can use the same function for all if you want)
    attn_cfg_func = None
    
    attn_test = AttnSettings()
    
    # Uncomment any of the Q, K, V lines below (can be used together)
    
    # attn_test.q = AttnQSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.k = AttnKSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    # attn_test.v = AttnVSettings(weight=attn_weight, cfg_func=attn_cfg_func)
    
    # Uncomment one line below, first line uses every single attention layer and 2nd line only uses the very last attention layer
    
    # chip.attn_layer_settings = [attn_test] * len(chip.attn_layers) # testing every attention layer
    # chip.attn_layer_settings[len(chip.attn_layers) - 1] = attn_test # testing only the last attention
    
    
        
    return chip
