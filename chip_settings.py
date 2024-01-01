from extensions.brainhackingchip.settings_classes import LayerSettings

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
    """
    
    # The weight of the CFG for thoughts in this example, change this value to change how strongly thoughts are affected by negative prompts
    # The amount of weight to use seems to depend on how many layers you are putting it on
    # It seems like once you accumulate 0.5 weight among all layers or more, things can get weird. The default puts 0.2 weight into two different layers.
    thought_weight = 0.2
    
    chip.layer_settings[last_kv_layer - 1] = LayerSettings(weight=thought_weight)
    chip.layer_settings[last_kv_layer + 1] = LayerSettings(weight=thought_weight)
    
    

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
    
    return chip
