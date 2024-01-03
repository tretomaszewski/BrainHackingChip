

class HackingchipSettings:
    def __init__(self, layer_count, attn_layers):
        self.on = True
        self.output_prompts = False
        self.output_extra_samples = False
        
        self.layer_settings = [None] * layer_count
        self.attn_layer_settings = [None] * len(attn_layers)
        self.attn_layers = attn_layers # Stores the layer index of each attention layer, for conversion from attention layer idx to layer idx

class LayerSettingsBase:
    def __init__(self, weight=0.0, cfg_func=None):
        self.weight = weight
        self.cfg_func = cfg_func

class LayerSettings(LayerSettingsBase):
    def __init__(self, weight=0.0, cfg_func=None):
        super().__init__(weight, cfg_func)
        
class AttnSettings:
    def __init__(self, q=None, k=None, v=None): # Use AttnQSettings, AttnKSettings, AttnVSettings respectively, in case each needs special handling
        self.q = q # AttnQSettings
        self.k = k # AttnKSettings
        self.v = v # AttnVSettings
        
class AttnQSettings(LayerSettingsBase):
    def __init__(self, weight=0.0, cfg_func=None):
        super().__init__(weight, cfg_func)
        
class AttnKSettings(LayerSettingsBase):
    def __init__(self, weight=0.0, cfg_func=None):
        super().__init__(weight, cfg_func)
        
class AttnVSettings(LayerSettingsBase):
    def __init__(self, weight=0.0, cfg_func=None):
        super().__init__(weight, cfg_func)
        