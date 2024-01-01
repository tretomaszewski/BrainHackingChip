

class HackingchipSettings:
    def __init__(self, head_layer):
        self.layer_settings = [None] * (head_layer + 1)
        self.on = True
        self.output_prompts = False

class LayerSettings:
    def __init__(self, weight=0.0):
        self.weight = weight