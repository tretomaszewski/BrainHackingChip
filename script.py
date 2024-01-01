import importlib

def custom_generate_chat_prompt(user_input, state, **kwargs):
    chip = importlib.import_module("extensions.brainhackingchip.chip")
    importlib.reload(chip)
    
    chip_settings = importlib.import_module("extensions.brainhackingchip.chip_settings")
    importlib.reload(chip_settings)
    
    prompt = chip.gen_full_prompt(chip_settings, user_input, state, **kwargs)
    
    return prompt
