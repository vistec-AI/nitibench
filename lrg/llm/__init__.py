from typing import Dict
from .collections.claude import  ClaudeConfig, ClaudeModel
from .collections.gemini import  GeminiConfig, GeminiModel
from .collections.openai import OpenAIConfig, OpenAIModel
MAP_MODEL = {"gpt": (OpenAIConfig, OpenAIModel),
             "claude": (ClaudeConfig, ClaudeModel),
             "gemini": (GeminiConfig, GeminiModel),
             "o1": (OpenAIConfig, OpenAIModel),
             "aisingapore/gemma2": (OpenAIConfig, OpenAIModel),
             "typhoon": (OpenAIConfig, OpenAIModel),
             "x": (OpenAIConfig, OpenAIModel)}

def init_llm(config: Dict):
    
    name = config["model"].split("-")[0]
    
    assert name in MAP_MODEL, "Unrecognize model name: {}".format(model_name)
    config_class, model_class = MAP_MODEL[name]
    
    model_config = config_class(**config)
    print(model_config.model_dump())
    model = model_class(config=model_config)
    
    return model