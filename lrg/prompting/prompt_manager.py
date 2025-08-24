## Contain function to parse schema(json file) to actual pydantic class (need to do this for openai)
from typing import List, Dict, Any, Tuple

import json
import os
import re
from pydantic import create_model

from typing_extensions import TypedDict
from types import new_class


from jinja2 import Environment, FileSystemLoader, meta
#Structuring Prompt Manager
class PromptManager(object):
    
    def __init__(self, template_dir: str = "/app/LRG/lrg/prompting/templates", main_template: str = "default_query.md") -> None:
        '''
        Initialise the templates and stuff. Load all template in-memory
        '''
        # Set up the Jinja2 environment. This is fixed
        self.template_dir = template_dir
        env = Environment(loader=FileSystemLoader(template_dir))
        self.template = env.get_template(main_template)
        self.o1_template = env.get_template("o1_query.md")
        
        self.DATASET_NAMES = ["tax", "wangchan"]
        
        self.TASK_NAMES = {"response": "/app/LRG/lrg/prompting/structured_outputs/system_response.json", 
                           "coverage-contradiction": "/app/LRG/lrg/prompting/structured_outputs/system_eval.json",
                           "response-long":
 "/app/LRG/lrg/prompting/structured_outputs/system_response.json" ,
                          "response-pure": "/app/LRG/lrg/prompting/structured_outputs/system_response.json",
                          "response-o1": "/app/LRG/lrg/prompting/structured_outputs/system_response.json"}
        
        self._init_template()
        
    def _init_template(self) -> None:
        '''
        Initialise all required templates necessary for all model and all task
        '''
        response_structure = dict()
        turn_prompts = dict()
        number_pattern = re.compile(r'\d+')
        
        for task in self.TASK_NAMES:
            #Read structure
            if os.path.exists(self.TASK_NAMES[task]):
                with open(self.TASK_NAMES[task], "r") as f:
                    json_schema = json.load(f)
                    pydantic_schema = PromptManager.parse_schema_to_pydantic(json_schema["input_schema"], task.replace("-", "").capitalize())
                    typing_schema = PromptManager.parse_schema_to_typed_dict(json_schema["input_schema"], task.replace("-", "").capitalize())
                    response_structure[task] = (json_schema, pydantic_schema, typing_schema)
                
            
            for dataset in self.DATASET_NAMES:      
                turn_prompts[f"{task}-{dataset}"] = [os.path.join(f"{task}-{dataset}", file) for file in os.listdir(os.path.join(self.template_dir, f"{task}-{dataset}")) if file.endswith(".md")]
                turn_prompts[f"{task}-{dataset}"] = sorted(turn_prompts[f"{task}-{dataset}"], key = lambda x: int(re.search(number_pattern, x).group()))
                
        self.response_structure = response_structure
        self.turn_files = turn_prompts
            
        
    @staticmethod
    def parse_schema_to_pydantic(schema: Dict[str, Any], model_name: str):
        """Converts JSON schema to Pydantic model."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields = {}
        # Use $defs instead of definitions
        definitions = schema.get("$defs", {})

        # Process each property in the schema
        for prop_name, prop_info in properties.items():
            field_type = prop_info.get("type")

            # Handle string fields
            if field_type == "string":
                fields[prop_name] = (str, ...)

            # Handle integer fields
            elif field_type == "integer":
                fields[prop_name] = (int, ...)

            # Handle boolean fields
            elif field_type == "boolean":
                fields[prop_name] = (bool, ...)

            # Handle array (list) fields
            elif field_type == "array":
                items = prop_info.get("items")
                if "$ref" in items:
                    ref = items["$ref"].split("/")[-1]  # Get the model name from $ref
                    ref_model = PromptManager.parse_schema_to_pydantic(definitions[ref], ref)
                    fields[prop_name] = (List[ref_model], ...)
                else:
                    # Handle simple item types if needed
                    item_type = items.get("type")
                    if item_type == "string":
                        fields[prop_name] = (List[str], ...)
                    elif item_type == "integer":
                        fields[prop_name] = (List[int], ...)

            # Handle nested objects via references
            elif "$ref" in prop_info:
                ref = prop_info["$ref"].split("/")[-1]
                ref_model = PromptManager.parse_schema_to_pydantic(definitions[ref], ref)
                fields[prop_name] = (ref_model, ...)

        # Include required fields
        for req in required:
            if req in fields:
                fields[req] = (fields[req][0], ...)

        # Dynamically create the Pydantic model
        return create_model(model_name, **fields)
    
    @staticmethod
    def parse_schema_to_typed_dict(schema: Dict[str, Any], model_name: str) -> TypedDict:
        """Converts JSON schema to TypedDict."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        fields = {}
        definitions = schema.get("$defs", {})

        # Process each property in the schema
        for prop_name, prop_info in properties.items():
            field_type = prop_info.get("type")
            is_required = prop_name in required
            
            # Handle string fields
            if field_type == "string":
                fields[prop_name] = str

            # Handle integer fields
            elif field_type == "integer":
                fields[prop_name] = int
            
            # Handle boolean fields
            elif field_type == "boolean":
                fields[prop_name] = bool

            # Handle array fields
            elif field_type == "array":
                items = prop_info.get("items")
                if "$ref" in items:
                    ref = items["$ref"].split("/")[-1]  # Get the model name from $ref
                    fields[prop_name] = List[PromptManager.parse_schema_to_typed_dict(definitions[ref], ref)]
                else:
                    item_type = items.get("type", "string")
                    if item_type == "string":
                        fields[prop_name] = List[str]
                    elif item_type == "integer":
                        fields[prop_name] = List[int]
                    # Add more types as necessary

            # Handle nested objects via references
            elif "$ref" in prop_info:
                ref = prop_info["$ref"].split("/")[-1]
                fields[prop_name] = PromptManager.parse_schema_to_typed_dict(definitions[ref], ref)

        # Create the TypedDict dynamically
        TypedDictModel = TypedDict(model_name, {k: (v, NotRequired) if not is_required else v for k, v in fields.items()})
        
        return TypedDictModel
    
    

    def get_prompt_rendered(self, task: str, dataset: str, query: str):
        
        assert task in self.TASK_NAMES, "{} not found in TASK_NAMES".format(task)
        assert dataset in self.DATASET_NAMES, "{} not found in DATASET_NAMES".format(dataset)
        
        extra_task = task.split("-")[1] if len(task.split("-")) > 1 else ""
        
        system_path = ""
        if extra_task in ["long", "pure"]:
            system_path = f"system_prompt_{dataset}_{extra_task}.md"
        elif extra_task != "o1":
            system_path = f"system_prompt_{dataset}.md"
        
        if system_path == "":
            data = {"turn_files": self.turn_files[f"{task}-{dataset}"],
                "query": query}
            
            return self.o1_template.render(data)
        else:
            data = {"system_prompt": system_path,
                "turn_files": self.turn_files[f"{task}-{dataset}"],
                "query": query}
            

            return self.template.render(data)

    def get_prompt(self, prompt_str):
        prompts = []
        for line in prompt_str.split("\n"):
            if re.search(r"^<system>", line.strip()):
                prompts.append({"role": "system", "content": [line.replace("<system>", "").strip()]})
            elif re.search(r"^<user>", line.strip()):
                prompts.append({"role": "user", "content": [line.replace("<user>", "").strip()]})
            elif re.search(r"^<assistant>", line.strip()):
                prompts.append({"role": "assistant", "content": [line.replace("<assistant>", "").strip()]})
            else:
                prompts[-1]["content"].append(line)

        for prompt in prompts:
            prompt["content"] = "\n".join(prompt["content"]).strip()

        return prompts

    def format_gemini_prompt(self, prompts: List[Dict[str, str]], task: str) -> Tuple[str, List[Dict[str, str]]]:
        pattern = r'```json(.*?)```'

        #Normally, the first thing in the list is system prompt
        #Next two are considered instruction prompt and should be put in the message as is
        #The last one is the actual query
        new_prompts = prompts[1:]
        system_prompt = prompts[0]["content"]

        for prompt in new_prompts:
            if prompt["role"] == "assistant":
                prompt["role"] = "model"

            prompt["parts"] = [{"text": prompt["content"]}]
            del prompt["content"]


        return {"system": system_prompt, "messages": new_prompts}
    
    def format_claude_prompt(self, prompts: List[Dict[str, str]], task: str) -> Tuple[str, List[Dict[str, str]]]:
        pattern = r'```json(.*?)```'

        #Normally, the first thing in the list is system prompt
        #Next two are considered instruction prompt and should be put in the message as is
        #The last one is the actual query
        new_prompts = prompts[1:3]
        system_prompt = [{"type": "text", "text": prompts[0]["content"], "cache_control": {"type": "ephemeral"}}]
        counter = 0

        for i, prompt in enumerate(prompts[3:len(prompts)-1]):

            if prompt["role"] == "assistant":
                tmp = prompt["content"]
                match = re.search(pattern, tmp, re.DOTALL)
                #Prepare tool use 
                if match:
                    tool_input = eval(match.group(1).strip())
                else:
                    #If not match, append as is
                    new_prompts.append([{"role": prompt["role"], "content": [{"type": "text", "text": prompt["content"]}]}])
                    continue

                tool_id = "toolu_{:04d}".format(counter)
                content = [{"type": "tool_use", "id": tool_id, "name": task, "input": tool_input}]
                prompt["content"] = content

                new_prompts.append(prompt)
                new_prompts.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "success"}]})
                new_prompts.append({"role": "assistant", "content": [{"type": "text", "text": "Great Success"}]})
                counter += 1

            else:
                new_prompts.append({"role": prompt["role"], "content": [{"type": "text", "text": prompt["content"]}]})
                

        #The last one should be cached
        new_prompts[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}

        new_prompts.append(prompts[-1])


        return {"system": system_prompt, "messages": new_prompts}
    
    def format_gpt_prompt(self, prompts: List[Dict[str, str]], task: str) -> Tuple[str, List[Dict[str, str]]]:
        #For gpt, no need to do anything
        return {"messages": prompts}

    #Then, need to have a set of functions to format the prompt for each model
    #Get everything and return prompt ready for parsing
    def get_formatted_prompt(self, query: str, task: str, model: str, dataset: str):
       
        assert task in self.TASK_NAMES, "{} not found in TASK_NAMES".format(task)
        assert dataset in self.DATASET_NAMES, "{} not found in DATASET_NAMES".format(dataset)
        
        prompt_str = self.get_prompt_rendered(task=task, dataset=dataset, query=query).strip()

        prompts = self.get_prompt(prompt_str=prompt_str)
        
        type_name = model.split('-')[0]
        if type_name not in ("claude", "gemini"):
            type_name = "gpt"

        return eval(f"self.format_{type_name}_prompt(prompts, task= task)")
            
