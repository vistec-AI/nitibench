from llama_index.core.base.base_retriever import BaseRetriever
from typing import List, Dict, Any, Optional, Union

from ..llm import GeminiModel, OpenAIModel, ClaudeModel, MAP_MODEL
from ..data import EvalDataset
from ..prompting import PromptManager
from ..augmenter import NitiLinkAugmenter

from pydantic import ValidationError
import time

import asyncio

class Ragger(object):
    """
    This class needs to handle 3 things:
    1. Normal RAG pipeline: Take in only query and go through the whole RAG process
    2. Long Context Pipeline: Take in only query and skip through retrieval and augmentation process straight to prompt formatting
    3. Golden Context Pipeline (Only work if strat is golden) : Take in query and nodes. Input to the process during augmenter (change retrieved nodes to input nodes)
    """
    
    def __init__(self,
                 dataset: EvalDataset,
                 prompt_manager: PromptManager,
                 llm: Union[GeminiModel, OpenAIModel, ClaudeModel],
                 retriever: Optional[BaseRetriever] = None,
                 augmenter: Optional[NitiLinkAugmenter] = None,
                 max_retries: int = 5
                ):
    
        #Set attributes
        self.dataset = dataset
        self.prompt_manager = prompt_manager
        self.llm = llm
        self.retriever = retriever
        self.augmenter = augmenter
        self.pure = False
        self.o1 = False
        
        
        #Create node map for easy access
        self.id_to_node = {n.id_: n for n in self.dataset.text_nodes}
        self.model_name = self.llm.model_name.split("-")[0]
        assert self.model_name in MAP_MODEL, "Unrecognized model name: {}".format(self.model_name)
        
        self.strat_name = self.dataset.strat_name
        self.max_retries = max_retries
        print("Max Retries: {}".format(self.max_retries))
        
        if hasattr(self.llm, "long_context"):
            self.long_context = self.llm.long_context
            
        else:
            self.long_context = False
        
        if (self.retriever is None) and (self.augmenter is None):
            self.pure = True
            
        if "o1" in self.llm.model_name:
            self.o1 = True
            
        # if long_context:
        #     assert isinstance(self.llm, GeminiModel), "Long context is activated but the model provided is not Gemini"
            
        
    def get_prompt_structure(self,
            query: str,
            relevant_laws: List[Dict] = None,
            dataset_name: str = "tax"):
    
        """
        For dealing with both normal RAG pipeline and Golden Context Pipeline. If relevant laws are parsed use nodes from it. If not, go through normal RAG pipeline
        """
        
        retrieve_query = query
        
        
            
        if dataset_name == "tax":
            query = f"<ข้อหารือ> {retrieve_query} </ข้อหารือ>"
        else:
            query = f"<question> {retrieve_query} </question>"
        
        
        retriever_time = 0
        if self.long_context or self.pure:
            retrieved_nodes = []
            augmented_query = query
            
        elif (relevant_laws is not None):
            #Then retrieved nodes are the one we use from relevant laws
            retrieved_nodes = [self.id_to_node[f"{l['law']}-{l['sections']}"] for l in relevant_laws if f"{l['law']}-{l['sections']}" in self.id_to_node]
            augmented_query = self.augmenter(query, retrieved_nodes)
            
            
        else:
            #Otherwise, retrieve node normally with the retriever
            assert self.retriever is not None, "Please provide a retriever in case of normal rag pipeline"
            start_time = time.time()
            retrieved_nodes = self.retriever.retrieve(retrieve_query)
            retriever_time = time.time() - start_time
            #Then, augment the query
            augmented_query = self.augmenter(query, retrieved_nodes)
            
        
        name = self.model_name.split("-")[0]
        
        task = "response"
        if self.long_context:
            task = "response-long"
        elif self.pure:
            task = "response-pure"
        elif self.o1:
            task = "response-o1"
            
        formatted_prompt = self.prompt_manager.get_formatted_prompt(query=augmented_query, task=task, dataset=dataset_name, model=self.model_name)

        if name == "claude":
            structure = self.prompt_manager.response_structure["response"][0]
        elif name == "typhoon":
            structure = None
        else:
            structure = self.prompt_manager.response_structure["response"][1]
            
        
        
        
        return formatted_prompt, structure, retrieved_nodes, retriever_time
    
    async def rag(self, 
                  index: str, 
                  query: str,
                  relevant_laws: List[Dict] = None,
                  dataset_name: str = "tax"):
        
        formatted_prompt, structure, retrieved_nodes, retriever_time = self.get_prompt_structure(query=query, relevant_laws=relevant_laws, dataset_name=dataset_name)
        
        main_structure = self.prompt_manager.response_structure["response"][1]
        
        counter = 0
        
        
        if self.o1:
            structure = None
        
        if self.long_context:
            response = await self.llm.complete_lc(**formatted_prompt, structure=structure)
            
            
        else:
            
            response = None
            while counter < self.max_retries:
                
                try:
                    response = await self.llm.complete(**formatted_prompt, structure=structure)
                    tmp = main_structure(**response["content"])
                    break

                except Exception as e:
                    print(e)
                    counter += 1
                    continue
            # if (counter == self.max_retries) and (response is None):
            #     response = {"content": {}, "usage": {}}
                   
        
        #Save retrieve ids as well
        response["retrieved_ids"] = [n.id_ for n in retrieved_nodes]
        
        response["idx"] = index
        response["tries"] = counter + 1
        response["retriever_time"] = retriever_time
               
        return response
    
    async def rag_multi(self,
                        indices: List[str],
                        queries: List[str],
                        dataset_names: List[str],
                        relevant_laws: List[List[Dict]]):
        
        #If the model name is not claude, just return the job of rag
        # if self.model_name != "claude":
        return await asyncio.gather(*(self.rag(i, q, r, d) for i, q, r, d in zip(indices, queries, relevant_laws, dataset_names)))
                                    
                                    
        
        
        
        
        
        
            
        
        
            
            
            
                 
                 
                 