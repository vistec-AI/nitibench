import argparse
import sys
import asyncio
import os
import yaml
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/app/LRG/setting.env")

if "/app/LRG" not in sys.path:
    sys.path.append("/app/LRG")

from lrg.data import EvalDataset
from lrg.retrieval.retrieval_init import init_retriever
from lrg.prompting import PromptManager
from lrg.llm import init_llm
from lrg.augmenter import NitiLinkAugmenterConfig, NitiLinkAugmenter
from lrg.e2e import Ragger

from tqdm import tqdm
import torch

import time

import asyncio
import gc

tqdm.pandas()

async def evaluate_ragger(ragger: Ragger, golden_retriever: bool = False, batch_size: int = 1, setting_name: str = ""):
    
    os.makedirs(setting_name, exist_ok=True)
    
    tax_df = ragger.dataset.tax_df
    wangchan_df = ragger.dataset.wangchan_df
    
    #First, do tax
    tax_results = []
    if os.path.exists(os.path.join(setting_name, "tax_response.json")):
        with open(os.path.join(setting_name, "tax_response.json"), "r") as f:
            tax_results = json.load(f)
    for i in tqdm(range(len(tax_results), tax_df.shape[0], batch_size)):
        
        job_params = tax_df.iloc[i: i+batch_size][["idx", "ข้อหารือ", "actual_relevant_laws"]].to_dict(orient="records")
        if isinstance(job_params, dict):
            job_params = [job_params]

        indices = [p["idx"] for p in job_params]
        queries = [p["ข้อหารือ"] for p in job_params]
        
        if golden_retriever:
            relevant_laws = [p["actual_relevant_laws"] for p in job_params]
            
        else:
            relevant_laws = [None] * len(indices)
            
        dataset_names = ["tax"] * len(indices)
               
        start = time.time()
        jobs = ragger.rag_multi(indices=indices, queries=queries, relevant_laws=relevant_laws, dataset_names=dataset_names)
            
        results = await jobs
        
        tax_results.extend(results)
        with open(os.path.join(setting_name, "tax_response.json"), "w") as f:
            json.dump(tax_results, f)
        
        time.sleep(max(0, 30 - (time.time() - start)))
        

        
    #Next, do wangchan
    wangchan_results = []
    if os.path.exists(os.path.join(setting_name, "wangchan_response.json")):
        with open(os.path.join(setting_name, "wangchan_response.json"), "r") as f:
            wangchan_results = json.load(f)
    
    print(len(wangchan_results))
    for i in tqdm(range(len(wangchan_results), wangchan_df.shape[0], batch_size)):
        
        job_params = wangchan_df.iloc[i: i+batch_size][["idx", "question", "relevant_laws"]].to_dict(orient="records")
        if isinstance(job_params, dict):
            job_params = [job_params]
                
        indices = [p["idx"] for p in job_params]
        queries = [p["question"] for p in job_params]
        
        if golden_retriever:
            relevant_laws = [p["relevant_laws"] for p in job_params]
            
        else:
            relevant_laws = [None] * len(indices)
                   
        dataset_names = ["wangchan"] * len(indices)
            
        jobs = ragger.rag_multi(indices=indices, queries=queries, relevant_laws=relevant_laws, dataset_names=dataset_names)
        
        start = time.time()
        results = await jobs
        
        wangchan_results.extend(results)
        
        with open(os.path.join(setting_name, "wangchan_response.json"), "w") as f:
            json.dump(wangchan_results, f)
        
        time.sleep(max(0, 60 - (time.time() - start)))
        

        
    torch.cuda.empty_cache()

async def main(args):
    
    #Read config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        
    #Iterate through each config
    data_config = config["data_config"]
    retriever_config = config["retriever_config"]
    augmenter_config = config["augmenter_config"]
    llm_config = config["llm_config"]
    
    batch_size = config.get("batch_size", 1)
    output_path = config["output_path"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("device", "0"))
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    
    #Makedirs
    os.makedirs(output_path, exist_ok=True)
    
    pm = PromptManager()
    
    #Some concern, if the setting is golden retriever -> Ignore data config. if the setting is lclm -> ignore retriever, augmenter, data. if the augmenter have reference -> do only golden and if chunk is not golden, ignore augmenter that uses referencer
    
    #1. LCLM
    lclm_keys = [k for k in llm_config if "-lc" in k]
    # print(llm_config)
    
    if len(lclm_keys) > 0:
        
        #Dataset dont matter
        dataset = EvalDataset(**list(data_config.values())[0])
        #No need to parse retriever or augmenter
        for k in lclm_keys:
            print("Doing {}...".format(k))
            llm = init_llm(config = llm_config[k])
            
            ragger = Ragger(dataset = dataset,
                            prompt_manager = pm,
                            llm = llm)
            
            await evaluate_ragger(ragger, batch_size = batch_size, setting_name = os.path.join(output_path, k))
            
            del llm_config[k]
            del ragger
            gc.collect()
            torch.cuda.empty_cache()
            
    #1.5 Parametric Knowledge
    pr_keys = [k for k in retriever_config if "no-retriever" in k]
    
    if len(pr_keys) > 0:
        #Dataset dont matter
        dataset = EvalDataset(**list(data_config.values())[0])
            
        for lc in llm_config:
            llm = init_llm(llm_config[lc])

            setting_name = f"parametric-{lc}"

            print("Doing {}...".format(setting_name))

            ragger = Ragger(dataset = dataset,
                        prompt_manager = pm,
                        llm = llm)

            await evaluate_ragger(ragger, batch_size = batch_size, setting_name = os.path.join(output_path, setting_name))

            del ragger
            gc.collect()
            torch.cuda.empty_cache()
            
        del retriever_config["no-retriever"]
            
            
    #2 Golden Retriever
    gr_keys = [k for k in retriever_config if "golden-retriever" in k]
    
    if len(gr_keys) > 0:
        #Dataset dont matter
        dataset = EvalDataset(**list(data_config.values())[0])
        
        for ac in augmenter_config:
            augmenter = NitiLinkAugmenter(dataset = dataset, config = NitiLinkAugmenterConfig(**augmenter_config[ac], strat_name=dataset.strat_name))
            
            for lc in llm_config:
                llm = init_llm(llm_config[lc])
                
                setting_name = f"{ac}-golden-retriever-{lc}"
                
                print("Doing {}...".format(setting_name))
                
                ragger = Ragger(dataset = dataset,
                            prompt_manager = pm,
                            llm = llm,
                            augmenter=augmenter)
                
                await evaluate_ragger(ragger, batch_size = batch_size, golden_retriever=True, setting_name = os.path.join(output_path, setting_name))
                
                del ragger
                gc.collect()
                torch.cuda.empty_cache()
            
        del retriever_config["golden-retriever"]
        
    #3. Augment with referencer
    ref_keys = [k for k in augmenter_config if "ref-depth" in k]
    
    if len(ref_keys) > 0:
        #Dataset is golden only
        dataset = EvalDataset(**data_config["golden"])
        
        for key in ref_keys:
            augmenter = NitiLinkAugmenter(dataset = dataset, config = NitiLinkAugmenterConfig(**augmenter_config[key], strat_name=dataset.strat_name))
            
            #Init the retriever as well
            for rc in retriever_config:
                retriever = init_retriever(dataset=dataset, strat_name = dataset.strat_name, **retriever_config[rc])
                
                for lc in llm_config:
                    print(llm_config[lc])
                    llm = init_llm(llm_config[lc])
                    
                    setting_name = f"golden-{rc}-{key}-{lc}"
                    
                    print("Doing {}...".format(setting_name))
                    
                    ragger = Ragger(dataset = dataset,
                                prompt_manager = pm,
                                llm = llm,
                                augmenter=augmenter,
                                retriever=retriever)

                    await evaluate_ragger(ragger, batch_size = batch_size, golden_retriever=False, setting_name = os.path.join(output_path, setting_name))
                    del ragger
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            del augmenter_config[key]
            
    #4. Everything else. Just loop inside loop normally
    for dc in data_config:
        dataset = EvalDataset(**data_config[dc])
        
        for rc in retriever_config:
            retriever = init_retriever(dataset=dataset, strat_name=dataset.strat_name, **retriever_config[rc])
            
            for ac in augmenter_config:
                augmenter = NitiLinkAugmenter(dataset = dataset, config = NitiLinkAugmenterConfig(**augmenter_config[ac], strat_name=dataset.strat_name))
                
                for lc in llm_config:
                    llm = init_llm(llm_config[lc])
                    
                    setting_name = f"{dc}-{rc}-{ac}-{lc}"
                    
                    print("Doing {}...".format(setting_name))
                    
                    ragger = Ragger(dataset = dataset,
                                prompt_manager = pm,
                                llm = llm,
                                augmenter=augmenter,
                                retriever=retriever)
                    
                    

                    await evaluate_ragger(ragger, batch_size = batch_size, golden_retriever=False, setting_name = os.path.join(output_path, setting_name))
                    
                    del ragger
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
    
    
                
                
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/app/LRG/config/all_e2e.yaml")
    args = parser.parse_args()
    asyncio.run(main(args))