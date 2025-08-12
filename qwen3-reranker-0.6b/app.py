#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2023/11/7 22:45
@Author: zhidong
@File: qwen_reranker.py
@Desc: Qwen3-Reranker API服务
"""
import os
import torch
import logging
import uvicorn
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()
security = HTTPBearer()
env_bearer_token = 'sk-xxxx'

class QADocs(BaseModel):
    query: Optional[str]
    documents: Optional[List[str]]
    instruction: Optional[str] = 'Given a web search query, retrieve relevant passages that answer the query'

class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class QwenReranker(metaclass=Singleton):
    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        # 如需使用flash_attention_2和半精度，取消下面这行注释
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
    
    def format_instruction(self, instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    
    @torch.no_grad()
    def compute_scores(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()
    
    def rerank(self, query_docs: QADocs) -> List:
        if query_docs is None or not query_docs.documents:
            return []
        
        pairs = [self.format_instruction(query_docs.instruction, query_docs.query, doc) 
                for doc in query_docs.documents]
        
        inputs = self.process_inputs(pairs)
        scores = self.compute_scores(inputs)
        
        results = []
        for index, (doc, score) in enumerate(zip(query_docs.documents, scores)):
            results.append({
                "index": index,
                "text": doc,
                "score": score
            })
        
        # 按分数降序排序
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return [{"index": item["index"], "relevance_score": item["score"]} for item in sorted_results]

reranker = QwenReranker()

@app.post('/v1/rerank')
async def handle_rerank_request(docs: QADocs, credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if env_bearer_token is not None and token != env_bearer_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        results = reranker.rerank(docs)
        return {"results": results}
    except Exception as e:
        logging.error(f"Reranking error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    token = os.getenv("ACCESS_TOKEN")
    if token is not None:
        env_bearer_token = token
    try:
        uvicorn.run(app, host='0.0.0.0', port=6006)
    except Exception as e:
        print(f"API启动失败！\n报错：\n{e}")