
"""
Standalone Application for RAG system
"""


import os
import time
import json
import uuid
import random
import hashlib
import warnings
from datetime import datetime

# Data processing and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Machine Learning and NLP
import torch
import faiss
import nltk
import wikipediaapi
import gradio as gr
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bert_score import score
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Initial setup
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
hf_token = ""


class DenseRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, corpus):
        print("Building dense vector index...")
        self.chunks = corpus
        texts = [chunk['chunk_text'] for chunk in corpus]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"Faiss Index built with {self.index.ntotal} vectors")
        return self.index

    def search(self, query, k=10):
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                chunk = self.chunks[idx].copy()
                chunk['dense_score'] = float(dist)
                chunk['dense_rank'] = i + 1
                results.append(chunk)
        return results

class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens

    def build_index(self, corpus):
        print("Building sparse BM25 index...")
        self.chunks = corpus
        tokenized_corpus = [self.preprocess(chunk['chunk_text']) for chunk in tqdm(corpus)]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built for {len(corpus)} chunks")

    def search(self, query, k=10):
        query_tokens = self.preprocess(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx].copy()
            chunk['sparse_score'] = float(scores[idx])
            chunk['sparse_rank'] = rank + 1
            results.append(chunk)
        return results

class ReciprocalRankFusion:
    def __init__(self, k=60):
        self.k = k

    def fuse(self, dense_results, sparse_results, top_n=10):
        dense_scores = {res['chunk_id']: res['dense_rank'] for res in dense_results}
        sparse_scores = {res['chunk_id']: res['sparse_rank'] for res in sparse_results}
        all_chunks = set(dense_scores.keys()) | set(sparse_scores.keys())

        # Create a mapping that merges data from both retrievers
        chunk_data = {}
        for res in dense_results:
            chunk_data[res['chunk_id']] = res.copy()
        for res in sparse_results:
            if res['chunk_id'] in chunk_data:
                # Update existing chunk with sparse score/rank without losing dense info
                chunk_data[res['chunk_id']].update(res)
            else:
                chunk_data[res['chunk_id']] = res.copy()

        rrf_scores = {}
        for chunk_id in all_chunks:
            score = 0
            if chunk_id in dense_scores: score += 1.0 / (self.k + dense_scores[chunk_id])
            if chunk_id in sparse_scores: score += 1.0 / (self.k + sparse_scores[chunk_id])
            rrf_scores[chunk_id] = score

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        final_results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks):
            result = chunk_data[chunk_id].copy()
            result['rrf_score'] = float(score)
            result['final_rank'] = rank + 1
            final_results.append(result)
        return final_results



from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class ResponseGenerator:
    def __init__(self, model_name='meta-llama/Llama-2-7b-chat-hf', hf_token=None):
        self.model_name = model_name

        load_kwargs = {"token": hf_token, "device_map": "auto"}

        load_kwargs.update({
            "trust_remote_code": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        })
        task, model_class = "text-generation", AutoModelForCausalLM


        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        # Llama-2 doesn't have a default pad token; setting it prevents generation errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = model_class.from_pretrained(model_name, **load_kwargs)

        # ADVANCED OPTIMIZATION: Compile the model (Requires PyTorch 2.0+)
        # This can take ~1-2 mins initially but speeds up every subsequent generation.
        # self.model = torch.compile(self.model)

        self.generator_pipeline = pipeline(task, model=self.model, tokenizer=self.tokenizer, truncation=True)

    def generate(self, query, context_chunks, max_new_tokens=256):
        context = "\n\n".join([chunk['chunk_text'] for chunk in context_chunks])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "max_length": None,          # Added to resolve conflict with max_new_tokens
            "use_cache": True,           # CRITICAL: Reuses previous token calculations
            "num_beams": 1,              # Greedy search is the fastest decoding metho
            "temperature": 0.2,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        prompt = f"[INST] <<SYS>>\nAnswer based ONLY on given context.\n<</SYS>>\n\nContext:{context}\nQuestion: {query} [/INST]"

        outputs = self.generator_pipeline(prompt, return_full_text=False, **gen_kwargs)
        return outputs[0]['generated_text'].strip()

generator = ResponseGenerator(model_name="meta-llama/Llama-2-7b-chat-hf", hf_token=hf_token)
print("Generator load success")

"""### Complete Hybrid RAG System"""

class HybridRAGSystem:
    def __init__(self, dense_retriever, sparse_retriever, rrf_fusion, generator):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_fusion = rrf_fusion
        self.generator = generator

    def query(self, question, top_k=20, top_n=5):
        """Execute full RAG pipeline"""
        start_time = time.time()

        # Step 1: Dense retrieval
        dense_results = self.dense_retriever.search(question, k=top_k)

        # Step 2: Sparse retrieval
        sparse_results = self.sparse_retriever.search(question, k=top_k)

        # Step 3: RRF fusion
        fused_results = self.rrf_fusion.fuse(dense_results, sparse_results, top_n=top_n)

        # Step 4: Generate response
        answer = self.generator.generate(question, fused_results)

        end_time = time.time()
        response_time = end_time - start_time

        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': fused_results,
            'dense_results': dense_results[:5],  # Top 5 for display
            'sparse_results': sparse_results[:5],  # Top 5 for display
            'response_time': response_time
        }

    def format_response(self, response):
        """Format response for display"""
        output = f"Question: {response['question']}\n\n"
        output += f"Answer: {response['answer']}\n\n"
        output += f"Response Time: {response['response_time']:.2f} seconds\n\n"

        output += "Top Retrieved Chunks:\n"'source_title'
        for i, chunk in enumerate(response['retrieved_chunks']):
            output += f"\n{i+1}. [{chunk.get('final_rank', i+1)}] "
            output += f"Dense Score: {chunk.get('dense_score', 0):.4f} | "
            output += f"Sparse Score: {chunk.get('sparse_score', 0):.4f} | "
            output += f"RRF Score: {chunk.get('rrf_score', 0):.4f} | "
            output += f"Title: {chunk.get('chunk_title', 'N/A')}\n"
            output += f"   Source: {chunk.get('chunk_url', 'N/A')}\n"
            output += f"   Preview: {chunk['chunk_text'][:150]}...\n"

        return output


# --- Gradio Interface ---

def rag_dashboard(rag_system, user_query):
    response = rag_system.query(user_query)
    retrieved_chunks_details = []
    for chunk in response['retrieved_chunks']:
        retrieved_chunks_details.append({
            "Dense Score": f"{chunk.get('dense_score', 0.0):.4f}",
            "Sparse Score": f"{chunk.get('sparse_score', 0.0):.4f}",
            "RRF Score": f"{chunk.get('rrf_score', 0.0):.4f}",
            "URL": chunk['chunk_url'],
            "Title": chunk['chunk_title'],
            "Text Snippet": chunk['chunk_text'][:100].replace('\n', ' ') + '...'
        })
    df = pd.DataFrame(retrieved_chunks_details)
    full_answer = response['answer']
    elapsed_time = f"{response['response_time']:.2f}s"
    return full_answer,df,elapsed_time
    # partial_answer = ""
    # for char in full_answer:
    #     partial_answer += char
    #     yield partial_answer, df, elapsed_time
    #     time.sleep(0.01)


def main():
    corpus = ""
    # Initialize retrievers
    corpus_path="corpus_data.json"
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
            print("corpus loaded: ",len(corpus))
    else:
        print("corpus file doesn't exist")
        return

    dense_retriever = DenseRetriever('all-mpnet-base-v2')
    vector_index = dense_retriever.build_index(corpus)

    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(corpus)
    print("Retrieval indices ready.")

    # Initialize complete system
    rrf = ReciprocalRankFusion(k=60)
    rag_system = HybridRAGSystem(dense_retriever, sparse_retriever, rrf, generator)
    print(" RAG System Loaded")
    # # Test the system
    # test_question = "What is Mathematics?"
    # response = rag_system.query(test_question)
    # print(rag_system.format_response(response))

# Custom CSS to style the answer box and ensure text visibility
    custom_css = """
    #answer-container {
        background-color: #0b0f19;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #f0f7ff;
        min-height: 100px;
    }
    #answer-container * {
        color: #f0f7ff !important;
    }
    """

    with gr.Blocks(title="RAG Dashboard", css=custom_css) as demo:
        gr.Markdown("# RAG System Dashboard")
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(label="Input User Query", placeholder="What is AI?", lines=3)
                submit_btn = gr.Button("Generate Output", variant="primary")
            with gr.Column(scale=1):
                resp_time = gr.Label(label="Query Response Time")

        gr.Markdown("## Generated Answer")
        with gr.Column(elem_id="answer-container"):
            output_answer = gr.Markdown()

        gr.Markdown("## Retrieval Details")
        source_table = gr.Dataframe(
            headers=["Dense Score", "Sparse Score", "RRF Score", "URL", "Title", "Text Snippet"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )

        def predict_wrapper(user_query):
            # We call the generator and return it directly to Gradio
            return rag_dashboard(rag_system, user_query)

        submit_btn.click(
            fn=predict_wrapper, 
            inputs=[query_input], 
            outputs=[output_answer, source_table, resp_time]
        )

    demo.launch(debug=False)


if __name__ == "__main__":
    main()