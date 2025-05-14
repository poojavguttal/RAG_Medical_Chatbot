

import argparse
import os
import pickle

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from huggingface_hub import login

def load_data(csv_path: str):
    # same function from ingest.py
    # loading the csv file
    df = pd.read_csv(csv_path)
    docs = df['Answer'].fillna('').astype(str).tolist()
    metas = [
        {
            'doc_id': row['Document_ID'],
            'source': row['Document_Source'],
            'url': row['Document_URL'],
        }
        for _, row in df.iterrows()
    ]
    return docs, metas

def chunk_text(text: str, size: int, overlap: int):
    tokens = text.split()
    chunks = []
    for start in range(0, len(tokens), size - overlap):
        chunk = tokens[start:start + size]
        if not chunk:
            break
        chunks.append(' '.join(chunk))
        if start + size >= len(tokens):
            break
    return chunks

def embed_query(text: str, tokenizer: AutoTokenizer, model: AutoModel, device: torch.device) -> np.ndarray:
    # Embeding single query and L2-normalize (for IndexFlatIP / cosine sim)
    model.eval()
    toks = tokenizer(text, return_tensors='pt', truncation=True, padding='longest').to(device)
    with torch.no_grad():
        out = model(**toks).last_hidden_state      # (1, seq_len, D)
        mask = toks['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
        summed = (out * mask).sum(dim=1)            # (1, D)
        counts = mask.sum(dim=1)                    # (1, 1)
        pooled = (summed / counts).cpu().numpy()    # (1, D)
    faiss.normalize_L2(pooled)
    return pooled

def main():
    parser = argparse.ArgumentParser("RAG Retrieval + Llama-2 Generation")
    parser.add_argument("--csv_path",    "-c", required=True,
                        help="Your consolidated CSV file")
    parser.add_argument("--index_dir",   "-i", default="index_data",
                        help="Where ingest.py saved faiss.index + metadata.pkl")
    parser.add_argument("--embed_model", "-e",
                        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                        help="HuggingFace PubMedBERT model for embeddings")
    parser.add_argument("--llm_model",   "-l",
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="HuggingFace Llama-2 chat model")
    parser.add_argument("--chunk_size",  type=int, default=200,
                        help="Must match ingest.py")
    parser.add_argument("--overlap",     type=int, default=50,
                        help="Must match ingest.py")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="How many chunks to retrieve")
    parser.add_argument("--query",       "-q", required=True,
                        help="Your question")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Generation length")
    parser.add_argument("--hf_token",
                        help="Hugging Face API token")
    args = parser.parse_args()

    # if provided, log in to HF to access private models
    if args.hf_token:
        login(token=args.hf_token)

    # 1) Load FAISS index + metadata
    idx = faiss.read_index(os.path.join(args.index_dir, "faiss.index"))
    with open(os.path.join(args.index_dir, "metadata.pkl"), "rb") as f:
        chunk_metas = pickle.load(f)

    # 2) Reload & re-chunk docs
    docs, _ = load_data(args.csv_path)
    all_chunks = []
    for doc_text in docs:
        all_chunks.extend(chunk_text(doc_text, args.chunk_size, args.overlap))

    # 3) Prepare embed model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    embed_model     = AutoModel.from_pretrained(args.embed_model).to(device)

    # 4) Embed query & search
    q_vec = embed_query(args.query, embed_tokenizer, embed_model, device)
    distances, indices = idx.search(q_vec, args.top_k)

    # 5) Collect retrieved chunks
    retrieved = []
    for dist, idx_ in zip(distances[0], indices[0]):
        retrieved.append({
            "score": float(dist),
            "meta":  chunk_metas[idx_],
            "text":  all_chunks[idx_],
        })

    # 6) Build prompt
    context = "\n\n".join(
        f"[{i+1}] Source: {r['meta']['source']} | Text: {r['text']}"
        for i, r in enumerate(retrieved)
    )
    system_prompt = (
        "You are a knowledgeable medical assistant. "
        "Use the following retrieved context to answer the user's question."
    )
    user_prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {args.query}\n"
        f"Answer:"
    )

    # 7) Load & quantize Llama-2 chat model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        use_fast=True,
        use_auth_token=args.hf_token
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=args.hf_token
    ).eval()

    # 8) Tokenize & generate
    inputs = llm_tokenizer(user_prompt, return_tensors="pt").to(llm_model.device)
    output_ids = llm_model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        eos_token_id=llm_tokenizer.eos_token_id,
        pad_token_id=llm_tokenizer.pad_token_id,
    )
    answer = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 9) Display results
    print("\n=== Retrieved Contexts ===")
    for r in retrieved:
        print(f"• (score={r['score']:.4f}) {r['text'][:200]}…\n")
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
