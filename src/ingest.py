

import argparse
import os
import pickle

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def load_data(csv_path: str):
   

    """
    Loading the csv file
    - docs: the text to chunk (e.g. the Answer field)
    - metas: a list of dicts with metadata 
    """
    df = pd.read_csv(csv_path)
    # adjust column names as needed
    docs = df['Answer'].fillna('').astype(str).tolist()
    metas = [
        {
            'doc_id': row['Document_ID'],
            'question': row['Question'],
            'source': row['Document_Source'],
            'url': row['Document_URL']
        }
        for _, row in df.iterrows()
    ]
    return docs, metas


def chunk_text(text: str, size: int, overlap: int):
    """
    Converting data into Simple word-based chunking with overlap.
    """
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


def embed_chunks(
    chunks: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device
) -> np.ndarray:
    """
    Tokenizing + forward-pass each chunk, mean-pool the last hidden states.
    Returns an (N × D) array.
    """
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for text in chunks:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='longest'
            ).to(device)

            outputs = model(**tokens)
            last_hidden = outputs.last_hidden_state    # (1, seq_len, D)
            mask = tokens['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
            # mean-pool only over non-padded tokens
            summed = (last_hidden * mask).sum(dim=1)      # (1, D)
            counts = mask.sum(dim=1)                      # (1, 1)
            pooled = summed / counts                      # (1, D)
            all_embeds.append(pooled.squeeze(0).cpu().numpy())

    return np.vstack(all_embeds)  # shape (N, D)


def build_and_save_index(
    embeddings: np.ndarray,
    metas: list[dict],
    out_dir: str
):
    """
    Normalizing embeddings for inner-product similarity,
    building a FlatIP index, save index + metadata.
    """
    # normalizing to unit length for IP = cosine
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, 'faiss.index'))

    with open(os.path.join(out_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metas, f)


def main(args):
    # 1) loading the arguments 
    docs, base_metas = load_data(args.csv_path)

    # 2) chunking + assembling metadata
    all_chunks = []
    chunk_metas = []
    for doc_text, meta in zip(docs, base_metas):
        chunks = chunk_text(doc_text, args.chunk_size, args.overlap)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            m = meta.copy()
            m['chunk_id'] = i
            chunk_metas.append(m)

    # 3) loading the model + embedding all the chunks 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    embeddings = embed_chunks(all_chunks, tokenizer, model, device)

    # 4) building datastore FAISS 
    build_and_save_index(embeddings, chunk_metas, args.out_dir)
    print(f"Indexed {len(all_chunks)} chunks. Index + metadata saved to '{args.out_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG ingestion: CSV → chunks → PUBMEDBERT → FAISS"
    )
    parser.add_argument(
        "csv_path",
        help="path to your consolidated CSV (e.g. /content/all_data.csv)"
    )
    parser.add_argument(
        "--out_dir", "-o",
        default="index_data",
        help="where to write faiss.index + metadata.pkl"
    )
    parser.add_argument(
        "--chunk_size", "-c", type=int, default=200,
        help="max words per chunk"
    )
    parser.add_argument(
        "--overlap", "-l", type=int, default=50,
        help="words overlap between consecutive chunks"
    )
    parser.add_argument(
        "--model", "-m",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        help="HuggingFace model ID for PubMedBERT"
    )
    args = parser.parse_args()
    main(args)
