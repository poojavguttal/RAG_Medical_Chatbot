# RAG_Medical_Chatbot

## 🚀 Usage

1. **Ingest & index** your MedQuAD CSV

   ```bash
   python src/ingest.py data/data.csv \
     --out_dir outputs \
     --chunk_size 200 \
     --overlap 50
   ```

2. **Run retrieval + generation**

   ```bash
   python src/qa.py \
     --csv_path data/data.csv \
     --index_dir outputs \
     --query "What are the symptoms of L-arginine:glycine amidinotransferase deficiency?" \
     --top_k 5 \
     --max_new_tokens 512
   ```


---

## 🔧 Configuration

Both scripts accept these flags (see `--help` for full list):

| Flag               | Default                                                | Description                                 |
| ------------------ | ------------------------------------------------------ | ------------------------------------------- |
| `--chunk_size`     | 200                                                    | Words per chunk                             |
| `--overlap`        | 50                                                     | Overlap (words) between chunks              |
| `--top_k`          | 5                                                      | Number of chunks to retrieve for each query |
| `--max_new_tokens` | 512                                                    | Maximum tokens to generate with Llama-2     |
| `--embed_model`    | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | Model for embedding chunks & queries        |
| `--llm_model`      | `meta-llama/Llama-2-7b-chat-hf`                        | Chat model for answer generation            |


