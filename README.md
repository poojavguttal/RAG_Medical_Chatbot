# RAG_Medical_Chatbot

FactMed.AI is an interactive medical assistant built using React (frontend) and Flask (backend).
It uses Retrieval-Augmented Generation (RAG) with biomedical data and Google Gemini API to provide accurate, context-aware answers.

Features:
Chat interface with conversation history

Medical answers generated using Gemini (Google GenAI)

FAISS-based semantic search over biomedical data


Tech Stack:
Frontend: React, JavaScript, CSS, Axios

Backend: Python, Flask, Flask-CORS, pyngrok

ML/NLP: HuggingFace Transformers, PubMedBERT, FAISS, Google Gemini

Hosting: Google Colab or Localhost with Ngrok tunnel



## Usage

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

## Configuration

Both scripts accept these flags (see `--help` for full list):

| Flag               | Default                                                | Description                                 |
| ------------------ | ------------------------------------------------------ | ------------------------------------------- |
| `--chunk_size`     | 200                                                    | Words per chunk                             |
| `--overlap`        | 50                                                     | Overlap (words) between chunks              |
| `--top_k`          | 5                                                      | Number of chunks to retrieve for each query |
| `--max_new_tokens` | 512                                                    | Maximum tokens to generate with Llama-2     |
| `--embed_model`    | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | Model for embedding chunks & queries        |
| `--llm_model`      | `meta-llama/Llama-2-7b-chat-hf`                        | Chat model for answer generation            |



Here’s the full updated **README content** with Gemini API usage, formatted for direct use:

---

# FactMed.AI - Medical Q\&A Chatbot with RAG + Gemini

**FactMed.AI** is an interactive medical assistant built using **React** (frontend) and **Flask** (backend).
It uses **Retrieval-Augmented Generation (RAG)** with biomedical data and **Google Gemini API** to provide accurate, context-aware answers.

---

### ✅ Features:

* Chat interface with conversation history
* Medical answers generated using Gemini (Google GenAI)
* FAISS-based semantic search over biomedical data
* Real-time interaction via ngrok or local hosting

---

### ⚙️ Tech Stack:

* **Frontend**: React, JavaScript, CSS, Axios
* **Backend**: Python, Flask, Flask-CORS, pyngrok
* **ML/NLP**: HuggingFace Transformers, PubMedBERT, FAISS, Google Gemini
* **Hosting**: Google Colab or Localhost with Ngrok tunnel

---

### 🛠️ Setup Instructions:

#### 1. Clone the Repository

```bash
git clone https://github.com/poojavguttal/factmed-ai.git
cd factmed-ai
```


#### 2. Backend (Colab or Local)

##### Google Colab Steps:

* Upload `all_data.csv`
* Run

* ⚠️ Update the API URL in `App.js`:

```js
fetch("https://<your-ngrok-url>.ngrok-free.app/api/ask", {...})
```

#### 2. Frontend (React)

```bash
cd frontend
npm install
npm start
```

### 🧪 Sample Query:

```
What causes L-arginine:glycine amidinotransferase deficiency?
```


---

### 📝 Notes:

* If ngrok fails: visit [https://dashboard.ngrok.com/agents](https://dashboard.ngrok.com/agents) to kill older sessions
* To switch from Gemini to OpenAI or LLaMA2, comment/uncomment respective lines in `retrieve_generate.py`

---









