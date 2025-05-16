# RAG_Medical_Chatbot

FactMed.AI is an interactive medical assistant built using React (frontend) and Flask (backend).
It uses Retrieval-Augmented Generation (RAG) with biomedical data and Google Gemini API to provide accurate, context-aware answers.

---
### ✅ Features:

* Chat interface with conversation history
* Medical answers generated using Gemini (Google GenAI)
* FAISS-based semantic search over biomedical data

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
git clone https://github.com/poojavguttal/RAG_Medical_Chatbot.git
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









