# RAG_Medical_Chatbot
FactMed.AI is an interactive medical assistant built with a React frontend and RAG (Retrieval Augmented Generation) based backend, designed to provide accurate, context aware healthcare information. It incorporates Retrieval Augmented Generation (RAG) over trusted biomedical over the MedQuAD datasets (https://github.com/abachaa/MedQuAD?utm_source=chatgpt.com) and integrates the Google Gemini API to reduce hallucinations and ensure reliability. The system addresses the challenge of inconsistent and misleading medical information available online by grounding responses in verified sources. FactMed.AI enhances accessibility, helping patients, caregivers, and professionals obtain reliable medical insights quickly and effectively.

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
cd cd factmed-ai-frontend
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









