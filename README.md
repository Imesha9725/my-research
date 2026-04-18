# Emotion-Aware Mental Health Chatbot

A chatbot for mental health support with **text and voice input**, optional **LLM-based empathetic responses**, and **IEMOCAP-based emotion recognition** (research: MCS 3204).

**Paths for this setup:**
- **Project:** `D:\My Research Project\my-research`
- **Dataset:** `D:\My Research Project\IEMOCAP_full_release`

---

## Quick start (chatbot only, no backend)

```bash
cd D:\My Research Project\my-research
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000). The chatbot works with built-in rule-based responses (no API key needed).

---

## Full setup (with IEMOCAP + LLM)

### Step 1: Train the SER model and extract dataset responses (one time)

Open PowerShell and run:

```powershell
cd D:\My Research Project\my-research
```

Install Python dependencies:

```powershell
cd ml
pip install -r requirements.txt
cd ..
```

Set the dataset path and train the SER model:

```powershell
$env:IEMOCAP_PATH="D:\My Research Project\IEMOCAP_full_release"
python ml/train_ser.py
```

Extract responses from the dataset (so the chatbot uses IEMOCAP dialogue as replies):

```powershell
$env:IEMOCAP_PATH="D:\My Research Project\IEMOCAP_full_release"
python ml/extract_dataset_responses.py
```

This creates `ml/models/dataset_responses.json`. The chatbot will use these dataset responses first; if none match, it falls back to built-in responses. If your IEMOCAP has no transcript files, you can create `ml/models/dataset_responses.json` manually with format: `{"sad": ["response1", "response2"], "happy": [...], "angry": [...], "neutral": [...]}`.

### Optional: Train **text** emotion (BERT / DistilBERT) — text-first research

Speech stays on **SER** (`train_ser.py`). For **text** input, you can train a small transformer classifier (recommended if you focus on text):

```powershell
pip install -r ml/requirements.txt -r ml/requirements-text-emotion.txt
python ml/train_text_emotion.py
```

This reads user lines from `dataset_responses.json` plus `ml/data/text_emotion_augment.csv`, saves **`ml/models/text_emotion/`**, and **`ml/app.py`** loads it automatically when present (otherwise keyword fallback). **Response generation** remains LoRA/GPT as in the section below.

### Optional: LoRA fine-tuning for empathetic generation (adapts to unseen questions)

This teaches a **small generative** model (GPT-style completion) a **listener style**—validation, reassurance, gentle questions—using curated multi-turn examples in **`ml/data/empathic_support_train.jsonl`**. It is **not** “store every possible user sentence”; add more JSONL lines in the same `messages` format to strengthen behavior before you train.

```powershell
cd D:\My Research Project\my-research
pip install -r ml/requirements.txt -r ml/requirements-train.txt
python ml/finetune_lora_support.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
```

**GPU** (~6–8GB+) recommended. Output: **`ml/models/lora_empathic_support/`** (gitignored). Next steps: **merge** LoRA into the base weights (Hugging Face `PeftModel.merge_and_unload`) or serve with **vLLM / Ollama / LM Studio** (OpenAI-compatible API), then set **`OPENAI_BASE_URL`** and **`OPENAI_MODEL`** in `server/.env` so your existing Node server calls the adapted model instead of only commercial GPT.

---

### Step 2: Create environment files (one time)

**`server\.env`** – create this file in `D:\My Research Project\my-research\server\`:

```
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
# Optional: local generative model (OpenAI-compatible API, e.g. Ollama, vLLM, LLaMA)
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_MODEL=llama3.2
EMOTION_API_URL=http://localhost:5002
JWT_SECRET=use-a-long-random-string-in-production
# Optional: set to false to turn off IEMOCAP line-matching in non-LLM fallback only
# USE_IEMOCAP_RETRIEVAL=true
```

**Generative vs lookup:** With `OPENAI_API_KEY` set, replies are **generated** each turn (GPT or any OpenAI-compatible model via `OPENAI_BASE_URL`)—the app does **not** paste answers from `dataset_responses.json` in that path. IEMOCAP JSON is only used in the **fallback** chain when the LLM is unavailable, unless `USE_IEMOCAP_RETRIEVAL=false`.

Replace `sk-your-openai-key-here` with your real OpenAI API key from https://platform.openai.com/api-keys. Set **`JWT_SECRET`** to a long random string for login tokens (optional for local dev; the server falls back to a dev default with a warning).

**Accounts & chat history:** With `REACT_APP_CHAT_API_URL` set, the app shows **Register / Log in**. Messages are stored in **`server/data/chat.db`** (SQLite), one row per user and bot turn, including **emotion** on user messages. Logged-in users get **emotional memory** in the LLM system prompt: saved emotion trajectories from the DB are merged with this-session keyword cues, plus **multi-turn reasoning** (recent user lines summarized) and **personalized adaptation** (returning-user depth). Guests still get session-only keyword emotion + multi-turn prompts. The database file is gitignored.

**Project root `.env`** – create this file in `D:\My Research Project\my-research\`:

```
REACT_APP_CHAT_API_URL=http://localhost:5001
```

---

### Step 3: Run the project (every time)

Open **three** terminals, all starting from:

```powershell
cd D:\My Research Project\my-research
```

**Terminal 1 – Emotion API (Python):**
```powershell
python -m uvicorn ml.app:app --reload --port 5002
```
Leave it running.

**Terminal 2 – Node backend:**
```powershell
cd server
npm install
npm start
```
Leave it running.

**Terminal 3 – React frontend:**
```powershell
npm install
npm start
```

The browser will open at [http://localhost:3000](http://localhost:3000). You can chat there; responses use the IEMOCAP-based emotion model and the LLM.

---

## Quick run (next time you use the project)

You do **not** need to train again. Just start the three services:

1. **Terminal 1:** `cd D:\My Research Project\my-research` then `python -m uvicorn ml.app:app --reload --port 5002`
2. **Terminal 2:** `cd D:\My Research Project\my-research\server` then `npm start`
3. **Terminal 3:** `cd D:\My Research Project\my-research` then `npm start`

---

## Optional: Use the scripts (Windows)

**Train once (edit path first):**  
Open `scripts\train-and-check.bat`, change the line:

```
set IEMOCAP_PATH=D:\My Research Project\IEMOCAP_full_release
```

Then double-click the file.

**Start both backends:**  
Double-click `scripts\start-backends.bat`, then in a terminal run `npm start` for the React app.

---

## Dataset structure

The loader expects the dataset at `D:\My Research Project\IEMOCAP_full_release` in one of these shapes:

- `IEMOCAP_full_release\IEMOCAP\Session1\`, `Session2\`, …  
  with `sentences\wav\` and `dialog\EmoEvaluation\`

or

- `IEMOCAP_full_release\Session1\`, `Session2\`, …  
  with the same subfolders

---

## Environment variables

| File | Variable | Description |
|------|----------|-------------|
| `server\.env` | `OPENAI_API_KEY` | Your OpenAI API key (required for LLM) |
| `server\.env` | `EMOTION_API_URL` | `http://localhost:5002` |
| Root `.env` | `REACT_APP_CHAT_API_URL` | `http://localhost:5001` |

---

## Project structure

- `src/components/Chat.js` – Chat UI, voice recording
- `src/chatResponses.js` – Fallback rule-based responses
- `server/index.js` – Chat API (emotion + LLM)
- `ml/iemocap_loader.py` – Load IEMOCAP data
- `ml/train_ser.py` – Train SER model
- `ml/app.py` – Emotion API (voice + text)
