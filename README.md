# Emotion-Aware Mental Health Chatbot

A chatbot for mental health support with **text and voice input**, optional **LLM-based empathetic responses**, and **IEMOCAP-based emotion recognition** (research: MCS 3204).

**Paths for this setup:**
- **Project:** `D:\My Research Project\my-research`
- **Dataset:** `D:\My Research Project\IEMOCAP_full_release`

### Windows / PowerShell notes (avoids common copy-paste failures)

- **PowerShell vs Git Bash:** **`$env:NAME="value"`** only works in **PowerShell**. In **Git Bash** (or WSL), use **`export NAME="value"`** or a one-line prefix: **`NAME="value" python ...`**. If Bash prints `No such file or directory` for a line starting with `$env:`, you are not in PowerShell—switch shell or use the Bash examples under Step 1.
- **Paths with spaces:** Always quote the directory, e.g. `cd "D:\My Research Project\my-research"`. The same applies in **Command Prompt** and **Git Bash**.
- **`&&` in PowerShell:** Built-in **Windows PowerShell 5.1** does **not** support `&&` between commands (you get a parser error). Run **one command per line**, or chain with **`;`**, or use **[PowerShell 7+](https://github.com/PowerShell/PowerShell)** where `&&` works.
- **`python` not found:** Try `py` instead (Windows launcher), e.g. `py -m pip install -r ml/requirements.txt` and `py -m uvicorn ml.app:app --reload --port 5002`.
- **Port 5002 already in use:** Stop the other terminal that is running the emotion API, or start uvicorn on another port (e.g. `--port 5003`) and set **`EMOTION_API_URL`** in `server\.env` to match (e.g. `http://localhost:5003`).

---

## How to run the project

Pick how much of the stack you need.

### A. Frontend only (no Python, no login API)

From the project root:

```powershell
cd "D:\My Research Project\my-research"
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000). Replies use **built-in rules** in the browser (`src/chatResponses.js`). No `server/` or `ml/` required.

### B. Full stack (three terminals)

Runs the **emotion API** (Python), **chat API** (Node in `server/`), and **React** app together. Use this after cloning or after new preprocessing / training (public datasets merge, `--extra-csv`, etc.). **Prerequisites (one time per machine):**

1. **Root:** `cd "D:\My Research Project\my-research"` then `npm install` (React app).
2. **Server:** `cd "D:\My Research Project\my-research\server"` then `npm install`.
3. **Python emotion API:** from project root, `pip install -r ml/requirements.txt` (add `-r ml/requirements-text-emotion.txt` if you trained **`ml/train_text_emotion.py`**).
4. **Env files:** create **`server\.env`** and **root `.env`** as in [Step 2](#step-2-create-environment-files-one-time) (`EMOTION_API_URL`, `REACT_APP_CHAT_API_URL`, optional `OPENAI_API_KEY`).
5. **Trained artifacts (optional but recommended for your research setup):**
   - **Voice emotion:** run [Step 1](#step-1-train-the-ser-model-and-extract-dataset-responses-one-time) (`train_ser.py` + `extract_dataset_responses.py`) so `ml/models/` has SER weights and `dataset_responses.json`.
   - **Public text data + merged LoRA data:** run **`python ml/preprocess_external_datasets.py --merge-base ml/data/empathic_support_train.jsonl`**, then train text emotion / LoRA as in [Optional: Public text datasets](#optional-public-text-datasets-goemotions-empatheticdialogues-counselchat).

**Every time you run the app (three terminals)** — start from project root in **PowerShell**, or in **Git Bash** use `cd "/d/My Research Project/my-research"` (same folder).

| Terminal | Directory | Command |
|----------|-----------|---------|
| **1 – Emotion API** | project root (folder that contains `ml`) | `python -m uvicorn ml.app:app --reload --port 5002` |
| **2 – Chat API** | `server` | `npm start` |
| **3 – React** | project root | `npm start` |

Then open [http://localhost:3000](http://localhost:3000). The UI talks to **`REACT_APP_CHAT_API_URL`** (default `http://localhost:5001`); the server calls **`EMOTION_API_URL`** (default `http://localhost:5002`) for voice/text emotion.

**After you change ML files:** stop Terminal 1 and start uvicorn again so **`ml/app.py`** reloads SER / text-emotion weights from `ml/models/`.

**Optional chat quality:** with **`OPENAI_API_KEY`** in `server\.env`, replies are **LLM-generated**. Without it, the server uses **rules + empathy** fallback (IEMOCAP line paste is **off** unless you set `USE_IEMOCAP_RETRIEVAL=true`).

### C. Quick reference (same as B, copy-paste)

**Terminal 1**

```powershell
cd "D:\My Research Project\my-research"
python -m uvicorn ml.app:app --reload --port 5002
```

**Terminal 2**

```powershell
cd "D:\My Research Project\my-research\server"
npm start
```

**Terminal 3**

```powershell
cd "D:\My Research Project\my-research"
npm start
```

---

## Quick start (chatbot only, no backend)

```bash
cd "D:/My Research Project/my-research"
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000). The chatbot works with built-in rule-based responses (no API key needed).

---

## Full setup (with IEMOCAP + LLM)

### Step 1: Train the SER model and extract dataset responses (one time)

Open PowerShell, go to the project folder (quoted path), then run each block **line by line** (or use `;` instead of `&&` if you combine commands on one line):

```powershell
cd "D:\My Research Project\my-research"
```

Install Python dependencies:

```powershell
cd ml
pip install -r requirements.txt
cd ..
```

Set the dataset path and train the SER model.

**PowerShell:**

```powershell
$env:IEMOCAP_PATH="D:\My Research Project\IEMOCAP_full_release"
python ml/train_ser.py
```

**Git Bash** (same machine; use `export` instead of `$env:`):

```bash
cd "/d/My Research Project/my-research"
export IEMOCAP_PATH="/d/My Research Project/IEMOCAP_full_release"
python ml/train_ser.py
```

(You can use Windows-style paths in Git Bash too, e.g. `export IEMOCAP_PATH="D:/My Research Project/IEMOCAP_full_release"`.)

Extract IEMOCAP dialogue lines into JSON (used for **optional** scripted retrieval and for **text-emotion training**, not as the main reply path when the LLM is on).

**PowerShell:**

```powershell
$env:IEMOCAP_PATH="D:\My Research Project\IEMOCAP_full_release"
python ml/extract_dataset_responses.py
```

**Git Bash:**

```bash
export IEMOCAP_PATH="/d/My Research Project/IEMOCAP_full_release"
python ml/extract_dataset_responses.py
```

This creates `ml/models/dataset_responses.json`. With **`USE_IEMOCAP_RETRIEVAL=true`** in `server\.env`, the **non-LLM** fallback can paste a matching IEMOCAP line when word overlap is strong; **by default that is off**, and without the LLM the server uses **rules + empathy** instead. The JSON file is still useful for `train_text_emotion.py` and related tooling. If your IEMOCAP has no transcript files, you can create `ml/models/dataset_responses.json` manually with format: `{"sad": ["response1", "response2"], "happy": [...], "angry": [...], "neutral": [...]}`.

### Optional: Public text datasets (GoEmotions, EmpatheticDialogues, CounselChat)

For stronger **text emotion** labels and more **SFT / LoRA** dialogue, this repo includes a downloader/normalizer that pulls open datasets from the **Hugging Face Hub** (needs internet; optional `HF_TOKEN` for higher rate limits):

```powershell
cd "D:\My Research Project\my-research"
pip install -r ml/requirements.txt -r ml/requirements-text-emotion.txt
python ml/preprocess_external_datasets.py --merge-base ml/data/empathic_support_train.jsonl
```

- **`go_emotions` (simplified)** → `ml/data/processed/goemotions_for_text_emotion.csv` mapped to the same nine labels as `train_text_emotion.py` (coarse mapping from 28 fine Reddit labels).
- **EmpatheticDialogues** → the legacy Hub dataset `empathetic_dialogues` (Python script) is **not** loadable on recent `datasets` versions; **`ml/preprocess_external_datasets.py`** instead loads the parquet mirror **`pixelsandpointers/empathetic_dialogues_for_lm`** (multi-turn lines turned into user/assistant pairs) and writes `empathetic_dialogues_sft.jsonl`.
- **`nbertagnolli/counsel-chat`** → therapist-style Q&A as `counsel_chat_sft.jsonl`.
- With **`--merge-base`**, you also get **`merged_empathic_support_train.jsonl`**: your hand-tuned `empathic_support_train.jsonl` plus the two JSONL sources, shuffled for training.

**Train text emotion with GoEmotions:**

```powershell
python ml/train_text_emotion.py --extra-csv ml/data/processed/goemotions_for_text_emotion.csv
```

**LoRA on merged data:**

```powershell
pip install -r ml/requirements.txt -r ml/requirements-train.txt
python ml/finetune_lora_support.py --data ml/data/processed/merged_empathic_support_train.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
```

Use **`--max-per-source N`** on the preprocess script for a quick dry run. Generated files under `ml/data/processed/` are **gitignored** (large); re-run the script if you clone fresh.

### Optional: Train **text** emotion (BERT / DistilBERT) — text-first research

Speech stays on **SER** (`train_ser.py`). For **text** input, you can train a small transformer classifier (recommended if you focus on text):

```powershell
pip install -r ml/requirements.txt -r ml/requirements-text-emotion.txt
python ml/train_text_emotion.py
```

This reads user lines from `dataset_responses.json` plus `ml/data/text_emotion_augment.csv`, and any **`--extra-csv`** files (e.g. processed GoEmotions), saves **`ml/models/text_emotion/`**, and **`ml/app.py`** loads it automatically when present (otherwise keyword fallback). **Response generation** remains LoRA/GPT as in the section below.

### Optional: LoRA fine-tuning for empathetic generation (adapts to unseen questions)

This teaches a **small generative** model (GPT-style completion) a **listener style**—validation, reassurance, gentle questions—using curated multi-turn examples in **`ml/data/empathic_support_train.jsonl`**. It is **not** “store every possible user sentence”; add more JSONL lines in the same `messages` format to strengthen behavior before you train.

```powershell
cd "D:\My Research Project\my-research"
pip install -r ml/requirements.txt -r ml/requirements-train.txt
python ml/finetune_lora_support.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
```

**GPU** (~6–8GB+) recommended. Output: **`ml/models/lora_empathic_support/`** (gitignored). Next steps: **merge** LoRA into the base weights (Hugging Face `PeftModel.merge_and_unload`) or serve with **vLLM / Ollama / LM Studio** (OpenAI-compatible API), then set **`OPENAI_BASE_URL`** and **`OPENAI_MODEL`** in `server/.env` so your existing Node server calls the adapted model instead of only commercial GPT.

---

### Step 2: Create environment files (one time)

**`server\.env`** – create this file in `D:\My Research Project\my-research\server\` (same folder as `index.js`). The server loads this file **by path**, so `OPENAI_API_KEY` applies even if your terminal’s current directory is not `server/`.

```
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
# Optional: local generative model (OpenAI-compatible API, e.g. Ollama, vLLM, LLaMA)
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_MODEL=llama3.2
EMOTION_API_URL=http://localhost:5002
JWT_SECRET=use-a-long-random-string-in-production
# Optional: set to true to paste IEMOCAP lines in non-LLM fallback when word overlap is strong (usually not recommended)
# USE_IEMOCAP_RETRIEVAL=false
```

**Generative vs lookup:** With `OPENAI_API_KEY` set, replies are **generated** each turn (GPT or any OpenAI-compatible model via `OPENAI_BASE_URL`)—the app does **not** paste answers from `dataset_responses.json` in that path. When the LLM is **unavailable**, **IEMOCAP line retrieval is off by default** (scripted lines often mismatch users). Set **`USE_IEMOCAP_RETRIEVAL=true`** only if you explicitly want that behavior.

Replace `sk-your-openai-key-here` with your real OpenAI API key from https://platform.openai.com/api-keys. Set **`JWT_SECRET`** to a long random string for login tokens (optional for local dev; the server falls back to a dev default with a warning).

**Accounts & chat history:** With `REACT_APP_CHAT_API_URL` set, the app shows **Register / Log in**. Use **Continue without an account** for guest mode (no saving). For signed-in users, messages go to **`server/data/chat.db`** (SQLite), one row per user and bot turn, including **emotion** on user messages. **Emotional memory** in the LLM system prompt merges saved emotion trajectories from the DB with this-session keyword cues, plus **multi-turn reasoning** (recent user lines summarized) and **personalized adaptation** (returning-user depth). Guests still get session-only keyword emotion + multi-turn prompts. The database file is gitignored.

**Project root `.env`** – create this file in `D:\My Research Project\my-research\`:

```
REACT_APP_CHAT_API_URL=http://localhost:5001
```

---

### Step 3: Run the project (every time)

Use **[How to run the project, section B](#b-full-stack-three-terminals)** (three terminals + `.env` files). You do **not** need to re-train or re-preprocess unless you changed data or scripts.

---

## Quick run (next time you use the project)

Same as **section B / C** in [How to run the project](#how-to-run-the-project): three terminals (uvicorn on **5002**, Node server in **`server/`**, `npm start` at **root**). Ensure **`server\.env`** and **root `.env`** still exist.

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
