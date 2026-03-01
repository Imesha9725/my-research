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

### Step 1: Train the SER model (one time)

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

Set the dataset path and train:

```powershell
$env:IEMOCAP_PATH="D:\My Research Project\IEMOCAP_full_release"
python ml/train_ser.py
```

You should see something like: `Loaded XXXXX samples` and `Saved model and config to ...\ml\models`.

---

### Step 2: Create environment files (one time)

**`server\.env`** – create this file in `D:\My Research Project\my-research\server\`:

```
OPENAI_API_KEY=sk-your-openai-key-here
EMOTION_API_URL=http://localhost:5002
```

Replace `sk-your-openai-key-here` with your real OpenAI API key from https://platform.openai.com/api-keys.

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
