# Emotion-Aware Mental Health Chatbot

A chatbot for mental health support with **text and voice input**, optional **LLM-based empathetic responses**, and **IEMOCAP-based emotion recognition** (research: MCS 3204).

## Quick start

```bash
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000). The chatbot works with built-in rule-based responses (no API key needed).

---

## Using the IEMOCAP dataset (text + voice emotion)

Your research uses the [IEMOCAP emotion speech database](https://www.kaggle.com/datasets/samuelsamsudinng/iemocap-emotion-speech-database). The project is set up to use it in two ways:

1. **Train a Speech Emotion Recognition (SER) model** on IEMOCAP audio (labels: neutral, happy, angry, sad).
2. **Run an emotion API** that uses that model for **voice** emotion and keyword-based **text** emotion, so the chatbot response is driven by both text and voice data.

### 1. Place the dataset

After downloading from Kaggle, put the data so the project can see it. Either:

- **Option A:** Extract so the project root has an `IEMOCAP` folder:
  ```
  my-research/
  ├── IEMOCAP/
  │   ├── Session1/
  │   │   ├── sentences/wav/...
  │   │   └── dialog/EmoEvaluation/...
  │   ├── Session2/
  │   ...
  ```
- **Option B:** Extract anywhere (e.g. `D:\Datasets\iemocap-emotion-speech-database`) and set that path when training (see below).

The loader supports both `root/IEMOCAP/Session1..5/` and `root/Session1..5/` (e.g. Kaggle extract).

### 2. Train the SER model (Python)

From the **project root**:

```bash
cd ml
pip install -r requirements.txt
cd ..
```

Set the path to the folder that contains `IEMOCAP` or `Session1..5` (use `.\path` on Windows):

```bash
# Windows (PowerShell)
$env:IEMOCAP_PATH="D:\path\to\folder\containing\IEMOCAP"
python ml/train_ser.py

# Linux / macOS
export IEMOCAP_PATH=/path/to/folder/containing/IEMOCAP
python ml/train_ser.py
```

If you put IEMOCAP inside the project (e.g. `my-research/data/IEMOCAP`):

```bash
$env:IEMOCAP_PATH="data"   # Windows
python ml/train_ser.py
```

This script:

- Loads IEMOCAP (audio paths + labels from `dialog/EmoEvaluation/`).
- Maps labels to 4 classes: **neutral**, **happy**, **angry**, **sad** (exc→happy, fru→angry).
- Extracts MFCCs, trains an MLP classifier, and saves the model under `ml/models/` (`ser_model.joblib`, `ser_scaler.joblib`, `ser_label_encoder.joblib`, `ser_config.json`).

### 3. Run the emotion API

Start the emotion service (uses the trained SER model for voice; text uses keyword-based emotion):

```bash
# From project root
python -m uvicorn ml.app:app --reload --port 5002
```

Or:

```bash
cd ml && uvicorn app:app --host 0.0.0.0 --port 5002
```

- **Voice:** POST audio (WAV or, if supported, browser webm) → SER model → emotion (neutral/happy/angry/sad).
- **Text:** Optional text in the same request → keyword-based emotion.
- Response includes `primary` emotion (voice preferred when both are sent).

### 4. Wire the chatbot to the emotion API and LLM

1. **Node backend** – In `server/.env` add:
   ```env
   OPENAI_API_KEY=sk-your-key
   EMOTION_API_URL=http://localhost:5002
   ```
   Then start the server: `cd server && npm install && npm start` (port 5001).

2. **Frontend** – In the project root `.env` (or `.env.local`):
   ```env
   REACT_APP_CHAT_API_URL=http://localhost:5001
   ```
   Restart the React app (`npm start`).

When you **type**, the backend gets text-only and uses the emotion API’s text emotion (or its own keyword fallback). When you **use the mic** and then send, the frontend sends the **recorded audio** (as base64) plus the **transcript**; the Node server calls the emotion API with both; the API returns emotion from **voice** (IEMOCAP-trained SER) and optionally **text**; the backend uses that emotion in the LLM system prompt and returns the reply. So the chatbot uses **both text and voice data** (IEMOCAP for voice) to drive responses.

---

## LLM-based responses (emotion-aware)

To use an **LLM** for contextually appropriate and empathetic replies:

1. **Get an API key** from [OpenAI API keys](https://platform.openai.com/api-keys).
2. **Run the Node backend** (see above); set `OPENAI_API_KEY` in `server/.env`.
3. **Set `REACT_APP_CHAT_API_URL=http://localhost:5001`** in the project root `.env` and restart the React app.

The backend builds a system prompt that includes the **detected emotion** (from the emotion API when `EMOTION_API_URL` is set, else keyword-based) so the LLM adapts its tone. If you also run the IEMOCAP-trained emotion API, that emotion comes from **text and voice** (IEMOCAP data).

### Optional server env

| Variable            | Description                                              |
|---------------------|----------------------------------------------------------|
| `OPENAI_API_KEY`    | Required for LLM.                                       |
| `OPENAI_MODEL`      | Model name (default: `gpt-4o-mini`).                   |
| `EMOTION_API_URL`   | IEMOCAP emotion API URL (e.g. `http://localhost:5002`).|
| `PORT`              | Server port (default: 5001).                            |

---

## Project structure

- `src/components/Chat.js` – Chat UI, voice recording, sends text + optional audio to backend.
- `src/chatResponses.js` – Fallback rule-based responses.
- `server/index.js` – Chat API: calls emotion API (text + audio), then LLM with emotion in the prompt.
- `ml/iemocap_loader.py` – Load IEMOCAP (sessions, WAV paths, labels from EmoEvaluation).
- `ml/train_ser.py` – Train SER model (MFCCs + MLP) on IEMOCAP, save to `ml/models/`.
- `ml/app.py` – Emotion API: voice (SER) + text (keywords), returns primary emotion.

---

## Research (MCS 3204)

This supports your objectives:

- **IEMOCAP text and voice data:** SER model is trained on IEMOCAP; the emotion API uses it for voice and keywords for text. The chatbot response is driven by both.
- **LLM-based, contextually appropriate responses:** The backend uses detected emotion (from IEMOCAP-based pipeline when the emotion API is running) in the system prompt so the LLM generates empathetic, adaptive replies.
- **Contextually-aware psychological support:** The backend builds an **emotional history** from the conversation (one emotion per user message) and passes it to the LLM with instructions to dynamically adjust responses based on the user’s emotional trajectory and conversation context, for more personalized and empathetic support.
