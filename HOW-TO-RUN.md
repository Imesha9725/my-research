# How to Run – Step by Step

Do these steps **in order**. Use **PowerShell** or **Command Prompt** (Windows) or **Terminal** (Mac/Linux).

---

## First time vs next day

| When | What to do |
|------|------------|
| **First time only** | Install deps (Step 2), train the model (Step 3), create `server/.env` and root `.env` (Steps 5 & 6). Do this once. |
| **Every time you run the project** (next day, new session) | **Do NOT train again.** Just start the 3 running parts: Emotion API, Node backend, React app (Steps 4, 5, 6). |

So **the next day you only start the servers and the website** – no training, no `IEMOCAP_PATH`, no installing again (unless you changed something).

---

## Quick run – next day (Windows)

1. Double‑click **`scripts\start-backends.bat`** (opens Emotion API + Node backend in two windows; leave them open).
2. In a terminal in the project folder run: **`npm start`**.  
   Browser opens at http://localhost:3000.

That’s it. No training, no `.env` editing (unless you changed your API key).

---

## Quick run – first time (Windows)

1. **Train once** (edit path inside the script first):  
   Double‑click `scripts\train-and-check.bat`
2. **Create** `server/.env` and root `.env` (see Steps 5 & 6 below).
3. Then use “Quick run – next day” above whenever you run the project.

---

## Before you start

1. You must have **downloaded and extracted** the IEMOCAP dataset (e.g. from Kaggle).
2. Know the **full path** to the folder that contains `Session1`, `Session2`, etc.  
   Example: `D:\Datasets\iemocap-emotion-speech-database`
3. You need an **OpenAI API key** from https://platform.openai.com/api-keys

---

## Step 1: Open a terminal in the project folder

1. Open File Explorer and go to: `D:\My Research Project\my-research`
2. In the address bar type `powershell` and press Enter (or right‑click → “Open in Terminal”).
3. You should see a prompt like: `PS D:\My Research Project\my-research>`

---

## Step 2: Install Python dependencies (one time)

In that same terminal, run:

```powershell
cd ml
pip install -r requirements.txt
cd ..
```

Wait until it finishes. If you get an error, make sure Python is installed and `pip` works.

---

## Step 3: Set the dataset path and train the model (one time)

**Replace the path below with YOUR actual IEMOCAP folder path.**

In the same terminal (project root `my-research`), run **one** of these:

**If your dataset is at `D:\Datasets\iemocap-emotion-speech-database`:**
```powershell
$env:IEMOCAP_PATH="D:\Datasets\iemocap-emotion-speech-database"
python ml/train_ser.py
```

**If your dataset is inside the project, e.g. `my-research\data\IEMOCAP`:**
```powershell
$env:IEMOCAP_PATH="data"
python ml/train_ser.py
```

**If your dataset is somewhere else (e.g. `E:\IEMOCAP`):**
```powershell
$env:IEMOCAP_PATH="E:\IEMOCAP"
python ml/train_ser.py
```

Wait until you see something like: `Saved model and config to ...\ml\models`.  
If you see “IEMOCAP not found”, fix the path and run the two lines again.

---

## Step 4: Start the Emotion API (Python)

Keep the terminal open. Run:

```powershell
python -m uvicorn ml.app:app --reload --port 5002
```

Leave this running. You should see: `Uvicorn running on http://0.0.0.0:5002`.

**Do not close this terminal.**

---

## Step 5: Open a SECOND terminal for the Node backend

1. Open another PowerShell/Terminal (same project folder: `D:\My Research Project\my-research`).
2. Go to the server folder and install:

```powershell
cd server
npm install
```

3. Create the `.env` file:

**PowerShell:**
```powershell
Copy-Item .env.example .env
notepad .env
```

**Or:** Create a new file named `.env` inside the `server` folder with this content (paste your real OpenAI key):

```
OPENAI_API_KEY=sk-your-actual-openai-key-here
EMOTION_API_URL=http://localhost:5002
```

Save and close the file.

4. Start the Node backend:

```powershell
npm start
```

You should see: `Server running at http://localhost:5001`.  
**Leave this terminal running.**

---

## Step 6: Open a THIRD terminal for the React app (frontend)

1. Open another PowerShell/Terminal in the project folder: `D:\My Research Project\my-research`.

2. Create a `.env` file in the **project root** (next to `package.json`):

**PowerShell:**
```powershell
Set-Content -Path .env -Value "REACT_APP_CHAT_API_URL=http://localhost:5001"
```

**Or:** Create a file named `.env` in `my-research` with this single line:

```
REACT_APP_CHAT_API_URL=http://localhost:5001
```

3. Install and start the frontend (if you haven’t already):

```powershell
npm install
npm start
```

The browser will open at http://localhost:3000. You can chat there; responses use the dataset (emotion) and the LLM.

---

## Summary – what should be running

| Terminal / window | Command / app              | Port |
|-------------------|----------------------------|------|
| 1                 | `uvicorn ml.app:app ...`   | 5002 |
| 2                 | `cd server` then `npm start` | 5001 |
| 3                 | `npm start` (React)        | 3000 |

Use the app in the browser at **http://localhost:3000**.

**Next day:** Run only the 3 commands above (or use `scripts\start-backends.bat` + `npm start`). Do not run training again.

---

## Optional: run training with a script (Windows)

You can use the script so you only set the path once:

1. Open `scripts/train-and-check.bat` in a text editor.
2. Change the line `set IEMOCAP_PATH=...` to your real path.
3. Double‑click the file or run in terminal: `scripts\train-and-check.bat`

This only **trains the model**. You still need to start the two servers (Steps 4 and 5) and the frontend (Step 6) yourself.
