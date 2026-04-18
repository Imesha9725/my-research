# How to Train Your Model Very Well

Your project has **three main parts** that affect response quality. Here's how to improve each.

---

## 1. SER Model (Speech Emotion Recognition) – Voice Emotion

**Current:** MLPClassifier on IEMOCAP MFCCs, trained with `train_ser.py`

### How to train

```powershell
cd "D:\My Research Project\my-research"
$env:IEMOCAP_PATH="D:\My Research Project\IEMOCAP_full_release"
python ml/train_ser.py
```

### Ways to improve SER

| Approach | How |
|----------|-----|
| **More epochs** | Edit `train_ser.py`: increase `max_iter=200` to `500` or `1000` |
| **Deeper network** | Change `hidden_layer_sizes=(256, 128)` to `(512, 256, 128)` |
| **More features** | Add prosody (pitch, energy) in addition to MFCCs |
| **Data augmentation** | Add noise, speed change, pitch shift to audio |
| **Cross-validation** | Use 5-fold instead of single train/test split |
| **Save best model** | Retrain until accuracy improves, keep best model |

---

## 2. Text Emotion Detection – Text Input

**Current:** Keyword-based (no ML model). Text uses the same keywords as the SER fallback.

### Option A: Improve keywords (fast)

- Add more phrases to `KEYWORD_MAP` in `server/index.js` and `TEXT_KEYWORDS_ORDERED` in `ml/app.py`
- Use synonyms and common phrasings

### Option B: Train a text classifier (implemented)

**Script:** `ml/train_text_emotion.py` — **DistilBERT** (default) multi-class on:

- User lines from `ml/models/dataset_responses.json` (IEMOCAP emotions: `neutral`, `sad`, `happy`, `angry`)
- Plus `ml/data/text_emotion_augment.csv` (seed phrases for `anxious`, `stress`, `lonely`, `tired`, `fear`)

**Install:** `pip install -r ml/requirements.txt -r ml/requirements-text-emotion.txt`

**Run:**

```powershell
cd D:\My Research Project\my-research
python ml/train_text_emotion.py
```

**Output:** `ml/models/text_emotion/` — restart **uvicorn** (`ml/app.py`); health shows `text_emotion_model_loaded: true`. Text then uses the **neural** classifier; **speech** still uses **SER** (joblib). Add more rows to the CSV or use GoEmotions-style data for stronger text-focused research.

---

## 3. Response Selection – Matching User to Reply

**Current:** Rule-based keyword matching (topics + emotions)

### Ways to improve

| Approach | How |
|----------|-----|
| **More keywords** | Add more variations (e.g. "got the medicine", "went near vet") |
| **Use conversation history** | Pass last 2–3 messages so “what to do for my pet?” uses prior “my cat is sick” context |
| **Use LLM** | Set `OPENAI_API_KEY` – LLM can infer context and give better replies |
| **Train retrieval model** | Embed user text and responses, train to score best match |
| **LoRA fine-tuning** | Train a small instruct model on `ml/data/empathic_support_train.jsonl` (`ml/finetune_lora_support.py`) for empathy style on *unseen* questions—see README “Optional: LoRA fine-tuning” |

---

## 4. Generative model (LoRA) – adapt without memorizing every answer

**Goal:** Better replies when the user’s exact words are **not** in IEMOCAP or fixed rules.

1. Add or edit lines in **`ml/data/empathic_support_train.jsonl`** (each line: `{"messages":[{"role":"system"|"user"|"assistant","content":"..."}, ...]}`).
2. Install training deps: `pip install -r ml/requirements.txt -r ml/requirements-train.txt`
3. Run: `python ml/finetune_lora_support.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3`
4. Serve the merged model or adapter via an **OpenAI-compatible** API and set **`OPENAI_BASE_URL`** in `server/.env`.

---

## Quick Checklist for “Very Well” Training

1. **SER**
   - [ ] Run `train_ser.py` with IEMOCAP path
   - [ ] Try `max_iter=500` and deeper MLP
   - [ ] Keep the model with highest test accuracy

2. **Keywords**
   - [ ] Add variations for common user phrases
   - [ ] Test with phrases like “went to vet”, “got medicine”, “cat is sick”

3. **Context**
   - [ ] Use last 1–2 user messages when choosing topic/response

4. **LLM**
   - [ ] Add `OPENAI_API_KEY` in `server/.env` for stronger responses

5. **LoRA (optional)**
   - [ ] Expand `ml/data/empathic_support_train.jsonl` with your best multi-turn examples
   - [ ] Run `ml/finetune_lora_support.py` on a GPU; point the server at your served model

---

## Files to Edit for Training

| File | Purpose |
|------|---------|
| `ml/train_ser.py` | SER model (MLP, epochs, features) |
| `ml/app.py` | Load SER model, text keywords |
| `server/index.js` | Topic keywords, response mapping |
| `ml/finetune_lora_support.py` | LoRA SFT on empathic JSONL |
| `ml/data/empathic_support_train.jsonl` | Training conversations |
