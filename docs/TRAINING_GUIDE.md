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

### Option B: Train a text classifier (better)

1. **Dataset:** Use IEMOCAP transcriptions + labels from `dataset_responses.json` (each pair has emotion).
2. **Model:** Simple BERT or TF-IDF + Logistic Regression.
3. **Train:** Label each “user” text with its emotion and train a classifier.

Example steps:

```python
# Pseudo: load user texts from dataset_responses.json, map to emotion
# Train sklearn LogisticRegression or use transformers for BERT
# Save model, use in ml/app.py for text emotion prediction
```

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

---

## Files to Edit for Training

| File | Purpose |
|------|---------|
| `ml/train_ser.py` | SER model (MLP, epochs, features) |
| `ml/app.py` | Load SER model, text keywords |
| `server/index.js` | Topic keywords, response mapping |
