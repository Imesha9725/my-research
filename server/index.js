/**
 * LLM-based chat API for emotion-aware mental health support.
 * Uses detected emotion to tailor the system prompt for empathetic, context-aware responses.
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors({ origin: true }));
app.use(express.json());

// ----- Emotion detection (same logic as frontend; can be replaced with a trained model) -----
const KEYWORD_MAP = [
  { keywords: ['sad', 'sadness', 'depressed', 'down', 'miserable', 'hopeless', 'not good', 'not well', 'not okay', 'not great', 'not fine', 'no good', "don't feel good", "don't feel well", "i'm not good", "feeling not good", "am not good", "not doing good", "not doing well"], emotion: 'sad' },
  { keywords: ['anxious', 'anxiety', 'nervous', 'worried', 'panic'], emotion: 'anxious' },
  { keywords: ['stress', 'stressed', 'overwhelmed', 'pressure'], emotion: 'stress' },
  { keywords: ['angry', 'anger', 'mad', 'frustrated', 'annoyed'], emotion: 'angry' },
  { keywords: ['lonely', 'alone', 'isolated', 'left out'], emotion: 'lonely' },
  { keywords: ['happy', 'good', 'great', 'better', 'relieved', 'hopeful'], emotion: 'happy' },
  { keywords: ['tired', 'exhausted', 'drained', 'burned out'], emotion: 'tired' },
  { keywords: ['scared', 'afraid', 'fear', 'frightened'], emotion: 'fear' },
];

const CRISIS_KEYWORDS = ['suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself', 'self harm'];

function getEmotionFromText(text) {
  if (!text || typeof text !== 'string') return null;
  const lower = text.toLowerCase().trim();
  for (const { keywords, emotion } of KEYWORD_MAP) {
    if (keywords.some((k) => lower.includes(k))) return emotion;
  }
  return null;
}

function isCrisisMessage(text) {
  if (!text || typeof text !== 'string') return false;
  return CRISIS_KEYWORDS.some((k) => text.toLowerCase().includes(k));
}

/**
 * Build emotional history from conversation: one emotion per user message (from text).
 * Used for contextually-aware, personalized responses per the research proposal.
 */
function getEmotionalHistory(history) {
  if (!Array.isArray(history) || history.length === 0) return [];
  const emotions = [];
  for (const m of history) {
    if (m.role === 'user' && (m.text || m.content)) {
      const e = getEmotionFromText(m.text || m.content);
      emotions.push(e || 'neutral');
    }
  }
  return emotions;
}

function buildSystemPrompt(emotion, isCrisis, emotionalHistory = []) {
  const base = `You are a supportive, empathetic mental health companion. You provide emotional support and listen without judgment. You are NOT a therapist or doctor—do not diagnose or prescribe. If someone is in crisis or mentions self-harm or suicide, encourage them to contact a crisis helpline (e.g. Sumithrayo 1926 in Sri Lanka) and a professional. Keep responses concise (2–4 short paragraphs max), warm, and natural. Use "you" and "I" naturally. Do not use bullet points or formal lists unless helpful.`;

  let contextBlock = '';
  if (emotionalHistory.length > 0) {
    const historyStr = emotionalHistory.join(' → ');
    contextBlock = `\n\nContextually-aware support: Use the user's emotional history and conversation context to personalize your response.
- Emotional history this conversation (oldest to newest): ${historyStr}.
- Current message emotion: ${emotion || 'neutral'}.
Dynamically adjust your tone and content: acknowledge the emotional trajectory if it has changed, reference what they shared earlier when relevant, and offer continuity (e.g. "You mentioned...", "Last time you said..."). Avoid repeating the same advice; build on the conversation. Offer more personalized and empathetic responses based on this context.`;
  } else if (emotion) {
    contextBlock = `\n\nThe user appears to be feeling: ${emotion}. Adapt your tone and response to be contextually appropriate and empathetic for this emotional state.`;
  }

  if (isCrisis) {
    return `${base}\n\nImportant: The user's message may indicate a crisis. Prioritize safety: express care, validate feelings, and strongly encourage contacting a crisis helpline or trusted person. Be calm and supportive.${contextBlock}`;
  }
  return base + contextBlock;
}

// ----- Emotion API (IEMOCAP-based voice + text) -----
const EMOTION_API_URL = process.env.EMOTION_API_URL || '';

async function getEmotionFromAPI(text, audioBase64) {
  if (!EMOTION_API_URL) return null;
  try {
    const body = { text: text || undefined, audio_base64: audioBase64 || undefined };
    const res = await fetch(`${EMOTION_API_URL}/api/predict_emotion_json`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.primary || null;
  } catch (_) {
    return null;
  }
}

// ----- Chat endpoint -----
app.post('/api/chat', async (req, res) => {
  const { message, history = [], audioBase64 } = req.body;
  const text = typeof message === 'string' ? message.trim() : '';

  if (!text) {
    return res.status(400).json({ error: 'message is required' });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(503).json({
      error: 'LLM not configured',
      hint: 'Set OPENAI_API_KEY in server/.env to enable LLM responses.',
    });
  }

  let emotion = null;
  if (EMOTION_API_URL && (text || audioBase64)) {
    emotion = await getEmotionFromAPI(text, audioBase64);
  }
  if (emotion == null) {
    emotion = getEmotionFromText(text);
  }
  const crisis = isCrisisMessage(text);
  const emotionalHistory = getEmotionalHistory(history);
  const systemPrompt = buildSystemPrompt(emotion, crisis, emotionalHistory);

  const openai = new OpenAI({ apiKey });

  const messages = [
    { role: 'system', content: systemPrompt },
    ...history.slice(-12).map((m) => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.text || m.content,
    })),
    { role: 'user', content: text },
  ];

  try {
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      messages,
      max_tokens: 400,
      temperature: 0.7,
    });

    const reply = completion.choices[0]?.message?.content?.trim();
    if (!reply) {
      return res.status(502).json({ error: 'Empty reply from LLM' });
    }

    res.json({ reply, emotion: emotion || undefined });
  } catch (err) {
    console.error('OpenAI error:', err.message);
    const status = err.status === 401 ? 401 : err.status === 429 ? 429 : 502;
    res.status(status).json({
      error: 'LLM request failed',
      hint: err.message || 'Check OPENAI_API_KEY and try again.',
    });
  }
});

// Health check
app.get('/api/health', (_, res) => {
  res.json({
    ok: true,
    llm: !!process.env.OPENAI_API_KEY,
  });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  if (!process.env.OPENAI_API_KEY) {
    console.warn('OPENAI_API_KEY not set. Set it in server/.env to enable LLM responses.');
  }
});
