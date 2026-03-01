/**
 * LLM-based chat API for emotion-aware mental health support.
 * Uses IEMOCAP dataset for relevant scenarios (strong match + suitable response).
 * Fallback to curated topic/emotion responses when no suitable IEMOCAP match.
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATASET_PATH = path.resolve(__dirname, '../ml/models/dataset_responses.json');
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

// ----- Fallback when LLM fails (e.g. 429 quota) -----
const CRISIS_RESPONSES = [
  "I'm really glad you're reaching out. What you're feeling sounds very heavy. Please consider talking to someone who can support you right now. You can contact a crisis helpline—they're there 24/7 for you. In Sri Lanka: 1926 (Sumithrayo). You matter, and you don't have to face this alone.",
  "Thank you for sharing this with me. Your safety is important. I'm not a substitute for professional help in a crisis. Please reach out to a crisis line or a trusted person today. Sumithrayo (Sri Lanka): 1926. You deserve support.",
];
// Topic-based responses for problems not in IEMOCAP (exam, work, family, etc.)
const TOPIC_KEYWORDS = [
  { keywords: ['exam', 'exams', 'test', 'tests', 'study', 'studying', 'assignment', 'project', 'grade', 'marks', 'university', 'college', 'school'], topic: 'exam' },
  { keywords: ['work', 'job', 'office', 'boss', 'colleague', 'career', 'deadline', 'meeting', 'salary', 'promotion'], topic: 'work' },
  { keywords: ['family', 'parents', 'mother', 'father', 'mom', 'dad', 'sibling', 'brother', 'sister', 'child', 'children', 'kid', 'relationship with family'], topic: 'family' },
  { keywords: ['boyfriend', 'girlfriend', 'partner', 'relationship', 'breakup', 'divorce', 'dating', 'love'], topic: 'relationship' },
  { keywords: ['health', 'sick', 'ill', 'pain', 'doctor', 'hospital', 'sleep', 'insomnia', 'tired', 'exhausted'], topic: 'health' },
  { keywords: ['money', 'financial', 'debt', 'bills', 'salary', 'poor', 'afford'], topic: 'money' },
  { keywords: ['cat', 'dog', 'pet', 'animal', 'puppy', 'kitten'], topic: 'pet' },
];

const TOPIC_RESPONSES = {
  exam: ["Exams can be really overwhelming. You're not alone in feeling this way. What would help—breaking it into smaller parts, or talking through what's stressing you most?", "I hear you. Study pressure is real. Remember, your worth isn't measured by one exam. What subject or part feels hardest right now?", "That sounds tough. It might help to take short breaks and be kind to yourself. Would you like to talk about what's making it feel so heavy?"],
  work: ["Work stress can drain us. You're doing your best. What would help—venting about it, or thinking of one small thing you could change?", "I understand. Job pressure is real. Is there a specific situation that's weighing on you most?", "That sounds exhausting. It's okay to need a break. What would give you some relief right now—talking it out, or stepping back for a moment?"],
  family: ["Family issues can be really painful. I'm here to listen. Would you like to share what's going on?", "I hear you. Family dynamics are complex. You're allowed to feel how you feel. What's the hardest part right now?", "That sounds difficult. It's brave of you to reach out. What would feel helpful—talking through it, or just being heard?"],
  relationship: ["Relationship struggles hurt. I'm here for you. Would you like to talk about what's going on?", "I hear you. It's okay to feel upset. What would help—venting, or thinking about what you need from this situation?", "That sounds really tough. You're not alone. What would feel supportive right now?"],
  health: ["Health concerns can be scary. I'm here to listen. How have you been feeling? Would talking about it help?", "I hear you. Taking care of your body and mind matters. What would feel supportive right now?", "That sounds draining. Be gentle with yourself. Is there anything that usually helps you feel a bit better?"],
  money: ["Financial stress is heavy. You're not alone. What would help—talking through it, or brainstorming one small step?", "I hear you. Money worries can affect everything. It's okay to feel overwhelmed. What's on your mind most?", "That sounds really tough. Would it help to talk about what's stressing you, or just to be heard?"],
  pet: ["I'm sorry to hear that. Losing a pet or having them go missing is really hard. How long has it been? Would it help to talk about it?", "That must be so worrying. Pets are family. Have you been able to search or put up flyers? I'm here to listen.", "I hear you. That's a difficult situation. Would you like to talk about what happened, or what you've tried so far?"],
};

const EMOTION_RESPONSES = {
  sad: ["It's okay to feel sad sometimes. Would you like to talk more about what's going on? I'm here to listen.", "I hear you. Sadness can feel overwhelming. Remember, it's okay to take things one step at a time. What would feel helpful right now—talking, or a small suggestion?", "Thank you for sharing that. Feeling down is valid. Is there something that usually helps you feel a bit better, or would you like to just vent?"],
  anxious: ["Anxiety can be really exhausting. You're not alone in feeling this way. Would it help to name one small thing you can control right now?", "I understand. Anxiety often makes everything feel bigger. Try to take a slow breath with me. What's one thing that's actually okay in this moment?", "It's brave of you to say that. When anxiety spikes, grounding can help—name 3 things you can see, 2 you can hear, 1 you can touch. Want to try?"],
  stress: ["Stress can pile up quickly. It might help to break things down: what's one small task or decision you could set aside for later?", "I hear you. When things feel like too much, even a short pause can help. What would give you a 5-minute break right now?", "That sounds really stressful. You don't have to solve everything at once. What feels most urgent to you right now?"],
  angry: ["Anger is a valid emotion. It often comes from something that matters to us. Would you like to talk about what's behind it?", "I get it—sometimes we just need to feel heard. You're in a safe space here. What happened?", "It's okay to feel angry. What would feel helpful right now—venting, or thinking about what you need?"],
  lonely: ["Loneliness is painful, and it's more common than we think. Reaching out here is a step. Would you like to talk about what's been missing?", "I hear you. Connection matters. Even small steps—like writing here—count. What would you want from a friend right now?", "Thank you for sharing that. You're not alone in feeling alone. Is there someone you've been meaning to reach out to, or would you rather just chat here for a bit?"],
  happy: ["I'm really glad to hear that. Celebrating the good moments matters. What's been going well?", "That's wonderful. It's important to notice when things feel good. Want to share more?", "So good that you're feeling better. Remember this moment when things get tough—you've had good days before and you can again."],
  tired: ["It sounds like you're carrying a lot. Rest is valid. Is there a way you could give yourself permission to slow down today?", "Being tired—emotionally or physically—is real. What would rest look like for you right now, even for a few minutes?", "I hear you. Sometimes the best we can do is get through the day. That's enough. Be gentle with yourself."],
  fear: ["Fear can be overwhelming. You're safe here to talk about it. What are you most afraid of right now?", "It's okay to be scared. Naming the fear can sometimes make it a bit easier. Would you like to try?", "I understand. When fear takes over, small steps help. What's one thing that would make you feel a tiny bit safer?"],
  default: ["I'm here to listen. You can share as much or as little as you like. How are you really doing today?", "Thank you for messaging. Whatever you're feeling is valid. Would you like to tell me more?", "I'm glad you're here. Take your time. What's on your mind?", "I'm listening. Sometimes just writing things out helps. What would you like to talk about?", "You're not alone. I'm here for you. How can I support you right now?"],
};

function getTopicFromText(text) {
  if (!text || typeof text !== 'string') return null;
  const lower = text.toLowerCase().trim();
  for (const { keywords, topic } of TOPIC_KEYWORDS) {
    if (keywords.some((k) => lower.includes(k))) return topic;
  }
  return null;
}

function isGreeting(text) {
  if (!text || typeof text !== 'string') return false;
  const lower = text.toLowerCase().trim();
  const greetStart = /^(hi|hey|hello|helo|hola|namaste)[\s,!.]*/i;
  const hasIntro = /\b(i'?m|i am|my name is)\s+\w+/i.test(lower);
  if (greetStart.test(lower) && (hasIntro || lower.split(/\s+/).length <= 4)) return true;
  if (/^(i'?m|i am|my name is)\s+\w+\.?$/i.test(lower)) return true;
  const short = lower.replace(/[^\w\s]/g, '').trim();
  if (/^(hi|hey|hello)$/.test(short)) return true;
  return false;
}

const GREETING_RESPONSES = [
  "Hi! Nice to meet you. I'm here to listen whenever you'd like to talk. How are you doing today?",
  "Hello! I'm glad you're here. You can share as much or as little as you like. What's on your mind?",
  "Hey there! I'm here for you. How are you feeling today?",
];

// IEMOCAP: only use responses suitable for mental health support (exclude drama/bureaucracy scripts)
const IEMOCAP_BLOCKLIST = ['brandy', 'bored stiff', 'day care', 'supervisor', 'direct line', 'rigmarole', 'z.x.four', 'passport', 'birth certificate', 'bank account', 'overdue fee', 'statement in a few days', 'put that into the computer', 'we keep it on file', 'fill out', 'this form', 'different form of id', 'get in this line', 'do you have your forms', 'wallet was stolen', 'try to do that for you', 'i can try to do'];

function isSuitableIEMOCAPResponse(response) {
  if (!response || typeof response !== 'string') return false;
  const lower = response.toLowerCase();
  return !IEMOCAP_BLOCKLIST.some((phrase) => lower.includes(phrase));
}

const EMOTION_TO_DATASET = { anxious: 'sad', stress: 'sad', tired: 'sad', fear: 'sad', lonely: 'sad' };
const STOPWORDS = new Set(['i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'am', 'and', 'but', 'or', 'for', 'so', 'in', 'on', 'at', 'to', 'of', 'with', 'that', 'this', 'what', 'how', 'why', 'when', 'who', 'im', 'dont', 'cant', 'wont', 'its', 'thats']);

function contentWords(s) {
  return (s || '').toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter((w) => w && !STOPWORDS.has(w) && w.length > 1);
}

function wordOverlapScore(a, b) {
  if (a.length === 0 || b.length === 0) return 0;
  const setB = new Set(b);
  let overlap = 0;
  for (const w of new Set(a)) {
    if (setB.has(w)) overlap++;
  }
  return overlap / Math.max(new Set(a).size, setB.size);
}

function getDatasetResponse(userInput, emotion) {
  try {
    if (!fs.existsSync(DATASET_PATH)) return null;
    const data = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'));
    const emo = (emotion || 'neutral').toLowerCase();
    const lookup = EMOTION_TO_DATASET[emo] || emo;
    const pairs = data[lookup] || data[emo] || data.neutral;
    if (!Array.isArray(pairs) || pairs.length === 0) return null;
    const userContent = contentWords(userInput);
    if (userContent.length === 0) return null;
    let bestScore = 0;
    let bestResponse = null;
    for (const p of pairs) {
      if (!p.user || !p.response) continue;
      if (!isSuitableIEMOCAPResponse(p.response)) continue;
      const score = wordOverlapScore(userContent, contentWords(p.user));
      if (score > bestScore) {
        bestScore = score;
        bestResponse = p.response;
      }
    }
    if (bestScore >= 0.55 && bestResponse) return bestResponse;
    return null;
  } catch (_) {
    return null;
  }
}

/**
 * Priority: crisis → greeting → topic → emotion → IEMOCAP (only relevant) → default.
 * IEMOCAP used only when strong match (0.55+) and response passes suitability filter.
 */
function getFallbackResponse(text, emotion) {
  const trimmed = (text || '').trim();
  if (!trimmed) return "I'm here when you're ready. You can type anything you'd like to share.";
  if (isCrisisMessage(text)) return CRISIS_RESPONSES[Math.floor(Math.random() * CRISIS_RESPONSES.length)];
  if (isGreeting(text)) return GREETING_RESPONSES[Math.floor(Math.random() * GREETING_RESPONSES.length)];
  const topic = getTopicFromText(trimmed);
  if (topic && TOPIC_RESPONSES[topic]) return TOPIC_RESPONSES[topic][Math.floor(Math.random() * TOPIC_RESPONSES[topic].length)];
  if (emotion && EMOTION_RESPONSES[emotion]) return EMOTION_RESPONSES[emotion][Math.floor(Math.random() * EMOTION_RESPONSES[emotion].length)];
  const fromDataset = getDatasetResponse(trimmed, emotion);
  if (fromDataset) return fromDataset;
  return EMOTION_RESPONSES.default[Math.floor(Math.random() * EMOTION_RESPONSES.default.length)];
}

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
  let emotion = null;
  if (EMOTION_API_URL && (text || audioBase64)) {
    emotion = await getEmotionFromAPI(text, audioBase64);
  }
  if (emotion == null) {
    emotion = getEmotionFromText(text);
  }

  if (!apiKey) {
    const fallback = getFallbackResponse(text, emotion);
    return res.json({ reply: fallback, emotion: emotion || undefined, fallback: true });
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
      const fallback = getFallbackResponse(text, emotion);
      return res.json({ reply: fallback, emotion: emotion || undefined, fallback: true });
    }

    res.json({ reply, emotion: emotion || undefined });
  } catch (err) {
    console.warn('OpenAI error (using fallback):', err.message);
    const fallback = getFallbackResponse(text, emotion);
    res.json({ reply: fallback, emotion: emotion || undefined, fallback: true });
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
