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
  { keywords: ['sad', 'sadness', 'depressed', 'down', 'miserable', 'hopeless', 'rejected', 'not good', 'not well', 'not okay', 'not great', 'not fine', 'no good', "don't feel good", "don't feel well", "i'm not good", "feeling not good", "am not good", "not doing good", "not doing well"], emotion: 'sad' },
  { keywords: ['anxious', 'anxiety', 'nervous', 'worried', 'panic'], emotion: 'anxious' },
  { keywords: ['stress', 'stressed', 'overwhelmed', 'pressure'], emotion: 'stress' },
  { keywords: ['angry', 'anger', 'mad', 'frustrated', 'annoyed'], emotion: 'angry' },
  { keywords: ['lonely', 'alone', 'isolated', 'left out'], emotion: 'lonely' },
  { keywords: ['happy', 'good', 'great', 'better', 'relieved', 'hopeful'], emotion: 'happy' },
  { keywords: ['tired', 'exhausted', 'drained', 'burned out'], emotion: 'tired' },
  { keywords: ['scared', 'afraid', 'fear', 'frightened', 'worried about future', 'future scares'], emotion: 'fear' },
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
  { keywords: ['failed', 'didnt pass', "didn't pass", 'bad grade', 'failed exam', 'flunked'], topic: 'academic_failure' },
  { keywords: ['exam', 'exams', 'test', 'tests', 'study', 'studying', 'assignment', 'project', 'grade', 'marks', 'university', 'college', 'school'], topic: 'exam' },
  { keywords: ['work', 'job', 'office', 'boss', 'colleague', 'career', 'deadline', 'meeting', 'salary', 'promotion'], topic: 'work' },
  { keywords: ['going to other country', 'going to another country', 'going abroad', 'going overseas', 'going away', 'leaving for', 'went to', 'moved to another', 'flying to', 'flying overseas', 'moving abroad', 'moving overseas', 'i am flying', 'im flying', 'i will fly', 'father going', 'mother going', 'mom going', 'dad going', 'brother going', 'sister going', 'husband going', 'wife going', 'son going', 'daughter going', 'parent going', 'spouse going', 'partner going', 'family member going', 'loved one going'], topic: 'family_separation' },
  { keywords: ['family', 'parents', 'mother', 'father', 'mom', 'dad', 'sibling', 'brother', 'sister', 'child', 'children', 'kid', 'relationship with family'], topic: 'family' },
  { keywords: ['boyfriend', 'girlfriend', 'partner', 'relationship', 'breakup', 'divorce', 'dating', 'love'], topic: 'relationship' },
  { keywords: ['health', 'sick', 'ill', 'pain', 'doctor', 'hospital', 'sleep', 'insomnia', 'tired', 'exhausted'], topic: 'health' },
  { keywords: ['money', 'financial', 'debt', 'bills', 'salary', 'poor', 'afford'], topic: 'money' },
  { keywords: ['cat', 'dog', 'pet', 'animal', 'puppy', 'kitten'], topic: 'pet' },
  { keywords: ['lost someone', 'passed away', 'died', 'death', 'grief', 'mourning', 'bereavement', 'left the world', 'left us', 'no longer with', 'gone forever', 'father died', 'mother died', 'mom died', 'dad died', 'brother died', 'sister died', 'spouse died', 'husband died', 'wife died', 'child died', 'friend died', 'close person died', 'family member died'], topic: 'grief' },
  { keywords: ['not good enough', 'worthless', 'failure', 'useless', 'stupid', "can't do anything", 'cant do anything'], topic: 'self_doubt' },
  { keywords: ["don't know", 'dont know', 'something wrong', 'something off', 'something feels', 'cant explain', 'confused about feelings'], topic: 'vague_distress' },
  { keywords: ['guilty', 'guilt', 'ashamed', 'shame', 'regret'], topic: 'guilt' },
  { keywords: ['future', 'tomorrow', 'scared about', 'worried about what will', 'uncertain about future'], topic: 'future_anxiety' },
  { keywords: ['nobody likes', 'rejected', 'friends left', 'excluded', 'left out', 'dont fit in', "don't fit in", 'people avoid'], topic: 'rejection_social' },
  { keywords: ['lost my job', 'fired', 'laid off', 'unemployed', 'job loss', 'got fired', 'lost job'], topic: 'job_loss' },
  { keywords: ['bullied', 'bullying', 'picked on', 'people are mean', 'they mock', 'teasing me'], topic: 'bullying' },
  { keywords: ['had a fight', 'we argued', 'big argument', 'fought with', 'fight with', 'we had a fight'], topic: 'conflict_fight' },
  { keywords: ['disappointed', 'didnt work out', "didn't work out", 'let down', 'disappointment', 'things didnt go'], topic: 'disappointment' },
  { keywords: ['cant stop thinking', "can't stop thinking", 'overthinking', 'racing thoughts', 'mind wont stop', "mind won't stop", 'cant shut off'], topic: 'overthinking' },
  { keywords: ['everything annoys', 'irritable', 'on edge', 'short fuse', 'snapping at', 'easily annoyed'], topic: 'irritable' },
  { keywords: ['feel nothing', 'empty inside', 'numb', 'empty', 'dont feel anything', "don't feel anything"], topic: 'numb_empty' },
  { keywords: ['miss them', 'miss my', 'missing someone', 'i miss', 'longing for'], topic: 'missing_someone' },
  { keywords: ['graduating', 'moved to', 'new job', 'everything changed', 'life change', 'big change', 'transition'], topic: 'life_change' },
  { keywords: ['kids are', 'parenting', 'overwhelmed with kids', 'my child', 'being a parent'], topic: 'parenting_stress' },
];

const TOPIC_RESPONSES = {
  exam: ["Exams can be really overwhelming. You're not alone in feeling this way. What would help—breaking it into smaller parts, or talking through what's stressing you most?", "I hear you. Study pressure is real. Remember, your worth isn't measured by one exam. What subject or part feels hardest right now?", "That sounds tough. It might help to take short breaks and be kind to yourself. Would you like to talk about what's making it feel so heavy?"],
  work: ["Work stress can drain us. You're doing your best. What would help—venting about it, or thinking of one small thing you could change?", "I understand. Job pressure is real. Is there a specific situation that's weighing on you most?", "That sounds exhausting. It's okay to need a break. What would give you some relief right now—talking it out, or stepping back for a moment?"],
  family_separation: ["I understand—it's really hard when your father, mother, or any family member goes overseas, or when you're the one flying to another country. Your sadness is valid. Whether they're leaving or you are, you'll see each other again after work or studies are done. How long will the separation be? I'm here to listen.", "I hear you. Saying goodbye—whether your parent, sibling, or spouse is going, or you're flying abroad yourself—is so difficult. It's okay to feel sad. You'll be together again. Would you like to talk about how you're feeling? I'm here for you.", "That must be really tough. Whether someone close is leaving or you're moving overseas, the separation is painful. Your feelings matter. You'll reunite—and until then, we can chat anytime you need a friend. How are you coping?"],
  family: ["Family issues can be really painful. I'm here to listen. Would you like to share what's going on?", "I hear you. Family dynamics are complex. You're allowed to feel how you feel. What's the hardest part right now?", "That sounds difficult. It's brave of you to reach out. What would feel helpful—talking through it, or just being heard?"],
  relationship: ["Relationship struggles hurt. I'm here for you. Would you like to talk about what's going on?", "I hear you. It's okay to feel upset. What would help—venting, or thinking about what you need from this situation?", "That sounds really tough. You're not alone. What would feel supportive right now?"],
  health: ["Health concerns can be scary. I'm here to listen. How have you been feeling? Would talking about it help?", "I hear you. Taking care of your body and mind matters. What would feel supportive right now?", "That sounds draining. Be gentle with yourself. Is there anything that usually helps you feel a bit better?"],
  money: ["Financial stress is heavy. You're not alone. What would help—talking through it, or brainstorming one small step?", "I hear you. Money worries can affect everything. It's okay to feel overwhelmed. What's on your mind most?", "That sounds really tough. Would it help to talk about what's stressing you, or just to be heard?"],
  pet: ["I'm sorry to hear that. Losing a pet or having them go missing is really hard. How long has it been? Would it help to talk about it?", "That must be so worrying. Pets are family. Have you been able to search or put up flyers? I'm here to listen.", "I hear you. That's a difficult situation. Would you like to talk about what happened, or what you've tried so far?"],
  grief: ["I'm so sorry. Losing your father, mother, family member, or anyone close—when they leave this world—is one of the hardest things. Your sadness is valid. There's no right way to grieve. I'm here to listen. Would you like to talk about them?", "I hear you. When someone we love passes away, the pain can feel unbearable. Whether it's your parent, sibling, spouse, or a close friend—your feelings matter. Take the time you need. I'm here for you.", "That must be so painful. Losing someone who has left the world is devastating. You're allowed to feel sad, angry, lost—whatever you feel. Would it help to talk about them or how you're coping? I'm here."],
  self_doubt: ["I hear you. Those thoughts can be really loud. You're more capable than you feel right now. Would you like to talk about what's making you feel this way?", "Feeling like you're not good enough is painful—and it's not the truth. What would it mean to be a little kinder to yourself today?", "You're not a failure. Everyone struggles sometimes. What's one small thing you've done recently that you're proud of?"],
  vague_distress: ["It's okay if you can't put it into words yet. Sometimes we feel off before we know why. I'm here—take your time. What's the first thing that comes to mind?", "That fuzzy, confused feeling is valid. You don't have to understand everything to be heard. Want to try describing it in any way that feels right?", "I'm listening. Even 'something feels wrong' is enough to start. What's been going on lately?"],
  guilt: ["Guilt is heavy. You're human, and everyone makes mistakes. Would it help to talk about what happened?", "I hear you. Carrying guilt can be exhausting. What would it mean to forgive yourself a little—or at least talk it through?", "That sounds really hard. Guilt often means we care. Would you like to share what's on your mind?"],
  future_anxiety: ["Worrying about the future is exhausting. You don't have to have it all figured out. What's one small thing you can control right now?", "The future can feel scary when it's uncertain. I hear you. What would help—talking through it, or focusing on today?", "It's okay to be scared. Many people feel that way. What's the biggest worry on your mind?"],
  rejection_social: ["Feeling rejected or left out hurts a lot. I hear you. Would you like to talk about what happened? You matter, regardless of how others act.", "That sounds really painful. Connection and belonging matter—and you deserve both. What would feel supportive right now?", "I'm sorry you're going through this. Rejection can make us question ourselves, but it says more about the situation than about you. Want to talk about it?"],
  job_loss: ["Losing a job is a huge blow—to your routine, identity, and security. I'm sorry. How are you coping? It's okay to feel whatever you're feeling.", "That's really hard. Job loss can trigger a lot of emotions. Would it help to talk through it, or brainstorm one small next step?", "I hear you. This is a difficult transition. Be gentle with yourself. What would feel helpful right now—venting, or thinking about what comes next?"],
  bullying: ["I'm sorry you're going through that. Nobody deserves to be treated that way. How long has this been going on? Would it help to talk about it?", "Bullying can make you feel small and alone. You're not—and reaching out here takes courage. What would feel supportive right now?", "That sounds really tough. Have you been able to talk to anyone you trust about it? I'm here to listen."],
  conflict_fight: ["Fights and arguments can leave us shaken. What happened? Sometimes just talking it through helps.", "Conflict is exhausting. It's okay to feel upset or confused. Would you like to vent, or think about how to repair things?", "I hear you. Arguments often come from both sides caring. What would help right now—getting it off your chest, or working through it?"],
  disappointment: ["Disappointment is heavy. It's okay to sit with it. Would you like to talk about what didn't work out?", "I hear you. When things don't go as hoped, it can feel like a letdown. What would feel helpful—talking it through, or just being heard?", "That sounds hard. Disappointment is valid. Sometimes the best we can do is acknowledge it. Want to share what happened?"],
  overthinking: ["Racing thoughts are exhausting. You're not alone. Would it help to try a grounding exercise—name 3 things you see, 2 you hear, 1 you can touch?", "I hear you. When your mind won't stop, it can feel overwhelming. What's one small thing that sometimes helps you slow down?", "Overthinking can spiral. It's brave to notice it. Would talking it out help, or do you need to shift focus for a bit?"],
  irritable: ["Feeling irritable and on edge is draining. It often means we're carrying too much. What's been going on lately?", "I hear you. When everything annoys us, something underneath is usually going on. Would it help to talk about it?", "That short fuse can be exhausting. Be gentle with yourself. What would give you a bit of relief right now?"],
  numb_empty: ["Feeling numb or empty can be your mind's way of protecting you. It's valid. Would you like to talk about what might be underneath?", "I hear you. That hollow feeling is real. Sometimes it helps to name it. What's been going on in your life lately?", "Feeling nothing can be scary. You're not broken. Would it help to talk, or just sit with it for now?"],
  missing_someone: ["Missing someone hurts. Whether they're far away or gone, that longing is real. Would you like to talk about them?", "I hear you. Missing people we love can be really heavy. What would feel supportive—sharing memories, or just being heard?", "That ache of missing someone is valid. You're allowed to feel it. Want to talk about what you miss most?"],
  life_change: ["Big life changes—graduating, moving, new job—can stir up a lot. How are you handling the transition?", "Change can be exciting and scary at once. I hear you. What's the hardest part right now?", "Transitions are tough. It's okay to feel unsettled. Would it help to talk through what's changing?"],
  parenting_stress: ["Parenting is one of the hardest jobs. It's okay to feel overwhelmed. What's been toughest lately?", "I hear you. Kids can be exhausting—and it doesn't mean you're a bad parent. What would give you a moment of relief?", "That sounds really demanding. You're doing your best. Would it help to talk about what's weighing on you?"],
  academic_failure: ["Not passing or getting a bad grade can feel like a big setback. Your worth isn't defined by one result. Would you like to talk about it?", "I hear you. Academic pressure and failure can be crushing. What would help—venting, or thinking about what you can do next?", "That sounds really discouraging. It's okay to feel upset. Many people have been there. Want to talk through it?"],
};

const EMOTION_RESPONSES = {
  sad: ["It's okay to feel sad sometimes. Would you like to talk more about what's going on? I'm here to listen.", "I hear you. Sadness can feel overwhelming. Remember, it's okay to take things one step at a time. What would feel helpful right now—talking, or a small suggestion?", "Thank you for sharing that. Feeling down is valid. Is there something that usually helps you feel a bit better, or would you like to just vent?"],
  anxious: ["Anxiety can be really exhausting. You're not alone in feeling this way. Would it help to name one small thing you can control right now?", "I understand. Anxiety often makes everything feel bigger. Try to take a slow breath with me. What's one thing that's actually okay in this moment?", "It's brave of you to say that. When anxiety spikes, grounding can help—name 3 things you can see, 2 you can hear, 1 you can touch. Want to try?"],
  stress: ["Stress can pile up quickly. It might help to break things down: what's one small task or decision you could set aside for later?", "I hear you. When things feel like too much, even a short pause can help. What would give you a 5-minute break right now?", "That sounds really stressful. You don't have to solve everything at once. What feels most urgent to you right now?"],
  angry: ["Anger is a valid emotion. It often comes from something that matters to us. Would you like to talk about what's behind it?", "I get it—sometimes we just need to feel heard. You're in a safe space here. What happened?", "It's okay to feel angry. What would feel helpful right now—venting, or thinking about what you need?"],
  lonely: ["Loneliness is painful, and it's more common than we think. Reaching out here is a step. Would you like to talk about what's been missing?", "I hear you. Connection matters. Even small steps—like writing here—count. What would you want from a friend right now?", "Thank you for sharing that. You're not alone in feeling alone. Is there someone you've been meaning to reach out to, or would you rather just chat here for a bit?"],
  happy: ["That's wonderful! I'm glad to hear it. What's making you so happy today? I'd love to hear!", "So great! It's nice when things feel good. Why are you happy? Tell me more!", "I'm really happy for you! What's going well? I'd like to celebrate with you."],
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
