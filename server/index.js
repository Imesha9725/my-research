/**
 * Emotion-aware mental health chat API.
 *
 * Primary path (generative): OpenAI-compatible chat completion (GPT, or local LLaMA/T5 via
 * OPENAI_BASE_URL + OPENAI_MODEL). The model generates new replies from context—it does not
 * look up answers from dataset rows.
 *
 * Fallback path (no API key / API error): rule-based topics + emotions + optional IEMOCAP
 * retrieval (USE_IEMOCAP_RETRIEVAL) when overlap is strong, plus curated topic/emotion lines and
 * general empathy—not a substitute for a generative model.
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import {
  findUserByEmail,
  createUser,
  insertMessage,
  getRecentHistory,
  getAllMessagesForUser,
  getPersistedUserEmotions,
  getUserMessageCount,
} from './db.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATASET_PATH = path.resolve(__dirname, '../ml/models/dataset_responses.json');
const app = express();
const PORT = process.env.PORT || 5001;
const JWT_SECRET = process.env.JWT_SECRET || 'dev-only-change-JWT_SECRET-in-production';

function signToken(user) {
  return jwt.sign({ sub: String(user.id), email: user.email }, JWT_SECRET, { expiresIn: '30d' });
}

function getUserIdFromReq(req) {
  const h = req.headers.authorization;
  if (!h || typeof h !== 'string' || !h.startsWith('Bearer ')) return null;
  try {
    const payload = jwt.verify(h.slice(7), JWT_SECRET);
    const id = Number(payload.sub);
    return Number.isFinite(id) && id > 0 ? id : null;
  } catch {
    return null;
  }
}

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

function dominantEmotionFromList(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const counts = {};
  for (const e of arr) {
    const k = String(e || 'neutral').toLowerCase();
    counts[k] = (counts[k] || 0) + 1;
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

function countEmotionSwitches(arr) {
  if (!Array.isArray(arr) || arr.length < 2) return 0;
  let n = 0;
  for (let i = 1; i < arr.length; i += 1) {
    if (String(arr[i]) !== String(arr[i - 1])) n += 1;
  }
  return n;
}

/** Short textual anchor for multi-turn reasoning in the system prompt. */
function buildRecentUserThemesSnippet(history, maxTurns = 4, maxLen = 130) {
  if (!Array.isArray(history) || history.length === 0) return '';
  const users = history.filter((m) => m.role === 'user' && (m.text || m.content)).slice(-maxTurns);
  if (users.length === 0) return '';
  return users
    .map((m, i) => {
      const raw = (m.text || m.content || '').replace(/\s+/g, ' ').trim();
      const t = raw.length > maxLen ? `${raw.slice(0, maxLen)}…` : raw;
      return `(${i + 1}) "${t}"`;
    })
    .join('\n');
}

function buildMemoryContext(userId, history) {
  const sessionKeywordEmotions = getEmotionalHistory(history);
  if (!userId) {
    return {
      userId: null,
      persistedEmotions: [],
      sessionKeywordEmotions,
      totalUserMessagesApprox: sessionKeywordEmotions.length,
    };
  }
  return {
    userId,
    persistedEmotions: getPersistedUserEmotions(userId, 40),
    sessionKeywordEmotions,
    totalUserMessagesApprox: getUserMessageCount(userId),
  };
}

/** Recent user turns that mention exams, study, or related stress (for adaptive fallback / prompt). */
function countRecentExamStressUserTurns(history) {
  if (!Array.isArray(history)) return 0;
  const examRe = /\b(exam|exams|test|tests|stud(y|ying|ied)|final|midterm|assignment|grade|marks|gpa|semester|prepared|unprepared|revision|panic|fail|failing|parent|parents|mom|dad|forget|forgetting)\b/i;
  const userMsgs = history.filter((m) => m.role === 'user' && (m.text || m.content)).slice(-8);
  let n = 0;
  for (const m of userMsgs) {
    if (examRe.test(m.text || m.content || '')) n++;
  }
  return n;
}

function getExamStressIntensity(text) {
  const lower = (text || '').toLowerCase();
  if (
    /\b(panic|terrified|going to fail|will fail|i'?ll fail|disappoint my|disappointing my|hopeless|can'?t calm|cant calm|heart racing|ruin my future)\b/.test(lower)
  ) {
    return 'high';
  }
  if (
    /\b(really stressed|very stressed|so stressed|super stressed|anxious|scared|forgetting|can'?t remember|cant remember|keep forgetting|not prepared|not ready|overwhelmed)\b/.test(lower)
  ) {
    return 'medium';
  }
  return 'low';
}

function isExamAcademicStressScenario(text, history = []) {
  const userBits = (history || [])
    .filter((m) => m.role === 'user' && (m.text || m.content))
    .slice(-4)
    .map((m) => (m.text || m.content || '').toLowerCase())
    .join(' ');
  const blob = `${userBits} ${(text || '').toLowerCase()}`;
  return /\b(exam|exams|final|midterm|test|tests|stud(y|ying|ied)|assignment|gpa|grade|marks|semester|unprepared|revision|course|class)\b/.test(blob);
}

function buildSystemPrompt(
  emotion,
  isCrisis,
  emotionalHistory = [],
  text = '',
  history = [],
  memoryContext = {}
) {
  const base = `You are a supportive, empathetic mental health companion. You provide emotional support and listen without judgment. You are NOT a therapist or doctor—do not diagnose or prescribe. If someone is in crisis or mentions self-harm or suicide, encourage them to contact a crisis helpline (e.g. Sumithrayo 1926 in Sri Lanka) and a professional. Keep responses concise (2–4 short paragraphs max), warm, and natural. Use "you" and "I" naturally. Do not use bullet points or formal lists unless helpful.

Generative model (critical): You are a generative model—you compose original language for each turn. You do NOT retrieve, match, or copy lines from a fixed dialogue database or script. Do not behave like search-over-templates: infer meaning, emotion, and scenario from the user's words and conversation history, then generate a new response. This is the same idea as GPT, T5, or LLaMA-style generation, not lookup.

Novel inputs: Situations may not appear verbatim in any training-style list. That is normal. Generate a fresh, empathetic reply; validate feelings; offer gentle support and, if helpful, one open question. Never refuse because something was "not in your data" or "not in training."`;

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

  if (isExamAcademicStressScenario(text, history)) {
    const examIntensity = getExamStressIntensity(text);
    const examTurns = countRecentExamStressUserTurns(history);
    let level = 'low';
    if (examIntensity === 'high' || examTurns >= 3) level = 'high';
    else if (examIntensity === 'medium' || examTurns >= 2) level = 'medium';
    contextBlock += `\n\nExam / study stress (scenario: exam_stress): Use an empathetic listener style—validate first (e.g. that the pressure feels overwhelming), normalize anxiety about memory and performance, gently separate self-worth from one exam when they fear failure or disappointing family. Offer light coping (short breaks, self-kindness) without sounding preachy. Estimated intensity for this turn: ${level}. User messages in this chat touching exam/study stress recently: about ${examTurns}. If intensity is medium or high, or they have brought this up multiple times, offer slightly deeper reassurance and acknowledge that the worry has been building.`;
  }

  const mc = memoryContext || {};
  const persist = Array.isArray(mc.persistedEmotions) ? mc.persistedEmotions : [];
  const sessionKw = Array.isArray(mc.sessionKeywordEmotions) ? mc.sessionKeywordEmotions : [];
  const blended = [...persist.slice(-14), ...sessionKw].slice(-22);
  const dominantBlend = dominantEmotionFromList(
    blended.length ? blended : sessionKw.length ? sessionKw : emotion ? [emotion] : ['neutral']
  );
  const volatility = countEmotionSwitches(blended.length >= 2 ? blended : sessionKw);

  let memoryBlock = '';
  if (persist.length > 0 || sessionKw.length > 0) {
    memoryBlock = `\n\nEmotional memory (per-user + session): Use this to adapt tone and depth, not to label the user clinically.
- Saved trajectory (from stored user turns, multimodal/text emotion at send time): ${persist.length ? persist.join(' → ') : 'none yet'}.
- This chat window (keyword-inferred cues): ${sessionKw.length ? sessionKw.join(' → ') : 'sparse'}.
- Working summary: dominant tone ≈ "${dominantBlend}"; emotion switches in this blended window ≈ ${volatility}. High switching → steady, grounding replies; stable heavy tone → sustained validation without sounding repetitive.`;
  }

  const themeSnippet = buildRecentUserThemesSnippet(history, 4, 130);
  const multiTurnBlock = themeSnippet
    ? `\n\nMulti-turn reasoning: Recent user lines in this thread (chronological in window):\n${themeSnippet}\nConnect the latest message to these earlier points. Maintain continuity; do not contradict prior validation; avoid "resetting" empathy as if the conversation just started unless the user clearly pivots topic.`
    : '';

  let personalizeBlock = '';
  if (mc.userId && typeof mc.totalUserMessagesApprox === 'number' && mc.totalUserMessagesApprox > 2) {
    personalizeBlock = `\n\nPersonalized adaptation: Returning user with about ${mc.totalUserMessagesApprox} saved user messages. You may acknowledge recurring themes gently if the transcript suggests it, deepen trust-appropriate continuity, and vary phrasing so repetition feels caring—not robotic.`;
  } else if (!mc.userId && sessionKw.length >= 3) {
    personalizeBlock = `\n\nSession depth: Several emotional cues already appear in this anonymous thread—thread concerns together and avoid generic openers that ignore what they already shared.`;
  }

  const tailBlocks = memoryBlock + multiTurnBlock + personalizeBlock;

  if (isCrisis) {
    return `${base}\n\nImportant: The user's message may indicate a crisis. Prioritize safety: express care, validate feelings, and strongly encourage contacting a crisis helpline or trusted person. Be calm and supportive.${contextBlock}${tailBlocks}`;
  }
  return base + contextBlock + tailBlocks;
}

// ----- Emotion API (IEMOCAP-based voice + text) -----
const EMOTION_API_URL = process.env.EMOTION_API_URL || '';

/** When false, fallback chain skips IEMOCAP string retrieval (pure rules + empathy pool). */
const USE_IEMOCAP_RETRIEVAL = !/^(0|false|no|off)$/i.test(String(process.env.USE_IEMOCAP_RETRIEVAL ?? 'true').trim());

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
  { keywords: ['going to other country', 'going to another country', 'going abroad', 'going overseas', 'going away', 'leaving for', 'went to another', 'went to other country', 'moved to another', 'flying to', 'flying overseas', 'moving abroad', 'moving overseas', 'i am flying', 'im flying', 'i will fly', 'father going', 'mother going', 'mom going', 'dad going', 'brother going', 'sister going', 'husband going', 'wife going', 'son going', 'daughter going', 'parent going', 'spouse going', 'partner going', 'family member going', 'loved one going'], topic: 'family_separation' },
  { keywords: ['family', 'parents', 'mother', 'father', 'mom', 'dad', 'sibling', 'brother', 'sister', 'child', 'children', 'kid', 'relationship with family'], topic: 'family' },
  { keywords: ['boyfriend', 'girlfriend', 'partner', 'relationship', 'breakup', 'divorce', 'dating', 'love'], topic: 'relationship' },
  { keywords: ['father is sick', 'mother is sick', 'father sick', 'mother sick', 'parent sick', 'parent is sick', 'sibling sick', 'spouse sick', 'husband sick', 'wife sick', 'child sick', 'family member sick', 'someone close is sick', 'loved one sick'], topic: 'someone_sick' },
  { keywords: ['cat is sick', 'dog is sick', 'pet is sick', 'cat sick', 'dog sick', 'my cat sick', 'my dog sick', 'pet ill', 'cat ill', 'dog ill'], topic: 'pet_sick' },
  { keywords: ['what to do for my pet', 'what i have to do for my pet', 'do for my pet', 'how to help my pet', 'what should i do for my pet', 'give medicine', 'what can i do for my pet'], topic: 'pet_sick_advice' },
  { keywords: ['went to vet', 'went to the vet', 'went near to vet', 'got medicine', 'got the medicine', 'got medicine for', 'got the medicine for', 'medicine for my cat', 'medicine for my dog', 'medicine for my pet', 'took to vet', 'saw the vet'], topic: 'pet_sick_update' },
  { keywords: ['pet left', 'pet gone', 'pet missing', 'pet ran away', 'lost my pet', 'cat left', 'dog left', 'pet die', 'pet died', 'cat died', 'dog died', 'cat', 'dog', 'pet', 'animal', 'puppy', 'kitten'], topic: 'pet' },
  { keywords: ['health', 'sick', 'ill', 'pain', 'doctor', 'hospital', 'sleep', 'insomnia', 'tired', 'exhausted'], topic: 'health' },
  { keywords: ['money', 'financial', 'debt', 'bills', 'salary', 'poor', 'afford'], topic: 'money' },
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
  family_separation: ["I understand—it's really hard when someone you love leaves. Take a deep breath. Your feelings are valid. Whether they're going overseas or you are, you'll see each other again. Try to stay calm—you're not alone. I'm here to listen. How long will the separation be?", "I hear you. Saying goodbye is painful. Remember to breathe. It's okay to feel sad. They will come back, and you'll be together again. Try to relax your mind—things will be okay. Would you like to talk? I'm here for you.", "That must be tough. When someone leaves, it hurts. Your feelings matter. Take a moment to breathe. You'll reunite—until then, we can chat anytime. You're going to be okay. How are you holding up?"],
  family: ["Family issues can be really painful. I'm here to listen. Would you like to share what's going on?", "I hear you. Family dynamics are complex. You're allowed to feel how you feel. What's the hardest part right now?", "That sounds difficult. It's brave of you to reach out. What would feel helpful—talking through it, or just being heard?"],
  relationship: ["Relationship struggles hurt. I'm here for you. Would you like to talk about what's going on?", "I hear you. It's okay to feel upset. What would help—venting, or thinking about what you need from this situation?", "That sounds really tough. You're not alone. What would feel supportive right now?"],
  someone_sick: ["I'm so sorry to hear someone close to you is sick. Take a deep breath. It's natural to feel scared and worried. Try to stay calm—your love and care mean a lot to them. You're not alone. I'm here. Would you like to talk about it? Breathe slowly; things will be okay.", "I hear you. When a family member or loved one is ill, it's really hard. Your feelings are valid. Try to relax your mind—you're doing what you can. They're in good hands with you by their side. I'm here whenever you need to talk.", "That must be so worrying. When someone we love is sick, our minds can race. Take a moment to breathe. Stay calm—your presence and care matter. You're not alone in this. I'm here to listen. How are you holding up?"],
  pet_sick: ["I'm so sorry your cat or dog is sick. Take a deep breath. That's scary—pets are family. Try to stay calm. Have you been able to see a vet and get medicine? That would be the best step. I'm here to listen. Breathe; we'll get through this together.", "I hear you. When our pet is unwell, it's natural to feel scared. Try to relax your mind—you're doing what you can. Have you gone to the vet or gotten medicine for them? I'm here for you. Your feelings are valid.", "That must be so worrying. A sick pet can make us feel helpless. Take a breath. Try to stay calm. Have you taken them to the vet or got medicine? That can help a lot. I'm here whenever you need to talk."],
  pet_sick_advice: ["For a sick pet, the best steps are: (1) Take them to a vet if you haven't yet—they can diagnose and prescribe medicine. (2) Give any medicine the vet prescribed, exactly as directed. (3) Keep them comfortable—quiet place, water, rest. (4) Watch for changes. I'm here if you need to talk.", "Here's what you can do: First, see a vet—they'll tell you what medicine or treatment your pet needs. Follow their advice on feeding and medicine. Keep your pet warm and calm. You're doing your best—I'm here for you.", "When your pet is sick: take them to the vet, give medicine as the vet prescribes, keep them in a calm place with water, and monitor them. Don't give human medicine unless the vet says so. You're doing what you can. I'm here to listen."],
  pet_sick_update: ["That's great that you went to the vet and got medicine. You're doing the right thing for your cat. Give the medicine as the vet prescribed, keep them comfortable, and they should start feeling better. I'm glad you took care of them. How are they doing now?", "Good job taking them to the vet and getting medicine. Follow the vet's instructions—give the medicine on time and keep your pet restful. You're a great pet parent. Let me know if you need to talk.", "I'm glad you got them to the vet and have medicine. That's the best thing you could do. Keep giving the medicine as directed and watch how they respond. You're doing your best."],
  health: ["Health concerns can be scary. I'm here to listen. How have you been feeling? Would talking about it help?", "I hear you. Taking care of your body and mind matters. What would feel supportive right now?", "That sounds draining. Be gentle with yourself. Is there anything that usually helps you feel a bit better?"],
  money: ["Financial stress is heavy. You're not alone. What would help—talking through it, or brainstorming one small step?", "I hear you. Money worries can affect everything. It's okay to feel overwhelmed. What's on your mind most?", "That sounds really tough. Would it help to talk about what's stressing you, or just to be heard?"],
  pet: ["I'm sorry to hear that. Losing a pet or having them go missing is really hard. Take a deep breath. Try to relax your mind. How long has it been? Would it help to talk? I'm here. You're not alone.", "That must be so worrying. Pets are family. Try to stay calm. Have you been able to search or put up flyers? I'm here to listen. Breathe—things will work out. You're doing your best.", "I hear you. That's a difficult situation. Take a moment to breathe. Would you like to talk about what happened? I'm here for you. Try to relax—you're not alone."],
  grief: ["I'm so sorry. Losing someone who left this world is one of the hardest things. Take a deep breath. Your sadness is valid. Try to relax your mind—there's no right way to grieve. You're not alone. I'm here. Would you like to talk about them?", "I hear you. When someone we love passes away, the pain can feel unbearable. Breathe slowly. Your feelings matter. Take the time you need. Try to stay calm—you will get through this. I'm here for you.", "That must be so painful. Losing someone is devastating. Take a moment to breathe. You're allowed to feel whatever you feel. Try to relax your mind—you're not alone. I'm here whenever you need to talk."],
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

/** Scenario exam_stress: validation → normalization → reassurance → gentle coping (research-style dialogue). */
const EXAM_STRESS_BASE = [
  "That sounds really overwhelming. Exams can put a lot of pressure on you.",
  "Exams can be really overwhelming. You're not alone in feeling this way. What would help most right now—breaking study into smaller chunks, or talking through what's stressing you most?",
  "I hear you. Study pressure is real. Remember, your worth isn't measured by one exam. What subject or part feels hardest right now?",
  "That sounds tough. It might help to take short breaks and be kind to yourself. What's making it feel heaviest today?",
];

const EXAM_STRESS_INTENSITY_MEDIUM = [
  "I understand how frustrating that can be. It's normal to feel anxious when you're worried about remembering everything.",
  "Studying hard and still feeling like things slip away is exhausting—and that can make panic spike. Many people go through that; it doesn't mean you're failing.",
];

const EXAM_STRESS_INTENSITY_HIGH = [
  "It makes sense that you're feeling that pressure. Your effort matters, and one exam doesn't define your future—or your worth.",
  "Fear of disappointing people you love can feel huge. You're allowed to want their pride and still be gentle with yourself right now.",
  "I'm really glad you're saying this out loud. Being scared you might fail doesn't mean you will—and it doesn't cancel the work you've already put in.",
];

const EXAM_STRESS_CALM_FOCUS = [
  "Maybe taking short breaks or doing something small and calming could help you reset. You've already put in effort—try to be kind to yourself too.",
  "When you wish you could calm down and focus, sometimes a few slow breaths, a short walk, or switching topics for five minutes can help more than pushing nonstop.",
];

const EXAM_STRESS_ESCALATED = [
  "It sounds like exam stress has shown up in several messages—I'm glad you're sharing it. We can take this one worry at a time.",
  "You've been carrying a lot of this worry lately. That doesn't mean you're weak—it means you care. What's one small thing that might ease the pressure today, even a little?",
];

function getExamStressAdaptiveResponse(text, history = []) {
  const intensity = getExamStressIntensity(text);
  const repeats = countRecentExamStressUserTurns(history);
  const pool = [...EXAM_STRESS_BASE];
  if (intensity === 'medium' || intensity === 'high') {
    pool.push(...EXAM_STRESS_INTENSITY_MEDIUM);
  }
  if (intensity === 'high') {
    pool.push(...EXAM_STRESS_INTENSITY_HIGH, ...EXAM_STRESS_CALM_FOCUS);
  }
  if (repeats >= 2) {
    pool.push(...EXAM_STRESS_ESCALATED);
  }
  return pool[Math.floor(Math.random() * pool.length)];
}

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

/** When no topic rule, no keyword emotion, and no strong IEMOCAP match—general empathy (not dataset lookup). */
const NO_DATASET_MATCH_RESPONSES = [
  "That sounds really difficult, and I'm glad you shared it with me. I'm here for you—could you tell me a bit more about what's going on?",
  "I hear you. Even when it's hard to put into words, what you're feeling matters. What part of this weighs on you most right now?",
  "Thank you for trusting me with that. You don't need a perfect explanation for your feelings to be valid. How are you coping day to day?",
  "It makes sense that you'd reach out. I'm not here to judge—just to listen. What would feel most helpful: talking it through, or simply being heard?",
  "Whatever you're facing, you don't have to face it alone in this chat. Can you say a little more about what happened or what you're afraid might happen?",
  ...EMOTION_RESPONSES.default,
];

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

function isThanks(text) {
  if (!text || typeof text !== 'string') return false;
  const lower = text.toLowerCase().trim();
  const thanksPatterns = /^(thank(s| you)?|thanks a lot|thank you so much|many thanks|really appreciate|i appreciate|grateful|that helps|so helpful|bye|goodbye|take care)[\s,!.]*$/i;
  if (thanksPatterns.test(lower)) return true;
  const short = lower.replace(/[^\w\s]/g, '').trim();
  return /^(thank you|thanks|bye|goodbye|take care)$/.test(short) || (short.split(/\s+/).length <= 5 && /\b(thanks|thank|appreciate|grateful|helpful)\b/i.test(short));
}

const THANKS_RESPONSES = [
  "It was truly my pleasure to be here with you. Remember, you're never alone—I'm here whenever you need to talk. Take care of yourself, and reach out anytime.",
  "Thank you for trusting me with your feelings. That takes courage. I'm always here when you need support. Wishing you peace and kindness. Take care.",
  "I'm so glad I could be here for you. You matter, and you deserve to feel supported. Come back anytime—I'll be here. Take gentle care of yourself.",
  "It means a lot that you shared with me. You're not alone in this journey. Whenever you need a listening ear, I'm here. Be kind to yourself.",
  "Thank you for chatting with me. I hope our conversation helped a little. Remember, reaching out is a strength. I'm here whenever you need. Take care.",
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

/** How many distinct content words from `a` appear in `b` (reduces false IEMOCAP matches on one shared word). */
function distinctOverlapCount(a, b) {
  if (a.length === 0 || b.length === 0) return 0;
  const setB = new Set(b);
  let n = 0;
  for (const w of new Set(a)) {
    if (setB.has(w)) n++;
  }
  return n;
}

/** Stricter retrieval: only use IEMOCAP when overlap is strong (hybrid: generative/fallback covers the rest). */
const IEMOCAP_MIN_SCORE = Number(process.env.IEMOCAP_MIN_SCORE) || 0.62;
const IEMOCAP_MIN_OVERLAP_WORDS = Number(process.env.IEMOCAP_MIN_OVERLAP_WORDS) || 2;

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
    let bestOverlap = 0;
    for (const p of pairs) {
      if (!p.user || !p.response) continue;
      if (!isSuitableIEMOCAPResponse(p.response)) continue;
      const wordsB = contentWords(p.user);
      const score = wordOverlapScore(userContent, wordsB);
      const overlapWords = distinctOverlapCount(userContent, wordsB);
      if (score > bestScore || (score === bestScore && overlapWords > bestOverlap)) {
        bestScore = score;
        bestOverlap = overlapWords;
        bestResponse = p.response;
      }
    }
    if (
      bestResponse &&
      bestScore >= IEMOCAP_MIN_SCORE &&
      bestOverlap >= IEMOCAP_MIN_OVERLAP_WORDS
    ) {
      return bestResponse;
    }
    return null;
  } catch (_) {
    return null;
  }
}

/**
 * Build context from recent user messages for better topic inference (e.g. "what to do for my pet" + previous "my cat is sick").
 */
function getContextText(history) {
  if (!Array.isArray(history) || history.length === 0) return '';
  const userTexts = history
    .filter((m) => m.role === 'user' && (m.text || m.content))
    .slice(-2)
    .map((m) => (m.text || m.content || '').trim())
    .filter(Boolean);
  return userTexts.join(' ');
}

/**
 * Hybrid fallback (no LLM): crisis → greeting → thanks → topic → keyword emotion →
 * IEMOCAP only if strong match → general empathy (no dataset).
 * IEMOCAP is optional retrieval, not the only source of answers.
 *
 * Priority: crisis → greeting → topic (use context) → emotion → IEMOCAP → default.
 * Uses conversation history so follow-ups like "what to do for my pet?" get sick-pet context.
 */
function getFallbackResponse(text, emotion, history = [], _memoryContext = {}) {
  const trimmed = (text || '').trim();
  if (!trimmed) return "I'm here when you're ready. You can type anything you'd like to share.";
  if (isCrisisMessage(text)) return CRISIS_RESPONSES[Math.floor(Math.random() * CRISIS_RESPONSES.length)];
  if (isGreeting(text)) return GREETING_RESPONSES[Math.floor(Math.random() * GREETING_RESPONSES.length)];
  if (isThanks(text)) return THANKS_RESPONSES[Math.floor(Math.random() * THANKS_RESPONSES.length)];
  const contextText = getContextText(history) + ' ' + trimmed;
  const topic = getTopicFromText(trimmed) || getTopicFromText(contextText);
  if (topic === 'exam') {
    return getExamStressAdaptiveResponse(trimmed, history);
  }
  if (topic && TOPIC_RESPONSES[topic]) return TOPIC_RESPONSES[topic][Math.floor(Math.random() * TOPIC_RESPONSES[topic].length)];
  if (emotion && EMOTION_RESPONSES[emotion]) return EMOTION_RESPONSES[emotion][Math.floor(Math.random() * EMOTION_RESPONSES[emotion].length)];
  if (USE_IEMOCAP_RETRIEVAL) {
    const fromDataset = getDatasetResponse(trimmed, emotion);
    if (fromDataset) return fromDataset;
  }
  return NO_DATASET_MATCH_RESPONSES[Math.floor(Math.random() * NO_DATASET_MATCH_RESPONSES.length)];
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

async function computeChatReply(text, history, audioBase64, memoryContext = {}) {
  const apiKey = process.env.OPENAI_API_KEY;
  let emotion = null;
  if (EMOTION_API_URL && (text || audioBase64)) {
    emotion = await getEmotionFromAPI(text, audioBase64);
  }
  if (emotion == null) {
    emotion = getEmotionFromText(text);
  }

  const memoryMerged = {
    ...memoryContext,
    sessionKeywordEmotions: getEmotionalHistory(history),
  };

  if (isThanks(text)) {
    const thanksReply = THANKS_RESPONSES[Math.floor(Math.random() * THANKS_RESPONSES.length)];
    return { reply: thanksReply, emotion: emotion || undefined, fallback: false };
  }

  if (!apiKey) {
    const fallback = getFallbackResponse(text, emotion, history, memoryMerged);
    return { reply: fallback, emotion: emotion || undefined, fallback: true };
  }

  const crisis = isCrisisMessage(text);
  const emotionalHistory = getEmotionalHistory(history);
  const systemPrompt = buildSystemPrompt(emotion, crisis, emotionalHistory, text, history, memoryMerged);
  const openaiBaseURL = (process.env.OPENAI_BASE_URL || '').trim();
  const openai = new OpenAI({
    apiKey,
    ...(openaiBaseURL ? { baseURL: openaiBaseURL } : {}),
  });

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
      const fallback = getFallbackResponse(text, emotion, history, memoryMerged);
      return { reply: fallback, emotion: emotion || undefined, fallback: true };
    }
    return { reply, emotion: emotion || undefined, fallback: false };
  } catch (err) {
    console.warn('OpenAI error (using fallback):', err.message);
    const fallback = getFallbackResponse(text, emotion, history, memoryMerged);
    return { reply: fallback, emotion: emotion || undefined, fallback: true };
  }
}

// ----- Auth (SQLite: one account per email; messages stored per user_id) -----
app.post('/api/auth/register', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password || typeof email !== 'string' || typeof password !== 'string') {
    return res.status(400).json({ error: 'email and password required' });
  }
  const e = email.trim().toLowerCase();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e)) {
    return res.status(400).json({ error: 'invalid email' });
  }
  if (password.length < 6) {
    return res.status(400).json({ error: 'password must be at least 6 characters' });
  }
  if (findUserByEmail(e)) {
    return res.status(409).json({ error: 'email already registered' });
  }
  const hash = bcrypt.hashSync(password, 10);
  const user = createUser(e, hash);
  const token = signToken(user);
  res.json({ token, user: { id: user.id, email: user.email } });
});

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password || typeof email !== 'string' || typeof password !== 'string') {
    return res.status(400).json({ error: 'email and password required' });
  }
  const user = findUserByEmail(email);
  if (!user || !bcrypt.compareSync(password, user.password_hash)) {
    return res.status(401).json({ error: 'invalid email or password' });
  }
  const token = signToken({ id: user.id, email: user.email });
  res.json({ token, user: { id: user.id, email: user.email } });
});

app.get('/api/chat/history', (req, res) => {
  const userId = getUserIdFromReq(req);
  if (!userId) return res.status(401).json({ error: 'unauthorized' });
  const rows = getAllMessagesForUser(userId, 200);
  res.json({
    messages: rows.map((r) => ({
      id: String(r.id),
      role: r.role,
      text: r.text,
      emotion: r.emotion || undefined,
      createdAt: r.created_at,
    })),
  });
});

function normalizeClientHistory(arr) {
  if (!Array.isArray(arr)) return [];
  return arr
    .filter((m) => m && (m.text || m.content))
    .map((m) => {
      const role = m.role === 'assistant' ? 'bot' : m.role;
      const t = (m.text || m.content || '').trim();
      return { role, text: t };
    })
    .filter((m) => m.text && (m.role === 'user' || m.role === 'bot'));
}

/**
 * Logged-in users used to send history: [] so the server only read SQLite. That broke context when DB lagged
 * or for emotional-memory heuristics expecting transcript shape. Merge: prefer the richer source (usually UI).
 */
function mergeHistoryForUser(dbHistory, clientHistory, maxLen = 24) {
  const db = Array.isArray(dbHistory) ? dbHistory.filter((m) => m?.text) : [];
  const cl = normalizeClientHistory(clientHistory);
  if (cl.length === 0) return db.slice(-maxLen);
  if (db.length === 0) return cl.slice(-maxLen);
  if (cl.length >= db.length) return cl.slice(-maxLen);
  return db.slice(-maxLen);
}

// ----- Chat endpoint -----
app.post('/api/chat', async (req, res) => {
  const { message, history: clientHistory = [], audioBase64 } = req.body;
  const text = typeof message === 'string' ? message.trim() : '';

  if (!text) {
    return res.status(400).json({ error: 'message is required' });
  }

  const userId = getUserIdFromReq(req);
  let history = normalizeClientHistory(clientHistory);
  if (userId) {
    const fromDb = getRecentHistory(userId, 24);
    history = mergeHistoryForUser(fromDb, clientHistory, 24);
  }

  const memoryContext = buildMemoryContext(userId, history);
  const { reply, emotion, fallback } = await computeChatReply(text, history, audioBase64, memoryContext);
  if (userId) {
    insertMessage(userId, 'user', text, emotion || null);
    insertMessage(userId, 'bot', reply, null);
  }

  res.json({ reply, emotion, fallback: fallback || undefined });
});

// Health check
app.get('/api/health', (_, res) => {
  res.json({
    ok: true,
    generative: !!process.env.OPENAI_API_KEY,
    llm: !!process.env.OPENAI_API_KEY,
    openaiBaseURL: !!(process.env.OPENAI_BASE_URL || '').trim(),
    iemocapRetrievalFallback: USE_IEMOCAP_RETRIEVAL,
  });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  if (!process.env.OPENAI_API_KEY) {
    console.warn('OPENAI_API_KEY not set. Set it in server/.env to enable generative (GPT/compatible) responses.');
  } else if ((process.env.OPENAI_BASE_URL || '').trim()) {
    console.log('Generative API: using OPENAI_BASE_URL (e.g. local LLaMA/Ollama/vLLM). Model:', process.env.OPENAI_MODEL || 'gpt-4o-mini');
  } else {
    console.log('Generative API: OpenAI. Model:', process.env.OPENAI_MODEL || 'gpt-4o-mini');
  }
  if (!USE_IEMOCAP_RETRIEVAL) {
    console.log('IEMOCAP retrieval disabled in fallback (USE_IEMOCAP_RETRIEVAL).');
  }
  if (!process.env.JWT_SECRET) {
    console.warn('JWT_SECRET not set. Using a dev default; set JWT_SECRET in server/.env for production.');
  }
});
