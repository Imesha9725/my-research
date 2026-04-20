/**
 * Empathetic response logic for the mental health support chatbot.
 * Uses simple keyword matching; can be replaced later with NLP/LLM per research.
 */

const CRISIS_KEYWORDS = ['suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself', 'self harm'];
const CRISIS_RESPONSES = [
  "I'm really glad you're reaching out. What you're feeling sounds very heavy. Please consider talking to someone who can support you right now. You can contact a crisis helpline—they're there 24/7 for you. In Sri Lanka: 1926 (Sumithrayo). You matter, and you don't have to face this alone.",
  "Thank you for sharing this with me. Your safety is important. I'm not a substitute for professional help in a crisis. Please reach out to a crisis line or a trusted person today. Sumithrayo (Sri Lanka): 1926. You deserve support.",
];

const EMOTION_RESPONSES = {
  sad: [
    "It's okay to feel sad sometimes. Would you like to talk more about what's going on? I'm here to listen.",
    "I hear you. Sadness can feel overwhelming. Remember, it's okay to take things one step at a time. What would feel helpful right now—talking, or a small suggestion?",
    "Thank you for sharing that. Feeling down is valid. Is there something that usually helps you feel a bit better, or would you like to just vent?",
  ],
  anxious: [
    "Anxiety can be really exhausting. You're not alone in feeling this way. Would it help to name one small thing you can control right now?",
    "I understand. Anxiety often makes everything feel bigger. Try to take a slow breath with me. What's one thing that's actually okay in this moment?",
    "It's brave of you to say that. When anxiety spikes, grounding can help—name 3 things you can see, 2 you can hear, 1 you can touch. Want to try?",
  ],
  stress: [
    "Stress can pile up quickly. It might help to break things down: what's one small task or decision you could set aside for later?",
    "I hear you. When things feel like too much, even a short pause can help. What would give you a 5-minute break right now?",
    "That sounds really stressful. You don't have to solve everything at once. What feels most urgent to you right now?",
  ],
  angry: [
    "Anger is a valid emotion. It often comes from something that matters to us. Would you like to talk about what's behind it?",
    "I get it—sometimes we just need to feel heard. You're in a safe space here. What happened?",
    "It's okay to feel angry. What would feel helpful right now—venting, or thinking about what you need?",
    "When you're mad at a parent, partner, or friend, it can feel extra heavy—care and frustration often show up together. What sits with you most right now?",
    "It makes sense to feel angry when someone close to you lets you down or crosses a line. I'm not here to judge. Would you like to talk through what happened?",
  ],
  /** When user is mad at someone close (mirrors server topic fallbacks for frontend-only mode). */
  relational_anger_parents: [
    "It makes sense you'd feel angry when things with your parents feel unfair or hurtful. I'm not here to take sides. What happened that stirred this up most?",
    "Anger toward family can feel messy. Your feelings are valid. Would it help to vent about what they did—or what keeps replaying in your mind?",
    "When parents push our buttons, it can hit hard. You're allowed to be mad. What would feel safest next—cooling off, writing it down, or planning what you might say later?",
  ],
  relational_anger_partner: [
    "Anger toward a partner often comes from feeling let down or unheard. What happened between you two that's sitting with you most?",
    "When someone you're close to crosses a line, anger is a natural response. What would feel fair to you in this situation?",
    "If you're worried you might say something you regret, taking a little space before you respond can help. Do you want to express this to them soon, or do you need time first?",
  ],
  relational_anger_friend: [
    "Fights with a friend can hurt a lot—especially when you trusted them. What happened from your side? I'm listening.",
    "It's rough when a friend lets you down. You can care about them and still be furious. What would feel most honest to get off your chest?",
    "When you're done with how they're acting, that anger can be protective. You don't have to fix the whole friendship tonight. What would feel kindest to yourself in the next day or two?",
  ],
  lonely: [
    "Loneliness is painful, and it's more common than we think. Reaching out here is a step. Would you like to talk about what's been missing?",
    "I hear you. Connection matters. Even small steps—like writing here—count. What would you want from a friend right now?",
    "Thank you for sharing that. You're not alone in feeling alone. Is there someone you've been meaning to reach out to, or would you rather just chat here for a bit?",
  ],
  happy: [
    "I'm really glad to hear that. Celebrating the good moments matters. What's been going well?",
    "That's wonderful. It's important to notice when things feel good. Want to share more?",
    "So good that you're feeling better. Remember this moment when things get tough—you've had good days before and you can again.",
  ],
  tired: [
    "It sounds like you're carrying a lot. Rest is valid. Is there a way you could give yourself permission to slow down today?",
    "Being tired—emotionally or physically—is real. What would rest look like for you right now, even for a few minutes?",
    "I hear you. Sometimes the best we can do is get through the day. That's enough. Be gentle with yourself.",
  ],
  fear: [
    "Fear can be overwhelming. You're safe here to talk about it. What are you most afraid of right now?",
    "It's okay to be scared. Naming the fear can sometimes make it a bit easier. Would you like to try?",
    "I understand. When fear takes over, small steps help. What's one thing that would make you feel a tiny bit safer?",
  ],
  default: [
    "I'm here to listen. You can share as much or as little as you like. How are you really doing today?",
    "Thank you for messaging. Whatever you're feeling is valid. Would you like to tell me more?",
    "I'm glad you're here. Take your time. What's on your mind?",
    "I'm listening. Sometimes just writing things out helps. What would you like to talk about?",
    "You're not alone. I'm here for you. How can I support you right now?",
  ],
};

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
  const lower = text.toLowerCase();
  return CRISIS_KEYWORDS.some((k) => lower.includes(k));
}

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

/** Rough match: upset with parents / partner / friend (no server required). */
function getRelationalAngerTopic(text) {
  const lower = (text || '').toLowerCase();
  const hasAnger =
    /\b(angry|mad|furious|pissed|resent|frustrat|fed up|so done)\b/.test(lower) || lower.includes("can't stand") || lower.includes('cant stand');
  if (!hasAnger) return null;
  if (
    /\b(parents?|mom|mother|dad|father)\b/.test(lower) ||
    lower.includes('at my parents') ||
    lower.includes('with my parents')
  ) {
    return 'relational_anger_parents';
  }
  if (
    /\b(boyfriend|girlfriend|husband|wife)\b/.test(lower) ||
    (/\bpartner\b/.test(lower) && (lower.includes('my partner') || lower.includes('at my partner')))
  ) {
    return 'relational_anger_partner';
  }
  if (/\bfriend\b/.test(lower) || lower.includes('my friend')) {
    return 'relational_anger_friend';
  }
  return null;
}

export function getBotResponse(userMessage) {
  const trimmed = (userMessage || '').trim();
  if (!trimmed) {
    return "I'm here when you're ready. You can type anything you'd like to share.";
  }

  const normalized = trimmed
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (normalized === 'okay i will try then') {
    return 'I think you can do it!';
  }

  if (isCrisisMessage(trimmed)) {
    return pickRandom(CRISIS_RESPONSES);
  }

  const relTopic = getRelationalAngerTopic(trimmed);
  if (relTopic && EMOTION_RESPONSES[relTopic]) {
    return pickRandom(EMOTION_RESPONSES[relTopic]);
  }

  const emotion = getEmotionFromText(trimmed);
  const responses = emotion ? EMOTION_RESPONSES[emotion] : EMOTION_RESPONSES.default;
  return pickRandom(responses);
}
