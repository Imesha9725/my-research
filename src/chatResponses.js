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

export function getBotResponse(userMessage) {
  const trimmed = (userMessage || '').trim();
  if (!trimmed) {
    return "I'm here when you're ready. You can type anything you'd like to share.";
  }

  if (isCrisisMessage(trimmed)) {
    return pickRandom(CRISIS_RESPONSES);
  }

  const emotion = getEmotionFromText(trimmed);
  const responses = emotion ? EMOTION_RESPONSES[emotion] : EMOTION_RESPONSES.default;
  return pickRandom(responses);
}
