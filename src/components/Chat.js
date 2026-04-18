import React, { useState, useRef, useEffect } from 'react';
import { getBotResponse } from '../chatResponses';
import './Chat.css';

const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

// welcome messgae for my chetbot
const WELCOME_MESSAGE = {
  id: 'welcome',
  role: 'bot',
  text: "Hi, I'm here to support you. This is a safe space to share how you're feeling. You can talk about your day, what's on your mind, or anything that's bothering you. How are you doing today?",
  timestamp: new Date(),
};

function Chat({ authToken, userEmail, onLogout }) {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const lastRecordedBlobRef = useRef(null);
  const audioReadyRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const apiUrl = process.env.REACT_APP_CHAT_API_URL || '';

  useEffect(() => {
    if (!apiUrl || !authToken) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${apiUrl}/api/chat/history`, {
          headers: { Authorization: `Bearer ${authToken}` },
        });
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        const loaded = (data.messages || []).map((m) => ({
          id: `db-${m.id}`,
          role: m.role,
          text: m.text,
          timestamp: m.createdAt ? new Date(m.createdAt) : new Date(),
        }));
        setMessages(loaded.length === 0 ? [WELCOME_MESSAGE] : loaded);
      } catch (_) {
        /* keep welcome */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiUrl, authToken]);

  useEffect(() => {
    if (!SpeechRecognitionAPI) return;
    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (e) => {
      const last = e.results.length - 1;
      const transcript = e.results[last][0].transcript;
      if (e.results[last].isFinal) {
        setInput((prev) => (prev ? `${prev} ${transcript}` : transcript).trim());
      }
    };

    recognition.onend = () => setIsListening(false);
    recognition.onerror = (e) => {
      setIsListening(false);
      if (e.error === 'not-allowed') {
        setVoiceError('Microphone access was denied.');
      } else if (e.error === 'no-speech') {
        setVoiceError(null);
      } else {
        setVoiceError('Voice input failed. You can type instead.');
      }
    };

    recognitionRef.current = recognition;
    return () => {
      try {
        recognition.abort();
      } catch (_) {}
    };
  }, []);

  const toggleVoice = async () => {
    if (!SpeechRecognitionAPI) {
      setVoiceError('Voice input is not supported in this browser. Try Chrome or Edge.');
      return;
    }
    setVoiceError(null);
    const recognition = recognitionRef.current;
    if (!recognition) return;
    if (isListening) {
      recognition.stop();
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      streamRef.current?.getTracks?.().forEach((t) => t.stop());
      streamRef.current = null;
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const chunks = [];
      const mr = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg' });
      let resolveAudio;
      audioReadyRef.current = new Promise((r) => { resolveAudio = r; });
      mr.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
      mr.onstop = () => {
        if (chunks.length > 0) {
          const blob = new Blob(chunks, { type: mr.mimeType });
          lastRecordedBlobRef.current = blob;
          resolveAudio(blob);
        } else {
          resolveAudio(null);
        }
      };
      mr.start(200);
      mediaRecorderRef.current = mr;
      recognition.start();
      setIsListening(true);
    } catch (err) {
      setVoiceError('Could not start microphone. Check permissions.');
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isTyping) return;

    if (isListening) {
      recognitionRef.current?.stop();
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      streamRef.current?.getTracks?.().forEach((t) => t.stop());
      streamRef.current = null;
      setIsListening(false);
    }

    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      text,
      timestamp: new Date(),
    };
    const historyForApi = messages
      .filter((m) => m.role === 'user' || m.role === 'bot')
      .map((m) => ({ role: m.role, text: m.text }));
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    let audioBase64 = null;
    let blob = lastRecordedBlobRef.current;
    if (!blob && audioReadyRef.current) {
      try {
        blob = await Promise.race([audioReadyRef.current, new Promise((r) => setTimeout(() => r(null), 1500))]);
      } catch (_) {}
    }
    if (blob) {
      try {
        audioBase64 = await new Promise((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => {
            const dataUrl = r.result;
            const base64 = dataUrl.indexOf(',') >= 0 ? dataUrl.split(',')[1] : dataUrl;
            resolve(base64);
          };
          r.onerror = reject;
          r.readAsDataURL(blob);
        });
      } catch (_) {}
      lastRecordedBlobRef.current = null;
    }

    const payload = { message: text, history: historyForApi };
    if (audioBase64) payload.audioBase64 = audioBase64;

    const chatHeaders = { 'Content-Type': 'application/json' };
    if (authToken) chatHeaders.Authorization = `Bearer ${authToken}`;

    let botText;
    if (apiUrl) {
      try {
        const res = await fetch(`${apiUrl}/api/chat`, {
          method: 'POST',
          headers: chatHeaders,
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (res.ok && data.reply) {
          botText = data.reply;
        } else {
          botText = getBotResponse(text);
        }
      } catch (_) {
        botText = getBotResponse(text);
      }
    } else {
      botText = getBotResponse(text);
    }

    const botMsg = {
      id: `bot-${Date.now()}`,
      role: 'bot',
      text: botText,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, botMsg]);
    setIsTyping(false);
  };

  return (
    <div className="chat-app">
      <div className="chat-header">
        <div className="chat-header-brand">
          <div className="chat-header-icon" aria-hidden>
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none"/>
              <path d="M10 14c0-1.5 1.5-3 4-3s4 1.5 4 3v6c0 1.5-1.5 3-4 3s-4-1.5-4-3v-6z" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
              <path d="M16 11v2M16 19v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="chat-header-text">
            <h1>Mental Health Support</h1>
            <p>Emotion-aware chatbot · Text & voice</p>
          </div>
        </div>
        <div className="chat-header-actions">
          {userEmail && (
            <span className="chat-header-user" title={userEmail}>
              {userEmail.split('@')[0]}
            </span>
          )}
          {onLogout && (
            <button type="button" className="chat-header-logout" onClick={onLogout}>
              Log out
            </button>
          )}
          {/* <span className="chat-header-badge">Research · MCS 3204</span> */}
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chat-message chat-message--${msg.role}`}
            role={msg.role === 'bot' ? 'article' : 'none'}
          >
            {msg.role === 'bot' && <span className="chat-message-avatar" aria-hidden>🤖</span>}
            <div className="chat-message-bubble">
              <p className="chat-message-text">{msg.text}</p>
            </div>
            {msg.role === 'user' && <span className="chat-message-avatar chat-message-avatar--user" aria-hidden>You</span>}
          </div>
        ))}
        {isTyping && (
          <div className="chat-message chat-message--bot" aria-live="polite">
            <span className="chat-message-avatar" aria-hidden>🤖</span>
            <div className="chat-message-bubble chat-message-bubble--typing">
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          placeholder="Type or use the mic to speak..."
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            if (voiceError) setVoiceError(null);
          }}
          disabled={isTyping}
          maxLength={2000}
          aria-label="Your message"
        />
        <button
          type="button"
          className={`chat-mic ${isListening ? 'chat-mic--active' : ''}`}
          onClick={toggleVoice}
          disabled={isTyping}
          aria-label={isListening ? 'Stop listening' : 'Start voice input'}
          title={SpeechRecognitionAPI ? (isListening ? 'Stop listening' : 'Speak your message') : 'Voice not supported'}
        >
          <span className="chat-mic-icon" aria-hidden>
            {isListening ? '⏹' : '🎤'}
          </span>
        </button>
        <button
          type="submit"
          className="chat-send"
          disabled={!input.trim() || isTyping}
          aria-label="Send message"
        >
          Send
        </button>
      </form>
      {voiceError && (
        <p className="chat-voice-error" role="alert">
          {voiceError}
        </p>
      )}

      <p className="chat-disclaimer">
        This is a supportive companion, not a substitute for professional care. If you're in crisis, please contact a helpline (e.g. Sumithrayo 1926).
      </p>
    </div>
  );
}

export default Chat;
