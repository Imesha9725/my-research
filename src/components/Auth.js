import React, { useState } from 'react';
import './Auth.css';

function Auth({ apiUrl, onLoggedIn, onContinueGuest }) {
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    const path = mode === 'register' ? '/api/auth/register' : '/api/auth/login';
    setLoading(true);
    try {
      const res = await fetch(`${apiUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.error || 'Something went wrong');
        return;
      }
      if (data.token && data.user) {
        onLoggedIn(data.token, data.user.email);
      } else {
        setError('Invalid response from server');
      }
    } catch (_) {
      setError('Cannot reach server. Is it running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-app">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-header-icon" aria-hidden>
            <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none" />
              <path d="M10 14c0-1.5 1.5-3 4-3s4 1.5 4 3v6c0 1.5-1.5 3-4 3s-4-1.5-4-3v-6z" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <h1 className="auth-title">Mental Health Support</h1>
            <p className="auth-subtitle">Sign in to save your chats and emotional memory</p>
          </div>
        </div>

        <div className="auth-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            className={`auth-tab ${mode === 'login' ? 'auth-tab--active' : ''}`}
            onClick={() => { setMode('login'); setError(null); }}
          >
            Log in
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            className={`auth-tab ${mode === 'register' ? 'auth-tab--active' : ''}`}
            onClick={() => { setMode('register'); setError(null); }}
          >
            Register
          </button>
        </div>

        <form className="auth-form" onSubmit={submit}>
          <label className="auth-label">
            Email
            <input
              type="email"
              className="auth-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
              disabled={loading}
            />
          </label>
          <label className="auth-label">
            Password
            <input
              type="password"
              className="auth-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
              minLength={mode === 'register' ? 6 : undefined}
              required
              disabled={loading}
            />
          </label>
          {mode === 'register' && (
            <p className="auth-hint">Use at least 6 characters. Your data is stored locally on the server database.</p>
          )}
          {error && (
            <p className="auth-error" role="alert">
              {error}
            </p>
          )}
          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? 'Please wait…' : mode === 'register' ? 'Create account' : 'Log in'}
          </button>
        </form>

        <button type="button" className="auth-guest" onClick={onContinueGuest} disabled={loading}>
          Continue without an account
        </button>
        <p className="auth-guest-note">Guest mode works the same, but messages are not saved when you leave.</p>
      </div>
    </div>
  );
}

export default Auth;
