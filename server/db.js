import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

const dbPath = path.join(dataDir, 'chat.db');
const db = new Database(dbPath);

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'bot')),
    text TEXT NOT NULL,
    emotion TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
  );
  CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, created_at);
`);

export function findUserByEmail(email) {
  const row = db.prepare('SELECT id, email, password_hash FROM users WHERE email = ? COLLATE NOCASE').get(email.trim().toLowerCase());
  return row || null;
}

export function createUser(email, passwordHash) {
  const e = email.trim().toLowerCase();
  const r = db.prepare('INSERT INTO users (email, password_hash) VALUES (?, ?)').run(e, passwordHash);
  return { id: r.lastInsertRowid, email: e };
}

export function insertMessage(userId, role, text, emotion) {
  db.prepare('INSERT INTO messages (user_id, role, text, emotion) VALUES (?, ?, ?, ?)').run(
    userId,
    role,
    text,
    emotion || null
  );
}

/** Last N messages for LLM / emotional context (oldest first). */
export function getRecentHistory(userId, limit = 24) {
  const rows = db
    .prepare(
      `SELECT role, text FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?`
    )
    .all(userId, limit);
  return rows.reverse().map((r) => ({ role: r.role, text: r.text }));
}

/** Last N messages for UI (oldest first). */
export function getAllMessagesForUser(userId, limit = 200) {
  const rows = db
    .prepare(
      `SELECT id, role, text, emotion, created_at FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?`
    )
    .all(userId, limit);
  return rows.reverse();
}

export default db;
