const Database = require('better-sqlite3');
const path = require('path');
const bcrypt = require('bcrypt');

const db = new Database(path.join(__dirname, 'simora.db'));

// Enable foreign keys
db.pragma('foreign_keys = ON');

// Helper wrappers (synchronous)
function run(sql, params = []) {
  const stmt = db.prepare(sql);
  return stmt.run(...params);
}

function get(sql, params = []) {
  const stmt = db.prepare(sql);
  return stmt.get(...params);
}

function all(sql, params = []) {
  const stmt = db.prepare(sql);
  return stmt.all(...params);
}

// Initialize tables
const initDB = () => {
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT UNIQUE,
      password TEXT,
      mine TEXT,
      location TEXT,
      expiry TEXT,
      role TEXT DEFAULT 'requester',
      created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS subscriptions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT,
      tier TEXT,
      date TEXT
    );
    CREATE TABLE IF NOT EXISTS contacts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT,
      message TEXT,
      date TEXT
    );
    CREATE TABLE IF NOT EXISTS data_requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      mine TEXT,
      location TEXT,
      data_needed TEXT,
      date TEXT,
      approved INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS gallery (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      url TEXT,
      caption TEXT,
      timestamp TEXT,
      mine TEXT
    );
    CREATE TABLE IF NOT EXISTS institution_subs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      institution TEXT,
      contact_name TEXT,
      email TEXT,
      date TEXT
    );
  `);

  // Insert default admin if not exists
  const admin = get(`SELECT * FROM users WHERE email = ?`, ['admin@simora.com']);
  if (!admin) {
    const hashed = bcrypt.hashSync('admin123', 10);
    run(`INSERT INTO users (name, email, password, role, created_at) VALUES (?, ?, ?, ?, ?)`,
      ['Admin', 'admin@simora.com', hashed, 'admin', new Date().toISOString()]);
  }
  console.log('Database initialised');
};

module.exports = { db, run, get, all, initDB };