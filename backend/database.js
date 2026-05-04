const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const db = new sqlite3.Database(path.join(__dirname, 'simora.db'));

// Promise wrapper
db.runAsync = (sql, params=[]) => new Promise((resolve, reject) => {
  db.run(sql, params, function(err) { if(err) reject(err); else resolve(this); });
});
db.getAsync = (sql, params=[]) => new Promise((resolve, reject) => {
  db.get(sql, params, (err, row) => err ? reject(err) : resolve(row));
});
db.allAsync = (sql, params=[]) => new Promise((resolve, reject) => {
  db.all(sql, params, (err, rows) => err ? reject(err) : resolve(rows));
});

// Initialisation: create tables
const initDB = async () => {
  await db.runAsync(`
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
    )
  `);
  await db.runAsync(`
    CREATE TABLE IF NOT EXISTS subscriptions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT,
      tier TEXT,
      date TEXT
    )
  `);
  await db.runAsync(`
    CREATE TABLE IF NOT EXISTS contacts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT,
      message TEXT,
      date TEXT
    )
  `);
  await db.runAsync(`
    CREATE TABLE IF NOT EXISTS data_requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      mine TEXT,
      location TEXT,
      data_needed TEXT,
      date TEXT,
      approved BOOLEAN DEFAULT 0
    )
  `);
  await db.runAsync(`
    CREATE TABLE IF NOT EXISTS gallery (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      url TEXT,
      caption TEXT,
      timestamp TEXT,
      mine TEXT
    )
  `);
  await db.runAsync(`
    CREATE TABLE IF NOT EXISTS institution_subs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      institution TEXT,
      contact_name TEXT,
      email TEXT,
      date TEXT
    )
  `);
  // Insert default admin if not exists
  const admin = await db.getAsync(`SELECT * FROM users WHERE email = 'admin@simora.com'`);
  if (!admin) {
    const bcrypt = require('bcrypt');
    const hashed = await bcrypt.hash('admin123', 10);
    await db.runAsync(
      `INSERT INTO users (name, email, password, role, created_at) VALUES (?, ?, ?, ?, ?)`,
      ['Admin', 'admin@simora.com', hashed, 'admin', new Date().toISOString()]
    );
  }
  console.log('Database initialised');
};

module.exports = { db, initDB };