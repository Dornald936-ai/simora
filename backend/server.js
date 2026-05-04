const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { run, get, all, initDB } = require('./database');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const SECRET = process.env.JWT_SECRET || 'simora_secret_2026';

// Middleware to verify JWT
const verifyToken = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({ error: 'No token provided' });
  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, SECRET);
    req.user = decoded;
    next();
  } catch(err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

// ---------- Auth ----------
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = get(`SELECT * FROM users WHERE email = ?`, [email]);
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });
    const valid = await bcrypt.compare(password, user.password);
    if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
    const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, SECRET, { expiresIn: '7d' });
    res.json({ token, user: { id: user.id, name: user.name, email: user.email, role: user.role, expiry: user.expiry, mine: user.mine } });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Subscriptions ----------
app.post('/api/subscriptions', verifyToken, (req, res) => {
  const { name, email, tier } = req.body;
  if (!email) return res.status(400).json({ error: 'Email required' });
  try {
    run(`INSERT INTO subscriptions (name, email, tier, date) VALUES (?, ?, ?, ?)`, [name, email, tier, new Date().toISOString()]);
    res.json({ success: true });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/subscriptions', verifyToken, (req, res) => {
  try {
    const rows = all(`SELECT * FROM subscriptions ORDER BY date DESC`);
    res.json(rows);
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Contacts ----------
app.post('/api/contacts', (req, res) => {
  const { name, email, message } = req.body;
  try {
    run(`INSERT INTO contacts (name, email, message, date) VALUES (?, ?, ?, ?)`, [name, email, message, new Date().toISOString()]);
    res.json({ success: true });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/contacts', verifyToken, (req, res) => {
  try {
    const rows = all(`SELECT * FROM contacts ORDER BY date DESC`);
    res.json(rows);
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Data Requests ----------
app.post('/api/data-requests', (req, res) => {
  const { name, mine, location, dataNeeded } = req.body;
  try {
    run(`INSERT INTO data_requests (name, mine, location, data_needed, date) VALUES (?, ?, ?, ?, ?)`,
      [name, mine, location, dataNeeded, new Date().toISOString()]);
    res.json({ success: true });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/data-requests', verifyToken, (req, res) => {
  try {
    const rows = all(`SELECT * FROM data_requests ORDER BY date DESC`);
    res.json(rows);
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/data-requests/:id/approve', verifyToken, async (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const id = req.params.id;
  const { days } = req.body;
  try {
    const reqRecord = get(`SELECT * FROM data_requests WHERE id = ?`, [id]);
    if (!reqRecord) return res.status(404).json({ error: 'Not found' });
    const expiry = new Date();
    expiry.setDate(expiry.getDate() + days);
    const password = Math.random().toString(36).substring(2, 8);
    const hashed = await bcrypt.hash(password, 10);
    run(`INSERT INTO users (name, email, password, mine, location, expiry, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [reqRecord.name, `user_${Date.now()}@simora.com`, hashed, reqRecord.mine, reqRecord.location, expiry.toISOString(), 'requester', new Date().toISOString()]);
    run(`UPDATE data_requests SET approved = 1 WHERE id = ?`, [id]);
    res.json({ success: true, email: `user_${Date.now()}@simora.com`, password });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Gallery ----------
app.get('/api/gallery', (req, res) => {
  try {
    const rows = all(`SELECT * FROM gallery ORDER BY timestamp DESC`);
    res.json(rows);
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/gallery', verifyToken, (req, res) => {
  const { url, caption, mine } = req.body;
  if (!url) return res.status(400).json({ error: 'URL required' });
  try {
    run(`INSERT INTO gallery (url, caption, timestamp, mine) VALUES (?, ?, ?, ?)`, [url, caption, new Date().toISOString(), mine || 'Great Dyke']);
    res.json({ success: true });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Institutional Subscriptions ----------
app.post('/api/inst-subscriptions', (req, res) => {
  const { institution, contactName, email } = req.body;
  try {
    run(`INSERT INTO institution_subs (institution, contact_name, email, date) VALUES (?, ?, ?, ?)`, [institution, contactName, email, new Date().toISOString()]);
    res.json({ success: true });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/inst-subscriptions', verifyToken, (req, res) => {
  try {
    const rows = all(`SELECT * FROM institution_subs ORDER BY date DESC`);
    res.json(rows);
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Users list ----------
app.get('/api/users', verifyToken, (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  try {
    const rows = all(`SELECT id, name, email, mine, expiry, role FROM users`);
    res.json(rows);
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- Admin clear data ----------
app.delete('/api/admin/clear/:type', verifyToken, (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const type = req.params.type;
  const tableMap = { subs: 'subscriptions', reqs: 'data_requests', contacts: 'contacts', gallery: 'gallery', inst: 'institution_subs', users: 'users' };
  const table = tableMap[type];
  if (!table) return res.status(400).json({ error: 'Invalid type' });
  try {
    run(`DELETE FROM ${table}`);
    res.json({ success: true });
  } catch(err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------- InSAR endpoint (simulated, but ready for real API) ----------
app.get('/api/insar/:lat/:lon', async (req, res) => {
  // Simulated data – replace with actual Sentinel‑1 call later
  const simulated = 1.5 + Math.random() * 4.5;
  res.json({
    deformation_mm: simulated.toFixed(2),
    source: 'Sentinel-1 (simulated - add credentials later)',
    timestamp: new Date().toISOString()
  });
});

// Start server
initDB();
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));