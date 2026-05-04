const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { db, initDB } = require('./database');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const SECRET = process.env.JWT_SECRET || 'simora_secret_2026';

// Middleware to verify JWT
const verifyToken = (req, res, next) => {
  const token = req.headers['authorization'];
  if (!token) return res.status(401).json({ error: 'No token' });
  try {
    const decoded = jwt.verify(token.split(' ')[1], SECRET);
    req.user = decoded;
    next();
  } catch(e) { res.status(401).json({ error: 'Invalid token' }); }
};

// ---------- Auth routes ----------
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await db.getAsync(`SELECT * FROM users WHERE email = ?`, [email]);
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });
    const valid = await bcrypt.compare(password, user.password);
    if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
    const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, SECRET, { expiresIn: '7d' });
    res.json({ token, user: { id: user.id, name: user.name, email: user.email, role: user.role, expiry: user.expiry, mine: user.mine } });
  } catch(e) { res.status(500).json({ error: e.message }); }
});

// ---------- Subscription routes ----------
app.post('/api/subscriptions', verifyToken, async (req, res) => {
  const { name, email, tier } = req.body;
  if (!email) return res.status(400).json({ error: 'Email required' });
  await db.runAsync(`INSERT INTO subscriptions (name, email, tier, date) VALUES (?, ?, ?, ?)`, [name, email, tier, new Date().toISOString()]);
  res.json({ success: true });
});
app.get('/api/subscriptions', verifyToken, async (req, res) => {
  const rows = await db.allAsync(`SELECT * FROM subscriptions ORDER BY date DESC`);
  res.json(rows);
});

// Contacts
app.post('/api/contacts', async (req, res) => {
  const { name, email, message } = req.body;
  await db.runAsync(`INSERT INTO contacts (name, email, message, date) VALUES (?, ?, ?, ?)`, [name, email, message, new Date().toISOString()]);
  res.json({ success: true });
});
app.get('/api/contacts', verifyToken, async (req, res) => {
  const rows = await db.allAsync(`SELECT * FROM contacts ORDER BY date DESC`);
  res.json(rows);
});

// Data requests (public)
app.post('/api/data-requests', async (req, res) => {
  const { name, mine, location, dataNeeded } = req.body;
  await db.runAsync(`INSERT INTO data_requests (name, mine, location, data_needed, date) VALUES (?, ?, ?, ?, ?)`, [name, mine, location, dataNeeded, new Date().toISOString()]);
  res.json({ success: true });
});
app.get('/api/data-requests', verifyToken, async (req, res) => {
  const rows = await db.allAsync(`SELECT * FROM data_requests ORDER BY date DESC`);
  res.json(rows);
});
app.put('/api/data-requests/:id/approve', verifyToken, async (req, res) => {
  const { days } = req.body;
  const expiry = new Date(); expiry.setDate(expiry.getDate() + days);
  // get request
  const reqRecord = await db.getAsync(`SELECT * FROM data_requests WHERE id = ?`, [req.params.id]);
  if (!reqRecord) return res.status(404).json({ error: 'Not found' });
  const password = Math.random().toString(36).substring(2,8);
  const hashed = await bcrypt.hash(password, 10);
  await db.runAsync(`INSERT INTO users (name, email, password, mine, location, expiry, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [reqRecord.name, `user_${Date.now()}@simora.com`, hashed, reqRecord.mine, reqRecord.location, expiry.toISOString(), 'requester', new Date().toISOString()]);
  await db.runAsync(`UPDATE data_requests SET approved = 1 WHERE id = ?`, [req.params.id]);
  res.json({ success: true, email: `user_${Date.now()}@simora.com`, password });
});

// Gallery
app.get('/api/gallery', async (req, res) => {
  const rows = await db.allAsync(`SELECT * FROM gallery ORDER BY timestamp DESC`);
  res.json(rows);
});
app.post('/api/gallery', verifyToken, async (req, res) => {
  const { url, caption, mine } = req.body;
  if (!url) return res.status(400).json({ error: 'URL required' });
  await db.runAsync(`INSERT INTO gallery (url, caption, timestamp, mine) VALUES (?, ?, ?, ?)`, [url, caption, new Date().toISOString(), mine || 'Great Dyke']);
  res.json({ success: true });
});

// Institution subscriptions
app.post('/api/inst-subscriptions', async (req, res) => {
  const { institution, contactName, email } = req.body;
  await db.runAsync(`INSERT INTO institution_subs (institution, contact_name, email, date) VALUES (?, ?, ?, ?)`, [institution, contactName, email, new Date().toISOString()]);
  res.json({ success: true });
});
app.get('/api/inst-subscriptions', verifyToken, async (req, res) => {
  const rows = await db.allAsync(`SELECT * FROM institution_subs ORDER BY date DESC`);
  res.json(rows);
});

// Users list (admin)
app.get('/api/users', verifyToken, async (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const rows = await db.allAsync(`SELECT id, name, email, mine, expiry, role FROM users`);
  res.json(rows);
});

// Delete all data (clear) – admin only
app.delete('/api/admin/clear/:type', verifyToken, async (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const type = req.params.type;
  const tableMap = { subs: 'subscriptions', reqs: 'data_requests', contacts: 'contacts', gallery: 'gallery', inst: 'institution_subs', users: 'users' };
  const table = tableMap[type];
  if (!table) return res.status(400).json({ error: 'Invalid type' });
  await db.runAsync(`DELETE FROM ${table}`);
  res.json({ success: true });
});

// ---------- REAL EO DATA: Sentinel Hub ----------
// You need to register at https://www.sentinel-hub.com/ and get an OAuth client.
// We'll implement a generic wrapper that returns simulated data until real credentials are added.
const getInSAR = async (lat, lon) => {
  // Real implementation: call Sentinel Hub Process API to get backscatter/coherence.
  // For now, we simulate a realistic value that changes slowly.
  // To enable real data, uncomment the axios call and fill credentials.
  const simulated = 1.5 + Math.random() * 4.5;
  return { deformation_mm: simulated.toFixed(2), source: 'Sentinel-1 (simulated - add credentials)' };
};
app.get('/api/insar/:lat/:lon', async (req, res) => {
  const { lat, lon } = req.params;
  const data = await getInSAR(lat, lon);
  res.json({ ...data, timestamp: new Date().toISOString() });
});

// Start server
const PORT = process.env.PORT || 5000;
initDB().then(() => {
  app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
});