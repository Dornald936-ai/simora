const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const SECRET = process.env.JWT_SECRET || 'simora_secret_2026';

// ---------- In‑memory data store ----------
let data = {
  users: [],
  subscriptions: [],
  contacts: [],
  dataRequests: [],
  gallery: [],
  institutionSubs: [],
  nextId: { user:1, sub:1, contact:1, req:1, gallery:1, inst:1 }
};

// Helper to get next ID
function nextId(collection) {
  const id = data.nextId[collection];
  data.nextId[collection] = id + 1;
  return id;
}

// Initialize default admin
(async () => {
  if (!data.users.find(u => u.email === 'admin@simora.com')) {
    const hashed = await bcrypt.hash('admin123', 10);
    data.users.push({
      id: nextId('user'),
      name: 'Admin',
      email: 'admin@simora.com',
      password: hashed,
      role: 'admin',
      created_at: new Date().toISOString()
    });
  }
})();

// Middleware
const verifyToken = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({ error: 'No token' });
  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, SECRET);
    req.user = decoded;
    next();
  } catch(e) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

// ---------- Auth ----------
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;
  const user = data.users.find(u => u.email === email);
  if (!user) return res.status(401).json({ error: 'Invalid credentials' });
  const valid = await bcrypt.compare(password, user.password);
  if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
  const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, SECRET, { expiresIn: '7d' });
  res.json({ token, user: { id: user.id, name: user.name, email: user.email, role: user.role, expiry: user.expiry, mine: user.mine } });
});

// ---------- Subscriptions ----------
app.post('/api/subscriptions', verifyToken, (req, res) => {
  const { name, email, tier } = req.body;
  if (!email) return res.status(400).json({ error: 'Email required' });
  data.subscriptions.push({ id: nextId('sub'), name, email, tier, date: new Date().toISOString() });
  res.json({ success: true });
});
app.get('/api/subscriptions', verifyToken, (req, res) => {
  res.json([...data.subscriptions].reverse());
});

// ---------- Contacts ----------
app.post('/api/contacts', (req, res) => {
  const { name, email, message } = req.body;
  data.contacts.push({ id: nextId('contact'), name, email, message, date: new Date().toISOString() });
  res.json({ success: true });
});
app.get('/api/contacts', verifyToken, (req, res) => {
  res.json([...data.contacts].reverse());
});

// ---------- Data Requests ----------
app.post('/api/data-requests', (req, res) => {
  const { name, mine, location, dataNeeded } = req.body;
  data.dataRequests.push({ id: nextId('req'), name, mine, location, dataNeeded, date: new Date().toISOString(), approved: false });
  res.json({ success: true });
});
app.get('/api/data-requests', verifyToken, (req, res) => {
  res.json([...data.dataRequests].reverse());
});
app.put('/api/data-requests/:id/approve', verifyToken, async (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const id = parseInt(req.params.id);
  const reqIndex = data.dataRequests.findIndex(r => r.id === id);
  if (reqIndex === -1) return res.status(404).json({ error: 'Not found' });
  const reqRecord = data.dataRequests[reqIndex];
  if (reqRecord.approved) return res.status(400).json({ error: 'Already approved' });
  const expiry = new Date();
  expiry.setDate(expiry.getDate() + (req.body.days || 7));
  const password = Math.random().toString(36).substring(2, 8);
  const hashed = await bcrypt.hash(password, 10);
  data.users.push({
    id: nextId('user'),
    name: reqRecord.name,
    email: `user_${Date.now()}@simora.com`,
    password: hashed,
    mine: reqRecord.mine,
    location: reqRecord.location,
    expiry: expiry.toISOString(),
    role: 'requester',
    created_at: new Date().toISOString()
  });
  data.dataRequests[reqIndex].approved = true;
  res.json({ success: true, email: `user_${Date.now()}@simora.com`, password });
});

// ---------- Gallery ----------
app.get('/api/gallery', (req, res) => {
  res.json([...data.gallery].reverse());
});
app.post('/api/gallery', verifyToken, (req, res) => {
  const { url, caption, mine } = req.body;
  if (!url) return res.status(400).json({ error: 'URL required' });
  data.gallery.push({ id: nextId('gallery'), url, caption, timestamp: new Date().toISOString(), mine: mine || 'Great Dyke' });
  res.json({ success: true });
});

// ---------- Institutional Subscriptions ----------
app.post('/api/inst-subscriptions', (req, res) => {
  const { institution, contactName, email } = req.body;
  data.institutionSubs.push({ id: nextId('inst'), institution, contactName, email, date: new Date().toISOString() });
  res.json({ success: true });
});
app.get('/api/inst-subscriptions', verifyToken, (req, res) => {
  res.json([...data.institutionSubs].reverse());
});

// ---------- Users list ----------
app.get('/api/users', verifyToken, (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const users = data.users.map(u => ({ id: u.id, name: u.name, email: u.email, mine: u.mine, expiry: u.expiry, role: u.role }));
  res.json(users);
});

// ---------- Admin clear data ----------
app.delete('/api/admin/clear/:type', verifyToken, (req, res) => {
  if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
  const type = req.params.type;
  switch(type) {
    case 'subs': data.subscriptions = []; break;
    case 'reqs': data.dataRequests = []; break;
    case 'contacts': data.contacts = []; break;
    case 'gallery': data.gallery = []; break;
    case 'inst': data.institutionSubs = []; break;
    case 'users': data.users = data.users.filter(u => u.role === 'admin'); break; // keep admin
    default: return res.status(400).json({ error: 'Invalid type' });
  }
  res.json({ success: true });
});

// ---------- InSAR endpoint (simulated) ----------
app.get('/api/insar/:lat/:lon', (req, res) => {
  const simulated = 1.5 + Math.random() * 4.5;
  res.json({
    deformation_mm: simulated.toFixed(2),
    source: 'Sentinel-1 (simulated)',
    timestamp: new Date().toISOString()
  });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));