import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { 
  Activity, Cpu, Satellite, Wifi, TrendingUp, AlertTriangle, Brain, RefreshCw,
  Mail, FileText, Send, CreditCard, UserCheck, Eye, Download, Trash2,
  Lock, LogIn, Image, CheckCircle, XCircle, Calendar
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

// ---------- TensorFlow.js Model ----------
interface TrainingRun {
  id: string;
  timestamp: Date;
  lossHistory: number[];
  finalLoss: number;
  accuracy: number;
  description: string;
}

class RealMLModel {
  private model: tf.Sequential | null = null;
  private isTraining = false;
  private currentAccuracy = 0;
  private trainingRuns: TrainingRun[] = [];
  private currentRunId: string | null = null;

  async init() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [3] }));
    this.model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    await this.initialTrain();
  }

  private generateSyntheticData() {
    const inputs = [
      [0.5, 0.2, 0.1], [1.2, 0.4, 0.2], [2.0, 0.7, 0.3], [3.5, 1.2, 0.5], [5.0, 1.8, 0.8],
      [0.8, 0.3, 0.15], [1.5, 0.5, 0.25], [2.5, 0.9, 0.4], [4.0, 1.4, 0.6], [6.0, 2.0, 0.9]
    ];
    const outputs = inputs.map(inp => {
      let risk = (inp[0] / 6) * 0.5 + (inp[1] / 2) * 0.3 + inp[2] * 0.2;
      return [Math.min(0.98, Math.max(0.02, risk))];
    });
    return { xs: tf.tensor2d(inputs), ys: tf.tensor2d(outputs) };
  }

  private async initialTrain() {
    const { xs, ys } = this.generateSyntheticData();
    const history = await this.model!.fit(xs, ys, { epochs: 150, verbose: 0, validationSplit: 0.2 });
    const lossHistory = history.history.loss as number[];
    const finalLoss = lossHistory[lossHistory.length - 1];
    await this.evaluateAccuracy();
    this.addTrainingRun(lossHistory, finalLoss, this.currentAccuracy, 'Initial synthetic training');
    xs.dispose(); ys.dispose();
  }

  private async evaluateAccuracy() {
    const { xs, ys } = this.generateSyntheticData();
    const predictions = this.model!.predict(xs) as tf.Tensor;
    const predValues = await predictions.data();
    const trueValues = await ys.data();
    let correct = 0;
    for (let i = 0; i < predValues.length; i++) {
      if (Math.abs(predValues[i] - trueValues[i]) <= 0.1) correct++;
    }
    this.currentAccuracy = (correct / predValues.length) * 100;
    xs.dispose(); ys.dispose(); predictions.dispose();
  }

  private addTrainingRun(lossHistory: number[], finalLoss: number, accuracy: number, description: string) {
    const run: TrainingRun = {
      id: Date.now().toString(),
      timestamp: new Date(),
      lossHistory: [...lossHistory],
      finalLoss,
      accuracy,
      description
    };
    this.trainingRuns.push(run);
    this.currentRunId = run.id;
  }

  async predict(insar: number, vibration: number, rainfall: number): Promise<number> {
    if (!this.model) return 50;
    const input = tf.tensor2d([[insar, vibration, rainfall]]);
    const output = this.model.predict(input) as tf.Tensor;
    const risk = (await output.data())[0] * 100;
    output.dispose(); input.dispose();
    return Math.round(risk);
  }

  async retrain(newData: { insar: number, vibration: number, rainfall: number, risk: number }[]) {
    if (!this.model || this.isTraining) return;
    this.isTraining = true;
    try {
      const inputs = newData.map(d => [d.insar, d.vibration, d.rainfall]);
      const outputs = newData.map(d => [d.risk / 100]);
      const { xs: synthXs, ys: synthYs } = this.generateSyntheticData();
      const allXs = tf.concat([synthXs, tf.tensor2d(inputs)], 0);
      const allYs = tf.concat([synthYs, tf.tensor2d(outputs)], 0);
      const history = await this.model.fit(allXs, allYs, { epochs: 50, verbose: 0, validationSplit: 0.2 });
      const lossHistory = history.history.loss as number[];
      const finalLoss = lossHistory[lossHistory.length - 1];
      await this.evaluateAccuracy();
      const desc = `Retrain with ${newData.length} new incident(s) (risk ~${newData[0].risk}%)`;
      this.addTrainingRun(lossHistory, finalLoss, this.currentAccuracy, desc);
      allXs.dispose(); allYs.dispose();
      synthXs.dispose(); synthYs.dispose();
    } catch (err) {
      console.error('Retrain error', err);
    } finally {
      this.isTraining = false;
    }
  }

  getCurrentAccuracy(): number { return this.currentAccuracy; }
  getTrainingRuns(): TrainingRun[] { return this.trainingRuns; }
  getCurrentRunId(): string | null { return this.currentRunId; }
  resetHistory() {
    this.trainingRuns = [];
    this.currentRunId = null;
    this.initialTrain();
  }
}

const mlModel = new RealMLModel();

// ---------- Storage Keys ----------
const STORAGE_KEYS = {
  subscriptions: 'simora_subscriptions',
  dataRequests: 'simora_data_requests',
  contacts: 'simora_contacts',
  users: 'simora_users',
  gallery: 'simora_gallery',
  instSubs: 'simora_inst_subs'
};

const saveToLocal = (key: string, data: any) => localStorage.setItem(key, JSON.stringify(data));
const loadFromLocal = (key: string) => {
  const raw = localStorage.getItem(key);
  return raw ? JSON.parse(raw) : [];
};

const isExpired = (expiry: string) => new Date(expiry) < new Date();

// ---------- Dashboard Component ----------
const Dashboard = () => {
  // Tabs
  const [activeTab, setActiveTab] = useState('dashboard');

  // Auth state
  const [adminLoggedIn, setAdminLoggedIn] = useState(false);
  const [userLoggedIn, setUserLoggedIn] = useState<{ id: string; name: string; email: string; expiry: string; mine: string } | null>(null);
  const [showAdminLogin, setShowAdminLogin] = useState(false);
  const [showUserLogin, setShowUserLogin] = useState(false);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  // ML state
  const [modelLoading, setModelLoading] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMine, setSelectedMine] = useState('Great Dyke - North Dam');
  const [riskPercent, setRiskPercent] = useState(50);
  const [riskHistory, setRiskHistory] = useState<{ time: string; risk: number }[]>([
    { time: '00:00', risk: 45 }, { time: '06:00', risk: 48 }
  ]);
  const [insarValue, setInsarValue] = useState(2.5);
  const [vibrationValue, setVibrationValue] = useState(1.1);
  const [rainfallValue, setRainfallValue] = useState(0.4);
  const [userCount, setUserCount] = useState(147);
  const [modelAccuracy, setModelAccuracy] = useState<number | null>(null);
  const [modelLoss, setModelLoss] = useState<number | null>(null);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [displayLossHistory, setDisplayLossHistory] = useState<number[]>([]);
  const [training, setTraining] = useState(false);
  const [fetchingEO, setFetchingEO] = useState(false);
  const [eoSource, setEoSource] = useState('Sentinel-1 (simulated)');
  const [lastFetch, setLastFetch] = useState('');

  // Form states
  const [subEmail, setSubEmail] = useState('');
  const [subName, setSubName] = useState('');
  const [subTier, setSubTier] = useState('free');
  const [subMessage, setSubMessage] = useState('');
  const [contactName, setContactName] = useState('');
  const [contactEmail, setContactEmail] = useState('');
  const [contactMsg, setContactMsg] = useState('');
  const [contactSuccess, setContactSuccess] = useState('');
  const [requestName, setRequestName] = useState('');
  const [requestMine, setRequestMine] = useState('');
  const [requestLocation, setRequestLocation] = useState('');
  const [requestData, setRequestData] = useState('');
  const [requestSuccess, setRequestSuccess] = useState('');

  // Admin/Storage data
  const [subscriptions, setSubscriptions] = useState<any[]>([]);
  const [dataRequests, setDataRequests] = useState<any[]>([]);
  const [contacts, setContacts] = useState<any[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [galleryImages, setGalleryImages] = useState<any[]>([]);
  const [instSubs, setInstSubs] = useState<any[]>([]);
  const [newImageUrl, setNewImageUrl] = useState('');
  const [newImageCaption, setNewImageCaption] = useState('');
  const [instSubName, setInstSubName] = useState('');
  const [instSubEmail, setInstSubEmail] = useState('');
  const [instSubInst, setInstSubInst] = useState('');
  const [instSubMessage, setInstSubMessage] = useState('');

  // Load stored data
  useEffect(() => {
    setSubscriptions(loadFromLocal(STORAGE_KEYS.subscriptions));
    setDataRequests(loadFromLocal(STORAGE_KEYS.dataRequests));
    setContacts(loadFromLocal(STORAGE_KEYS.contacts));
    setUsers(loadFromLocal(STORAGE_KEYS.users));
    setGalleryImages(loadFromLocal(STORAGE_KEYS.gallery));
    setInstSubs(loadFromLocal(STORAGE_KEYS.instSubs));
  }, []);

  // ML init and functions
  const refreshMetrics = () => {
    setModelAccuracy(mlModel.getCurrentAccuracy());
    const runs = mlModel.getTrainingRuns();
    setTrainingRuns(runs);
    const currentId = mlModel.getCurrentRunId();
    setSelectedRunId(currentId);
    const selectedRun = runs.find(r => r.id === currentId);
    setDisplayLossHistory(selectedRun?.lossHistory || []);
    if (runs.length && currentId) {
      const lastLoss = runs.find(r => r.id === currentId)?.finalLoss;
      setModelLoss(lastLoss ?? null);
    }
  };

  const updatePrediction = async () => {
    if (!modelReady) return;
    const newRisk = await mlModel.predict(insarValue, vibrationValue, rainfallValue);
    setRiskPercent(newRisk);
    setRiskHistory(prev => [...prev.slice(-5), { time: new Date().toLocaleTimeString(), risk: newRisk }]);
  };

  useEffect(() => {
    mlModel.init()
      .then(() => {
        setModelReady(true);
        setModelLoading(false);
        refreshMetrics();
        updatePrediction();
      })
      .catch(err => {
        setError(err.message);
        setModelLoading(false);
      });
  }, []);

  useEffect(() => {
    if (modelReady) updatePrediction();
  }, [insarValue, vibrationValue, rainfallValue, modelReady]);

  useEffect(() => {
    const interval = setInterval(() => {
      setUserCount(prev => Math.min(200, prev + 0.1));
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  const handleRetrain = async () => {
    if (!modelReady || training) return;
    setTraining(true);
    try {
      await mlModel.retrain([{ insar: insarValue, vibration: vibrationValue, rainfall: rainfallValue, risk: riskPercent }]);
      await updatePrediction();
      refreshMetrics();
      alert(`Model retrained! New accuracy: ${mlModel.getCurrentAccuracy().toFixed(1)}%`);
    } catch (err) {
      alert('Retraining failed');
    } finally {
      setTraining(false);
    }
  };

  const handleSelectRun = (runId: string) => {
    const run = trainingRuns.find(r => r.id === runId);
    if (run) {
      setDisplayLossHistory(run.lossHistory);
      setModelLoss(run.finalLoss);
      setModelAccuracy(run.accuracy);
    }
  };

  const handleResetHistory = () => {
    mlModel.resetHistory();
    refreshMetrics();
  };

  const fetchLiveInsar = () => {
    setFetchingEO(true);
    setTimeout(() => {
      const newInsar = Math.random() * 5 + 1;
      setInsarValue(parseFloat(newInsar.toFixed(2)));
      setEoSource('Sentinel-1 (live simulation)');
      setLastFetch(new Date().toLocaleTimeString());
      setFetchingEO(false);
    }, 800);
  };

  // Form handlers (save to localStorage)
  const handleSubscribe = (e: React.FormEvent) => {
    e.preventDefault();
    if (!subEmail) return;
    const newSub = { id: Date.now(), name: subName, email: subEmail, tier: subTier, date: new Date().toISOString() };
    const updated = [...subscriptions, newSub];
    setSubscriptions(updated);
    saveToLocal(STORAGE_KEYS.subscriptions, updated);
    setSubMessage(`Subscription successful (${subTier} tier). Welcome ${subName || 'miner'}!`);
    setTimeout(() => setSubMessage(''), 3000);
    setSubEmail(''); setSubName('');
  };

  const handleContact = (e: React.FormEvent) => {
    e.preventDefault();
    const newContact = { id: Date.now(), name: contactName, email: contactEmail, message: contactMsg, date: new Date().toISOString() };
    const updated = [...contacts, newContact];
    setContacts(updated);
    saveToLocal(STORAGE_KEYS.contacts, updated);
    setContactSuccess(`Thank you ${contactName}, we'll respond within 24 hours.`);
    setTimeout(() => setContactSuccess(''), 3000);
    setContactName(''); setContactEmail(''); setContactMsg('');
  };

  const handleDataRequest = (e: React.FormEvent) => {
    e.preventDefault();
    const newReq = { id: Date.now(), name: requestName, mine: requestMine, location: requestLocation, dataNeeded: requestData, date: new Date().toISOString(), approved: false };
    const updated = [...dataRequests, newReq];
    setDataRequests(updated);
    saveToLocal(STORAGE_KEYS.dataRequests, updated);
    setRequestSuccess(`Request submitted! Admin will review and send you login credentials.`);
    setTimeout(() => setRequestSuccess(''), 3000);
    setRequestName(''); setRequestMine(''); setRequestLocation(''); setRequestData('');
  };

  const handleInstSubscribe = (e: React.FormEvent) => {
    e.preventDefault();
    const newSub = { id: Date.now(), institution: instSubInst, contactName: instSubName, email: instSubEmail, date: new Date().toISOString() };
    const updated = [...instSubs, newSub];
    setInstSubs(updated);
    saveToLocal(STORAGE_KEYS.instSubs, updated);
    setInstSubMessage('Institutional subscription request sent. We will contact you.');
    setTimeout(() => setInstSubMessage(''), 3000);
    setInstSubName(''); setInstSubEmail(''); setInstSubInst('');
  };

  // Admin functions
  const handleAdminLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (loginEmail === 'admin@simora.com' && loginPassword === 'admin123') {
      setAdminLoggedIn(true);
      setShowAdminLogin(false);
      setLoginError('');
    } else {
      setLoginError('Invalid credentials');
    }
  };

  const handleUserLogin = (e: React.FormEvent) => {
    e.preventDefault();
    const user = users.find(u => u.email === loginEmail && u.password === loginPassword);
    if (user && !isExpired(user.expiry)) {
      setUserLoggedIn({ id: user.id, name: user.name, email: user.email, expiry: user.expiry, mine: user.mine });
      setShowUserLogin(false);
      setLoginError('');
    } else if (user && isExpired(user.expiry)) {
      setLoginError('Your access has expired. Please request new data.');
    } else {
      setLoginError('Invalid email or password');
    }
  };

  const approveDataRequest = (req: any, days: number) => {
    const expiryDate = new Date();
    expiryDate.setDate(expiryDate.getDate() + days);
    const newUser = {
      id: Date.now(),
      name: req.name,
      email: `user_${req.name.replace(/\s/g, '')}${Date.now()}@simora.com`,
      password: Math.random().toString(36).substring(2, 8),
      mine: req.mine,
      location: req.location,
      expiry: expiryDate.toISOString(),
      createdAt: new Date().toISOString()
    };
    const updatedUsers = [...users, newUser];
    setUsers(updatedUsers);
    saveToLocal(STORAGE_KEYS.users, updatedUsers);
    // Mark request as approved
    const updatedReqs = dataRequests.map(r => r.id === req.id ? { ...r, approved: true } : r);
    setDataRequests(updatedReqs);
    saveToLocal(STORAGE_KEYS.dataRequests, updatedReqs);
    alert(`User created!\nEmail: ${newUser.email}\nPassword: ${newUser.password}\nExpires: ${expiryDate.toLocaleString()}`);
  };

  const handleUploadImage = () => {
    if (!newImageUrl) return;
    const newImg = {
      id: Date.now(),
      url: newImageUrl,
      caption: newImageCaption || 'Mine area',
      timestamp: new Date().toISOString(),
      mine: selectedMine
    };
    const updated = [newImg, ...galleryImages];
    setGalleryImages(updated);
    saveToLocal(STORAGE_KEYS.gallery, updated);
    setNewImageUrl('');
    setNewImageCaption('');
  };

  const simulateRealTimeImage = () => {
    const fakeUrl = `https://picsum.photos/id/${Math.floor(Math.random() * 100)}/400/300`;
    const newImg = {
      id: Date.now(),
      url: fakeUrl,
      caption: `Live feed ${new Date().toLocaleTimeString()}`,
      timestamp: new Date().toISOString(),
      mine: selectedMine
    };
    const updated = [newImg, ...galleryImages];
    setGalleryImages(updated);
    saveToLocal(STORAGE_KEYS.gallery, updated);
  };

  const clearAdminData = (type: 'subs' | 'requests' | 'contacts' | 'users' | 'gallery' | 'instSubs') => {
    if (type === 'subs') { setSubscriptions([]); saveToLocal(STORAGE_KEYS.subscriptions, []); }
    if (type === 'requests') { setDataRequests([]); saveToLocal(STORAGE_KEYS.dataRequests, []); }
    if (type === 'contacts') { setContacts([]); saveToLocal(STORAGE_KEYS.contacts, []); }
    if (type === 'users') { setUsers([]); saveToLocal(STORAGE_KEYS.users, []); }
    if (type === 'gallery') { setGalleryImages([]); saveToLocal(STORAGE_KEYS.gallery, []); }
    if (type === 'instSubs') { setInstSubs([]); saveToLocal(STORAGE_KEYS.instSubs, []); }
  };

  if (error) return <div className="min-h-screen bg-gray-900 text-white p-6">Error: {error}</div>;
  if (modelLoading) return <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center"><Brain className="animate-pulse text-purple-400" size={48} /><span className="ml-2">Loading Simora AI…</span></div>;

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Top bar */}
      <div className="bg-gray-800/80 px-6 py-2 flex justify-end gap-3 text-sm">
        {!adminLoggedIn && <button onClick={() => setShowAdminLogin(true)} className="hover:text-blue-400"><Lock size={14} className="inline mr-1" /> Admin</button>}
        {!userLoggedIn && <button onClick={() => setShowUserLogin(true)} className="hover:text-green-400"><LogIn size={14} className="inline mr-1" /> Data Access Login</button>}
        {adminLoggedIn && <span className="text-green-400">Admin Mode</span>}
        {userLoggedIn && <span className="text-green-400">Logged in as {userLoggedIn.name} (expires: {new Date(userLoggedIn.expiry).toLocaleDateString()})</span>}
        {(adminLoggedIn || userLoggedIn) && <button onClick={() => { setAdminLoggedIn(false); setUserLoggedIn(null); }} className="text-red-400">Logout</button>}
      </div>

      {/* Login Modals */}
      {showAdminLogin && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-2xl w-96">
            <h2 className="text-xl mb-4">Admin Login</h2>
            <form onSubmit={handleAdminLogin}>
              <input type="email" placeholder="admin@simora.com" value={loginEmail} onChange={e=>setLoginEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <input type="password" placeholder="admin123" value={loginPassword} onChange={e=>setLoginPassword(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <button type="submit" className="bg-blue-600 w-full py-2 rounded">Login</button>
              {loginError && <p className="text-red-400 text-sm mt-2">{loginError}</p>}
            </form>
            <button onClick={()=>{setShowAdminLogin(false); setLoginError('');}} className="mt-2 text-gray-400">Cancel</button>
          </div>
        </div>
      )}
      {showUserLogin && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-2xl w-96">
            <h2 className="text-xl mb-4">Data Requester Login</h2>
            <form onSubmit={handleUserLogin}>
              <input type="email" placeholder="Email" value={loginEmail} onChange={e=>setLoginEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <input type="password" placeholder="Password" value={loginPassword} onChange={e=>setLoginPassword(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <button type="submit" className="bg-green-600 w-full py-2 rounded">Login</button>
              {loginError && <p className="text-red-400 text-sm mt-2">{loginError}</p>}
            </form>
            <button onClick={()=>{setShowUserLogin(false); setLoginError('');}} className="mt-2 text-gray-400">Cancel</button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur border-b border-gray-800 px-6 py-3 flex gap-4 overflow-x-auto">
        <button onClick={() => setActiveTab('dashboard')} className={`px-4 py-2 rounded-full ${activeTab === 'dashboard' ? 'bg-blue-600' : 'bg-gray-800'}`}>Dashboard</button>
        <button onClick={() => setActiveTab('subscribe')} className={`px-4 py-2 rounded-full ${activeTab === 'subscribe' ? 'bg-blue-600' : 'bg-gray-800'}`}>Subscribe</button>
        <button onClick={() => setActiveTab('contact')} className={`px-4 py-2 rounded-full ${activeTab === 'contact' ? 'bg-blue-600' : 'bg-gray-800'}`}>Contact</button>
        <button onClick={() => setActiveTab('dataRequest')} className={`px-4 py-2 rounded-full ${activeTab === 'dataRequest' ? 'bg-blue-600' : 'bg-gray-800'}`}>Data Request</button>
        <button onClick={() => setActiveTab('instSubscribe')} className={`px-4 py-2 rounded-full ${activeTab === 'instSubscribe' ? 'bg-blue-600' : 'bg-gray-800'}`}>Institution Sub</button>
        <button onClick={() => setActiveTab('gallery')} className={`px-4 py-2 rounded-full ${activeTab === 'gallery' ? 'bg-blue-600' : 'bg-gray-800'}`}>Mine Gallery</button>
        {adminLoggedIn && <button onClick={() => setActiveTab('adminPanel')} className="px-4 py-2 rounded-full bg-purple-600">Admin Panel</button>}
        {userLoggedIn && <button onClick={() => setActiveTab('myData')} className="px-4 py-2 rounded-full bg-green-600">My Data</button>}
      </div>

      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">MineGuard™ · Real-time Risk Intelligence</h2>
              <div className="flex gap-2"><div className="bg-gray-800/50 rounded-full px-3 py-1 text-sm"><Satellite size={14} className="inline mr-1"/> Sentinel-1</div><div className="bg-gray-800/50 rounded-full px-3 py-1 text-sm"><Wifi size={14} className="inline mr-1"/> IoT</div><div className="bg-purple-800/70 rounded-full px-3 py-1 text-sm"><Brain size={14} className="inline mr-1"/> TF.js</div></div>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="md:col-span-2 bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700">
                <label className="text-sm text-gray-400">Active Mine</label>
                <select value={selectedMine} onChange={e=>setSelectedMine(e.target.value)} className="w-full mt-1 bg-gray-900 border-gray-700 rounded-lg p-2"><option>Great Dyke - North Dam</option><option>Great Dyke - South Pit</option><option>HWange Colliery</option></select>
                <div className="mt-4 space-y-3">
                  <div><div className="flex justify-between"><label>InSAR (mm/day): {insarValue.toFixed(2)}</label><button onClick={fetchLiveInsar} disabled={fetchingEO} className="text-xs bg-blue-800 px-2 py-1 rounded"><RefreshCw size={12} className={fetchingEO ? 'animate-spin inline' : 'inline'} /> Refresh EO</button></div><input type="range" min="0.5" max="8" step="0.1" value={insarValue} onChange={e=>setInsarValue(parseFloat(e.target.value))} className="w-full" />{lastFetch && <div className="text-xs text-gray-400">Source: {eoSource} | Last fetch: {lastFetch}</div>}</div>
                  <div><label>IoT vibration (g): {vibrationValue.toFixed(2)}</label><input type="range" min="0.1" max="2.5" step="0.05" value={vibrationValue} onChange={e=>setVibrationValue(parseFloat(e.target.value))} className="w-full" /></div>
                  <div><label>Rainfall (mm/h): {rainfallValue.toFixed(2)}</label><input type="range" min="0" max="1" step="0.01" value={rainfallValue} onChange={e=>setRainfallValue(parseFloat(e.target.value))} className="w-full" /></div>
                </div>
                <div className="mt-6 flex justify-between items-center"><div><div className="text-5xl font-bold text-red-500">{riskPercent}%</div><div className="text-gray-400">collapse risk in 48h</div></div><AlertTriangle size={48} className="text-red-500 animate-pulse" /></div>
                <div className="mt-4 h-2 bg-gray-700 rounded-full"><div className="h-full bg-gradient-to-r from-yellow-500 to-red-500 rounded-full" style={{ width: `${riskPercent}%` }}></div></div>
                <div className="flex gap-3 mt-6"><button onClick={updatePrediction} className="flex-1 bg-blue-600 py-2 rounded flex items-center justify-center gap-2"><Activity size={18}/> Run ML Prediction</button><button onClick={handleRetrain} className="flex-1 bg-purple-700 py-2 rounded flex items-center justify-center gap-2"><Brain size={18}/> {training ? 'Retraining...' : 'Retrain with New Data'}</button></div>
              </div>
              <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-2xl p-6 border border-blue-500/30"><div className="text-blue-400"><TrendingUp size={18} className="inline mr-1"/> Live Traction</div><div className="text-3xl font-bold mt-2">{Math.floor(userCount)} <span className="text-sm font-normal">active users</span></div><div className="text-sm mt-1">+23% this quarter</div><div className="mt-4 text-xs">5 mines • 12 villages</div></div>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gray-800/40 rounded-2xl p-4"><h3 className="font-semibold"><Satellite size={16} className="inline mr-1"/> InSAR Time Series</h3><ResponsiveContainer width="100%" height={200}><LineChart data={Array.from({length:10},(_,i)=>({time:`${i*2}h`, mm: insarValue+(Math.random()-0.5)*0.5}))}><CartesianGrid stroke="#374151"/><XAxis dataKey="time" stroke="#9ca3af"/><YAxis stroke="#9ca3af"/><Tooltip/><Line type="monotone" dataKey="mm" stroke="#f97316" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></div>
              <div className="bg-gray-800/40 rounded-2xl p-4"><h3 className="font-semibold"><Activity size={16} className="inline mr-1"/> Predicted Risk Trend</h3><ResponsiveContainer width="100%" height={200}><LineChart data={riskHistory}><CartesianGrid stroke="#374151"/><XAxis dataKey="time" stroke="#9ca3af"/><YAxis domain={[0,100]} stroke="#9ca3af"/><Tooltip/><Line type="monotone" dataKey="risk" stroke="#ef4444" strokeWidth={2}/></LineChart></ResponsiveContainer></div>
            </div>
            <div className="bg-gray-800/30 rounded-2xl p-4 flex flex-wrap justify-between"><div><Cpu size={20} className="inline text-green-400"/> Simora‑Net (TF.js 3‑layer)</div><div className="text-xs text-green-300">Model active</div></div>
            <div className="bg-gray-800/40 rounded-2xl p-4 flex flex-wrap justify-between border border-purple-500/30"><div><span className="text-sm">Model accuracy:</span> <span className="text-xl font-bold text-purple-300">{modelAccuracy !== null ? modelAccuracy.toFixed(1) : '—'}%</span></div><div><span className="text-sm">Last loss:</span> <span className="text-xl font-bold text-orange-300">{modelLoss !== null ? modelLoss.toFixed(4) : '—'}</span></div><button onClick={refreshMetrics} className="bg-purple-800 px-3 py-1 rounded text-xs">↻ Evaluate</button></div>
            <div className="bg-gray-800/40 rounded-2xl p-4 border border-green-500/30"><div className="flex justify-between"><h3 className="font-semibold"><Brain size={16} className="inline text-green-400"/> Training Loss Curve</h3><select value={selectedRunId || ''} onChange={e=>handleSelectRun(e.target.value)} className="bg-gray-900 text-sm rounded p-1">{trainingRuns.map(r=><option key={r.id} value={r.id}>{r.timestamp.toLocaleTimeString()} - {r.description.slice(0,30)}</option>)}</select><button onClick={handleResetHistory} className="bg-red-800 px-2 py-1 rounded text-xs">Reset</button></div>{displayLossHistory.length?<ResponsiveContainer width="100%" height={220}><LineChart data={displayLossHistory.map((l,i)=>({epoch:i+1,loss:l}))}><CartesianGrid stroke="#374151"/><XAxis dataKey="epoch"/><YAxis/><Tooltip/><Line type="monotone" dataKey="loss" stroke="#22c55e" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer>:<div className="text-center py-8">No history. Click Retrain.</div>}</div>
          </motion.div>
        )}

        {/* Subscribe Tab */}
        {activeTab === 'subscribe' && (
          <motion.div className="bg-gray-800/40 rounded-2xl p-6 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold"><UserCheck className="inline mr-2"/> Subscription Plans</h2>
            <div className="grid md:grid-cols-2 gap-4 my-6"><div className="bg-gray-800/60 p-4 rounded-xl border border-blue-500/30"><h3 className="text-blue-400 font-bold">Free</h3><p>Weekly alerts, basic map</p><p className="text-lg font-bold mt-2">$0/mo</p></div><div className="bg-gray-800/60 p-4 rounded-xl border border-purple-500/30"><h3 className="text-purple-400 font-bold">Pro</h3><p>Daily analytics, API access</p><p className="text-lg font-bold mt-2">$29/mo</p></div></div>
            <form onSubmit={handleSubscribe} className="space-y-4"><input placeholder="Full name" value={subName} onChange={e=>setSubName(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><input type="email" placeholder="Email *" required value={subEmail} onChange={e=>setSubEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><select value={subTier} onChange={e=>setSubTier(e.target.value)} className="w-full bg-gray-900 p-2 rounded"><option value="free">Free</option><option value="pro">Pro ($29/mo)</option></select><button type="submit" className="bg-blue-600 px-6 py-2 rounded"><CreditCard className="inline mr-1"/> Subscribe</button>{subMessage && <p className="text-green-400 text-sm">{subMessage}</p>}</form>
          </motion.div>
        )}

        {/* Contact Tab */}
        {activeTab === 'contact' && (
          <motion.div className="bg-gray-800/40 rounded-2xl p-6 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold"><Mail className="inline mr-2"/> Contact Support</h2>
            <form onSubmit={handleContact} className="space-y-4 mt-4"><input placeholder="Your name" required value={contactName} onChange={e=>setContactName(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><input type="email" placeholder="Email" required value={contactEmail} onChange={e=>setContactEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><textarea placeholder="Message" rows={4} required value={contactMsg} onChange={e=>setContactMsg(e.target.value)} className="w-full bg-gray-900 p-2 rounded"></textarea><button type="submit" className="bg-purple-600 px-6 py-2 rounded"><Send className="inline mr-1"/> Send</button>{contactSuccess && <p className="text-green-400 text-sm">{contactSuccess}</p>}</form>
          </motion.div>
        )}

        {/* Data Request Tab (Peasant Miners) */}
        {activeTab === 'dataRequest' && (
          <motion.div className="bg-gray-800/40 rounded-2xl p-6 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold"><FileText className="inline mr-2"/> Peasant Miner Data Request</h2>
            <p className="text-gray-300 my-2">Request specific EO data for your small‑scale mine. Admin will review and send you login credentials.</p>
            <form onSubmit={handleDataRequest} className="space-y-4"><input placeholder="Your name / Cooperative" required value={requestName} onChange={e=>setRequestName(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><input placeholder="Mine name / location" required value={requestMine} onChange={e=>setRequestMine(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><input placeholder="Coordinates or area" required value={requestLocation} onChange={e=>setRequestLocation(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><textarea placeholder="What data do you need? (e.g., last 3 months InSAR)" rows={3} required value={requestData} onChange={e=>setRequestData(e.target.value)} className="w-full bg-gray-900 p-2 rounded"></textarea><button type="submit" className="bg-green-600 px-6 py-2 rounded"><FileText className="inline mr-1"/> Submit Request</button>{requestSuccess && <p className="text-green-400 text-sm">{requestSuccess}</p>}</form>
          </motion.div>
        )}

        {/* Institutional Subscription */}
        {activeTab === 'instSubscribe' && (
          <motion.div className="bg-gray-800/40 rounded-2xl p-6 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold">🏛️ Institutional Subscription</h2>
            <form onSubmit={handleInstSubscribe} className="space-y-4 mt-4"><input placeholder="Institution name" required value={instSubInst} onChange={e=>setInstSubInst(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><input placeholder="Contact person" required value={instSubName} onChange={e=>setInstSubName(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><input type="email" placeholder="Email" required value={instSubEmail} onChange={e=>setInstSubEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded" /><button type="submit" className="bg-blue-600 px-6 py-2 rounded">Request Subscription</button>{instSubMessage && <p className="text-green-400 text-sm">{instSubMessage}</p>}</form>
          </motion.div>
        )}

        {/* Mine Gallery */}
        {activeTab === 'gallery' && (
          <motion.div className="space-y-4">
            <h2 className="text-2xl font-bold"><Image className="inline mr-2"/> Mine Gallery – Real‑time & Historical</h2>
            {adminLoggedIn && (
              <div className="bg-gray-800 p-4 rounded-2xl"><h3>Admin: Upload or Simulate Live Feed</h3><input placeholder="Image URL" value={newImageUrl} onChange={e=>setNewImageUrl(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" /><input placeholder="Caption" value={newImageCaption} onChange={e=>setNewImageCaption(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" /><button onClick={handleUploadImage} className="bg-green-600 px-4 py-1 rounded mr-2">Upload</button><button onClick={simulateRealTimeImage} className="bg-yellow-600 px-4 py-1 rounded">Simulate Live Feed</button></div>
            )}
            <div className="grid md:grid-cols-3 gap-4">{galleryImages.map(img=><div key={img.id} className="bg-gray-800 rounded-xl overflow-hidden"><img src={img.url} className="w-full h-48 object-cover" /><div className="p-2"><p className="text-sm">{img.caption}</p><p className="text-xs text-gray-400">{new Date(img.timestamp).toLocaleString()}</p></div></div>)}</div>
          </motion.div>
        )}

        {/* Admin Panel (only when admin logged in) */}
        {adminLoggedIn && activeTab === 'adminPanel' && (
          <motion.div className="space-y-8">
            <h2 className="text-2xl font-bold">Admin Dashboard</h2>
            <div className="bg-gray-800 p-4 rounded-2xl"><h3 className="text-xl">Pending Data Requests</h3>{dataRequests.filter(r=>!r.approved).length===0?<p className="text-gray-400">None.</p>:dataRequests.filter(r=>!r.approved).map(req=><div key={req.id} className="border-b border-gray-700 py-2"><p><strong>{req.name}</strong> – {req.mine}, {req.location}</p><p className="text-sm">{req.dataNeeded}</p><div className="flex gap-2 mt-2"><button onClick={()=>approveDataRequest(req,7)} className="bg-green-600 px-2 py-1 text-sm">Approve (7 days)</button><button onClick={()=>approveDataRequest(req,30)} className="bg-blue-600 px-2 py-1 text-sm">Approve (30 days)</button></div></div>)}</div>
            <div className="bg-gray-800 p-4 rounded-2xl"><h3 className="text-xl">Users with Access</h3><table className="w-full text-sm"><thead><tr><th>Name</th><th>Email</th><th>Mine</th><th>Expiry</th><th>Status</th></tr></thead><tbody>{users.map(u=><tr key={u.id}><td>{u.name}</td><td>{u.email}</td><td>{u.mine}</td><td>{new Date(u.expiry).toLocaleDateString()}</td><td>{isExpired(u.expiry)?<XCircle size={14} className="text-red-400 inline"/>:<CheckCircle size={14} className="text-green-400 inline"/>}</td></tr>)}</tbody></table><button onClick={()=>clearAdminData('users')} className="mt-2 text-red-400 text-sm">Clear all users</button></div>
            <div className="bg-gray-800 p-4 rounded-2xl"><h3 className="text-xl">Institutional Subscriptions</h3><table className="w-full text-sm"><thead><tr><th>Institution</th><th>Contact</th><th>Email</th><th>Date</th></tr></thead><tbody>{instSubs.map(s=><tr key={s.id}><td>{s.institution}</td><td>{s.contactName}</td><td>{s.email}</td><td>{new Date(s.date).toLocaleDateString()}</td></tr>)}</tbody></table><button onClick={()=>clearAdminData('instSubs')} className="mt-2 text-red-400 text-sm">Clear all</button></div>
            <div className="bg-gray-800 p-4 rounded-2xl"><h3 className="text-xl">Other Data</h3><div className="flex gap-4 flex-wrap"><button onClick={()=>clearAdminData('subs')} className="bg-red-800 px-3 py-1 rounded text-sm">Clear Subscriptions</button><button onClick={()=>clearAdminData('requests')} className="bg-red-800 px-3 py-1 rounded text-sm">Clear Data Requests</button><button onClick={()=>clearAdminData('contacts')} className="bg-red-800 px-3 py-1 rounded text-sm">Clear Contacts</button><button onClick={()=>clearAdminData('gallery')} className="bg-red-800 px-3 py-1 rounded text-sm">Clear Gallery</button></div></div>
          </motion.div>
        )}

        {/* My Data (for logged-in requesters) */}
        {userLoggedIn && activeTab === 'myData' && (
          <motion.div className="bg-gray-800/40 rounded-2xl p-6">
            <h2 className="text-2xl font-bold">Your Data Access</h2>
            <p><strong>Name:</strong> {userLoggedIn.name}</p>
            <p><strong>Assigned Mine:</strong> {userLoggedIn.mine}</p>
            <p><strong>Access expires:</strong> {new Date(userLoggedIn.expiry).toLocaleString()}</p>
            <div className="mt-6"><h3 className="text-xl">Current Risk Report</h3><p className="text-gray-300">Collapse risk: <span className="text-red-400 font-bold">{riskPercent}%</span> (based on latest EO data)</p><p className="text-sm">InSAR: {insarValue} mm/day | Vibration: {vibrationValue} g | Rainfall: {rainfallValue} mm/h</p></div>
            <div className="mt-6"><h3 className="text-xl">Gallery for your mine</h3><div className="grid md:grid-cols-2 gap-4">{galleryImages.filter(img=>img.mine===userLoggedIn.mine).slice(0,4).map(img=><div key={img.id} className="bg-gray-800 rounded"><img src={img.url} className="h-40 w-full object-cover" /><p className="p-1 text-xs">{img.caption}</p></div>)}</div></div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;