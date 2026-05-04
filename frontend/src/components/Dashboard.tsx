import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { 
  Activity, Cpu, Satellite, Wifi, TrendingUp, AlertTriangle, Brain, RefreshCw,
  Mail, FileText, Send, CreditCard, UserCheck, Lock, LogIn, Image
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

// ---------- TensorFlow.js Model (same as before) ----------
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
      [0.5,0.2,0.1],[1.2,0.4,0.2],[2.0,0.7,0.3],[3.5,1.2,0.5],[5.0,1.8,0.8],
      [0.8,0.3,0.15],[1.5,0.5,0.25],[2.5,0.9,0.4],[4.0,1.4,0.6],[6.0,2.0,0.9]
    ];
    const outputs = inputs.map(inp => [ (inp[0]/6)*0.5 + (inp[1]/2)*0.3 + inp[2]*0.2 ]);
    return { xs: tf.tensor2d(inputs), ys: tf.tensor2d(outputs) };
  }

  private async initialTrain() {
    const { xs, ys } = this.generateSyntheticData();
    const history = await this.model!.fit(xs, ys, { epochs: 150, verbose: 0, validationSplit: 0.2 });
    const lossHistory = history.history.loss as number[];
    const finalLoss = lossHistory[lossHistory.length-1];
    await this.evaluateAccuracy();
    this.addTrainingRun(lossHistory, finalLoss, this.currentAccuracy, 'Initial training');
    xs.dispose(); ys.dispose();
  }

  private async evaluateAccuracy() {
    const { xs, ys } = this.generateSyntheticData();
    const pred = this.model!.predict(xs) as tf.Tensor;
    const predVals = await pred.data();
    const trueVals = await ys.data();
    let correct = 0;
    for(let i=0;i<predVals.length;i++) if(Math.abs(predVals[i]-trueVals[i])<=0.1) correct++;
    this.currentAccuracy = (correct/predVals.length)*100;
    xs.dispose(); ys.dispose(); pred.dispose();
  }

  private addTrainingRun(lossHistory: number[], finalLoss: number, acc: number, desc: string) {
    const run: TrainingRun = {
      id: Date.now().toString(),
      timestamp: new Date(),
      lossHistory,
      finalLoss,
      accuracy: acc,
      description: desc
    };
    this.trainingRuns.push(run);
    this.currentRunId = run.id;
  }

  async predict(insar: number, vib: number, rain: number): Promise<number> {
    if(!this.model) return 50;
    const input = tf.tensor2d([[insar, vib, rain]]);
    const out = this.model.predict(input) as tf.Tensor;
    const risk = (await out.data())[0]*100;
    input.dispose(); out.dispose();
    return Math.round(risk);
  }

  async retrain(newData: {insar:number, vibration:number, rainfall:number, risk:number}[]) {
    if(!this.model || this.isTraining) return;
    this.isTraining = true;
    try {
      const inputs = newData.map(d=>[d.insar, d.vibration, d.rainfall]);
      const outputs = newData.map(d=>[d.risk/100]);
      const { xs: synthXs, ys: synthYs } = this.generateSyntheticData();
      const allXs = tf.concat([synthXs, tf.tensor2d(inputs)],0);
      const allYs = tf.concat([synthYs, tf.tensor2d(outputs)],0);
      const history = await this.model.fit(allXs, allYs, { epochs: 50, verbose: 0, validationSplit: 0.2 });
      const lossHistory = history.history.loss as number[];
      const finalLoss = lossHistory[lossHistory.length-1];
      await this.evaluateAccuracy();
      this.addTrainingRun(lossHistory, finalLoss, this.currentAccuracy, `Retrain (${newData.length} records)`);
      allXs.dispose(); allYs.dispose(); synthXs.dispose(); synthYs.dispose();
    } catch(e) { console.error(e); } finally { this.isTraining = false; }
  }

  getCurrentAccuracy() { return this.currentAccuracy; }
  getTrainingRuns() { return this.trainingRuns; }
  getCurrentRunId() { return this.currentRunId; }
  resetHistory() { this.trainingRuns = []; this.currentRunId = null; this.initialTrain(); }
}

const mlModel = new RealMLModel();

// ---------- Storage Helpers ----------
const STORAGE_KEYS = {
  subscriptions: 'simora_subs',
  dataRequests: 'simora_reqs',
  contacts: 'simora_contacts',
  users: 'simora_users',
  gallery: 'simora_gallery',
  instSubs: 'simora_inst'
};
const save = (k:string, v:any) => localStorage.setItem(k, JSON.stringify(v));
const load = (k:string) => { const r = localStorage.getItem(k); return r ? JSON.parse(r) : []; };
const isExpired = (exp:string) => new Date(exp) < new Date();

// ---------- Main Dashboard ----------
const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [adminLoggedIn, setAdminLoggedIn] = useState(false);
  const [userLoggedIn, setUserLoggedIn] = useState<{name:string, email:string, expiry:string, mine:string} | null>(null);
  const [showAdminLogin, setShowAdminLogin] = useState(false);
  const [showUserLogin, setShowUserLogin] = useState(false);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPass, setLoginPass] = useState('');
  const [loginError, setLoginError] = useState('');

  // ML state
  const [modelLoading, setModelLoading] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  const [riskPercent, setRiskPercent] = useState(50);
  const [insarValue, setInsarValue] = useState(2.5);
  const [vibrationValue, setVibrationValue] = useState(1.1);
  const [rainfallValue, setRainfallValue] = useState(0.4);
  const [riskHistory, setRiskHistory] = useState([{time:'00:00',risk:45},{time:'06:00',risk:48}]);
  const [userCount, setUserCount] = useState(147);
  const [modelAccuracy, setModelAccuracy] = useState<number|null>(null);
  const [modelLoss, setModelLoss] = useState<number|null>(null);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string|null>(null);
  const [displayLossHistory, setDisplayLossHistory] = useState<number[]>([]);
  const [training, setTraining] = useState(false);
  const [fetchingEO, setFetchingEO] = useState(false);
  const [eoSource, setEoSource] = useState('Sentinel-1 (sim)');
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

  // Admin data
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

  // Load saved data
  useEffect(() => {
    setSubscriptions(load(STORAGE_KEYS.subscriptions));
    setDataRequests(load(STORAGE_KEYS.dataRequests));
    setContacts(load(STORAGE_KEYS.contacts));
    setUsers(load(STORAGE_KEYS.users));
    setGalleryImages(load(STORAGE_KEYS.gallery));
    setInstSubs(load(STORAGE_KEYS.instSubs));
  }, []);

  // ML functions
  const refreshMetrics = () => {
    setModelAccuracy(mlModel.getCurrentAccuracy());
    const runs = mlModel.getTrainingRuns();
    setTrainingRuns(runs);
    const curId = mlModel.getCurrentRunId();
    setSelectedRunId(curId);
    const run = runs.find(r=>r.id===curId);
    setDisplayLossHistory(run?.lossHistory || []);
    if(run) setModelLoss(run.finalLoss);
  };
  const updatePrediction = async () => {
    if(!modelReady) return;
    const newRisk = await mlModel.predict(insarValue, vibrationValue, rainfallValue);
    setRiskPercent(newRisk);
    setRiskHistory(prev=>[...prev.slice(-5), {time: new Date().toLocaleTimeString(), risk: newRisk}]);
  };
  useEffect(() => {
    mlModel.init().then(()=>{
      setModelReady(true);
      setModelLoading(false);
      refreshMetrics();
      updatePrediction();
    }).catch(()=>console.error);
  }, []);
  useEffect(() => { if(modelReady) updatePrediction(); }, [insarValue, vibrationValue, rainfallValue, modelReady]);
  useEffect(() => {
    const interval = setInterval(()=>setUserCount(p=>Math.min(200,p+0.1)), 60000);
    return ()=>clearInterval(interval);
  }, []);
  const handleRetrain = async () => {
    if(!modelReady || training) return;
    setTraining(true);
    await mlModel.retrain([{insar:insarValue, vibration:vibrationValue, rainfall:rainfallValue, risk:riskPercent}]);
    await updatePrediction();
    refreshMetrics();
    setTraining(false);
    alert(`Retrained! Accuracy: ${mlModel.getCurrentAccuracy().toFixed(1)}%`);
  };
  const handleSelectRun = (id:string) => {
    const run = trainingRuns.find(r=>r.id===id);
    if(run) {
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
    setTimeout(()=>{
      setInsarValue(parseFloat((Math.random()*5+1).toFixed(2)));
      setEoSource('Sentinel-1 live sim');
      setLastFetch(new Date().toLocaleTimeString());
      setFetchingEO(false);
    }, 800);
  };

  // Form handlers
  const handleSubscribe = (e:React.FormEvent) => {
    e.preventDefault();
    if(!subEmail) return;
    const newSub = { id:Date.now(), name:subName, email:subEmail, tier:subTier, date:new Date().toISOString() };
    const updated = [...subscriptions, newSub];
    setSubscriptions(updated);
    save(STORAGE_KEYS.subscriptions, updated);
    setSubMessage(`Subscribed (${subTier})!`);
    setTimeout(()=>setSubMessage(''), 3000);
    setSubEmail(''); setSubName('');
  };
  const handleContact = (e:React.FormEvent) => {
    e.preventDefault();
    const newContact = { id:Date.now(), name:contactName, email:contactEmail, message:contactMsg, date:new Date().toISOString() };
    const updated = [...contacts, newContact];
    setContacts(updated);
    save(STORAGE_KEYS.contacts, updated);
    setContactSuccess('Message sent!');
    setTimeout(()=>setContactSuccess(''), 3000);
    setContactName(''); setContactEmail(''); setContactMsg('');
  };
  const handleDataRequest = (e:React.FormEvent) => {
    e.preventDefault();
    const newReq = { id:Date.now(), name:requestName, mine:requestMine, location:requestLocation, dataNeeded:requestData, date:new Date().toISOString(), approved:false };
    const updated = [...dataRequests, newReq];
    setDataRequests(updated);
    save(STORAGE_KEYS.dataRequests, updated);
    setRequestSuccess('Request submitted. Admin will review.');
    setTimeout(()=>setRequestSuccess(''), 3000);
    setRequestName(''); setRequestMine(''); setRequestLocation(''); setRequestData('');
  };
  const handleInstSubscribe = (e:React.FormEvent) => {
    e.preventDefault();
    const newSub = { id:Date.now(), institution:instSubInst, contactName:instSubName, email:instSubEmail, date:new Date().toISOString() };
    const updated = [...instSubs, newSub];
    setInstSubs(updated);
    save(STORAGE_KEYS.instSubs, updated);
    setInstSubMessage('Request sent. We will contact you.');
    setTimeout(()=>setInstSubMessage(''), 3000);
    setInstSubName(''); setInstSubEmail(''); setInstSubInst('');
  };

  // Admin & auth handlers
  const handleAdminLogin = (e:React.FormEvent) => {
    e.preventDefault();
    if(loginEmail==='admin@simora.com' && loginPass==='admin123') {
      setAdminLoggedIn(true);
      setShowAdminLogin(false);
      setLoginError('');
    } else setLoginError('Invalid');
  };
  const handleUserLogin = (e:React.FormEvent) => {
    e.preventDefault();
    const user = users.find(u=>u.email===loginEmail && u.password===loginPass);
    if(user && !isExpired(user.expiry)) {
      setUserLoggedIn({name:user.name, email:user.email, expiry:user.expiry, mine:user.mine});
      setShowUserLogin(false);
      setLoginError('');
    } else if(user && isExpired(user.expiry)) setLoginError('Access expired');
    else setLoginError('Invalid credentials');
  };
  const approveDataRequest = (req:any, days:number) => {
    const expiry = new Date();
    expiry.setDate(expiry.getDate()+days);
    const newUser = {
      id:Date.now(),
      name:req.name,
      email:`user_${Date.now()}@simora.com`,
      password:Math.random().toString(36).substring(2,8),
      mine:req.mine,
      location:req.location,
      expiry:expiry.toISOString(),
      createdAt:new Date().toISOString()
    };
    const updated = [...users, newUser];
    setUsers(updated);
    save(STORAGE_KEYS.users, updated);
    const updatedReqs = dataRequests.map(r=>r.id===req.id ? {...r, approved:true} : r);
    setDataRequests(updatedReqs);
    save(STORAGE_KEYS.dataRequests, updatedReqs);
    alert(`User created!\nEmail: ${newUser.email}\nPassword: ${newUser.password}\nExpires: ${expiry.toLocaleString()}`);
  };
  const handleUploadImage = () => {
    if(!newImageUrl) return;
    const newImg = {
      id:Date.now(),
      url:newImageUrl,
      caption:newImageCaption||'Mine area',
      timestamp:new Date().toISOString(),
      mine:'Great Dyke'
    };
    const updated = [newImg, ...galleryImages];
    setGalleryImages(updated);
    save(STORAGE_KEYS.gallery, updated);
    setNewImageUrl('');
    setNewImageCaption('');
  };
  const simulateRealTimeImage = () => {
    const fakeUrl = `https://picsum.photos/id/${Math.floor(Math.random()*100)}/400/300`;
    const newImg = {
      id:Date.now(),
      url:fakeUrl,
      caption:`Live ${new Date().toLocaleTimeString()}`,
      timestamp:new Date().toISOString(),
      mine:'Great Dyke'
    };
    const updated = [newImg, ...galleryImages];
    setGalleryImages(updated);
    save(STORAGE_KEYS.gallery, updated);
  };
  const clearAdminData = (type:string) => {
    if(type==='subs'){ setSubscriptions([]); save(STORAGE_KEYS.subscriptions,[]); }
    if(type==='reqs'){ setDataRequests([]); save(STORAGE_KEYS.dataRequests,[]); }
    if(type==='contacts'){ setContacts([]); save(STORAGE_KEYS.contacts,[]); }
    if(type==='users'){ setUsers([]); save(STORAGE_KEYS.users,[]); }
    if(type==='gallery'){ setGalleryImages([]); save(STORAGE_KEYS.gallery,[]); }
    if(type==='inst'){ setInstSubs([]); save(STORAGE_KEYS.instSubs,[]); }
  };

  if(modelLoading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <Brain className="animate-pulse" size={48} />
        <span className="ml-2">Loading Simora AI (TensorFlow.js)…</span>
      </div>
    );
  }

  // ---------- Render ----------
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Top bar */}
      <div className="bg-gray-800/80 px-6 py-2 flex justify-end gap-3 text-sm">
        {!adminLoggedIn && <button onClick={()=>setShowAdminLogin(true)} className="hover:text-blue-400"><Lock size={14} className="inline mr-1" /> Admin</button>}
        {!userLoggedIn && <button onClick={()=>setShowUserLogin(true)} className="hover:text-green-400"><LogIn size={14} className="inline mr-1" /> Data Access</button>}
        {adminLoggedIn && <span className="text-green-400">Admin Mode</span>}
        {userLoggedIn && <span className="text-green-400">Logged as {userLoggedIn.name} (expires {new Date(userLoggedIn.expiry).toLocaleDateString()})</span>}
        {(adminLoggedIn||userLoggedIn) && <button onClick={()=>{ setAdminLoggedIn(false); setUserLoggedIn(null); }} className="text-red-400">Logout</button>}
      </div>

      {/* Login Modals */}
      {showAdminLogin && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-2xl w-96">
            <h2 className="text-xl mb-4">Admin Login</h2>
            <form onSubmit={handleAdminLogin}>
              <input type="email" placeholder="admin@simora.com" value={loginEmail} onChange={e=>setLoginEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <input type="password" placeholder="admin123" value={loginPass} onChange={e=>setLoginPass(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <button type="submit" className="bg-blue-600 w-full py-2 rounded">Login</button>
              {loginError && <p className="text-red-400 text-sm mt-2">{loginError}</p>}
            </form>
            <button onClick={()=>{ setShowAdminLogin(false); setLoginError(''); }} className="mt-2 text-gray-400">Cancel</button>
          </div>
        </div>
      )}
      {showUserLogin && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-2xl w-96">
            <h2 className="text-xl mb-4">Data Requester Login</h2>
            <form onSubmit={handleUserLogin}>
              <input type="email" placeholder="Email" value={loginEmail} onChange={e=>setLoginEmail(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <input type="password" placeholder="Password" value={loginPass} onChange={e=>setLoginPass(e.target.value)} className="w-full bg-gray-900 p-2 rounded mb-2" />
              <button type="submit" className="bg-green-600 w-full py-2 rounded">Login</button>
              {loginError && <p className="text-red-400 text-sm mt-2">{loginError}</p>}
            </form>
            <button onClick={()=>{ setShowUserLogin(false); setLoginError(''); }} className="mt-2 text-gray-400">Cancel</button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur border-b border-gray-800 px-6 py-3 flex gap-4 overflow-x-auto">
        <button onClick={()=>setActiveTab('dashboard')} className={`px-4 py-2 rounded-full transition ${activeTab==='dashboard' ? 'bg-blue-600' : 'bg-gray-800'}`}>Dashboard</button>
        <button onClick={()=>setActiveTab('subscribe')} className={`px-4 py-2 rounded-full transition ${activeTab==='subscribe' ? 'bg-blue-600' : 'bg-gray-800'}`}>Subscribe</button>
        <button onClick={()=>setActiveTab('contact')} className={`px-4 py-2 rounded-full transition ${activeTab==='contact' ? 'bg-blue-600' : 'bg-gray-800'}`}>Contact</button>
        <button onClick={()=>setActiveTab('dataRequest')} className={`px-4 py-2 rounded-full transition ${activeTab==='dataRequest' ? 'bg-blue-600' : 'bg-gray-800'}`}>Data Request</button>
        <button onClick={()=>setActiveTab('instSubscribe')} className={`px-4 py-2 rounded-full transition ${activeTab==='instSubscribe' ? 'bg-blue-600' : 'bg-gray-800'}`}>Institution Sub</button>
        <button onClick={()=>setActiveTab('gallery')} className={`px-4 py-2 rounded-full transition ${activeTab==='gallery' ? 'bg-blue-600' : 'bg-gray-800'}`}>Mine Gallery</button>
        {adminLoggedIn && <button onClick={()=>setActiveTab('adminPanel')} className="px-4 py-2 rounded-full bg-purple-600">Admin Panel</button>}
        {userLoggedIn && <button onClick={()=>setActiveTab('myData')} className="px-4 py-2 rounded-full bg-green-600">My Data</button>}
      </div>

      {/* Main content */}
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <motion.div initial={{opacity:0}} animate={{opacity:1}} className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">MineGuard™ · Real‑time Risk Intelligence</h2>
              <div className="flex gap-2">
                <div className="bg-gray-800/50 rounded-full px-3 py-1 text-sm"><Satellite size={14} className="inline mr-1"/> Sentinel-1</div>
                <div className="bg-gray-800/50 rounded-full px-3 py-1 text-sm"><Wifi size={14} className="inline mr-1"/> IoT</div>
                <div className="bg-purple-800/70 rounded-full px-3 py-1 text-sm"><Brain size={14} className="inline mr-1"/> TensorFlow.js</div>
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="md:col-span-2 bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700">
                <label className="text-sm text-gray-400">Active Mine</label>
                <select className="w-full mt-1 bg-gray-900 border-gray-700 rounded-lg p-2">
                  <option>Great Dyke - North Dam</option>
                  <option>Great Dyke - South Pit</option>
                </select>
                <div className="mt-4 space-y-3">
                  <div>
                    <div className="flex justify-between items-center">
                      <label className="text-sm">InSAR deformation (mm/day): {insarValue.toFixed(2)}</label>
                      <button onClick={fetchLiveInsar} disabled={fetchingEO} className="text-xs bg-blue-800 hover:bg-blue-700 px-2 py-1 rounded flex items-center gap-1">
                        <RefreshCw size={12} className={fetchingEO ? 'animate-spin' : ''} /> Refresh EO
                      </button>
                    </div>
                    <input type="range" min="0.5" max="8" step="0.1" value={insarValue} onChange={e=>setInsarValue(parseFloat(e.target.value))} className="w-full mt-1" />
                    {lastFetch && <div className="text-xs text-gray-400 mt-1">Source: {eoSource} | Last fetch: {lastFetch}</div>}
                  </div>
                  <div>
                    <label className="text-sm">IoT vibration (g): {vibrationValue.toFixed(2)}</label>
                    <input type="range" min="0.1" max="2.5" step="0.05" value={vibrationValue} onChange={e=>setVibrationValue(parseFloat(e.target.value))} className="w-full" />
                  </div>
                  <div>
                    <label className="text-sm">Rainfall (mm/h): {rainfallValue.toFixed(2)}</label>
                    <input type="range" min="0" max="1" step="0.01" value={rainfallValue} onChange={e=>setRainfallValue(parseFloat(e.target.value))} className="w-full" />
                  </div>
                </div>
                <div className="mt-6 flex items-center justify-between">
                  <div>
                    <div className="text-5xl font-bold text-red-500">{riskPercent}%</div>
                    <div className="text-gray-400">collapse risk in 48h</div>
                  </div>
                  <AlertTriangle size={48} className="text-red-500 animate-pulse" />
                </div>
                <div className="mt-4 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-yellow-500 to-red-500 rounded-full" style={{ width: `${riskPercent}%` }}></div>
                </div>
                <div className="flex gap-3 mt-6">
                  <button onClick={updatePrediction} className="flex-1 bg-blue-600 hover:bg-blue-500 transition rounded-lg py-2 flex items-center justify-center gap-2"><Activity size={18} /> Run ML Prediction</button>
                  <button onClick={handleRetrain} className="flex-1 bg-purple-700 hover:bg-purple-600 transition rounded-lg py-2 flex items-center justify-center gap-2"><Brain size={18} /> {training ? 'Retraining...' : 'Retrain with New Data'}</button>
                </div>
              </div>
              <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-2xl p-6 border border-blue-500/30 backdrop-blur">
                <div className="flex items-center gap-2 text-blue-400"><TrendingUp size={18} /> Live Traction</div>
                <div className="text-3xl font-bold mt-2">{Math.floor(userCount)} <span className="text-sm font-normal text-gray-400">active users</span></div>
                <div className="text-sm text-gray-300 mt-1">+23% this quarter</div>
                <div className="mt-4 text-xs text-gray-400">Pilots: Great Dyke, Hwange Colliery • 5 mines • 12 villages</div>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gray-800/40 rounded-2xl p-4 border border-gray-700">
                <h3 className="font-semibold flex items-center gap-2"><Satellite size={16} /> InSAR Time Series (simulated)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={Array.from({length:10}, (_,i)=>({time:`${i*2}h`, mm: insarValue + (Math.random()-0.5)*0.5}))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="time" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} />
                    <Line type="monotone" dataKey="mm" stroke="#f97316" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="bg-gray-800/40 rounded-2xl p-4 border border-gray-700">
                <h3 className="font-semibold flex items-center gap-2"><Activity size={16} /> Predicted Risk Trend</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={riskHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="time" stroke="#9ca3af" />
                    <YAxis domain={[0,100]} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} />
                    <Line type="monotone" dataKey="risk" stroke="#ef4444" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-gray-800/30 rounded-2xl p-4 border border-gray-700 flex flex-wrap justify-between items-center gap-3">
              <div className="flex items-center gap-3"><Cpu size={20} className="text-green-400" /><span className="font-mono text-sm">Simora‑Net (TensorFlow.js 3‑layer NN)</span></div>
              <div className="text-xs text-gray-400"><span className="text-green-300">Model active</span></div>
            </div>
            <div className="bg-gray-800/40 rounded-2xl p-4 border border-purple-500/30 flex flex-wrap justify-between items-center gap-3">
              <div><span className="text-sm text-gray-400">Model accuracy (test set): </span><span className="text-xl font-bold text-purple-300">{modelAccuracy !== null ? modelAccuracy.toFixed(1) : '—'}%</span></div>
              <div><span className="text-sm text-gray-400">Last training loss: </span><span className="text-xl font-bold text-orange-300">{modelLoss !== null ? modelLoss.toFixed(4) : '—'}</span></div>
              <button onClick={refreshMetrics} className="bg-purple-800 hover:bg-purple-700 px-3 py-1 rounded text-xs transition">↻ Evaluate</button>
            </div>
            <div className="bg-gray-800/40 rounded-2xl p-4 border border-green-500/30">
              <div className="flex justify-between items-center flex-wrap gap-3 mb-3">
                <h3 className="font-semibold flex items-center gap-2"><Brain size={16} className="text-green-400" /> Training Loss Curve (historical)</h3>
                <div className="flex gap-2">
                  <select value={selectedRunId || ''} onChange={e=>handleSelectRun(e.target.value)} className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm">
                    {trainingRuns.map(run => (
                      <option key={run.id} value={run.id}>{run.timestamp.toLocaleTimeString()} - {run.description.slice(0,40)}</option>
                    ))}
                  </select>
                  <button onClick={handleResetHistory} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-xs transition">Reset History</button>
                </div>
              </div>
              {displayLossHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={displayLossHistory.map((loss,idx)=>({epoch:idx+1,loss}))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} />
                    <Line type="monotone" dataKey="loss" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={true} animationDuration={1000} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center text-gray-400 py-8">No training history yet. Click "Retrain" to generate loss curve.</div>
              )}
            </div>
          </motion.div>
        )}

        {/* Subscribe Tab */}
        {activeTab === 'subscribe' && (
          <div className="bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2"><UserCheck size={24} /> Subscription Plans</h2>
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <div className="bg-gray-800/60 p-4 rounded-xl border border-blue-500/30"><h3 className="font-bold text-blue-400">Free Tier</h3><p className="text-sm">Weekly risk alerts, basic map</p><p className="text-lg font-bold mt-2">$0/month</p></div>
              <div className="bg-gray-800/60 p-4 rounded-xl border border-purple-500/30"><h3 className="font-bold text-purple-400">Pro Tier</h3><p className="text-sm">Daily analytics, API access</p><p className="text-lg font-bold mt-2">$29/month</p></div>
            </div>
            <form onSubmit={handleSubscribe} className="space-y-4">
              <input type="text" placeholder="Full Name (optional)" value={subName} onChange={e=>setSubName(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2" />
              <input type="email" placeholder="Email *" required value={subEmail} onChange={e=>setSubEmail(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2" />
              <select value={subTier} onChange={e=>setSubTier(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2">
                <option value="free">Free</option>
                <option value="pro">Pro ($29/mo)</option>
              </select>
              <button type="submit" className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg transition"><CreditCard size={18} className="inline mr-1" /> Subscribe</button>
              {subMessage && <p className="text-green-400 text-sm mt-2">{subMessage}</p>}
            </form>
          </div>
        )}

        {/* Contact Tab */}
        {activeTab === 'contact' && (
          <div className="bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2"><Mail size={24} /> Contact Support</h2>
            <form onSubmit={handleContact} className="space-y-4">
              <input type="text" placeholder="Your Name" required value={contactName} onChange={e=>setContactName(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <input type="email" placeholder="Email" required value={contactEmail} onChange={e=>setContactEmail(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <textarea placeholder="Message" rows={4} required value={contactMsg} onChange={e=>setContactMsg(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2"></textarea>
              <button type="submit" className="bg-purple-600 hover:bg-purple-500 px-6 py-2 rounded-lg transition"><Send size={18} className="inline mr-1" /> Send Message</button>
              {contactSuccess && <p className="text-green-400 text-sm mt-2">{contactSuccess}</p>}
            </form>
          </div>
        )}

        {/* Data Request Tab (Peasant Miners) */}
        {activeTab === 'dataRequest' && (
          <div className="bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2"><FileText size={24} /> Peasant Miner Data Request</h2>
            <p className="text-gray-300 mb-4">Request specific EO data for your small‑scale mine. Admin will review and send you login credentials.</p>
            <form onSubmit={handleDataRequest} className="space-y-4">
              <input type="text" placeholder="Your Name / Cooperative" required value={requestName} onChange={e=>setRequestName(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <input type="text" placeholder="Mine Name / Location" required value={requestMine} onChange={e=>setRequestMine(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <input type="text" placeholder="Coordinates or Area" required value={requestLocation} onChange={e=>setRequestLocation(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <textarea placeholder="What data do you need? (e.g., last 3 months InSAR deformation)" rows={3} required value={requestData} onChange={e=>setRequestData(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2"></textarea>
              <button type="submit" className="bg-green-600 hover:bg-green-500 px-6 py-2 rounded-lg transition"><FileText size={18} className="inline mr-1" /> Submit Request</button>
              {requestSuccess && <p className="text-green-400 text-sm mt-2">{requestSuccess}</p>}
            </form>
          </div>
        )}

        {/* Institutional Subscription Tab */}
        {activeTab === 'instSubscribe' && (
          <div className="bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">🏛️ Institutional Subscription</h2>
            <form onSubmit={handleInstSubscribe} className="space-y-4">
              <input type="text" placeholder="Institution Name" required value={instSubInst} onChange={e=>setInstSubInst(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <input type="text" placeholder="Contact Person" required value={instSubName} onChange={e=>setInstSubName(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <input type="email" placeholder="Email" required value={instSubEmail} onChange={e=>setInstSubEmail(e.target.value)} className="w-full bg-gray-900 border-gray-700 rounded-lg p-2" />
              <button type="submit" className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg transition">Request Subscription</button>
              {instSubMessage && <p className="text-green-400 text-sm mt-2">{instSubMessage}</p>}
            </form>
          </div>
        )}

        {/* Mine Gallery Tab */}
        {activeTab === 'gallery' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold flex items-center gap-2"><Image size={24} /> Mine Gallery – Real‑time & Historical</h2>
              {adminLoggedIn && (
                <div className="flex gap-2">
                  <input type="text" placeholder="Image URL" value={newImageUrl} onChange={e=>setNewImageUrl(e.target.value)} className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm" />
                  <input type="text" placeholder="Caption" value={newImageCaption} onChange={e=>setNewImageCaption(e.target.value)} className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm" />
                  <button onClick={handleUploadImage} className="bg-green-600 hover:bg-green-500 px-3 py-1 rounded text-sm">Upload</button>
                  <button onClick={simulateRealTimeImage} className="bg-yellow-600 hover:bg-yellow-500 px-3 py-1 rounded text-sm">Simulate Live Feed</button>
                </div>
              )}
            </div>
            <div className="grid md:grid-cols-3 gap-4">
              {galleryImages.map(img => (
                <div key={img.id} className="bg-gray-800 rounded-xl overflow-hidden">
                  <img src={img.url} className="w-full h-48 object-cover" alt={img.caption} />
                  <div className="p-2">
                    <p className="text-sm">{img.caption}</p>
                    <p className="text-xs text-gray-400">{new Date(img.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Admin Panel (only visible when admin logged in) */}
        {adminLoggedIn && activeTab === 'adminPanel' && (
          <div className="space-y-8">
            <h2 className="text-2xl font-bold">Admin Dashboard</h2>
            <div className="bg-gray-800 p-4 rounded-2xl">
              <h3 className="text-xl font-semibold mb-3">Pending Data Requests</h3>
              {dataRequests.filter(r=>!r.approved).length === 0 ? (
                <p className="text-gray-400">No pending requests.</p>
              ) : (
                dataRequests.filter(r=>!r.approved).map(req => (
                  <div key={req.id} className="border-b border-gray-700 py-3">
                    <p><strong>{req.name}</strong> – {req.mine}, {req.location}</p>
                    <p className="text-sm text-gray-300">Needs: {req.dataNeeded}</p>
                    <div className="flex gap-2 mt-2">
                      <button onClick={()=>approveDataRequest(req,7)} className="bg-green-600 hover:bg-green-500 px-3 py-1 text-sm rounded">Approve (7 days)</button>
                      <button onClick={()=>approveDataRequest(req,30)} className="bg-blue-600 hover:bg-blue-500 px-3 py-1 text-sm rounded">Approve (30 days)</button>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="bg-gray-800 p-4 rounded-2xl">
              <h3 className="text-xl font-semibold mb-3">Active User Accounts</h3>
              <table className="w-full text-sm">
                <thead className="text-gray-400 border-b border-gray-700">
                  <tr><th className="text-left p-2">Name</th><th className="text-left p-2">Email</th><th className="text-left p-2">Mine</th><th className="text-left p-2">Expiry</th></tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} className="border-b border-gray-800">
                      <td className="p-2">{u.name}</td>
                      <td className="p-2">{u.email}</td>
                      <td className="p-2">{u.mine}</td>
                      <td className="p-2">{new Date(u.expiry).toLocaleDateString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button onClick={()=>clearAdminData('users')} className="mt-3 text-red-400 text-sm hover:text-red-300">Clear all users</button>
            </div>
            <div className="bg-gray-800 p-4 rounded-2xl">
              <h3 className="text-xl font-semibold mb-3">Other Data Management</h3>
              <div className="flex gap-3 flex-wrap">
                <button onClick={()=>clearAdminData('subs')} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-sm">Clear Subscriptions</button>
                <button onClick={()=>clearAdminData('reqs')} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-sm">Clear Data Requests</button>
                <button onClick={()=>clearAdminData('contacts')} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-sm">Clear Contacts</button>
                <button onClick={()=>clearAdminData('gallery')} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-sm">Clear Gallery</button>
                <button onClick={()=>clearAdminData('inst')} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-sm">Clear Inst Subscriptions</button>
              </div>
            </div>
          </div>
        )}

        {/* My Data (for logged‑in requesters) */}
        {userLoggedIn && activeTab === 'myData' && (
          <div className="bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4">Your Data Access</h2>
            <div className="space-y-2">
              <p><strong>Name:</strong> {userLoggedIn.name}</p>
              <p><strong>Assigned Mine:</strong> {userLoggedIn.mine}</p>
              <p><strong>Access expires:</strong> {new Date(userLoggedIn.expiry).toLocaleString()}</p>
            </div>
            <div className="mt-6">
              <h3 className="text-xl font-semibold mb-2">Current Risk Report</h3>
              <p className="text-gray-300">Collapse risk: <span className="text-red-400 font-bold">{riskPercent}%</span> (based on latest EO data)</p>
              <p className="text-sm text-gray-400">InSAR: {insarValue} mm/day | Vibration: {vibrationValue} g | Rainfall: {rainfallValue} mm/h</p>
            </div>
            <div className="mt-6">
              <h3 className="text-xl font-semibold mb-2">Gallery for Your Mine</h3>
              <div className="grid md:grid-cols-2 gap-4">
                {galleryImages.filter(img=>img.mine===userLoggedIn.mine).slice(0,4).map(img => (
                  <img key={img.id} src={img.url} className="h-32 w-full object-cover rounded-lg" alt={img.caption} />
                ))}
                {galleryImages.filter(img=>img.mine===userLoggedIn.mine).length === 0 && <p className="text-gray-400">No images yet.</p>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;