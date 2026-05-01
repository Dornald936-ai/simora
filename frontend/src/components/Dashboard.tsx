import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, TrendingUp, AlertTriangle, Brain, Cpu, Wifi, Satellite } from 'lucide-react';
import { motion } from 'framer-motion';

// Simplified in‑memory ML model (no external file, no TensorFlow.js – just a realistic mock)
// This simulates a neural network that improves with retraining.
class SimulatedMLModel {
  private accuracy = 85.2;
  private loss = 0.045;
  private trainingRuns: Array<{ id: string; timestamp: Date; lossHistory: number[]; finalLoss: number; accuracy: number; description: string }> = [];
  private currentRunId: string | null = null;

  constructor() {
    // Initial synthetic training run
    this.addTrainingRun([0.12, 0.08, 0.05, 0.04, 0.035], 0.035, 85.2, 'Initial synthetic training');
  }

  private addTrainingRun(lossHistory: number[], finalLoss: number, accuracy: number, description: string) {
    const run = {
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
    // Realistic non‑linear formula that responds to sliders
    let risk = (insar / 8) * 50 + (vibration / 2.5) * 30 + rainfall * 20;
    risk = Math.min(95, Math.max(5, risk));
    return Math.round(risk);
  }

  async retrain(newData: { insar: number; vibration: number; rainfall: number; risk: number }[]) {
    // Simulate retraining: improve accuracy and reduce loss
    this.accuracy = Math.min(98, this.accuracy + 1.2);
    this.loss = Math.max(0.008, this.loss - 0.003);
    const newLossHistory = [0.08, 0.05, 0.03, 0.02, this.loss];
    const desc = `Retrain with ${newData.length} new incident(s) (risk ~${newData[0].risk}%)`;
    this.addTrainingRun(newLossHistory, this.loss, this.accuracy, desc);
  }

  getCurrentAccuracy() { return this.accuracy; }
  getTrainingRuns() { return this.trainingRuns; }
  getCurrentRunId() { return this.currentRunId; }
  resetHistory() {
    this.trainingRuns = [];
    this.accuracy = 85.2;
    this.loss = 0.045;
    this.addTrainingRun([0.12, 0.08, 0.05, 0.04, 0.035], 0.035, 85.2, 'Initial synthetic training');
  }
}

const mlModel = new SimulatedMLModel();

const Dashboard = () => {
  const [modelReady, setModelReady] = useState(true);
  const [riskPercent, setRiskPercent] = useState(50);
  const [insarValue, setInsarValue] = useState(2.5);
  const [vibrationValue, setVibrationValue] = useState(1.1);
  const [rainfallValue, setRainfallValue] = useState(0.4);
  const [riskHistory, setRiskHistory] = useState<{ time: string; risk: number }[]>([
    { time: '00:00', risk: 45 },
    { time: '06:00', risk: 48 },
  ]);
  const [userCount, setUserCount] = useState(147);
  const [modelAccuracy, setModelAccuracy] = useState<number | null>(85.2);
  const [modelLoss, setModelLoss] = useState<number | null>(0.045);
  const [trainingRuns, setTrainingRuns] = useState<any[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [displayLossHistory, setDisplayLossHistory] = useState<number[]>([]);
  const [training, setTraining] = useState(false);

  const refreshMetrics = () => {
    setModelAccuracy(mlModel.getCurrentAccuracy());
    const runs = mlModel.getTrainingRuns();
    setTrainingRuns(runs);
    const currentId = mlModel.getCurrentRunId();
    setSelectedRunId(currentId);
    const selectedRun = runs.find((r: any) => r.id === currentId);
    setDisplayLossHistory(selectedRun?.lossHistory || []);
    if (runs.length && currentId) {
      const lastLoss = runs.find((r: any) => r.id === currentId)?.finalLoss;
      setModelLoss(lastLoss ?? null);
    }
  };

  const updatePrediction = async () => {
    const newRisk = await mlModel.predict(insarValue, vibrationValue, rainfallValue);
    setRiskPercent(newRisk);
    setRiskHistory(prev => [...prev.slice(-5), { time: new Date().toLocaleTimeString(), risk: newRisk }]);
  };

  useEffect(() => {
    refreshMetrics();
    updatePrediction();
  }, []);

  useEffect(() => {
    updatePrediction();
  }, [insarValue, vibrationValue, rainfallValue]);

  useEffect(() => {
    const interval = setInterval(() => {
      setUserCount(prev => Math.min(200, prev + 0.1));
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  const handleRetrain = async () => {
    if (training) return;
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
    const run = trainingRuns.find((r: any) => r.id === runId);
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

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center flex-wrap gap-4">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Simora – MineGuard™ · AI Risk Intelligence
          </h1>
          <div className="flex gap-2">
            <div className="bg-gray-800/50 rounded-full px-3 py-1 text-sm flex items-center gap-1"><Satellite size={14}/> Sentinel-1</div>
            <div className="bg-gray-800/50 rounded-full px-3 py-1 text-sm flex items-center gap-1"><Wifi size={14}/> IoT</div>
            <div className="bg-purple-800/70 rounded-full px-3 py-1 text-sm flex items-center gap-1"><Brain size={14}/> AI Core</div>
          </div>
        </div>

        {/* Main panel */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/40 backdrop-blur rounded-2xl p-6 border border-gray-700">
            <label className="text-sm text-gray-400">Active Mine</label>
            <select className="w-full mt-1 bg-gray-900 border-gray-700 rounded-lg p-2">
              <option>Great Dyke - North Dam</option>
              <option>Great Dyke - South Pit</option>
              <option>HWange Colliery</option>
            </select>

            <div className="mt-4 space-y-3">
              <div>
                <label className="text-sm">InSAR deformation (mm/day): {insarValue.toFixed(2)}</label>
                <input type="range" min="0.5" max="8" step="0.1" value={insarValue} onChange={e => setInsarValue(parseFloat(e.target.value))} className="w-full mt-1" />
              </div>
              <div>
                <label className="text-sm">IoT vibration (g): {vibrationValue.toFixed(2)}</label>
                <input type="range" min="0.1" max="2.5" step="0.05" value={vibrationValue} onChange={e => setVibrationValue(parseFloat(e.target.value))} className="w-full" />
              </div>
              <div>
                <label className="text-sm">Rainfall (mm/h): {rainfallValue.toFixed(2)}</label>
                <input type="range" min="0" max="1" step="0.01" value={rainfallValue} onChange={e => setRainfallValue(parseFloat(e.target.value))} className="w-full" />
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
              <button onClick={updatePrediction} className="flex-1 bg-blue-600 hover:bg-blue-500 transition rounded-lg py-2 flex items-center justify-center gap-2">
                <Activity size={18} /> Run ML Prediction
              </button>
              <button onClick={handleRetrain} className="flex-1 bg-purple-700 hover:bg-purple-600 transition rounded-lg py-2 flex items-center justify-center gap-2">
                <Brain size={18} /> {training ? 'Retraining...' : 'Retrain with New Data'}
              </button>
            </div>
          </div>

          {/* Traction card */}
          <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-2xl p-6 border border-blue-500/30 backdrop-blur">
            <div className="flex items-center gap-2 text-blue-400"><TrendingUp size={18} /> Live Traction</div>
            <div className="text-3xl font-bold mt-2">{Math.floor(userCount)} <span className="text-sm font-normal text-gray-400">active users</span></div>
            <div className="text-sm text-gray-300 mt-1">+23% this quarter</div>
            <div className="mt-4 text-xs text-gray-400">Pilots: Great Dyke, Hwange Colliery • 5 mines • 12 villages</div>
          </div>
        </div>

        {/* Risk trend chart */}
        <div className="bg-gray-800/40 rounded-2xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-2 flex items-center gap-2"><Activity size={16} /> Predicted Risk Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={riskHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9ca3af" />
              <YAxis domain={[0,100]} stroke="#9ca3af" />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} />
              <Line type="monotone" dataKey="risk" stroke="#ef4444" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ML Model Info */}
        <div className="bg-gray-800/30 rounded-2xl p-4 border border-gray-700 flex flex-wrap justify-between items-center gap-3">
          <div className="flex items-center gap-2"><Cpu size={18} className="text-green-400" /><span className="text-sm font-mono">Simora‑Net (3‑layer NN simulated)</span></div>
          <div className="text-xs text-gray-400"><span className="text-green-300">Active</span></div>
        </div>

        {/* Accuracy & loss summary */}
        <div className="bg-gray-800/40 rounded-2xl p-4 border border-purple-500/30 flex flex-wrap justify-between items-center gap-3">
          <div><span className="text-sm text-gray-400">Model accuracy:</span> <span className="text-xl font-bold text-purple-300">{modelAccuracy !== null ? modelAccuracy.toFixed(1) : '—'}%</span></div>
          <div><span className="text-sm text-gray-400">Last training loss:</span> <span className="text-xl font-bold text-orange-300">{modelLoss !== null ? modelLoss.toFixed(4) : '—'}</span></div>
          <button onClick={refreshMetrics} className="bg-purple-800 hover:bg-purple-700 px-3 py-1 rounded text-xs">↻ Evaluate</button>
        </div>

        {/* Historical loss curve */}
        <div className="bg-gray-800/40 rounded-2xl p-4 border border-green-500/30">
          <div className="flex justify-between items-center flex-wrap gap-3 mb-3">
            <h3 className="font-semibold flex items-center gap-2"><Brain size={16} className="text-green-400" /> Training Loss Curve (historical)</h3>
            <div className="flex gap-2">
              <select value={selectedRunId || ''} onChange={e => handleSelectRun(e.target.value)} className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm">
                {trainingRuns.map((run: any) => (
                  <option key={run.id} value={run.id}>{run.timestamp.toLocaleTimeString()} – {run.description.slice(0, 40)}</option>
                ))}
              </select>
              <button onClick={handleResetHistory} className="bg-red-800 hover:bg-red-700 px-3 py-1 rounded text-xs">Reset History</button>
            </div>
          </div>
          {displayLossHistory.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={displayLossHistory.map((loss, idx) => ({ epoch: idx + 1, loss }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="epoch" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} />
                <Line type="monotone" dataKey="loss" stroke="#22c55e" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center text-gray-400 py-8">No history yet. Click "Retrain" to generate loss curve.</div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;