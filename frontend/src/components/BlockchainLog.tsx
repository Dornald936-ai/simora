import { useState } from 'react';

const mockPredictions = [
  { id: 1, mineId: 'Great Dyke', risk: '73%', hash: '0xabc...123', timestamp: '2025-03-15 14:32' },
  { id: 2, mineId: 'HWange Colliery', risk: '42%', hash: '0xdef...456', timestamp: '2025-03-14 09:17' },
];

const BlockchainLog = () => {
  const [predictions, setPredictions] = useState(mockPredictions);
  const [newPrediction, setNewPrediction] = useState({ mineId: '', risk: '' });

  const addMockPrediction = () => {
    if (!newPrediction.mineId || !newPrediction.risk) return;
    const newId = predictions.length + 1;
    const fakeHash = '0x' + Math.random().toString(36).substring(2, 8) + '...' + Math.random().toString(36).substring(2, 5);
    setPredictions([{
      id: newId,
      mineId: newPrediction.mineId,
      risk: newPrediction.risk,
      hash: fakeHash,
      timestamp: new Date().toLocaleString()
    }, ...predictions]);
    setNewPrediction({ mineId: '', risk: '' });
  };

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Blockchain-Verified Predictions</h2>
      <p className="text-gray-300 mb-4">Every hazard alert is logged on a permissioned blockchain (no third party). Validators: ZINGSA, Simora, Ministry of Mines.</p>
      <div className="bg-gray-800 p-4 rounded mb-4">
        <h3 className="font-bold">Add New Prediction (Validator only)</h3>
        <input placeholder="Mine ID" value={newPrediction.mineId} onChange={e => setNewPrediction({...newPrediction, mineId: e.target.value})} className="text-black p-1 mr-2" />
        <input placeholder="Risk level %" value={newPrediction.risk} onChange={e => setNewPrediction({...newPrediction, risk: e.target.value})} className="text-black p-1 mr-2" />
        <button onClick={addMockPrediction} className="bg-green-600 px-3 py-1 rounded">Log Prediction (Mock TX)</button>
      </div>
      <table className="w-full text-left">
        <thead><tr><th>Mine</th><th>Risk</th><th>Timestamp</th><th>TX Hash (immutable)</th></tr></thead>
        <tbody>
          {predictions.map(p => (
            <tr key={p.id} className="border-t border-gray-700">
              <td className="py-2">{p.mineId}</td><td>{p.risk}</td><td>{p.timestamp}</td><td className="font-mono text-xs">{p.hash}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 text-xs text-gray-400">🔗 Verified on local Ethereum network. No third-party oracle. Scalable to pan-Africa consortium.</div>
    </div>
  );
};

export default BlockchainLog;