import { useState } from 'react';
import { motion } from 'framer-motion';
import { PawPrint, Bell, MapPin, Users } from 'lucide-react';

const WildTrack = () => {
  const [alertSent, setAlertSent] = useState(false);
  const [activeConflict, setActiveConflict] = useState('Hwange East - Elephant corridor near Bubi mine');
  const conflictZones = ['Hwange East - Bubi mine', 'Sinamatella - Village #12', 'Victoria Falls - Zambezi lodge'];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-green-300">WildTrack™ · Conflict avoidance</h2>
        <div className="bg-gray-800 rounded-full px-3 py-1 text-sm flex items-center gap-1"><Users size={14} /> 38 village subscribers</div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-gray-800/40 rounded-2xl p-6 border border-green-500/30">
          <div className="flex items-center gap-2 text-green-400"><MapPin size={18} /> Active alert zone</div>
          <select value={activeConflict} onChange={e => setActiveConflict(e.target.value)} className="w-full mt-2 bg-gray-900 border border-gray-700 rounded-lg p-2">
            {conflictZones.map(z => <option key={z}>{z}</option>)}
          </select>
          <div className="mt-4 p-3 bg-red-900/30 rounded-lg border border-red-700/50">
            <p className="font-bold">⚠️ High risk</p>
            <p className="text-sm">Displaced elephant herd (12 individuals) detected within 1.8km of mining operation. Movement toward farms.</p>
          </div>
          <button onClick={() => setAlertSent(true)} className="mt-4 w-full bg-orange-600 hover:bg-orange-500 transition rounded-lg py-2 flex items-center justify-center gap-2"><Bell size={18} /> Send Alert to All Subscribers</button>
          {alertSent && <p className="text-green-400 text-sm mt-2 text-center">✅ SMS sent to 38 village contacts + local radio notice</p>}
        </div>

        <div className="bg-gray-800/40 rounded-2xl p-6 border border-gray-700">
          <h3 className="font-semibold flex items-center gap-2"><PawPrint size={18} /> EO‑based wildlife movement prediction</h3>
          <p className="text-sm text-gray-300 mt-2">Using NDVI decline + water source drying + mining expansion → predicted elephant exit corridor with 82% accuracy.</p>
          <div className="mt-4 h-40 bg-gray-700 rounded-lg flex items-center justify-center text-gray-400 text-sm">[dynamic map – GPS collar data overlay]</div>
          <div className="mt-3 text-xs text-gray-400">Last 7 days: 3 conflicts avoided via early warnings</div>
        </div>
      </div>
    </motion.div>
  );
};

export default WildTrack;