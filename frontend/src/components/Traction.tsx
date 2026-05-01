import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Users, Activity, Award, Globe } from 'lucide-react';

const userGrowth = [
  { month: 'Jan', users: 22 }, { month: 'Feb', users: 34 }, { month: 'Mar', users: 51 },
  { month: 'Apr', users: 87 }, { month: 'May', users: 118 }, { month: 'Jun', users: 147 }
];

const Traction = () => {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <h2 className="text-2xl font-bold">📈 Traction & Adoption</h2>
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-2xl p-4 text-center"><Users className="mx-auto text-blue-400" size={28} /><div className="text-2xl font-bold mt-2">147</div><div className="text-xs text-gray-400">active users</div></div>
        <div className="bg-gray-800 rounded-2xl p-4 text-center"><Activity className="mx-auto text-green-400" size={28} /><div className="text-2xl font-bold mt-2">5</div><div className="text-xs text-gray-400">mines piloting</div></div>
        <div className="bg-gray-800 rounded-2xl p-4 text-center"><Award className="mx-auto text-yellow-400" size={28} /><div className="text-2xl font-bold mt-2">73%</div><div className="text-xs text-gray-400">risk reduction (backtest)</div></div>
        <div className="bg-gray-800 rounded-2xl p-4 text-center"><Globe className="mx-auto text-purple-400" size={28} /><div className="text-2xl font-bold mt-2">12</div><div className="text-xs text-gray-400">communities alert-enabled</div></div>
      </div>
      <div className="bg-gray-800/40 rounded-2xl p-6">
        <h3 className="font-semibold mb-4">User growth (January – June 2026)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={userGrowth}><XAxis dataKey="month" stroke="#9ca3af" /><YAxis stroke="#9ca3af" /><Tooltip contentStyle={{ backgroundColor: '#1f2937' }} /><Line type="monotone" dataKey="users" stroke="#3b82f6" strokeWidth={3} /></LineChart>
        </ResponsiveContainer>
        <p className="text-sm text-gray-400 mt-4">Target: 200+ by Dec 2026 (all large mines in Zimbabwe)</p>
      </div>
    </motion.div>
  );
};

export default Traction;