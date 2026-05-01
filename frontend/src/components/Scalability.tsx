const Scalability = () => {
  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Scaling Simora Across Africa</h2>
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold text-lg">Phase 1 (2025-2026)</h3>
          <p>Zimbabwe: 15 large mines, 200 ASM co-ops, wildlife corridors. Infrastructure: Kubernetes + PostGIS. Blockchain: Permissioned PoA (validators: ZINGSA, Mines Min, Simora).</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold text-lg">Phase 2 (2027-2028)</h3>
          <p>Zambia & DRC: Add local validators to consortium blockchain. Cross-chain communication via Axelar. Scale ML inference using GPU clusters on AWS/Azure.</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold text-lg">Phase 3 (2029+)</h3>
          <p>Pan-African: 500+ mines, 1000+ communities. Blockchain moves to Proof-of-Stake with light clients for low-bandwidth areas. Integration with national space agencies.</p>
        </div>
      </div>
      <div className="mt-6 p-4 bg-blue-900 rounded">
        <p className="font-bold">Technical scalability metrics:</p>
        <ul className="list-disc ml-6 mt-2">
          <li>Data pipeline: Apache Kafka handles 1M events/min</li>
          <li>ML inference: Serverless GPU auto-scaling (0.5s latency)</li>
          <li>Blockchain: 1000+ TPS with validated nodes across borders</li>
          <li>User growth: 5,000 daily active users without degradation</li>
        </ul>
      </div>
    </div>
  );
};

export default Scalability;