import * as tf from '@tensorflow/tfjs';

export class SimpleRiskModel {
  private model: tf.Sequential | null = null;
  private ready = false;

  async init() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [3] }));
    this.model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    
    // Train with synthetic data (insar, vibration, rainfall) → risk
    const inputs = [[0.5,0.2,0.1], [2,0.7,0.3], [4,1.4,0.6], [6,2.0,0.9], [1,0.4,0.2]];
    const outputs = inputs.map(inp => [ (inp[0]/6)*0.5 + (inp[1]/2)*0.3 + inp[2]*0.2 ]);
    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(outputs);
    await this.model.fit(xs, ys, { epochs: 100, verbose: 0 });
    xs.dispose(); ys.dispose();
    this.ready = true;
  }

  async predict(insar: number, vibration: number, rainfall: number): Promise<number> {
    if (!this.model || !this.ready) return 50;
    const input = tf.tensor2d([[insar, vibration, rainfall]]);
    const output = this.model.predict(input) as tf.Tensor;
    const risk = (await output.data())[0] * 100;
    output.dispose(); input.dispose();
    return Math.round(risk);
  }

  isReady() { return this.ready; }
}

export const mlModel = new SimpleRiskModel();