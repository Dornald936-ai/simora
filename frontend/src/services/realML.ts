import * as tf from '@tensorflow/tfjs';

export interface TrainingRun {
  id: string;
  timestamp: Date;
  lossHistory: number[];
  finalLoss: number;
  accuracy: number;
  description: string;
}

export class RealRiskModel {
  private model: tf.Sequential | null = null;
  private isTraining = false;
  private currentAccuracy = 0;
  private trainingRuns: TrainingRun[] = [];
  private currentRunId: string | null = null;

  async init() {
    console.log('Initializing TensorFlow.js model...');
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [3] }));
    this.model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    await this.initialTrain();
    console.log('Model ready');
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