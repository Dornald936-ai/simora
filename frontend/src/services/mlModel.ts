import * as tf from '@tensorflow/tfjs';

export interface TrainingRun {
  id: string;
  timestamp: Date;
  lossHistory: number[];
  finalLoss: number;
  accuracy: number | null;
  description: string;
}

export class CollapseRiskModel {
  private model: tf.Sequential | null = null;
  private isTraining = false;
  private currentAccuracy: number | null = null;
  private trainingRuns: TrainingRun[] = [];
  private currentRunId: string | null = null;

  async init() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [3] }));
    this.model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['mae'] });
    await this.trainSynthetic();
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
    return { inputs, outputs };
  }

  private async trainSynthetic() {
    const { inputs, outputs } = this.generateSyntheticData();
    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(outputs);
    const history = await this.model!.fit(xs, ys, { epochs: 150, verbose: 0, validationSplit: 0.2 });
    const lossHistory = history.history.loss as number[];
    const finalLoss = lossHistory[lossHistory.length - 1];
    await this.evaluateAccuracy();
    this.addTrainingRun(lossHistory, finalLoss, 'Initial synthetic training');
    xs.dispose(); ys.dispose();
  }

  private async evaluateAccuracy() {
    if (!this.model) return;
    const { inputs, outputs } = this.generateSyntheticData();
    const testInputs = inputs.slice(-2);
    const testOutputs = outputs.slice(-2);
    const xs = tf.tensor2d(testInputs);
    const ys = tf.tensor2d(testOutputs);
    const predictions = this.model.predict(xs) as tf.Tensor;
    const predValues = await predictions.data();
    const trueValues = await ys.data();
    let correct = 0;
    for (let i = 0; i < predValues.length; i++) {
      if (Math.abs(predValues[i] - trueValues[i]) <= 0.1) correct++;
    }
    this.currentAccuracy = (correct / predValues.length) * 100;
    xs.dispose(); ys.dispose(); predictions.dispose();
  }

  private addTrainingRun(lossHistory: number[], finalLoss: number, description: string) {
    const run: TrainingRun = {
      id: Date.now().toString(),
      timestamp: new Date(),
      lossHistory: [...lossHistory],
      finalLoss,
      accuracy: this.currentAccuracy,
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
    const inputs = newData.map(d => [d.insar, d.vibration, d.rainfall]);
    const outputs = newData.map(d => [d.risk / 100]);
    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(outputs);
    const { inputs: synthInputs, outputs: synthOutputs } = this.generateSyntheticData();
    const allInputs = [...synthInputs, ...inputs];
    const allOutputs = [...synthOutputs, ...outputs];
    const xsAll = tf.tensor2d(allInputs);
    const ysAll = tf.tensor2d(allOutputs);
    const history = await this.model.fit(xsAll, ysAll, { epochs: 50, verbose: 0, validationSplit: 0.2 });
    const lossHistory = history.history.loss as number[];
    const finalLoss = lossHistory[lossHistory.length - 1];
    await this.evaluateAccuracy();
    const desc = `Retrain with ${newData.length} new incident(s) (risk ~${newData[0].risk}%)`;
    this.addTrainingRun(lossHistory, finalLoss, desc);
    this.isTraining = false;
    xs.dispose(); ys.dispose(); xsAll.dispose(); ysAll.dispose();
  }

  getCurrentAccuracy(): number | null { return this.currentAccuracy; }
  getTrainingRuns(): TrainingRun[] { return this.trainingRuns; }
  getCurrentRunId(): string | null { return this.currentRunId; }
  resetHistory() {
    this.trainingRuns = [];
    this.currentRunId = null;
    this.trainSynthetic(); // fresh start
  }
}

export const mlModel = new CollapseRiskModel();