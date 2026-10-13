import Link from 'next/link';
import TrainingMethodologyToggle from './methodology-toggle';

export default function TrainingPage() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center px-6 py-12">
      <h1 className="text-3xl font-bold mb-4 text-red-400">Resilient Health — MCD-rPPG Training Dashboard</h1>
      <p className="text-base text-neutral-400 mb-8 max-w-lg text-center">
        Train the SCNN_8rois model for remote photoplethysmography using the MCD-rPPG dataset.
      </p>

      <div className="w-full max-w-2xl bg-neutral-900 rounded-2xl border border-neutral-800 shadow-2xl shadow-red-900/20 overflow-hidden">
        <div className="p-6">
          <h2 className="text-xl font-bold mb-4 text-red-400">Model Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
              <p className="text-neutral-400 text-sm mb-1">Model Architecture</p>
              <p className="text-lg font-bold text-white">SCNN_8rois</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
              <p className="text-neutral-400 text-sm mb-1">Dataset</p>
              <p className="text-lg font-bold text-white">MCD-rPPG</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
              <p className="text-neutral-400 text-sm mb-1">Training Folds</p>
              <p className="text-lg font-bold text-white">5-Fold CV</p>
            </div>
            <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
              <p className="text-neutral-400 text-sm mb-1">Batch Size</p>
              <p className="text-lg font-bold text-white">36</p>
            </div>
          </div>

          <h2 className="text-xl font-bold mb-4 text-red-400">Training Pipeline</h2>
          <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700 mb-6">
            <p className="text-neutral-300 text-sm mb-2">
              The training pipeline processes facial video frames through 8 regions of interest (ROIs):
            </p>
            <ul className="text-neutral-400 text-sm space-y-1 ml-4">
              <li>• Face region extraction using facial landmarks</li>
              <li>• POS (Plane-Orthogonal-to-Skin) signal extraction</li>
              <li>• 1D Feature Pyramid Network with attention</li>
              <li>• PPG signal prediction and HR calculation</li>
            </ul>
          </div>

          <h2 className="text-xl font-bold mb-4 text-red-400">Results</h2>
          <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700 mb-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-neutral-400 text-sm mb-1">PPG MAE</p>
                <p className="text-2xl font-bold text-green-400">0.5556</p>
              </div>
              <div>
                <p className="text-neutral-400 text-sm mb-1">HR MAE (BPM)</p>
                <p className="text-2xl font-bold text-green-400">4.32</p>
              </div>
            </div>
          </div>

          <h2 className="text-xl font-bold mb-4 text-red-400">How to Train</h2>
          <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
            <pre className="text-sm text-neutral-300 overflow-x-auto">
{`# Install dependencies
pip install -r training/requirements.txt

# Download MCD-rPPG dataset from Hugging Face
# https://huggingface.co/datasets/kyegorov/mcd_rppg

# Run training
python training/run_training.py \\
  --model SCNN_8rois \\
  --dataset mcd_rppg \\
  --valid_fold 1 \\
  --test_fold 0 \\
  --device cuda`}
            </pre>
          </div>
        </div>
      </div>

      <TrainingMethodologyToggle />

      <div className="mt-6 flex gap-4">
        <Link href="/vital-monitoring" className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition">
           Start Scan
        </Link>
        <Link href="/" className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded-lg transition">
          Home
        </Link>
      </div>
    </main>
  );
}
