// Compare OLD Dart Direct-Form-II bandpass (pre-52d963c) vs current Dart SOS bandpass.
// A correct 0.5-3.5 Hz cardiac bandpass must PASS 1.2 Hz (cardiac) and REJECT 0.25 Hz (respiration).

const sampleRate = 30;

// OLD Dart filter (pre-52d963c): Direct Form II Transposed, 4th-order
const bOld = [0.0039, 0.0, -0.0078, 0.0, 0.0039];
const aOld = [1.0, -3.1806, 3.8612, -2.1122, 0.4351];
function applyBandpassOld(input: number[]): number[] {
  const length = input.length;
  const filtered = new Array(length).fill(0);
  const w = new Array(5).fill(0);
  for (let i = 0; i < length; i++) {
    let out = bOld[0] * input[i] + w[1];
    w[1] = bOld[1] * input[i] - aOld[1] * out + w[2];
    w[2] = bOld[2] * input[i] - aOld[2] * out + w[3];
    w[3] = bOld[3] * input[i] - aOld[3] * out + w[4];
    w[4] = bOld[4] * input[i] - aOld[4] * out;
    filtered[i] = out;
  }
  return filtered;
}

// NEW Dart filter (post-52d963c): broken SOS
const sos = [
  [0.00295136, 0.0, -0.00295136, 1.0, -1.891408, 0.897916],
  [1.0, 2.0, 1.0, 1.0, -1.882394, 0.906327],
];
function applyBandpassSOS(input: number[]): number[] {
  const length = input.length;
  let stage = input.slice();
  for (const section of sos) {
    const [b0, b1, b2, a0, a1, a2] = section;
    const out = new Array(length).fill(0);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < length; i++) {
      const x0 = stage[i];
      const y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
      out[i] = y0;
      x2 = x1; x1 = x0; y2 = y1; y1 = y0;
    }
    stage = out;
  }
  return stage;
}

function tone(f: number, seconds: number, amp = 15): number[] {
  const n = Math.floor(seconds * sampleRate);
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(110 + amp * Math.sin(2 * Math.PI * f * (i / fs)));
  return out;
}
const ac = (sig: number[]) => { let sq = 0; for (const v of sig) sq += v * v; return Math.sqrt(sq / sig.length); };
const acOnly = (sig: number[]) => { const m = sig.reduce((a,b)=>a+b,0)/sig.length; let sq=0; for (const v of sig) sq+=(v-m)**2; return Math.sqrt(sq/sig.length); };

for (const f of [1.2, 0.5, 0.25, 0.1, 2.5]) {
  const inTone = tone(f, 20, 10);
  const oldOut = applyBandpassOld(inTone);
  const sosOut = applyBandpassSOS(inTone);
  console.log(`f=${f}Hz: old gain=${(acOnly(oldOut)/10).toFixed(2)}  SOS gain=${(acOnly(sosOut)/10).toFixed(2)}  (cardiac=1.2 should pass, resp=0.25 should reject)`);
}
