// Test the web app's front-camera skin-ROI detection (0b83272) on synthetic faces.
// Replicates the exact YCbCr threshold logic from page.tsx.

function skinDetect(data: Uint8ClampedArray, label: string) {
  let skinR = 0, skinG = 0, skinB = 0, skinCount = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const i = (y * 64 + x) * 4;
      const pr = data[i]; const pg = data[i + 1]; const pb = data[i + 2];
      const brightness = (pr + pg + pb) / 3;
      if (brightness < 45 || brightness > 245) continue;
      const cb = 128 - 0.168736 * pr - 0.331264 * pg + 0.5 * pb;
      const cr = 128 + 0.5 * pr - 0.418688 * pg - 0.081312 * pb;
      if (cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173) {
        skinR += pr; skinG += pg; skinB += pb; skinCount++;
      }
    }
  }
  const hasSkin = skinCount >= 0.05 * 64 * 64;
  console.log(`${label}: skinCount=${skinCount} (${(skinCount/4096*100).toFixed(1)}%) hasSkin=${hasSkin} avgRGB=(${(skinR/Math.max(skinCount,1)).toFixed(0)},${(skinG/Math.max(skinCount,1)).toFixed(0)},${(skinB/Math.max(skinCount,1)).toFixed(0)})`);
  return hasSkin;
}

function makeCanvas(r: number, g: number, b: number, fillRect = true): Uint8ClampedArray {
  // 64x64, face occupies center 40x40 with skin color, border (hair/clothing) different
  const data = new Uint8ClampedArray(64 * 64 * 4);
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const i = (y * 64 + x) * 4;
      let rr = 40, gg = 40, bb = 40; // dark border (hair/cloth)
      if (fillRect && x >= 12 && x < 52 && y >= 12 && y < 52) {
        rr = r; gg = g; bb = b; // face
      }
      data[i] = rr; data[i+1] = gg; data[i+2] = bb; data[i+3] = 255;
    }
  }
  return data;
}

// Typical skin tones (R>G>B, reddish)
skinDetect(makeCanvas(180, 140, 110), "light skin R180G140B110");
skinDetect(makeCanvas(150, 115, 95), "medium skin R150G115B95");
skinDetect(makeCanvas(110, 85, 70), "darker skin R110G85B70");
skinDetect(makeCanvas(90, 70, 60), "very dark skin R90G70B60");
skinDetect(makeCanvas(200, 50, 40), "red-dominant (sunburn) R200G50B40");
skinDetect(makeCanvas(100, 100, 100), "neutral gray (no face) R100G100B100");
// Face only fills center 20x20 (small face)
const small = new Uint8ClampedArray(64*64*4);
for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
  const i = (y*64+x)*4;
  let rr=40,gg=40,bb=40;
  if (x>=22 && x<42 && y>=22 && y<42) { rr=180; gg=140; bb=110; }
  small[i]=rr; small[i+1]=gg; small[i+2]=bb; small[i+3]=255;
}
skinDetect(small, "small face (center 20x20)");
