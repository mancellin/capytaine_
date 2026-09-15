#!/usr/bin/env node
// Records a video straight from the interactive three.js hero animation
// (docs/_static/harmonic_mesh_viewer.js + boat_animation_data.bin), as an
// alternative to the offline vedo render in
// docs/examples/src/B7_boat_animation.py. This guarantees the video matches
// the live viewer's colors/lighting exactly, since it's literally the same
// renderer -- rather than approximating it in a second one.
//
// See README.md next to this file for setup, usage and why the loop point
// is found empirically from the recorded pixels (see findLoop() below)
// rather than computed from the animation's period: Playwright's video
// timestamps don't track the page's real animation clock precisely enough
// in this environment for computed cut points to land the loop cleanly.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, rm, rename, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const RAW_DIR = path.join(__dirname, '.raw');
const RAW_PATH = path.join(RAW_DIR, 'raw.webm');
const META_PATH = path.join(RAW_DIR, 'meta.json');

const OMEGA = 1.5; // must match `omega` in docs/examples/src/B7_boat_animation.py
const PERIOD = (2 * Math.PI) / OMEGA; // seconds for one wave cycle -- the loop length
const WIDTH = 800; // matches the resolution B7_boat_animation.py renders at,
const HEIGHT = 600; // which matches the page's `aspect-ratio: 4/3` hero box

const WARMUP = 0.4; // seconds of recording discarded up front (page settle time)
const BUFFER_PERIODS = 2.2; // how many periods' worth to record, for findLoop() to search
const TAIL_MARGIN = 0.3; // extra seconds recorded past what we need, as slack

// Loop-point search is done on small grayscale frames -- plenty to tell
// wave/boat phase apart, and fast to diff thousands of times over.
const ANALYSIS_WIDTH = 64;
const ANALYSIS_HEIGHT = 48;

const args = process.argv.slice(2);
const crossfadeArg = args.find((a) => a.startsWith('--crossfade='));
const crossfadeSeconds = crossfadeArg ? parseFloat(crossfadeArg.split('=')[1]) : 0.15;
const reEncodeOnly = args.includes('--re-encode');
const outputPath = path.resolve(
  args.find((a) => !a.startsWith('--')) ?? path.join(__dirname, 'front_page_animation.mp4'),
);

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.bin': 'application/octet-stream' };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function serveRepo() {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      try {
        const filePath = path.join(REPO_ROOT, decodeURIComponent(req.url.split('?')[0]));
        const data = await readFile(filePath);
        res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] ?? 'application/octet-stream' });
        res.end(data);
      } catch {
        res.writeHead(404);
        res.end();
      }
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

async function recordRaw() {
  await rm(RAW_DIR, { recursive: true, force: true });
  await mkdir(RAW_DIR, { recursive: true });

  const server = await serveRepo();
  const { port } = server.address();

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    recordVideo: { dir: RAW_DIR, size: { width: WIDTH, height: HEIGHT } },
  });
  const page = await context.newPage();

  await page.goto(`http://127.0.0.1:${port}/docs/_scripts/record_hero_animation/harness.html`);
  await page.waitForFunction(() => window.__viewer !== undefined);

  const recordedSeconds = WARMUP + BUFFER_PERIODS * PERIOD + TAIL_MARGIN;
  await sleep(recordedSeconds * 1000);

  await context.close(); // the video file is only flushed to disk on close
  await browser.close();
  server.close();

  // Playwright names the file after an internal id we don't otherwise know;
  // RAW_DIR was emptied above and this script only records one page at a
  // time, so there's exactly one file in it now. Rename it to a fixed name
  // so --re-encode can find it later without needing to search for it.
  const [rawVideo] = await readdir(RAW_DIR);
  await rename(path.join(RAW_DIR, rawVideo), RAW_PATH);
  return recordedSeconds;
}

function decodeGrayscaleFrames(rawPath) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let stderr = '';
    const ff = spawn('ffmpeg', [
      '-i', rawPath,
      '-vf', `scale=${ANALYSIS_WIDTH}:${ANALYSIS_HEIGHT}:flags=area,format=gray`,
      '-f', 'rawvideo',
      '-',
    ]);
    ff.stdout.on('data', (d) => chunks.push(d));
    ff.stderr.on('data', (d) => { stderr += d; });
    ff.on('error', reject);
    ff.on('close', (code) => {
      if (code !== 0) return reject(new Error(`ffmpeg frame decode failed (${code}): ${stderr}`));
      const buf = Buffer.concat(chunks);
      const frameSize = ANALYSIS_WIDTH * ANALYSIS_HEIGHT;
      const frameCount = Math.floor(buf.length / frameSize);
      const frames = [];
      for (let i = 0; i < frameCount; i++) frames.push(buf.subarray(i * frameSize, (i + 1) * frameSize));
      resolve(frames);
    });
  });
}

function meanAbsDiff(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

// Finds the loop point by autocorrelation of the recorded frames themselves,
// rather than by computing it from PERIOD and assumed timestamps: for a
// range of candidate loop lengths L (in frames), scores how well frame[i]
// matches frame[i+L] averaged across the whole recording, and keeps the L
// with the best (lowest-difference) match -- i.e. the recording's own,
// empirically observed period, however many frames it actually took the
// recorder to cover one wave cycle. Then, for that L, finds the single best
// start frame S. Both purely from pixel content, so it's correct regardless
// of whether the video's frame timestamps track real time accurately.
function findLoop(frames, recordedSeconds) {
  const total = frames.length;
  const estFps = total / recordedSeconds;
  const warmupFrames = Math.round(WARMUP * estFps);
  const centerL = Math.round(PERIOD * estFps);
  const searchMin = Math.max(5, Math.round(centerL * 0.5));
  const searchMax = Math.min(total - warmupFrames - 5, Math.round(centerL * 1.8));
  if (searchMax <= searchMin) {
    throw new Error(`Recording too short to search for a loop point (${total} frames) -- raise BUFFER_PERIODS`);
  }

  let bestL = null;
  let bestScore = Infinity;
  for (let L = searchMin; L <= searchMax; L++) {
    let sum = 0;
    let n = 0;
    for (let i = warmupFrames; i + L < total; i++) {
      sum += meanAbsDiff(frames[i], frames[i + L]);
      n++;
    }
    const score = sum / n;
    if (score < bestScore) {
      bestScore = score;
      bestL = L;
    }
  }

  // Refine the exact start frame for that L: average over a few consecutive
  // frames at each candidate start, to avoid picking a start that happens to
  // land on a compression-noise fluke.
  const K = 3;
  const startSearchEnd = Math.min(total - bestL - K, warmupFrames + centerL);
  let bestS = warmupFrames;
  let bestSScore = Infinity;
  for (let s = warmupFrames; s <= startSearchEnd; s++) {
    let sum = 0;
    for (let k = 0; k < K; k++) sum += meanAbsDiff(frames[s + k], frames[s + bestL + k]);
    const score = sum / K;
    if (score < bestSScore) {
      bestSScore = score;
      bestS = s;
    }
  }

  return { S: bestS, L: bestL, estFps, matchScore: bestSScore };
}

function buildFilterComplex(S, L, estFps) {
  const C = Math.max(1, Math.round(crossfadeSeconds * estFps)); // crossfade length, in frames
  const duration = C / estFps;
  return {
    filter:
      `[0:v]trim=start_frame=${S}:end_frame=${S + C},setpts=PTS-STARTPTS[head];` +
      `[0:v]trim=start_frame=${S + C}:end_frame=${S + L},setpts=PTS-STARTPTS[body];` +
      `[0:v]trim=start_frame=${S + L}:end_frame=${S + L + C},setpts=PTS-STARTPTS[tail];` +
      `[tail][head]xfade=transition=fade:duration=${duration.toFixed(4)}:offset=0[blended];` +
      `[blended][body]concat=n=2:v=1:a=0[out]`,
    crossfadeFrames: C,
  };
}

async function encode(S, L, estFps) {
  const { filter } = buildFilterComplex(S, L, estFps);
  await execFileAsync('ffmpeg', [
    '-y',
    '-i', RAW_PATH,
    '-filter_complex', filter,
    '-map', '[out]',
    '-an',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '28', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    outputPath,
  ]);
}

async function main() {
  let S, L, estFps, matchScore;

  if (reEncodeOnly) {
    ({ S, L, estFps, matchScore } = JSON.parse(await readFile(META_PATH, 'utf8')));
  } else {
    const recordedSeconds = await recordRaw();
    const frames = await decodeGrayscaleFrames(RAW_PATH);
    ({ S, L, estFps, matchScore } = findLoop(frames, recordedSeconds));
    await writeFile(META_PATH, JSON.stringify({ S, L, estFps, matchScore }, null, 2));
  }

  await encode(S, L, estFps);
  console.log(`Wrote ${outputPath}`);
  console.log(`(loop: ${L} frames at ~${estFps.toFixed(1)}fps, match score ${matchScore.toFixed(2)} ` +
    `(0 = identical, 255 = opposite) -- re-run with --re-encode --crossfade=SECONDS to tune without ` +
    `re-recording)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
