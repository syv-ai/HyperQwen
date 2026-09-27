// Render the demo film to video: headless Chrome draws each frame, ffmpeg encodes.
//
//   cd bench/demo && npm install && node render.mjs            # -> docs/media/demo.{mp4,avif}
//   node render.mjs --stills 12.5,40      # PNG stills into ./stills, for checking a frame
//   node render.mjs --from 20 --to 30     # a slice, for iterating on one scene
//
// The film is a pure function of time (demo.js), so frame N is drawn at exactly N/fps
// seconds -- no screen recording, no dropped frames, and no GPU needed: Chrome runs
// with its software rasterizer. Several pages render interleaved frames in parallel
// and are written to ffmpeg in order.
import { createServer } from "node:http";
import { readFile, mkdir, writeFile, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const HERE = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const FPS = Number(opt("--fps", 60));
const OUT = resolve(opt("--out", join(HERE, "../../docs/media/demo")));
const JOBS = Number(opt("--jobs", 6));
const STILLS = opt("--stills", null);

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
const server = createServer(async (req, res) => {
  try {
    const p = join(HERE, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (!p.startsWith(HERE)) throw new Error("outside");
    const body = await readFile(p);
    res.writeHead(200, { "Content-Type": TYPES[extname(p)] || "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
}).listen(0, "127.0.0.1");
await new Promise((r) => server.once("listening", r));
const URL_ = `http://127.0.0.1:${server.address().port}/index.html?render`;

const browser = await puppeteer.launch({
  headless: true,
  executablePath: process.env.CHROME || undefined,   // e.g. an installed Chrome instead of puppeteer's
  args: ["--disable-gpu", "--force-color-profile=srgb", "--font-render-hinting=none", "--hide-scrollbars"],
});
async function openPage() {
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto(URL_, { waitUntil: "networkidle0" });
  await page.waitForFunction("window.HQ && window.HQ.ready === true", { timeout: 60000 });
  return page;
}

// One frame as raw RGBA, moved out of the page as base64 (cheaper than a PNG encode).
async function grab(page, t) {
  const b64 = await page.evaluate((t) => {
    const px = window.HQ.rgba(t);
    let s = "";
    const CH = 0x8000;
    for (let i = 0; i < px.length; i += CH) s += String.fromCharCode.apply(null, px.subarray(i, i + CH));
    return btoa(s);
  }, t);
  return Buffer.from(b64, "base64");
}

const first = await openPage();
const duration = await first.evaluate(() => window.HQ.duration);
console.log(`film: ${duration.toFixed(2)} s`);
console.log((await first.evaluate(() => window.HQ.scenes)).map((s) => `  ${s.name.padEnd(16)} ${s.start.toFixed(2)}-${s.end.toFixed(2)}`).join("\n"));
const m = await first.evaluate(() => window.HQ.model);
console.log(`batch model: mean TPOT ${m.meanTpotMs.toFixed(2)} ms`);

if (STILLS) {
  const dir = join(HERE, "stills");
  await mkdir(dir, { recursive: true });
  for (const ts of STILLS.split(",").map(Number)) {
    await first.evaluate((t) => window.HQ.frame(t), ts);
    const png = await first.screenshot({ clip: { x: 0, y: 0, width: 1920, height: 1080 } });
    const f = join(dir, `t${ts.toFixed(2).padStart(6, "0")}.png`);
    await writeFile(f, png);
    console.log("wrote", f);
  }
  await browser.close(); server.close();
  process.exit(0);
}

const from = Number(opt("--from", 0)), to = Number(opt("--to", duration));
const n0 = Math.round(from * FPS), n1 = Math.floor(to * FPS);
const total = n1 - n0;
const master = OUT + (argv.includes("--from") || argv.includes("--to") ? `-${from}-${to}` : "") + ".mp4";
await mkdir(dirname(master), { recursive: true });

// Master: H.264 High, 60 fps. CRF 21 keeps 1080p text clean on this flat dark UI.
const ff = spawn("ffmpeg", ["-y", "-loglevel", "error",
  "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1920x1080", "-r", String(FPS), "-i", "-",
  "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-tune", "animation",
  "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
  "-vf", "scale=out_color_matrix=bt709:flags=lanczos",
  "-movflags", "+faststart", master], { stdio: ["pipe", "inherit", "inherit"] });
const ffDone = new Promise((r, j) => ff.on("close", (c) => c ? j(new Error("ffmpeg " + c)) : r()));

const pages = [first];
for (let i = 1; i < JOBS; i++) pages.push(await openPage());
const pending = new Map();
let next = n0, written = n0;
const t0 = Date.now();
async function write() {
  while (pending.has(written)) {
    const buf = pending.get(written); pending.delete(written);
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    written++;
    if ((written - n0) % 120 === 0) {
      const el = (Date.now() - t0) / 1000;
      process.stdout.write(`\r${written - n0}/${total} frames  ${((written - n0) / el).toFixed(1)} fps  eta ${((total - (written - n0)) / ((written - n0) / el)).toFixed(0)} s   `);
    }
  }
}
let writing = Promise.resolve();
await Promise.all(pages.map(async (page) => {
  while (true) {
    const f = next++;
    if (f >= n1) break;
    while (f - written > JOBS * 4) await new Promise((r) => setTimeout(r, 5));   // bound memory
    pending.set(f, await grab(page, f / FPS));
    writing = writing.then(write);
  }
}));
await writing;
ff.stdin.end();
await ffDone;
console.log(`\nwrote ${master} (${((await stat(master)).size / 1e6).toFixed(2)} MB) in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
await browser.close(); server.close();

if (!argv.includes("--from") && !argv.includes("--to") && !argv.includes("--no-avif")) {
  // README copy: GitHub will not play an .mp4 from the repo inline, but it shows an
  // animated image. AVIF carries AV1's inter-frame coding, so the whole film at 720p30
  // is a few MB -- an animated WebP of the same film was ~20 MB, a GIF worse.
  const avif = OUT + ".avif";
  await new Promise((r, j) => spawn("ffmpeg", ["-y", "-loglevel", "error", "-i", master,
    "-vf", "fps=30,scale=1280:-1:flags=lanczos,format=yuv420p", "-c:v", "libsvtav1",
    "-crf", opt("--avif-crf", "30"), "-preset", "5", "-svtav1-params", "tune=0:enable-overlays=1",
    "-loop", "0", "-f", "avif", avif], { stdio: ["ignore", "ignore", "inherit"] })
    .on("close", (c) => c ? j(new Error("avif " + c)) : r()));
  console.log(`wrote ${avif} (${((await stat(avif)).size / 1e6).toFixed(2)} MB)`);
}
