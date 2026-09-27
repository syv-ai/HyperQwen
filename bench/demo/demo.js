/* HyperQwen demo video -- the whole film is one pure function of time.
 *
 *   const film = HQDemo.create(window.HQ_DATA);
 *   film.draw(ctx, t);          // paint the 1920x1080 frame at t seconds
 *   film.duration               // seconds
 *
 * index.html plays it live in a browser; render.mjs steps it frame by frame in
 * headless Chrome and pipes the frames to ffmpeg. Same code, same pixels.
 *
 * What is on screen and where it comes from (data.json, see build_data.py):
 *
 *  - One request: two demo_capture.py recordings replayed at their recorded speed.
 *    At race time T a lane shows exactly the chunks whose recorded arrival was <= T.
 *    Nothing is sped up or interpolated, and token counts come from the tokenizer.
 *  - Sixty-four at once: `vllm bench serve` keeps no per-token arrivals, so the
 *    lanes replay a model fitted to that run's measured mean TTFT, mean TPOT and
 *    duration, and the frame says so. Every figure printed about the batch run is
 *    the measured one. The time-lapse is labelled with its speed on every frame.
 */
(function () {
  "use strict";

  const W = 1920, H = 1080;
  const SANS = "Inter, 'Helvetica Neue', Arial, sans-serif";
  const MONO = "'JetBrains Mono', Menlo, monospace";

  const C = {
    bg: "#06070A",
    ink: "#EEF0F4",
    ink2: "#D3D7DE",
    dim: "#8C939F",
    dim2: "#565C67",
    line: "rgba(255,255,255,0.075)",
    stock: "#5AA5EB",
    repo: "#FF9E44",
    repoHot: "#FFC27A",
    violet: "#9D8CFF",
  };
  // syntax colours, shared by both lanes so only the accent tells them apart
  const SYN = ["#CDD2DA", "#C5A3FF", "#A6D98A", "#626B7B", "#F4C27B",
               "#7CCBFF", "#F29BB5", "#8C939F", "#4E5460", "#FFFFFF", "#8CC4FF"];
  const S_DEF = 0, S_KW = 1, S_STR = 2, S_COM = 3, S_NUM = 4, S_FN = 5, S_BI = 6,
        S_PUN = 7, S_MARK = 8, S_HEAD = 9, S_LINK = 10;

  // ---------------------------------------------------------------- helpers
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, x) => a + (b - a) * x;
  const inv = (a, b, x) => clamp((x - a) / (b - a));
  const eOut = (x) => 1 - Math.pow(1 - clamp(x), 3);
  const eOut5 = (x) => 1 - Math.pow(1 - clamp(x), 5);
  const eIO = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  const smooth = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
  const fmt = (n, d = 0) => n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

  function upper(arr, x) {               // number of entries <= x in a sorted array
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
    return lo;
  }

  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function mix(h1, h2, x) {
    const a = parseInt(h1.slice(1), 16), b = parseInt(h2.slice(1), 16);
    const ch = (s) => Math.round(lerp((a >> s) & 255, (b >> s) & 255, x));
    return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
  }

  function rr(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function txt(ctx, s, x, y, o) {
    ctx.font = o.font;
    ctx.fillStyle = o.color || C.ink;
    ctx.textAlign = o.align || "left";
    ctx.textBaseline = o.base || "alphabetic";
    ctx.letterSpacing = o.ls || "0px";
    ctx.fillText(s, x, y);
    ctx.letterSpacing = "0px";
  }
  function measure(ctx, s, font, ls) {
    ctx.font = font; ctx.letterSpacing = ls || "0px";
    const w = ctx.measureText(s).width;
    ctx.letterSpacing = "0px";
    return w;
  }

  // Digits in fixed-width slots, so a counting number never jitters sideways.
  const _dw = {};
  function digitW(ctx, font) {
    if (!(font in _dw)) {
      ctx.font = font;
      let w = 0;
      for (let d = 0; d < 10; d++) w += ctx.measureText(String(d)).width;
      _dw[font] = w / 10;
    }
    return _dw[font];
  }
  function numW(ctx, s, font) {
    const dw = digitW(ctx, font);
    ctx.font = font;
    let w = 0;
    for (const ch of s) w += /\d/.test(ch) ? dw : ctx.measureText(ch).width;
    return w;
  }
  // a number that has stopped moving is set normally, with Inter's own spacing
  function numFinal(ctx, s, x, y, font, color, align = "left") {
    ctx.font = font; ctx.fillStyle = color; ctx.textBaseline = "alphabetic"; ctx.textAlign = align;
    ctx.fillText(s, x, y);
    ctx.textAlign = "left";
    return ctx.measureText(s).width;
  }
  function num(ctx, s, x, y, font, color, align = "left") {
    const dw = digitW(ctx, font);
    ctx.font = font; ctx.fillStyle = color; ctx.textBaseline = "alphabetic"; ctx.textAlign = "center";
    const w = numW(ctx, s, font);
    let cx = align === "right" ? x - w : align === "center" ? x - w / 2 : x;
    for (const ch of s) {
      const cw = /\d/.test(ch) ? dw : ctx.measureText(ch).width;
      ctx.fillText(ch, cx + cw / 2, y);
      cx += cw;
    }
    ctx.textAlign = "left";
    return w;
  }

  function pill(ctx, s, x, y, o) {
    const font = o.font || `600 15px ${SANS}`;
    const ls = o.ls || "1.5px";
    const padX = o.padX || 14, h = o.h || 32;
    const w = measure(ctx, s, font, ls) + padX * 2 + (o.dot ? 18 : 0);
    const x0 = o.align === "right" ? x - w : o.align === "center" ? x - w / 2 : x;
    rr(ctx, x0, y, w, h, h / 2);
    ctx.fillStyle = o.fill || "rgba(255,255,255,0.05)";
    ctx.fill();
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = 1; ctx.stroke(); }
    let tx = x0 + padX;
    if (o.dot) {
      ctx.beginPath(); ctx.arc(tx + 4, y + h / 2, 4, 0, Math.PI * 2);
      ctx.fillStyle = o.dot; ctx.fill();
      tx += 18;
    }
    txt(ctx, s, tx, y + h / 2 + 1, { font, color: o.color || C.ink2, base: "middle", ls });
    return w;
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ------------------------------------------------------------- highlight
  const PY_KW = new Set("def class return if else elif for in while not and or is None True False import from as with lambda try except raise pass yield assert break continue global nonlocal".split(" "));
  const PY_BI = new Set("sorted len max min print list int str range enumerate dict set tuple zip map sum abs isinstance".split(" "));

  function hlPython(s, out, from, to) {
    let i = from;
    const isId = (c) => /[A-Za-z0-9_]/.test(c);
    let prevWord = "";
    while (i < to) {
      const c = s[i];
      if (c === "#") { let j = i; while (j < to && s[j] !== "\n") out[j++] = S_COM; i = j; continue; }
      if (c === '"' || c === "'") {
        const tri = s.substr(i, 3) === c + c + c;
        const q = tri ? c + c + c : c;
        let j = i + q.length;
        while (j < to && s.substr(j, q.length) !== q && (tri || s[j] !== "\n")) j++;
        j = Math.min(to, j + q.length);
        for (let k = i; k < j; k++) out[k] = S_STR;
        i = j; continue;
      }
      if (/[0-9]/.test(c) && !(i > 0 && isId(s[i - 1]))) {
        let j = i; while (j < to && /[0-9.]/.test(s[j])) out[j++] = S_NUM; i = j; continue;
      }
      if (/[A-Za-z_]/.test(c)) {
        let j = i; while (j < to && isId(s[j])) j++;
        const w = s.slice(i, j);
        let k = j; while (k < to && s[k] === " ") k++;
        const st = PY_KW.has(w) ? S_KW : prevWord === "def" || prevWord === "class" ? S_FN
          : PY_BI.has(w) ? S_BI : s[k] === "(" ? S_FN : S_DEF;
        for (let m = i; m < j; m++) out[m] = st;
        prevWord = w; i = j; continue;
      }
      out[i] = /[()[\]{}:,.=<>+\-*/]/.test(c) ? S_PUN : S_DEF;
      i++;
    }
  }

  function highlight(s, kind) {
    const out = new Uint8Array(s.length);
    let pos = 0, inFence = false, fenceLang = "";
    for (const line of s.split("\n")) {
      const a = pos, b = pos + line.length;
      if (line.startsWith("```")) {
        out.fill(S_MARK, a, b);
        inFence = !inFence; fenceLang = line.slice(3).trim();
      } else if (inFence && (fenceLang === "python" || fenceLang === "py")) {
        hlPython(s, out, a, b);
      } else if (inFence) {
        out.fill(S_DEF, a, b);
      } else if (kind === "code") {
        hlPython(s, out, a, b);
      } else if (/^#{1,6} /.test(line)) {
        const h = line.indexOf(" ");
        out.fill(S_MARK, a, a + h); out.fill(S_HEAD, a + h, b);
      } else {
        out.fill(S_DEF, a, b);
        // tables, inline code, bold, links
        for (let i = a; i < b; i++) if (s[i] === "|") out[i] = S_MARK;
        const re = /`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\([^)\n]+\)/g;
        let m;
        const seg = s.slice(a, b);
        while ((m = re.exec(seg))) {
          const x = a + m.index, y = x + m[0].length;
          if (m[0][0] === "`") out.fill(S_STR, x, y);
          else if (m[0][0] === "*") { out.fill(S_HEAD, x, y); out.fill(S_MARK, x, x + 2); out.fill(S_MARK, y - 2, y); }
          else {
            const cut = m[0].indexOf("](");
            out.fill(S_LINK, x, x + cut + 1); out.fill(S_MARK, x + cut + 1, y);
          }
        }
        if (/^\s*([*-]|\d+\.)\s/.test(line)) {
          const mm = line.match(/^\s*([*-]|\d+\.)/);
          out.fill(S_MARK, a, a + mm[0].length);
        }
      }
      pos = b + 1;
    }
    return out;
  }

  // Greedy word wrap of the FINAL text, done once. Revealing a prefix of a stable
  // layout means a line never re-flows while the viewer is reading it.
  function wrap(s, cols) {
    const row = new Int32Array(s.length), col = new Int32Array(s.length);
    let r = 0, pos = 0;
    for (const line of s.split("\n")) {
      const indent = Math.min((line.match(/^ */)[0].length), 12);
      let i = 0, c = 0, first = true;
      while (i < line.length) {
        const avail = first ? cols : cols - indent - 2;
        let end = Math.min(line.length, i + avail);
        if (end < line.length) {
          const sp = line.lastIndexOf(" ", end);
          if (sp > i + 8) end = sp + 1;
        }
        c = first ? 0 : indent + 2;
        for (let k = i; k < end; k++) { row[pos + k] = r; col[pos + k] = c++; }
        i = end; first = false;
        if (i < line.length) r++;
      }
      if (line.length === 0) { /* empty line */ }
      if (pos + line.length < s.length) { row[pos + line.length] = r; col[pos + line.length] = -1; }
      pos += line.length + 1; r++;
    }
    return { row, col, rows: r };
  }

  // ------------------------------------------------------------------ film
  function create(DATA) {
    const single = DATA.single, B = DATA.batch;
    const P = {};
    for (const p of single.prompts) P[p.key] = p;

    const film = { W, H, duration: 0 };
    const scenes = [];
    let cursor = 0;
    function add(name, dur, draw, overlap = 0) {
      const start = Math.max(0, cursor - (scenes.length ? overlap : 0));
      scenes.push({ name, start, dur, end: start + dur, draw });
      cursor = start + dur;
      return scenes[scenes.length - 1];
    }

    // ----- lanes for the single-request races
    const BODY_FONT_PX = 16, LH = 23.5;
    let CW = 9.6;                        // measured once fonts are in
    function prepLane(L, kind) {
      const text = L.chunks.map((c) => c[1]).join("");
      const times = [], endChar = [], cumTok = [], nTok = [];
      let acc = 0, tok = 0;
      for (const [ms, s, n] of L.chunks) {
        acc += s.length; tok += n;
        times.push(ms / 1000); endChar.push(acc); cumTok.push(tok); nTok.push(n);
      }
      const charT = new Float32Array(text.length);
      let ci = 0;
      for (let i = 0; i < text.length; i++) { while (i >= endChar[ci]) ci++; charT[i] = times[ci]; }
      return { L, text, times, endChar, cumTok, nTok, charT, colors: highlight(text, kind),
               last: times[times.length - 1], total: tok, layout: null, cols: 0 };
    }
    const lanes = {};
    for (const k of ["code", "copy", "chat"]) {
      lanes[k] = { stock: prepLane(P[k].stock, "md"), repo: prepLane(P[k].repo, "md") };
    }
    const GMAX = 400;                  // one shared tok/s scale for every bar in the film

    function laneAt(ln, T) {
      const i = upper(ln.times, T);
      return { i, chars: i ? ln.endChar[i - 1] : 0, tokens: i ? ln.cumTok[i - 1] : 0,
               done: i === ln.times.length };
    }

    // running decode rate, same convention as demo_capture.py: (n-1)/(t since first)
    function rateAt(ln, T, st) {
      if (st.done) return ln.L.decode_tok_s;
      if (st.i < 2 || T < 0.1) return null;
      return (st.tokens - 1) / T;
    }

    // ----- offscreen buffer for masked text bodies
    let off = null, offCtx = null;
    // w, h in film pixels; sc is the device scale of the target, so the live player
    // stays sharp on a high-DPI screen (the video renders at sc = 1)
    function offscreen(w, h, sc) {
      if (!off) { off = document.createElement("canvas"); offCtx = off.getContext("2d"); }
      const pw = Math.ceil(w * sc), ph = Math.ceil(h * sc);
      if (off.width !== pw || off.height !== ph) { off.width = pw; off.height = ph; }
      offCtx.setTransform(1, 0, 0, 1, 0, 0);
      offCtx.clearRect(0, 0, pw, ph);
      offCtx.setTransform(sc, 0, 0, sc, 0, 0);
      return offCtx;
    }

    // ------------------------------------------------------------ backdrop
    let bgCanvas = null;
    function backdrop(ctx) {
      if (!bgCanvas) {
        bgCanvas = document.createElement("canvas");
        bgCanvas.width = W; bgCanvas.height = H;
        paintBackdrop(bgCanvas.getContext("2d"));
      }
      ctx.drawImage(bgCanvas, 0, 0, W, H);
    }
    function paintBackdrop(ctx) {
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
      const g1 = ctx.createRadialGradient(1500, -160, 60, 1500, -160, 1250);
      g1.addColorStop(0, rgba(C.repo, 0.12));
      g1.addColorStop(1, rgba(C.repo, 0));
      ctx.fillStyle = g1; ctx.fillRect(0, 0, W, H);
      const g2 = ctx.createRadialGradient(260, 1220, 40, 260, 1220, 1150);
      g2.addColorStop(0, rgba(C.stock, 0.075));
      g2.addColorStop(1, rgba(C.stock, 0));
      ctx.fillStyle = g2; ctx.fillRect(0, 0, W, H);
      // faint engineering grid, fading toward the edges
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.028)"; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x <= W; x += 64) { ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); }
      for (let y = 0; y <= H; y += 64) { ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); }
      ctx.stroke();
      const v = ctx.createRadialGradient(W / 2, H / 2, 300, W / 2, H / 2, 1150);
      v.addColorStop(0, "rgba(6,7,10,0)"); v.addColorStop(1, "rgba(6,7,10,0.92)");
      ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
      ctx.restore();
    }

    function logo(ctx, x, y, s, a = 1) {
      ctx.save(); ctx.globalAlpha *= a;
      const g = ctx.createLinearGradient(x, y, x + s, y + s);
      g.addColorStop(0, "#FFB15E"); g.addColorStop(1, "#FF7A2F");
      rr(ctx, x, y, s, s, s * 0.28); ctx.fillStyle = g; ctx.fill();
      // three speed bars, stepping up: tokens per step
      ctx.fillStyle = "#1A0E04";
      const bw = s * 0.14, gap = s * 0.08, base = y + s * 0.74;
      const hs = [0.22, 0.36, 0.5];
      let bx = x + s * 0.23;
      for (const h of hs) { rr(ctx, bx, base - s * h, bw, s * h, bw * 0.35); ctx.fill(); bx += bw + gap; }
      ctx.restore();
    }

    function chrome(ctx, a, sceneName) {
      if (a <= 0) return;
      ctx.save(); ctx.globalAlpha = a;
      logo(ctx, 72, 44, 34);
      txt(ctx, "HyperQwen", 118, 70, { font: `700 24px ${SANS}`, color: C.ink, ls: "-0.3px" });
      txt(ctx, "Qwen3.8-27B  ·  one RTX 3090  ·  24 GB  ·  250 W", W - 72, 68,
          { font: `500 17px ${SANS}`, color: C.dim, align: "right", ls: "0.3px" });
      ctx.restore();
    }

    function stamp(ctx, lines, a) {
      if (a <= 0) return;
      ctx.save(); ctx.globalAlpha = a;
      ctx.fillStyle = C.line; ctx.fillRect(72, 1000, W - 144, 1);
      lines.forEach((s, i) => txt(ctx, s, 72, 1030 + i * 22, { font: `400 15px ${SANS}`, color: C.dim }));
      ctx.restore();
    }

    // chapter progress, bottom right
    let chapters = [];
    function progress(ctx, t, a) {
      if (a <= 0 || !chapters.length) return;
      ctx.save(); ctx.globalAlpha = a;
      let x = W - 72;
      for (let i = chapters.length - 1; i >= 0; i--) {
        const ch = chapters[i];
        const f = inv(ch.a, ch.b, t);
        const on = t >= ch.a && t < ch.b;
        const w = Math.max(120, measure(ctx, ch.label, `600 13px ${SANS}`, "1.2px"));
        x -= w;
        rr(ctx, x, 1026, w, 4, 2); ctx.fillStyle = "rgba(255,255,255,0.10)"; ctx.fill();
        if (f > 0) { rr(ctx, x, 1026, w * f, 4, 2); ctx.fillStyle = on ? C.repo : rgba(C.repo, 0.55); ctx.fill(); }
        txt(ctx, ch.label, x, 1054, { font: `600 13px ${SANS}`, color: on ? C.ink2 : C.dim2, ls: "1.2px" });
        x -= 22;
      }
      ctx.restore();
    }

    // ------------------------------------------------------------ scene: intro
    add("intro", 4.4, (ctx, t, a) => {
      ctx.save(); ctx.globalAlpha = a;
      const k = eOut(inv(0.15, 1.3, t));
      const cy = 468;
      const titleFont = `800 148px ${SANS}`;
      const tw = measure(ctx, "HyperQwen", titleFont, "-4px");
      const ls = 124, gap = 40;
      const x0 = W / 2 - (tw + ls + gap) / 2;
      ctx.save();
      ctx.globalAlpha *= k;
      ctx.translate(0, (1 - k) * 26);
      // glow behind the mark
      const gg = ctx.createRadialGradient(x0 + ls / 2, cy - 50, 10, x0 + ls / 2, cy - 50, 260);
      gg.addColorStop(0, rgba(C.repo, 0.35 * k)); gg.addColorStop(1, rgba(C.repo, 0));
      ctx.fillStyle = gg; ctx.fillRect(x0 - 300, cy - 360, 700, 620);
      logo(ctx, x0, cy - 112, ls);
      txt(ctx, "HyperQwen", x0 + ls + gap, cy, { font: titleFont, color: C.ink, ls: "-4px" });
      ctx.restore();

      const k2 = eOut(inv(0.8, 1.8, t));
      ctx.save(); ctx.globalAlpha *= k2; ctx.translate(0, (1 - k2) * 14);
      txt(ctx, "Large Qwen models, served fast on the GPUs people actually own.", W / 2, cy + 92,
          { font: `400 36px ${SANS}`, color: C.dim, align: "center" });
      ctx.restore();

      {
        const ln = lanes.copy.repo, nb = ln.nTok.length;
        const span = 1180, bw = span / nb, base = 860;
        for (let i = 0; i < nb; i++) {
          const kk = eOut(inv(0.9 + i * 0.022, 1.5 + i * 0.022, t));
          if (kk <= 0) break;
          const h = (Math.min(16, ln.nTok[i]) / 16) * 110 * kk;
          ctx.fillStyle = rgba(C.repo, 0.10 + 0.22 * (ln.nTok[i] / 16));
          ctx.fillRect(W / 2 - span / 2 + i * bw + 1, base - h, Math.max(1, bw - 3), h);
        }
        const kc = eOut(inv(2.2, 3.0, t));
        txt(ctx, `tokens per step in one recorded request: ${fmt(ln.total)} tokens in ${nb} steps`, W / 2, base + 34,
            { font: `500 16px ${SANS}`, color: rgba("#8C939F", kc), align: "center", ls: "0.3px" });
      }
      const chips = ["Qwen3.8-27B", "one RTX 3090", "24 GB", "250 W power limit", "vLLM + a patch series"];
      ctx.font = `600 17px ${SANS}`;
      let widths = chips.map((c) => measure(ctx, c.toUpperCase(), `600 16px ${SANS}`, "1.6px") + 36);
      let total = widths.reduce((p, q) => p + q, 0) + 14 * (chips.length - 1);
      let cx = W / 2 - total / 2;
      chips.forEach((c, i) => {
        const kk = eOut(inv(1.35 + i * 0.09, 2.1 + i * 0.09, t));
        ctx.save(); ctx.globalAlpha *= kk; ctx.translate(0, (1 - kk) * 10);
        pill(ctx, c.toUpperCase(), cx, cy + 150, { font: `600 16px ${SANS}`, ls: "1.6px", h: 38, padX: 18,
          fill: "rgba(255,255,255,0.045)", stroke: "rgba(255,255,255,0.09)", color: C.ink2 });
        ctx.restore();
        cx += widths[i] + 14;
      });
      ctx.restore();
    }, 0);

    // --------------------------------------------------------- chapter cards
    function chapterCard(no, title, sub) {
      return (ctx, t, a) => {
        ctx.save(); ctx.globalAlpha = a;
        const k = eOut5(inv(0.05, 0.9, t));
        const x = 200;
        ctx.save(); ctx.globalAlpha *= k;
        txt(ctx, no, x, 520, { font: `800 220px ${SANS}`, color: rgba(C.repo, 0.95), ls: "-8px" });
        ctx.restore();
        const nw = measure(ctx, no, `800 220px ${SANS}`, "-8px");
        const lx = x + nw + 60;
        ctx.fillStyle = rgba(C.repo, 0.6);
        ctx.fillRect(lx, 382, 2, 150 * eOut(inv(0.15, 0.8, t)));
        const k2 = eOut(inv(0.2, 1.0, t));
        ctx.save(); ctx.globalAlpha *= k2; ctx.translate((1 - k2) * 30, 0);
        txt(ctx, title, lx + 50, 468, { font: `700 92px ${SANS}`, color: C.ink, ls: "-2px" });
        ctx.restore();
        const k3 = eOut(inv(0.45, 1.25, t));
        ctx.save(); ctx.globalAlpha *= k3; ctx.translate((1 - k3) * 30, 0);
        txt(ctx, sub, lx + 52, 526, { font: `400 30px ${SANS}`, color: C.dim });
        ctx.restore();
        ctx.restore();
      };
    }

    add("ch1", 2.2, chapterCard("01", "One request",
      "Stock vLLM against this repo. Same card, same model, same prompt."));

    // ----------------------------------------------------------- race scene
    const PANEL = { y: 206, h: 770, w: 872, gap: 32 };
    PANEL.xL = 72; PANEL.xR = 72 + PANEL.w + PANEL.gap;
    const BODY = { dx: 28, dy: 116, w: PANEL.w - 56, h: 336, pad: 16 };

    function drawBody(ctx, ln, T, px, py, accent, streaming) {
      const bx = px + BODY.dx, by = py + BODY.dy, bw = BODY.w, bh = BODY.h;
      rr(ctx, bx, by, bw, bh, 14);
      ctx.fillStyle = "rgba(0,0,0,0.34)"; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1; ctx.stroke();

      const cols = Math.floor((bw - BODY.pad * 2) / CW);
      if (!ln.layout || ln.cols !== cols) { ln.layout = wrap(ln.text, cols); ln.cols = cols; }
      const { row, col } = ln.layout;
      const st = laneAt(ln, T);
      const n = st.chars;
      const visRows = Math.floor((bh - BODY.pad * 2) / LH);

      // scroll: follow the cursor, smoothed by averaging the target over the last 0.3 s
      const target = (TT) => {
        const c = laneAt(ln, TT).chars;
        const r = c ? row[c - 1] : 0;
        return Math.max(0, r - (visRows - 3));
      };
      let top = 0;
      for (let k = 0; k < 8; k++) top += target(T - 0.3 * k / 7);
      top /= 8;

      const sc = ctx.getTransform().a;
      const o = offscreen(Math.ceil(bw), Math.ceil(bh), sc);
      o.font = `400 ${BODY_FONT_PX}px ${MONO}`;
      o.textBaseline = "alphabetic";
      const firstRow = Math.floor(top) - 1, lastRow = Math.ceil(top) + visRows + 1;
      // locate the first char of firstRow with a binary search on row[]
      let lo = 0, hi = n;
      while (lo < hi) { const m = (lo + hi) >> 1; if (row[m] < firstRow) lo = m + 1; else hi = m; }
      const yOf = (r) => BODY.pad + (r - top) * LH;

      // arrival glow: everything that landed in the last 0.45 s gets a tinted cell
      const FRESH = 0.45;
      for (let i = lo; i < n; i++) {
        const r = row[i]; if (r > lastRow) break;
        const age = T - ln.charT[i];
        if (age < FRESH && col[i] >= 0 && ln.text[i] !== " ") {
          let j = i;
          while (j + 1 < n && row[j + 1] === r && col[j + 1] >= 0 && ln.charT[j + 1] === ln.charT[i]) j++;
          const f = 1 - age / FRESH;
          o.fillStyle = rgba(accent, 0.30 * f * f);
          o.fillRect(BODY.pad + col[i] * CW - 1, yOf(r) + 3, (col[j] - col[i] + 1) * CW + 2, LH - 2);
          i = j;
        }
      }
      // glyphs, in runs of one colour
      for (let i = lo; i < n;) {
        const r = row[i]; if (r > lastRow) break;
        if (col[i] < 0) { i++; continue; }
        const cidx = ln.colors[i];
        let j = i;
        while (j + 1 < n && row[j + 1] === r && col[j + 1] === col[j] + 1 && ln.colors[j + 1] === cidx) j++;
        const age = T - ln.charT[i];
        o.fillStyle = age < 0.18 ? mix(SYN[cidx], "#FFFFFF", 1 - age / 0.18) : SYN[cidx];
        o.fillText(ln.text.slice(i, j + 1), BODY.pad + col[i] * CW, yOf(r) + LH - 6);
        i = j + 1;
      }
      // caret
      if (streaming) {
        const c = n ? n - 1 : 0;
        let cr = n ? row[c] : 0, cc = n ? (col[c] < 0 ? -1 : col[c]) + 1 : 0;
        if (n && ln.text[c] === "\n") { cr = row[c] + 1; cc = 0; }
        const blink = 0.55 + 0.45 * Math.cos(T * 9);
        o.fillStyle = rgba(accent, blink);
        o.fillRect(BODY.pad + cc * CW + 1, yOf(cr) + 5, CW - 2, LH - 6);
      }
      // soft fade at the top and bottom edges
      o.globalCompositeOperation = "destination-in";
      const m = o.createLinearGradient(0, 0, 0, bh);
      m.addColorStop(0, "rgba(0,0,0,0)"); m.addColorStop(0.07, "rgba(0,0,0,1)");
      m.addColorStop(0.9, "rgba(0,0,0,1)"); m.addColorStop(1, "rgba(0,0,0,0.15)");
      o.fillStyle = m; o.fillRect(0, 0, bw, bh);
      o.globalCompositeOperation = "source-over";
      ctx.drawImage(off, bx, by, off.width / sc, off.height / sc);
      return st;
    }

    function drawSteps(ctx, ln, T, x, y, w, h, axisS, accent) {
      // one bar per streamed step, on a shared time axis; height = tokens it carried
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(x, y + h, w, 1);
      for (const v of [8, 16]) {
        const yy = y + h - (v / 16) * h;
        ctx.fillStyle = "rgba(255,255,255,0.04)"; ctx.fillRect(x, Math.round(yy), w, 1);
      }
      const i1 = upper(ln.times, T);
      const bw = Math.max(1.6, Math.min(5, w / (axisS * 60)));
      for (let i = 0; i < i1; i++) {
        const tt = ln.times[i];
        if (tt > axisS) break;
        const bx = x + (tt / axisS) * (w - bw);
        const n = Math.max(1, ln.nTok[i]);
        const bh = Math.max(2, (Math.min(16, n) / 16) * h);
        const age = T - tt;
        ctx.fillStyle = age < 0.25 ? mix(accent, "#FFFFFF", 0.6 * (1 - age / 0.25)) : rgba(accent, 0.78);
        ctx.fillRect(bx, y + h - bh, bw, bh);
      }
    }

    function drawPanel(ctx, t, T, ln, side, opts) {
      const px = side === "L" ? PANEL.xL : PANEL.xR, py = PANEL.y;
      const accent = side === "L" ? C.stock : C.repo;
      const st0 = laneAt(ln, T);
      const started = T >= 0;
      const done = st0.done;

      rr(ctx, px, py, PANEL.w, PANEL.h, 22);
      const pg = ctx.createLinearGradient(px, py, px, py + PANEL.h);
      pg.addColorStop(0, "rgba(255,255,255,0.045)"); pg.addColorStop(1, "rgba(255,255,255,0.02)");
      ctx.fillStyle = pg; ctx.fill();
      ctx.strokeStyle = done && side === "R" ? rgba(C.repo, 0.35) : "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1.2; ctx.stroke();
      // accent hairline on top
      const hl = ctx.createLinearGradient(px, 0, px + PANEL.w, 0);
      hl.addColorStop(0, rgba(accent, 0)); hl.addColorStop(0.5, rgba(accent, 0.8)); hl.addColorStop(1, rgba(accent, 0));
      ctx.fillStyle = hl; ctx.fillRect(px + 40, py, PANEL.w - 80, 1.5);

      // header
      ctx.beginPath(); ctx.arc(px + 36, py + 50, 7, 0, Math.PI * 2); ctx.fillStyle = accent; ctx.fill();
      txt(ctx, opts.name, px + 56, py + 60, { font: `700 30px ${SANS}`, color: C.ink, ls: "-0.4px" });
      txt(ctx, opts.cmd, px + 30, py + 94, { font: `400 16px ${MONO}`, color: C.dim });
      let status, sfill, scol, sdot;
      if (!started) { status = "READY"; sfill = "rgba(255,255,255,0.05)"; scol = C.dim; sdot = C.dim2; }
      else if (done) { status = "DONE"; sfill = rgba(accent, 0.16); scol = side === "R" ? C.repoHot : "#9CCBF5"; sdot = accent; }
      else { status = "STREAMING"; sfill = "rgba(255,255,255,0.06)"; scol = C.ink2;
             sdot = rgba(accent, 0.55 + 0.45 * Math.cos(t * 7)); }
      pill(ctx, status, px + PANEL.w - 28, py + 34, { align: "right", fill: sfill, color: scol, dot: sdot,
        font: `700 14px ${SANS}`, ls: "1.6px", h: 32 });

      const st = drawBody(ctx, ln, Math.max(-1, T), px, py, accent, started && !done);

      // metrics
      const my = py + BODY.dy + BODY.h;
      const rate = started ? rateAt(ln, T, st) : null;
      const heroFont = `800 100px ${SANS}`;
      const heroStr = rate == null ? "0" : fmt(rate, rate < 100 ? 1 : 0);
      const hw = (done ? numFinal : num)(ctx, heroStr, px + 30, my + 118, heroFont,
                     rate == null ? C.dim2 : done ? (side === "R" ? C.repoHot : C.ink) : C.ink);
      txt(ctx, "tok/s", px + 30 + hw + 14, my + 118, { font: `600 30px ${SANS}`, color: C.dim });
      txt(ctx, done ? "DECODE, FULL-RUN AVERAGE" : "DECODE, RUNNING AVERAGE", px + 32, my + 34,
          { font: `600 13px ${SANS}`, color: C.dim2, ls: "1.6px" });

      // stats to the right
      const sx = px + PANEL.w - 30;
      const tokStr = fmt(st.tokens), totStr = fmt(ln.total);
      num(ctx, tokStr, sx, my + 76, `600 30px ${SANS}`, C.ink, "right");
      txt(ctx, "tokens", sx, my + 100, { font: `500 15px ${SANS}`, color: C.dim, align: "right" });
      const secs = started ? Math.min(T, ln.last) : 0;
      num(ctx, fmt(secs, 2) + " s", sx - 170, my + 76, `600 30px ${SANS}`, done ? accent : C.ink, "right");
      txt(ctx, done ? "to the last token" : "since the first token", sx - 170, my + 100,
          { font: `500 15px ${SANS}`, color: C.dim, align: "right" });

      // rate bar on the shared scale
      const barY = my + 146, barX = px + 30, barW = PANEL.w - 60;
      rr(ctx, barX, barY, barW, 10, 5); ctx.fillStyle = "rgba(255,255,255,0.07)"; ctx.fill();
      if (rate != null) {
        const f = clamp(rate / GMAX);
        const bg = ctx.createLinearGradient(barX, 0, barX + barW * f, 0);
        bg.addColorStop(0, rgba(accent, 0.45)); bg.addColorStop(1, accent);
        rr(ctx, barX, barY, Math.max(10, barW * f), 10, 5); ctx.fillStyle = bg; ctx.fill();
      }
      for (let v = 0; v <= GMAX; v += 100) {
        const xx = barX + (v / GMAX) * barW;
        txt(ctx, String(v), xx, barY + 30, { font: `500 13px ${SANS}`, color: C.dim2,
          align: v === 0 ? "left" : v === GMAX ? "right" : "center" });
      }

      // steps strip
      const sy = barY + 52;
      txt(ctx, "TOKENS PER STEP", barX, sy + 4, { font: `600 12px ${SANS}`, color: C.dim2, ls: "1.6px" });
      txt(ctx, "one bar per forward pass, height = tokens it produced (max 16)", barX + barW, sy + 4,
          { font: `400 13px ${SANS}`, color: C.dim2, align: "right" });
      drawSteps(ctx, ln, Math.max(-1, T), barX, sy + 14, barW, 62, opts.axisS, accent);
      return st;
    }

    function raceScene(key, o) {
      const L = lanes[key];
      const lead = 0.8;
      const window_ = o.window || Math.max(L.stock.last, L.repo.last);
      const hold = o.hold;
      const dur = lead + window_ + hold;
      const s = add("race-" + key, dur, (ctx, t, a) => {
        ctx.save(); ctx.globalAlpha = a;
        // a lane that finished inside the window keeps its own clock running (so its
        // arrival glow fades); one still streaming is frozen where the window closed
        const Tof = (ln) => ln.last <= window_ + 1e-6 ? t - lead : Math.min(t - lead, window_);
        const k = eOut(inv(0, 0.6, t));
        // header
        ctx.save(); ctx.globalAlpha *= k;
        pill(ctx, o.tag, 72, 112, { fill: rgba(C.repo, 0.13), color: C.repoHot, font: `700 14px ${SANS}`, ls: "1.8px", h: 32 });
        const tw = measure(ctx, o.tag, `700 14px ${SANS}`, "1.8px") + 28;
        txt(ctx, o.prompt, 72 + tw + 18, 135, { font: `500 23px ${SANS}`, color: C.ink2 });
        txt(ctx, o.meta, W - 72, 135, { font: `500 17px ${SANS}`, color: C.dim, align: "right" });
        ctx.restore();

        ctx.save(); ctx.globalAlpha *= k; ctx.translate(0, (1 - k) * 24);
        drawPanel(ctx, t, Tof(L.stock), L.stock, "L", { name: "Stock vLLM", cmd: "vllm serve   # no speculative decoding", axisS: window_ });
        drawPanel(ctx, t, Tof(L.repo), L.repo, "R", { name: "HyperQwen", cmd: "SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1", axisS: window_ });
        ctx.restore();

        // a lane that is still running when the window closes says where it would end
        const hk = eOut(inv(lead + window_, lead + window_ + 0.7, t));
        if (hk > 0 && L.stock.last > window_ + 0.01) {
          ctx.save(); ctx.globalAlpha *= hk;
          const px = PANEL.xL, bx = px + BODY.dx, by = PANEL.y + BODY.dy;
          rr(ctx, bx, by, BODY.w, BODY.h, 14);
          ctx.fillStyle = "#0B0C10"; ctx.fill();
          const frac = laneAt(L.stock, window_).tokens / L.stock.total;
          const lx = bx + 44;
          txt(ctx, "STILL STREAMING", lx, by + 104, { font: `700 15px ${SANS}`, color: "#9CCBF5", ls: "2px" });
          txt(ctx, `${Math.round(frac * 100)}% written`, lx, by + 158, { font: `700 44px ${SANS}`, color: C.ink, ls: "-0.8px" });
          txt(ctx, `finishes at ${fmt(L.stock.last, 1)} s`, lx, by + 198, { font: `500 26px ${SANS}`, color: C.ink2 });
          const pw = 520;
          rr(ctx, lx, by + 228, pw, 8, 4); ctx.fillStyle = "rgba(255,255,255,0.08)"; ctx.fill();
          rr(ctx, lx, by + 228, pw * frac, 8, 4); ctx.fillStyle = C.stock; ctx.fill();
          txt(ctx, "Same recording. The video stops watching here.", lx, by + 276,
              { font: `400 17px ${SANS}`, color: C.dim });
          ctx.restore();
        }

        // verdict badge once the result is in; the text bodies step back behind it
        const bk = eOut5(inv(lead + window_ + 0.25, lead + window_ + 1.05, t));
        if (bk > 0) {
          ctx.save(); ctx.globalAlpha *= bk;
          for (const px of [PANEL.xL, PANEL.xR]) {
            rr(ctx, px + BODY.dx, PANEL.y + BODY.dy, BODY.w, BODY.h, 14);
            ctx.fillStyle = "rgba(6,7,10,0.5)"; ctx.fill();
          }
          ctx.restore();
          const ratio = L.repo.L.decode_tok_s / L.stock.L.decode_tok_s;
          ctx.save(); ctx.globalAlpha *= bk;
          const cx = W / 2, cy = PANEL.y + BODY.dy + BODY.h / 2 + 8;
          const sc = lerp(0.86, 1, bk);
          ctx.translate(cx, cy); ctx.scale(sc, sc);
          const bw = 400, bh = 214;
          ctx.shadowColor = "rgba(0,0,0,0.6)"; ctx.shadowBlur = 50;
          rr(ctx, -bw / 2, -bh / 2, bw, bh, 28);
          ctx.fillStyle = "rgba(16,17,21,0.94)"; ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = rgba(C.repo, 0.55); ctx.lineWidth = 1.5; ctx.stroke();
          const g = ctx.createLinearGradient(0, -80, 0, 40);
          g.addColorStop(0, "#FFD29A"); g.addColorStop(1, C.repo);
          const rs = fmt(ratio, 1) + "×";
          txt(ctx, rs, 0, 30, { font: `800 118px ${SANS}`, color: g, align: "center", ls: "-3px" });
          txt(ctx, "faster decode", 0, 74, { font: `600 22px ${SANS}`, color: C.ink2, align: "center" });
          txt(ctx, `${fmt(L.repo.L.decode_tok_s, 1)} vs ${fmt(L.stock.L.decode_tok_s, 1)} tok/s`, 0, -78,
              { font: `600 15px ${SANS}`, color: C.dim, align: "center", ls: "1px" });
          ctx.restore();
        }

        stamp(ctx, o.stamp, k);
        ctx.restore();
      });
      return s;
    }

    const SINGLE_STAMP = [
      `Real time: both lanes replayed from recorded token arrivals, one RTX 3090 at 250 W, greedy, recorded ${single.date} (vLLM 0.27.1). Each clock starts at its lane's first token.`,
      "The lanes ran one after the other, not at once. The HyperQwen lane also serves an int4 lm_head, so the two answers are not word-for-word identical.",
    ];

    const race1 = raceScene("code", {
      tag: "PROMPT", hold: 2.6,
      prompt: "“Write a Python function that merges overlapping intervals. Include a short docstring and three test cases.”",
      meta: `${P.code.repo.prompt_tokens} prompt tokens`,
      stamp: SINGLE_STAMP,
    });

    add("copy-card", 2.5, (ctx, t, a) => {
      ctx.save(); ctx.globalAlpha = a;
      const k = eOut(inv(0.05, 0.8, t)), k2 = eOut(inv(0.35, 1.1, t));
      ctx.save(); ctx.globalAlpha *= k; ctx.translate(0, (1 - k) * 18);
      txt(ctx, "NEXT PROMPT", W / 2, 420, { font: `700 16px ${SANS}`, color: C.repoHot, align: "center", ls: "3px" });
      txt(ctx, "A 25,000-token document, and an answer that quotes it.", W / 2, 506,
          { font: `700 62px ${SANS}`, color: C.ink, align: "center", ls: "-1.2px" });
      ctx.restore();
      ctx.save(); ctx.globalAlpha *= k2;
      txt(ctx, "Code edits, RAG, rewrites and translation all copy long runs of their prompt.", W / 2, 572,
          { font: `400 30px ${SANS}`, color: C.dim, align: "center" });
      txt(ctx, "HyperQwen drafts those runs straight out of the prompt and checks up to 16 tokens per step.", W / 2, 616,
          { font: `400 30px ${SANS}`, color: C.dim, align: "center" });
      ctx.restore();
      ctx.restore();
    });

    const race2 = raceScene("copy", {
      tag: "PROMPT", hold: 3.0, window: 7.0,
      prompt: "[25,160-token document]  “Reproduce the first 60 lines of the document verbatim.”",
      meta: `prefill (${fmt(P.copy.repo.ttft_s, 1)} s on both lanes) not shown`,
      stamp: SINGLE_STAMP,
    });

    // ------------------------------------------------------ single summary
    add("single-summary", 4.8, (ctx, t, a) => {
      ctx.save(); ctx.globalAlpha = a;
      const k = eOut(inv(0.0, 0.7, t));
      ctx.save(); ctx.globalAlpha *= k;
      txt(ctx, "ONE REQUEST, THREE PROMPTS", 200, 214, { font: `700 16px ${SANS}`, color: C.repoHot, ls: "3px" });
      txt(ctx, "Decode speed, full-run averages", 200, 282, { font: `700 60px ${SANS}`, color: C.ink, ls: "-1.2px" });
      ctx.restore();
      ctx.save(); ctx.globalAlpha *= eOut(inv(0.2, 0.9, t));
      let lx = 640;
      for (const [nm, col] of [["Stock vLLM", C.stock], ["HyperQwen", C.repo]]) {
        ctx.beginPath(); ctx.arc(lx + 6, 358, 6, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
        txt(ctx, nm, lx + 20, 365, { font: `600 20px ${SANS}`, color: C.ink2 });
        lx += measure(ctx, nm, `600 20px ${SANS}`) + 56;
      }
      txt(ctx, "tok/s", 640 + 900, 365, { font: `600 20px ${SANS}`, color: C.dim, align: "right" });
      ctx.restore();
      const rows = [["chat", "Chat answer"], ["code", "Write code"], ["copy", "Quote a 25k-token document"]];
      const x0 = 200, x1 = 640, bw = 900, rowH = 150;
      rows.forEach(([key, label], i) => {
        const y = 420 + i * rowH;
        const kk = eOut(inv(0.25 + i * 0.18, 1.0 + i * 0.18, t));
        const kb = eIO(inv(0.45 + i * 0.18, 1.6 + i * 0.18, t));
        const s = P[key].stock, r = P[key].repo;
        ctx.save(); ctx.globalAlpha *= kk;
        txt(ctx, label, x0, y + 30, { font: `600 30px ${SANS}`, color: C.ink });
        txt(ctx, `${fmt(r.n_out)} tokens out`, x0, y + 62, { font: `400 19px ${SANS}`, color: C.dim });
        for (const [j, v, col, nm] of [[0, s.decode_tok_s, C.stock, "stock vLLM"], [1, r.decode_tok_s, C.repo, "HyperQwen"]]) {
          const yy = y + 8 + j * 44;
          rr(ctx, x1, yy, bw, 26, 13); ctx.fillStyle = "rgba(255,255,255,0.05)"; ctx.fill();
          const f = (v / GMAX) * kb;
          const g = ctx.createLinearGradient(x1, 0, x1 + bw * f, 0);
          g.addColorStop(0, rgba(col, 0.5)); g.addColorStop(1, col);
          rr(ctx, x1, yy, Math.max(26, bw * f), 26, 13); ctx.fillStyle = g; ctx.fill();
          if (kb > 0.02) num(ctx, fmt(v * kb, 1), x1 + Math.max(26, bw * f) + 16, yy + 21, `700 22px ${SANS}`, C.ink);
        }
        const ratio = r.decode_tok_s / s.decode_tok_s;
        txt(ctx, fmt(ratio, 1) + "×", W - 200, y + 60, { font: `800 64px ${SANS}`, color: mix(C.ink, C.repo, 0.4 + 0.2 * i), align: "right", ls: "-1.5px" });
        ctx.restore();
      });
      ctx.restore();
      stamp(ctx, [`tok/s after the first token. Greedy, one RTX 3090 at 250 W, recorded ${single.date}. HyperQwen lane: SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1.`,
                  "All three bars share one 0–400 tok/s scale, and each ratio is computed from the recordings."], a * eOut(inv(0, 0.6, t)));
    });

    add("ch2", 2.2, chapterCard("02", "Sixty-four at once",
      `Batch mode: ${B.requests} requests, ${B.concurrency} in flight at a time, ${B.input_len}-token prompts, ${B.output_len}-token answers.`));

    // -------------------------------------------------------------- batch
    // Model fitted to the measured run. Every request gets a TTFT drawn from a normal
    // matched to the run's TTFT (mean 2.88 s, P99 4.66 s). A lane's next request starts
    // when the previous one ends, and each lane's TPOT is set so its requests fill the
    // measured duration. Across lanes that averages out to the measured mean TPOT.
    const sim = (function () {
      const Cc = B.concurrency, N = B.requests, OUT = B.output_len, D = B.duration_s;
      const perLane = Math.round(N / Cc);
      const rnd = mulberry32(20260923);
      const gauss = () => {
        let u = 0, v = 0;
        while (u === 0) u = rnd();
        v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      };
      const mean = B.mean_ttft_ms / 1000, p99 = B.p99_ttft_ms / 1000;
      const sd = (p99 - mean) / 2.326;
      const ttfts = [];
      for (let i = 0; i < N; i++) ttfts.push(clamp(mean + sd * gauss(), 0.6, p99));
      const m0 = ttfts.reduce((p, q) => p + q, 0) / N;
      for (let i = 0; i < N; i++) ttfts[i] += mean - m0;
      const lanesB = [];
      const reqs = [];
      let id = 0;
      for (let l = 0; l < Cc; l++) {
        const tt = [];
        for (let k = 0; k < perLane; k++) tt.push(ttfts[l * perLane + k]);
        // lanes end within half a second of one another; the last one ends at D
        const endL = l === 0 ? D : D - 0.5 * rnd();
        const tpot = (endL - tt.reduce((p, q) => p + q, 0)) / (perLane * (OUT - 1));
        const rs = [];
        let s0 = 0;
        for (let k = 0; k < perLane; k++) {
          const first = s0 + tt[k], end = first + (OUT - 1) * tpot;
          const r = { lane: l, start: s0, first, end, tpot, id: 0 };
          rs.push(r); reqs.push(r); s0 = end;
        }
        lanesB.push(rs);
      }
      reqs.sort((a, b) => a.start - b.start || a.lane - b.lane);
      reqs.forEach((r) => { r.id = ++id; });
      const tpots = lanesB.map((rs) => rs[0].tpot);
      const meanTpot = tpots.reduce((p, q) => p + q, 0) / tpots.length;
      const endsSorted = reqs.map((r) => r.end).sort((a, b) => a - b);
      const tokensAt = (s, exact) => {
        let n = 0;
        for (const r of reqs) {
          if (s < r.first) continue;
          const k = (s - r.first) / r.tpot + 1;
          n += Math.min(OUT, exact ? k : Math.floor(k));
        }
        return n;
      };
      // aggregate rate over time, for the chart (0.1 s grid, 1 s trailing window)
      const series = [];
      for (let s = 0; s <= D + 1e-9; s += 0.1) series.push([s, s < 1 ? tokensAt(s, true) / 1 : tokensAt(s, true) - tokensAt(s - 1, true)]);
      return { lanes: lanesB, reqs, D, OUT, meanTpot, endsSorted, tokensAt, series, perLane };
    })();
    film.batchModel = { meanTpotMs: sim.meanTpot * 1000, series: sim.series };

    const BT = { intro: 0.9, real: 8.0, ramp: 1.6, speed: 14, hold: 5.2 };
    // wall seconds since the sim started -> sim seconds. Real time, then an eased ramp.
    const simOf = (w) => {
      if (w <= 0) return 0;
      if (w <= BT.real) return w;
      const x = Math.min(w - BT.real, BT.ramp) / BT.ramp;
      const ramp = BT.ramp * (x + (BT.speed - 1) * (x * x * x - x * x * x * x / 2));
      if (w <= BT.real + BT.ramp) return BT.real + ramp;
      return BT.real + ramp + (w - BT.real - BT.ramp) * BT.speed;
    };
    const speedOf = (w) => w <= BT.real ? 1 : w >= BT.real + BT.ramp ? BT.speed : 1 + (BT.speed - 1) * smooth((w - BT.real) / BT.ramp);
    let simWall = BT.real + BT.ramp;
    simWall += (sim.D - simOf(simWall)) / BT.speed;
    const wallOf = (ss) => {            // inverse of simOf, for effects that decay in wall time
      let lo = 0, hi = simWall;
      for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (simOf(m) < ss) lo = m; else hi = m; }
      return hi;
    };
    const batchDur = BT.intro + simWall + BT.hold;

    const LANE = { x: 72, y: 214, w: 1196, h: 760 };
    const RP = { x: 1318, w: W - 72 - 1318 };

    add("batch", batchDur, (ctx, t, a) => {
      ctx.save(); ctx.globalAlpha = a;
      const w = t - BT.intro;
      const s = Math.min(sim.D, simOf(w));
      const speed = w > 0 && s < sim.D ? speedOf(w) : 1;
      const k = eOut(inv(0, 0.7, t));

      // header
      ctx.save(); ctx.globalAlpha *= k;
      pill(ctx, "BATCH MODE", 72, 112, { fill: rgba(C.repo, 0.13), color: C.repoHot, font: `700 14px ${SANS}`, ls: "1.8px", h: 32 });
      const tw = measure(ctx, "BATCH MODE", `700 14px ${SANS}`, "1.8px") + 28;
      txt(ctx, `${B.requests} requests, ${B.concurrency} in flight  ·  ${B.input_len} tokens in, ${B.output_len} out  ·  docker compose --profile batch`,
          72 + tw + 18, 135, { font: `500 23px ${SANS}`, color: C.ink2 });
      // speed badge
      const lapse = speed > 1.01, over = w > 0 && s >= sim.D;
      const bs = over ? "RUN COMPLETE" : lapse ? `${fmt(speed, speed < 9.95 ? 1 : 0)}× TIME-LAPSE` : "REAL TIME";
      pill(ctx, bs, W - 72, 112, { align: "right", h: 32, font: `700 14px ${SANS}`, ls: "1.8px",
        fill: over ? rgba(C.repo, 0.16) : lapse ? rgba(C.violet, 0.2) : "rgba(255,255,255,0.06)",
        color: over ? C.repoHot : lapse ? "#CFC6FF" : C.ink2,
        dot: over ? C.repo : lapse ? C.violet : rgba("#6EE7A0", 0.6 + 0.4 * Math.cos(t * 6)) });
      ctx.restore();

      // lanes
      ctx.save(); ctx.globalAlpha *= k;
      rr(ctx, LANE.x, LANE.y - 8, LANE.w, LANE.h + 16, 22);
      ctx.fillStyle = "rgba(255,255,255,0.025)"; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.lineWidth = 1; ctx.stroke();
      const pitch = (LANE.h - 40) / B.concurrency;
      const tx0 = LANE.x + 64, tw0 = LANE.w - 64 - 92;
      txt(ctx, "SLOT", LANE.x + 24, LANE.y + 20, { font: `600 11px ${SANS}`, color: C.dim2, ls: "1.4px" });
      txt(ctx, `${sim.OUT} TOKENS`, tx0 + tw0, LANE.y + 20, { font: `600 11px ${SANS}`, color: C.dim2, align: "right", ls: "1.4px" });
      txt(ctx, "DONE", LANE.x + LANE.w - 24, LANE.y + 20, { font: `600 11px ${SANS}`, color: C.dim2, align: "right", ls: "1.4px" });
      for (let l = 0; l < B.concurrency; l++) {
        const y = LANE.y + 36 + l * pitch;
        const rs = sim.lanes[l];
        let cur = rs.findIndex((r) => s < r.end);
        if (cur < 0) cur = rs.length;
        const r = rs[Math.min(cur, rs.length - 1)];
        const th = 5;
        const yy = y + (pitch - th) / 2;
        ctx.fillStyle = "rgba(255,255,255,0.055)";
        ctx.fillRect(tx0, yy, tw0, th);
        if (l % 8 === 0 || l === B.concurrency - 1) txt(ctx, String(l + 1), LANE.x + 50, yy + 5.5,
          { font: `500 11px ${MONO}`, color: C.dim2, align: "right" });
        if (cur >= rs.length) {
          // every request in this slot is done: keep the last one on screen, at rest
          ctx.fillStyle = rgba(C.repo, 0.32);
          ctx.fillRect(tx0, yy, tw0, th);
        } else {
          if (s < r.first) {
            // waiting for its first token: queue + prefill
            const ph = ((t * 1.3 + l * 0.137) % 1);
            const g = ctx.createLinearGradient(tx0, 0, tx0 + 60, 0);
            g.addColorStop(0, rgba(C.violet, 0)); g.addColorStop(ph, rgba(C.violet, 0.75)); g.addColorStop(1, rgba(C.violet, 0));
            ctx.fillStyle = g; ctx.fillRect(tx0, yy, 60, th);
          } else {
            const n = Math.min(sim.OUT, Math.floor((s - r.first) / r.tpot) + 1);
            const fw = (n / sim.OUT) * tw0;
            const g = ctx.createLinearGradient(tx0, 0, tx0 + fw, 0);
            g.addColorStop(0, rgba(C.repo, 0.16)); g.addColorStop(Math.max(0, 1 - 90 / Math.max(fw, 90)), rgba(C.repo, 0.6)); g.addColorStop(1, C.repoHot);
            ctx.fillStyle = g; ctx.fillRect(tx0, yy, fw, th);
            ctx.fillStyle = "#FFE3C2";
            ctx.fillRect(tx0 + fw - 2, yy - 1, 3, th + 2);
          }
        }
        // a lane that just finished a request flashes (wall-time decay)
        for (const rq of rs) {
          const ageW = w - wallOf(rq.end);
          if (ageW >= 0 && ageW < 0.5) {
            ctx.fillStyle = rgba("#FFFFFF", 0.35 * (1 - ageW / 0.5));
            ctx.fillRect(tx0, yy - 1, tw0, th + 2);
          }
        }
        // completed requests on this lane, as dots
        const doneN = rs.filter((rq) => rq.end <= s + 1e-9).length;
        for (let d = 0; d < sim.perLane; d++) {
          ctx.beginPath();
          ctx.arc(LANE.x + LANE.w - 70 + d * 13, yy + th / 2, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = d < doneN ? C.repo : "rgba(255,255,255,0.10)";
          ctx.fill();
        }
      }
      ctx.restore();

      // right column
      ctx.save(); ctx.globalAlpha *= eOut(inv(0.15, 0.85, t));
      const x = RP.x, wR = RP.w;
      const aggNow = s <= 0 ? 0 : s < 1 ? sim.tokensAt(s, true) / Math.max(s, 1) : sim.tokensAt(s, true) - sim.tokensAt(s - 1, true);
      const finished = s >= sim.D - 1e-9;
      txt(ctx, finished ? "AGGREGATE DECODE, MEASURED" : "AGGREGATE DECODE, ALL STREAMS", x, 244, { font: `700 14px ${SANS}`, color: C.dim, ls: "2px" });
      const hero = finished ? B.decode_tok_s : aggNow;
      const heroStr = fmt(hero, 0);
      const hf = `800 132px ${SANS}`;
      const hwid = (finished ? numFinal : num)(ctx, heroStr, x - 4, 368, hf, finished ? C.repoHot : C.ink);
      txt(ctx, "tok/s", x + hwid + 10, 368, { font: `600 34px ${SANS}`, color: C.dim });
      const dec = sim.reqs.filter((r) => s >= r.first && s < r.end).length;
      const per = 1 / sim.meanTpot;
      txt(ctx, finished ? `${B.concurrency} × 1000 / median TPOT ${fmt(B.median_tpot_ms, 1)} ms` : `${dec} streams decoding × ~${fmt(per, 1)} tok/s each`,
          x, 422, { font: `500 20px ${SANS}`, color: C.dim });

      // chart
      const cx0 = x, cy0 = 452, cw = wR, ch = 196, ymax = 1200;
      rr(ctx, cx0, cy0, cw, ch, 14); ctx.fillStyle = "rgba(255,255,255,0.03)"; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.stroke();
      const px_ = (ss) => cx0 + 14 + (ss / sim.D) * (cw - 28);
      const py_ = (v) => cy0 + ch - 22 - (v / ymax) * (ch - 44);
      ctx.setLineDash([4, 6]); ctx.strokeStyle = rgba(C.repo, 0.35); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx0 + 14, py_(B.decode_tok_s)); ctx.lineTo(cx0 + cw - 14, py_(B.decode_tok_s)); ctx.stroke();
      ctx.setLineDash([]);
      txt(ctx, `${fmt(B.decode_tok_s, 0)} measured`, cx0 + cw - 16, py_(B.decode_tok_s) - 8, { font: `600 13px ${SANS}`, color: rgba(C.repo, 0.8), align: "right" });
      for (let ss = 0; ss <= sim.D; ss += 30) {
        txt(ctx, `${ss}s`, px_(ss), cy0 + ch - 6, { font: `500 12px ${SANS}`, color: C.dim2, align: ss === 0 ? "left" : "center" });
      }
      const ser = sim.series;
      const nPts = Math.min(ser.length, Math.floor(s / 0.1) + 1);
      if (nPts > 1) {
        const area = ctx.createLinearGradient(0, cy0, 0, cy0 + ch);
        area.addColorStop(0, rgba(C.repo, 0.35)); area.addColorStop(1, rgba(C.repo, 0));
        ctx.beginPath(); ctx.moveTo(px_(0), py_(0));
        for (let i = 0; i < nPts; i++) ctx.lineTo(px_(ser[i][0]), py_(ser[i][1]));
        ctx.lineTo(px_(ser[nPts - 1][0]), py_(0)); ctx.closePath();
        ctx.fillStyle = area; ctx.fill();
        ctx.beginPath();
        for (let i = 0; i < nPts; i++) { const X = px_(ser[i][0]), Y = py_(ser[i][1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
        ctx.strokeStyle = C.repo; ctx.lineWidth = 2; ctx.stroke();
        const lx = px_(ser[nPts - 1][0]), ly = py_(ser[nPts - 1][1]);
        ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fillStyle = "#FFE3C2"; ctx.fill();
      }
      txt(ctx, "tok/s over the run", cx0 + 16, cy0 + 24, { font: `600 13px ${SANS}`, color: C.dim2, ls: "0.5px" });

      // stats
      const tokNow = finished ? B.generated : sim.tokensAt(s, false);
      const doneReq = upper(sim.endsSorted, s + 1e-9);
      const stats = [
        ["TOKENS GENERATED", fmt(tokNow), ""],
        ["REQUESTS DONE", fmt(doneReq), ` / ${B.requests}`],
        ["ELAPSED", fmt(s, 1), " s"],
        ["IN FLIGHT", String(sim.reqs.filter((r) => s >= r.start && s < r.end).length), ` / ${B.concurrency}`],
      ];
      stats.forEach(([lab, v, suf], i) => {
        const sx = x + (i % 2) * (wR / 2), sy = 712 + Math.floor(i / 2) * 118;
        txt(ctx, lab, sx, sy, { font: `700 13px ${SANS}`, color: C.dim2, ls: "1.8px" });
        const vw = num(ctx, v, sx, sy + 52, `700 46px ${SANS}`, C.ink);
        if (suf) txt(ctx, suf, sx + vw + 4, sy + 52, { font: `500 24px ${SANS}`, color: C.dim });
      });
      ctx.restore();

      // finale card
      const fk = eOut5(inv(BT.intro + simWall + 0.3, BT.intro + simWall + 1.2, t));
      if (fk > 0) {
        ctx.save(); ctx.globalAlpha *= fk;
        const cw2 = 1010, ch2 = 360, x2 = LANE.x + LANE.w / 2 - cw2 / 2, y2 = LANE.y + LANE.h / 2 - ch2 / 2;
        ctx.translate(0, (1 - fk) * 20);
        ctx.shadowColor = "rgba(0,0,0,0.7)"; ctx.shadowBlur = 60;
        rr(ctx, x2, y2, cw2, ch2, 28); ctx.fillStyle = "rgba(14,15,19,0.95)"; ctx.fill();
        ctx.shadowBlur = 0; ctx.strokeStyle = rgba(C.repo, 0.5); ctx.lineWidth = 1.5; ctx.stroke();
        txt(ctx, "THE WHOLE RUN", x2 + 56, y2 + 70, { font: `700 15px ${SANS}`, color: C.repoHot, ls: "2.5px" });
        const g = ctx.createLinearGradient(0, y2 + 80, 0, y2 + 170);
        g.addColorStop(0, "#FFFFFF"); g.addColorStop(1, "#FFD9AE");
        txt(ctx, `${fmt(B.generated)} tokens in ${fmt(B.duration_s, 1)} s`, x2 + 54, y2 + 158, { font: `800 76px ${SANS}`, color: g, ls: "-2px" });
        const cols3 = [[fmt(B.decode_tok_s, 0), "tok/s decode", "64 × 1000 / median TPOT"],
                       [fmt(B.output_tok_s, 0), "tok/s end to end", "tokens ÷ wall clock"],
                       [fmt(B.median_tpot_ms, 1) + " ms", "median TPOT", `mean TTFT ${fmt(B.mean_ttft_ms / 1000, 2)} s`]];
        cols3.forEach(([v, l1, l2], i) => {
          const xx = x2 + 56 + i * 310, yy = y2 + 250;
          txt(ctx, v, xx, yy, { font: `700 44px ${SANS}`, color: i === 0 ? C.repoHot : C.ink });
          txt(ctx, l1, xx, yy + 34, { font: `600 19px ${SANS}`, color: C.ink2 });
          txt(ctx, l2, xx, yy + 60, { font: `400 16px ${SANS}`, color: C.dim });
        });
        ctx.restore();
      }

      stamp(ctx, [
        `Figures: vllm bench serve, ${B.requests} requests at concurrency ${B.concurrency}, random ${B.input_len}/${B.output_len}, one RTX 3090 at 250 W, vLLM 0.29, ${B.date}.`,
        `The bench keeps no per-token arrivals, so the lanes replay a model fitted to its mean TTFT (${fmt(B.mean_ttft_ms / 1000, 2)} s), mean TPOT (${fmt(B.mean_tpot_ms, 1)} ms) and duration (${fmt(B.duration_s, 1)} s).`,
      ], k);
      ctx.restore();
    });

    // --------------------------------------------------------------- outro
    add("outro", 6.4, (ctx, t, a) => {
      ctx.save(); ctx.globalAlpha = a;
      const k = eOut(inv(0.05, 0.9, t));
      ctx.save(); ctx.globalAlpha *= k;
      const tf = `800 84px ${SANS}`;
      const tw = measure(ctx, "HyperQwen", tf, "-2px");
      logo(ctx, W / 2 - (tw + 92) / 2, 172, 72);
      txt(ctx, "HyperQwen", W / 2 - (tw + 92) / 2 + 92, 238, { font: tf, color: C.ink, ls: "-2px" });
      txt(ctx, "Qwen3.8-27B on one RTX 3090 · 24 GB · 250 W", W / 2, 300, { font: `400 30px ${SANS}`, color: C.dim, align: "center" });
      ctx.restore();

      const cards = [
        { k: "ONE REQUEST", v: fmt(P.copy.repo.decode_tok_s, 0), u: "tok/s",
          l1: "quoting a 25k-token document", l2: `${fmt(P.code.repo.decode_tok_s, 0)} writing code · ${fmt(P.chat.repo.decode_tok_s, 0)} chatting` },
        { k: `${B.concurrency} REQUESTS AT ONCE`, v: fmt(B.decode_tok_s, 0), u: "tok/s",
          l1: "aggregate decode", l2: `${fmt(B.output_tok_s, 0)} end to end` },
      ];
      cards.forEach((c, i) => {
        const kk = eOut5(inv(0.35 + i * 0.2, 1.3 + i * 0.2, t));
        const cw = 700, chh = 330, x = W / 2 - cw - 20 + i * (cw + 40), y = 372;
        ctx.save(); ctx.globalAlpha *= kk; ctx.translate(0, (1 - kk) * 30);
        rr(ctx, x, y, cw, chh, 28);
        const g = ctx.createLinearGradient(x, y, x, y + chh);
        g.addColorStop(0, "rgba(255,255,255,0.055)"); g.addColorStop(1, "rgba(255,255,255,0.02)");
        ctx.fillStyle = g; ctx.fill();
        ctx.strokeStyle = rgba(C.repo, 0.28); ctx.lineWidth = 1.2; ctx.stroke();
        txt(ctx, c.k, x + 48, y + 66, { font: `700 16px ${SANS}`, color: C.repoHot, ls: "2.5px" });
        const vg = ctx.createLinearGradient(0, y + 100, 0, y + 210);
        vg.addColorStop(0, "#FFFFFF"); vg.addColorStop(1, "#FFC98E");
        const vw = measure(ctx, c.v, `800 136px ${SANS}`, "-4px");
        txt(ctx, c.v, x + 42, y + 206, { font: `800 136px ${SANS}`, color: vg, ls: "-4px" });
        txt(ctx, c.u, x + 42 + vw + 16, y + 206, { font: `600 38px ${SANS}`, color: C.dim });
        txt(ctx, c.l1, x + 48, y + 264, { font: `600 26px ${SANS}`, color: C.ink2 });
        txt(ctx, c.l2, x + 48, y + 298, { font: `400 22px ${SANS}`, color: C.dim });
        ctx.restore();
      });

      const k3 = eOut(inv(1.3, 2.1, t));
      ctx.save(); ctx.globalAlpha *= k3;
      const cmd = "docker compose --profile single up -d";
      const cf = `500 26px ${MONO}`;
      const cwid = measure(ctx, cmd, cf) + 80;
      rr(ctx, W / 2 - cwid / 2, 770, cwid, 64, 16);
      ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.stroke();
      txt(ctx, "$", W / 2 - cwid / 2 + 30, 811, { font: cf, color: C.repo });
      txt(ctx, cmd, W / 2 - cwid / 2 + 60, 811, { font: cf, color: C.ink });
      txt(ctx, "github.com/syv-ai/HyperQwen", W / 2, 900, { font: `600 32px ${SANS}`, color: C.ink2, align: "center", ls: "0.3px" });
      ctx.restore();
      ctx.restore();
    });

    film.duration = cursor + 0.6;       // a beat of black at the end so the loop breathes
    chapters = [
      { label: "01  ONE REQUEST", a: scenes[1].start, b: scenes.find((s) => s.name === "single-summary").end },
      { label: "02  SIXTY-FOUR AT ONCE", a: scenes.find((s) => s.name === "ch2").start, b: scenes.find((s) => s.name === "batch").end },
    ];
    film.scenes = scenes.map((s) => ({ name: s.name, start: s.start, end: s.end }));

    film.draw = function (ctx, t) {
      CW = measure(ctx, "0".repeat(20), `400 ${BODY_FONT_PX}px ${MONO}`) / 20;
      ctx.save();
      const inBatch = scenes.find((s) => s.name === "batch");
      backdrop(ctx);
      const fadeIn = 0.35;
      for (const s of scenes) {
        if (t < s.start || t > s.end) continue;
        const lt = t - s.start;
        // scenes hand over through the ground, never by double-exposing text on text
        const fin = s.name === "intro" ? 1 : smooth(lt / fadeIn);
        const fout = smooth((s.end - t) / 0.3);
        const a = Math.min(fin, fout);
        if (a <= 0.001) continue;
        s.draw(ctx, lt, a);
      }
      // chrome + chapter progress everywhere except the intro and outro
      const ci = scenes[1].start, co = scenes[scenes.length - 1].start + 0.35;
      const ca = Math.min(smooth((t - ci) / 0.5), smooth((co - t) / 0.4));
      chrome(ctx, ca);
      progress(ctx, t, ca);
      // global fade from and to black, so the loop point is invisible
      const edge = Math.min(smooth(t / 0.5), smooth((film.duration - t) / 0.6));
      if (edge < 1) { ctx.fillStyle = `rgba(6,7,10,${1 - edge})`; ctx.fillRect(0, 0, W, H); }
      ctx.restore();
      void inBatch;
    };
    return film;
  }

  window.HQDemo = { create, W, H };
})();
