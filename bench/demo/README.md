# The demo video

`docs/media/demo.mp4` and the `demo.avif` at the top of the README are drawn by
`demo.js`: a canvas renderer in plain JavaScript where the whole film is one pure
function of time. Open `index.html` in a browser and it plays live; `render.mjs`
steps the same function frame by frame in headless Chrome and hands the frames to
ffmpeg. Rendering needs no GPU.

```bash
cd bench/demo
npm install
node render.mjs                        # docs/media/demo.mp4 (1080p60) + demo.avif (720p30)
node render.mjs --stills 9,27.5,53     # PNG stills in ./stills
node render.mjs --from 36 --to 60      # one slice, for iterating on a scene
CHROME=/path/to/chrome node render.mjs # use an installed Chrome instead of puppeteer's
```

Where the pixels come from:

- **One request.** Two `bench/demo_capture.py` recordings, stock `vllm serve`
  and `SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1`, with every streamed chunk and
  the time it arrived. At race time T a lane shows exactly the chunks that had
  arrived by T. Nothing is sped up or interpolated. `build_data.py` tokenizes the
  text once and assigns each token to its chunk, so the token counts and the
  tokens-per-step bars are exact.
- **Sixty-four at once.** `vllm bench serve` (64 concurrent, 128 in / 512 out)
  records only a summary. The lanes therefore replay a model fitted to that run's
  mean TTFT, mean TPOT and duration, and the frame says so. Every number printed
  about the run is the measured one. The time-lapse part carries its speed on every
  frame.

Re-cut after a new capture: run `build_data.py` (it needs the model's
`tokenizer.json` and a `run_benchmarks.sh` batch log; see its docstring), then
`node render.mjs`.
