# Recording the hero animation

`record.mjs` records a video straight from the interactive three.js hero
animation (`docs/_static/harmonic_mesh_viewer.js` +
`docs/_static/boat_animation_data.bin`), as an alternative to the offline
vedo render in `docs/examples/src/B7_boat_animation.py`. Since it's
literally the same renderer as the live homepage, colors/lighting/gradient
match by construction — nothing to eyeball or re-tune by hand.

## Setup (once)

```sh
npm install --no-save playwright
npx playwright install chromium
```

`ffmpeg` must also be on `PATH` (used to find the loop point and encode the
final mp4).

## Usage

```sh
node docs/_scripts/record_hero_animation/record.mjs [output.mp4] [--crossfade=SECONDS] [--re-encode]
```

- `output.mp4` — where to write the result. Defaults to
  `front_page_animation.mp4` in this directory (deliberately **not**
  `docs/_static/front_page_animation.mp4`, so it doesn't clobber the current
  video before you've reviewed it — copy it over once you're happy).
- `--crossfade=SECONDS` — length of the dissolve used to hide the loop seam
  (default `0.15`). Raise it if the loop still isn't smooth, lower it if the
  dissolve itself becomes noticeable.
- `--re-encode` — skip re-recording from the browser and just re-run ffmpeg
  against the last raw capture and loop point (kept in `.raw/`, gitignored).
  Use this while tuning `--crossfade`.

## How the loop point is found

The video loops by finding, then blending over, a real match — not by
computing a cut point from the animation's period (`2π/1.5 ≈ 4.19s`) and
assumed frame timestamps. That was the first approach here, and it left a
visible jump of roughly a tenth of a period at the seam: Chromium's video
recording doesn't run at a clean, fixed 25fps tied to wall-clock time in
every environment (headless/sandboxed runs in particular) — `.raw/meta.json`
records what `findLoop()` actually measured last, which has come out well
above 25fps here.

So instead, `record.mjs`:

1. Records a few periods' worth of video (`BUFFER_PERIODS` in the script),
   with no attempt to time the cut.
2. Decodes it to small grayscale frames and, for a range of candidate loop
   lengths `L` (in frames, not seconds), scores how well frame `i` matches
   frame `i + L` averaged across the whole recording — i.e. an
   autocorrelation search for the recording's *own*, empirically observed
   period, however many frames it actually took. `console.log`'s
   `matchScore` reports the winning score (0 = identical frames, 255 =
   opposite) as a sanity check — it should be small.
3. Cuts the raw recording at that length using `trim`'s frame-indexed
   `start_frame`/`end_frame` (not time-based `start`/`end`), so the cut lines
   up with exactly the frames the search matched.
4. Crossfades a short overlap at the seam (see `buildFilterComplex()`) to
   smooth over what small residual mismatch remains (compression noise,
   mostly, now that the phase itself lines up).

Because this works from the recorded pixels rather than assumed timing, it
holds up regardless of the environment's actual capture rate.
