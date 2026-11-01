# No one cares about your research (yet!)

Seminar on research communication, Gabriel Weymouth (TU Delft), with the JFM paper in this repo as the running example. Same toolkit as StillDesigningShips: a plain [reveal.js](https://revealjs.com) deck, vendored, no build step. The outline lives in `outline.md`.

Published at <https://weymouth.github.io/KelvinFlatShip/talk/> (GitHub Pages from `main`, repo root, with `.nojekyll`). There the live demo doesn't run; slide 17 links back here instead. After changing `css/` or `js/`, bump the `?v=` tags in `index.html`: Pages lets browsers cache for about 10 minutes.

## Run it
Needs Python 3 and Julia 1.11 (tested 1.11.5). All commands run from the repo root.
```
julia --project=talk -e "using Pkg; Pkg.instantiate()"    # first time only, a few minutes
python talk/scripts/serve.py                              # the deck, on localhost:8000
julia --project=talk --threads=auto talk/server.jl        # the live demo, on localhost:8081
```
Then open <http://localhost:8000>. Start the Julia server a minute early: it compiles before printing `Serving wave fields…`. `talk/Manifest.toml` pins the tested versions (registered NeumannKelvin 0.9.3).

Keys: Esc overview · S speaker notes · L light/dark · H hide placeholders · F fullscreen. Use a full-size browser window: below a few hundred pixels wide, reveal.js switches to a scrolling layout.

## Live demo (slide 17)
`js/demo.js` asks `server.jl` for each frame; `server.jl` evaluates `src/` on a 160×80 grid (y-symmetric, so half is computed) with all threads, ≈0.5 s per frame on 4 cores.
- **Our method** redraws on every slider move. The *model* menu and *depth* slider switch to the point source, whose wake turns to noise as it nears the surface.
- **Brute force** fills the same frame one pixel at a time on one thread, with direct quadrature, so it never finishes; its clock shows the hours it would need.
- **Offline fallback**: without the server, the flat-ship frames come from `talk/assets/demo/` (status: "server offline: saved frames"). Regenerate them after changing the grid or slider ranges: `julia --project=talk --threads=auto talk/server.jl --bake` (≈2 min).
- **Check it** with the server running:
  ```
  curl "http://localhost:8081/field?F=0.50&beta=0.30&model=flat&z=0"
  curl "http://localhost:8081/brute?F=0.50&beta=0.30&k0=1200&n=1"
  ```
  The first returns JSON with `"time"` (seconds per frame) and `"z"`; the second one pixel after a few seconds.
- **Tuning**: frame size is `window(F; nx, ny)` in `server.jl`; slider ranges are in `index.html` (keep `--bake`'s loop in step); the colour range is `ZMAX` in `js/demo.js`.

## Renders
`julia --project=talk --threads=auto talk/scripts/renders.jl` remakes the talk-only figures in `assets/img`: the single point wake, both wave cuts (slides 12 and 14) and the two drag plots, in the deck's colours. The other paper figures are copied from `make_figures`/`tex/figures`.

## Editing
- `index.html` holds every slide in order: headline, graphic, and speaker notes in `<aside class="notes">`. Text inside `class="fragment"` steps in on a click.
- `js/figures.js` draws the coded graphics named by `data-figure`; `js/demo.js` and `server.jl` are the live demo.
- Media layers name their file; a missing one shows a placeholder. `python talk/scripts/assets.py --missing` lists them, and `python talk/scripts/commons.py fetch <name> "File:…"` fetches from Wikimedia Commons with its credit.
