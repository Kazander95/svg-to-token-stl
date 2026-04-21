# 🪙 SVG → Token STL

Turn any single-color SVG into a 3D-printable circular token — ready for the
table.

**🌐 Live app:** <https://svg-to-token-stl.streamlit.app/>

Mostly designed for **tabletop roleplaying games** (D&D, Pathfinder, etc.) —
drop in a character or monster icon, get back a printable token that fits
standard 1" / 2" / 3" grid squares.

## What it does

1. **Upload** a single-color SVG (a character portrait, monster icon, faction
   crest, spell symbol — anything with a clear silhouette).
2. **Crop** the art interactively by dragging a box over a live preview.
3. **Preview** the final token: a white disc with a raised black border and
   the art raised in relief on the face.
4. **Download** a watertight, 3D-print-ready STL, sized to exactly **1", 2",
   or 3" wide**.

The generated STL is a single manifold mesh (boolean-unioned via
[`manifold3d`](https://github.com/elalish/manifold)), so it imports cleanly
into any slicer — PrusaSlicer, OrcaSlicer, Bambu Studio, Cura, etc.

## Why

TTRPGs need a lot of tokens, fast. Art assets from sites like
[The Noun Project](https://thenounproject.com/) or
[Game-icons.net](https://game-icons.net/) are usually distributed as SVGs —
this tool composes them onto a bordered disc and spits out a print-ready STL
in seconds, at the exact size your battle map needs.

## Usage (hosted)

Just go to **<https://svg-to-token-stl.streamlit.app/>** and upload.

## Usage (local)

```bash
# Clone and set up with uv
git clone https://github.com/Kazander95/svg-to-token-stl
cd svg-to-token-stl
make install            # uv sync

# Run the Streamlit UI
make dev                # -> http://localhost:8501

# Or use the CLI directly
uv run python main.py path/to/icon.svg -s 1    # 1" token
uv run python main.py path/to/icon.svg -s 2    # 2" token
uv run python main.py path/to/icon.svg -s 3    # 3" token
```

### CLI options

```
positional:
  input                   Input SVG file.

options:
  -o, --output            Output STL path (default: <input>.stl).
  -s, --size {1,2,3}      Final disc diameter in inches. Default: 1.
  --base-thickness        Base disc thickness in mm (default: 2.0).
  --relief-height         Border/art relief height in mm (default: 0.6).
  --border-frac           Border width as fraction of disc radius (default: 0.06).
  --art-margin-frac       Gap between border and art as fraction of radius (default: 0.04).
```

## Print tips

- **Single filament:** prints fine as-is; the art will be embossed.
- **Two-color (AMS / MMU / manual swap):** in your slicer, add a filament
  change at the layer matching the base thickness (default `2.0 mm`), so the
  raised border and art print in a contrasting color.
- Tokens are oriented with the art face up, so no supports are needed.

## Stack

- [`svgelements`](https://pypi.org/project/svgelements/) — SVG parsing
- [`shapely`](https://shapely.readthedocs.io/) — 2D geometry (disc, border, art clipping)
- [`trimesh`](https://trimesh.org/) + [`manifold3d`](https://github.com/elalish/manifold) — extrusion + watertight boolean union
- [`streamlit`](https://streamlit.io/) + [`streamlit-cropper`](https://github.com/turner-anderson/streamlit-cropper) — web UI
