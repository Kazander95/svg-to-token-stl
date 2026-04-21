"""Streamlit UI for svg-to-token-stl.

Upload an SVG, preview the art, crop its bounding box with sliders, pick a
size (1" / 2" / 3"), then download a 3D-print-ready STL.

Run with:  uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box

from main import (
    DEFAULT_ART_MARGIN_FRAC,
    DEFAULT_BASE_THICKNESS_MM,
    DEFAULT_BORDER_FRAC,
    DEFAULT_RELIEF_HEIGHT_MM,
    MM_PER_INCH,
    build_token_from_art,
    load_svg_as_polygon,
)

st.set_page_config(page_title="SVG → Token STL", page_icon="🪙", layout="wide")
st.title("🪙 SVG → Token STL")
st.caption(
    "Upload a single-color SVG, crop it to taste, and download a 3D-printable "
    "circular token STL."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_polygon(ax, geom, **kwargs):
    """Draw a shapely (Multi)Polygon on a matplotlib axis."""
    if geom.is_empty:
        return
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for poly in polys:
        if not isinstance(poly, Polygon):
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kwargs)
        for interior in poly.interiors:
            ixs, iys = interior.xy
            ax.fill(ixs, iys, color="white", zorder=kwargs.get("zorder", 1) + 0.1)


def render_preview(art, crop_bbox=None, size_inches=1.0) -> plt.Figure:
    """Render the art + crop rectangle + final disc preview."""
    fig, (ax_src, ax_token) = plt.subplots(1, 2, figsize=(10, 5))

    # --- left: source art with crop rectangle ---
    ax_src.set_title("Source (crop region in red)")
    _plot_polygon(ax_src, art, color="black", zorder=1)
    if crop_bbox is not None:
        minx, miny, maxx, maxy = crop_bbox
        ax_src.add_patch(
            plt.Rectangle(
                (minx, miny),
                maxx - minx,
                maxy - miny,
                fill=False,
                edgecolor="red",
                linewidth=2,
                linestyle="--",
            )
        )
    ax_src.set_aspect("equal")
    ax_src.autoscale()

    # --- right: token preview (disc + border ring + cropped art fit inside) ---
    ax_token.set_title(f'Token preview ({size_inches}" disc)')
    target_diameter_mm = size_inches * MM_PER_INCH
    disc_radius = target_diameter_mm / 2.0
    border_width = disc_radius * DEFAULT_BORDER_FRAC
    art_margin = disc_radius * DEFAULT_ART_MARGIN_FRAC
    inner_safe = disc_radius - border_width - art_margin

    # Disc outline.
    ax_token.add_patch(Circle((0, 0), disc_radius, facecolor="white", edgecolor="black", linewidth=1))
    # Border ring (visual).
    ax_token.add_patch(Circle((0, 0), disc_radius, facecolor="none", edgecolor="black", linewidth=3))
    ax_token.add_patch(Circle((0, 0), disc_radius - border_width, facecolor="white", edgecolor="black", linewidth=0.5))

    # Fit cropped art into the safe inner area and plot it.
    cropped = art
    if crop_bbox is not None:
        cropped = art.intersection(box(*crop_bbox))
    if not cropped.is_empty:
        minx, miny, maxx, maxy = cropped.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        centered = affinity.translate(cropped, xoff=-cx, yoff=-cy)
        half_w, half_h = (maxx - minx) / 2, (maxy - miny) / 2
        current_r = max((half_w ** 2 + half_h ** 2) ** 0.5, 1e-9)
        fit = affinity.scale(centered, xfact=inner_safe / current_r, yfact=inner_safe / current_r, origin=(0, 0))
        _plot_polygon(ax_token, fit, color="black", zorder=2)

    lim = disc_radius * 1.1
    ax_token.set_xlim(-lim, lim)
    ax_token.set_ylim(-lim, lim)
    ax_token.set_aspect("equal")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

uploaded = st.file_uploader("Upload SVG", type=["svg"])

if uploaded is None:
    st.info("⬆️ Upload a single-color SVG to begin.")
    st.stop()

# Persist to a temp file because svgelements wants a path.
with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
    tmp.write(uploaded.getvalue())
    svg_path = Path(tmp.name)

try:
    art = load_svg_as_polygon(svg_path)
except Exception as exc:
    st.error(f"Could not parse SVG: {exc}")
    st.stop()

minx, miny, maxx, maxy = art.bounds
width = maxx - minx
height = maxy - miny
st.caption(f"Parsed SVG bounds: ({minx:.1f}, {miny:.1f}) → ({maxx:.1f}, {maxy:.1f})  |  {width:.1f} × {height:.1f} units")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Token settings")
    size_inches = st.selectbox("Disc size", [1, 2, 3], index=0, format_func=lambda v: f'{v}"')

    st.subheader("Crop")
    enable_crop = st.checkbox("Enable crop", value=False, help="Trim the SVG art's bounding box before it lands on the token.")

    if enable_crop:
        pad_x = width * 0.05
        pad_y = height * 0.05
        left = st.slider("Left", float(minx - pad_x), float(maxx), float(minx), step=width / 200)
        right = st.slider("Right", float(minx), float(maxx + pad_x), float(maxx), step=width / 200)
        bottom = st.slider("Bottom", float(miny - pad_y), float(maxy), float(miny), step=height / 200)
        top = st.slider("Top", float(miny), float(maxy + pad_y), float(maxy), step=height / 200)

        if right <= left or top <= bottom:
            st.warning("Crop rectangle is empty — adjust the sliders.")
            crop_bbox = None
        else:
            crop_bbox = (left, bottom, right, top)
    else:
        crop_bbox = None

    st.subheader("Advanced")
    with st.expander("Dimensions (mm)"):
        base_thickness = st.number_input("Base thickness", 0.4, 10.0, DEFAULT_BASE_THICKNESS_MM, 0.1)
        relief_height = st.number_input("Relief height", 0.2, 5.0, DEFAULT_RELIEF_HEIGHT_MM, 0.1)
        border_frac = st.slider("Border width (fraction of radius)", 0.0, 0.2, DEFAULT_BORDER_FRAC, 0.005)
        art_margin_frac = st.slider("Art margin (fraction of radius)", 0.0, 0.2, DEFAULT_ART_MARGIN_FRAC, 0.005)

# --- Preview ---
preview_art = art
if crop_bbox is not None:
    preview_art = art.intersection(box(*crop_bbox))
    if preview_art.is_empty:
        st.error("Crop removes all geometry — widen the crop window.")
        st.stop()

fig = render_preview(art, crop_bbox=crop_bbox, size_inches=size_inches)
st.pyplot(fig)

# --- Generate ---
col1, col2 = st.columns([1, 3])
with col1:
    generate = st.button("🔨 Generate STL", type="primary", use_container_width=True)

if generate:
    with st.spinner("Extruding and booleaning…"):
        try:
            mesh = build_token_from_art(
                preview_art,
                size_inches=float(size_inches),
                base_thickness=base_thickness,
                relief_height=relief_height,
                border_frac=border_frac,
                art_margin_frac=art_margin_frac,
            )
        except Exception as exc:
            st.error(f"Failed to build mesh: {exc}")
            st.stop()

        buf = io.BytesIO()
        mesh.export(buf, file_type="stl")
        buf.seek(0)

    ex = mesh.extents
    watertight = "✅ watertight" if mesh.is_watertight else "⚠️ NOT watertight"
    st.success(
        f"Built token: {len(mesh.faces):,} triangles · "
        f"{ex[0]:.2f} × {ex[1]:.2f} × {ex[2]:.2f} mm · {watertight}"
    )

    default_name = Path(uploaded.name).stem + f"_{size_inches}in.stl"
    st.download_button(
        "⬇️ Download STL",
        data=buf.getvalue(),
        file_name=default_name,
        mime="model/stl",
        use_container_width=True,
    )
