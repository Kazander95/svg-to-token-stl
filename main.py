"""Convert a single-color SVG into a circular token STL.

The script wraps the SVG art in a white disc with a black border ring,
extrudes the whole thing into a 3D token, and scales the final mesh so the
disc is exactly N inches across (default 1").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
from svgelements import SVG, Shape, Path as SvgPath

MM_PER_INCH = 25.4


def _rings_to_polygon(rings: list[list[tuple[float, float]]]):
    """Combine a list of closed rings into a shapely polygon using even-odd fill.

    Rings are classified as outer/hole based on how many other rings enclose
    them (depth). Even depth => outer; odd depth => hole.
    """
    from shapely.geometry import Polygon as _P
    polys: list[Polygon] = []
    for r in rings:
        if len(r) < 3:
            continue
        try:
            p = _P(r)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty or p.area <= 0:
                continue
            polys.append(p)
        except Exception:
            continue
    if not polys:
        return None

    depths = []
    for i, p in enumerate(polys):
        c = p.representative_point()
        d = 0
        for j, q in enumerate(polys):
            # Only count q as containing p if q is strictly larger. Without
            # the area guard, representative_point() for a donut-shaped outer
            # outline can land in its visual hole, making the inner counter
            # ring look like it "contains" the outer — which inverts letters
            # with counters (D, O, g, o, etc.).
            if i != j and q.area > p.area and q.contains(c):
                d += 1
        depths.append(d)

    outers = [p for p, d in zip(polys, depths) if d % 2 == 0]
    holes = [p for p, d in zip(polys, depths) if d % 2 == 1]
    if not outers:
        return None
    result = unary_union(outers)
    if holes:
        result = result.difference(unary_union(holes))
    if not result.is_valid:
        result = result.buffer(0)
    return result if not result.is_empty else None

# Defaults (mm). These are final, post-scale dimensions.
DEFAULT_BASE_THICKNESS_MM = 2.0
DEFAULT_RELIEF_HEIGHT_MM = 0.6
# Border width and art margin as a fraction of the disc radius.
DEFAULT_BORDER_FRAC = 0.06
DEFAULT_ART_MARGIN_FRAC = 0.04
# Text label defaults.
DEFAULT_TEXT_HEIGHT_FRAC = 0.75     # of border_width
DEFAULT_TEXT_PADDING_FRAC = 0.35    # of border_width (clearance around text)
DEFAULT_TEXT_MAX_WIDTH_FRAC = 0.75  # of disc diameter
DEFAULT_TEXT_TAPER_FRAC = 1.5       # of border_width (taper length per side)


def _curve_to_arc(geom, R_mid: float, position: str = "bottom", tolerance: float = 0.2):
    """Bend a geometry laid out flat around y=0 onto a circular arc.

    The flat-frame x-axis maps to arc length along the circle of radius
    ``R_mid``; y=0 sits on that circle. For ``position='top'`` the geometry is
    placed at angle +π/2 with +y pointing radially outward (ascenders point
    away from the disc center). For ``position='bottom'`` it sits at -π/2 with
    +y pointing radially inward (ascenders point toward the disc center) so
    the text reads right-side-up.
    """
    import shapely

    geom = shapely.segmentize(geom, max_segment_length=tolerance)

    if position == "top":
        def tx(coords):
            x = coords[:, 0]
            y = coords[:, 1]
            r = R_mid + y
            theta = np.pi / 2 - x / R_mid
            return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        def tx(coords):
            x = coords[:, 0]
            y = coords[:, 1]
            r = R_mid - y
            theta = -np.pi / 2 + x / R_mid
            return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    return shapely.transform(geom, tx)


def _build_tapered_plaque(half_w: float, border_width: float, pad: float, taper_len: float):
    """Build a hexagonal plaque centered on (0, 0) in the flat border frame.

    The plaque is `2 * half_w` wide at its middle, tapers to a point `taper_len`
    further out on each side, and extends `border_width/2 + pad` above/below
    the border centerline. Subtracting this (after bending) from the border
    ring leaves the border narrowing to a point on each side of the text.
    """
    from shapely.geometry import Polygon as _P
    outer = border_width / 2.0 + pad
    return _P([
        (-half_w - taper_len, 0.0),             # left apex
        (-half_w, -outer),                      # lower-left
        ( half_w, -outer),                      # lower-right
        ( half_w + taper_len, 0.0),             # right apex
        ( half_w, +outer),                      # upper-right
        (-half_w, +outer),                      # upper-left
    ])


# ---------------------------------------------------------------------------
# SVG loading
# ---------------------------------------------------------------------------

def _svg_shape_to_polygon(shape: Shape) -> Polygon | MultiPolygon | None:
    """Convert one svgelements Shape into a shapely (Multi)Polygon.

    Uses even-odd fill: each closed subpath toggles inside/outside so interior
    subpaths become holes in the enclosing polygon.
    """
    path = SvgPath(shape)
    try:
        path.approximate_arcs_with_cubics()
    except Exception:
        pass

    rings: list[list[tuple[float, float]]] = []
    for sub in path.as_subpaths():
        sp = SvgPath(sub)
        try:
            pts = sp.npoint(np.linspace(0.0, 1.0, 256))
        except Exception:
            continue
        coords: list[tuple[float, float]] = []
        for p in pts:
            x, y = float(p[0]), float(p[1])
            if not coords or coords[-1] != (x, y):
                coords.append((x, y))
        if len(coords) >= 3:
            rings.append(coords)

    if not rings:
        return None

    polys: list[Polygon] = []
    for r in rings:
        try:
            p = Polygon(r)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and p.area > 0:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return None

    result = polys[0]
    for p in polys[1:]:
        result = result.symmetric_difference(p)  # even-odd

    if result.is_empty:
        return None
    if not result.is_valid:
        result = result.buffer(0)
    return result


def load_svg_as_polygon(svg_path: Path):
    """Parse SVG and return a combined shapely (Multi)Polygon, Y-flipped for 3D."""
    svg = SVG.parse(str(svg_path))
    geoms = []
    for element in svg.elements():
        if isinstance(element, Shape):
            geom = _svg_shape_to_polygon(element)
            if geom is not None and not geom.is_empty:
                geoms.append(geom)
    if not geoms:
        raise ValueError(f"No drawable shapes found in {svg_path}")

    merged = unary_union(geoms)
    # SVG Y goes down; flip so the token reads correctly in 3D.
    merged = affinity.scale(merged, xfact=1.0, yfact=-1.0, origin=(0, 0))
    if not merged.is_valid:
        merged = merged.buffer(0)
    return merged


# ---------------------------------------------------------------------------
# Token composition
# ---------------------------------------------------------------------------

def fit_art_to_disc(art, inner_radius: float):
    """Center the art on the origin and scale it to fit inside `inner_radius`."""
    minx, miny, maxx, maxy = art.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    art = affinity.translate(art, xoff=-cx, yoff=-cy)

    half_w = (maxx - minx) / 2.0
    half_h = (maxy - miny) / 2.0
    current_radius = float(np.hypot(half_w, half_h))
    if current_radius <= 0:
        raise ValueError("Art has zero size after loading.")
    scale = inner_radius / current_radius
    return affinity.scale(art, xfact=scale, yfact=scale, origin=(0, 0))


def build_layers(art, disc_radius: float, border_width: float, art_margin: float):
    """Return (base_disc, border_ring, art_clipped) as shapely geometries."""
    disc = Point(0, 0).buffer(disc_radius, quad_segs=128)
    inner_disc = Point(0, 0).buffer(disc_radius - border_width, quad_segs=128)
    border = disc.difference(inner_disc)

    inner_safe_radius = disc_radius - border_width - art_margin
    if inner_safe_radius <= 0:
        raise ValueError("Border/margin leave no room for art.")

    art_fit = fit_art_to_disc(art, inner_safe_radius)
    art_clipped = art_fit.intersection(inner_disc)
    if art_clipped.is_empty:
        raise ValueError("Art did not survive clipping to the inner disc.")
    if not art_clipped.is_valid:
        art_clipped = art_clipped.buffer(0)
    return disc, border, art_clipped


# ---------------------------------------------------------------------------
# Extrusion
# ---------------------------------------------------------------------------

def _to_polygon_list(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def extrude_geom(geom, height: float, z_offset: float = 0.0) -> trimesh.Trimesh:
    meshes: list[trimesh.Trimesh] = []
    for poly in _to_polygon_list(geom):
        if poly.area <= 0:
            continue
        m = trimesh.creation.extrude_polygon(poly, height=height)
        if z_offset:
            m.apply_translation((0, 0, z_offset))
        meshes.append(m)
    if not meshes:
        raise ValueError("Nothing to extrude.")
    return trimesh.util.concatenate(meshes)


def build_token_mesh(base, border, art, base_thickness: float, relief_height: float,
                     extra_raised=None) -> trimesh.Trimesh:
    """Build a single watertight, manifold token mesh.

    Splits the disc into two non-overlapping 2D regions:
      * raised  = border ∪ art ∪ extra_raised  (extruded to base + relief)
      * flat    = disc − raised                (extruded to base only)

    Then boolean-unions the two prisms so the shared vertical wall between
    them is welded into a single 2-manifold surface.
    """
    raised_parts = [border, art]
    if extra_raised is not None and not extra_raised.is_empty:
        raised_parts.append(extra_raised)
    raised = unary_union(raised_parts)
    if not raised.is_valid:
        raised = raised.buffer(0)
    # Clip to disc (defensive) and compute the flat remainder.
    raised = raised.intersection(base)
    flat = base.difference(raised)

    total_height = base_thickness + relief_height
    raised_mesh = extrude_geom(raised, height=total_height, z_offset=0.0)
    flat_mesh = extrude_geom(flat, height=base_thickness, z_offset=0.0)

    # Boolean union welds the shared wall and removes internal faces, yielding
    # a single manifold mesh suitable for 3D printing / slicing.
    try:
        mesh = trimesh.boolean.union([flat_mesh, raised_mesh], engine="manifold")
    except Exception:
        mesh = trimesh.boolean.union([flat_mesh, raised_mesh])
    return mesh


def finalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean up a mesh so it exports as a watertight, printable STL."""
    mesh.process(validate=True)
    mesh.merge_vertices()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.fix_normals()
    return mesh


def build_token_from_art(
    art_geom,
    size_inches: float = 1.0,
    base_thickness: float = DEFAULT_BASE_THICKNESS_MM,
    relief_height: float = DEFAULT_RELIEF_HEIGHT_MM,
    border_frac: float = DEFAULT_BORDER_FRAC,
    art_margin_frac: float = DEFAULT_ART_MARGIN_FRAC,
    text: Optional[str] = None,
    text_position: str = "bottom",
    text_height_frac: float = DEFAULT_TEXT_HEIGHT_FRAC,
    text_padding_frac: float = DEFAULT_TEXT_PADDING_FRAC,
    text_max_width_frac: float = DEFAULT_TEXT_MAX_WIDTH_FRAC,
    text_taper_frac: float = DEFAULT_TEXT_TAPER_FRAC,
    font_family: str = "sans-serif",
    font_weight: str = "bold",
) -> trimesh.Trimesh:
    """High-level helper: shapely art polygon -> finalized token mesh.

    `art_geom` should already be Y-flipped (as returned by `load_svg_as_polygon`)
    and may be pre-cropped by the caller.

    If `text` is provided, a text label is embedded as a break in the border
    ring at `text_position` ("top" or "bottom"). The border is carved out
    around the text for visual separation.
    """
    target_diameter_mm = size_inches * MM_PER_INCH
    disc_radius = target_diameter_mm / 2.0
    border_width = disc_radius * border_frac
    art_margin = disc_radius * art_margin_frac

    base, border, art = build_layers(art_geom, disc_radius, border_width, art_margin)

    text_geom = None
    if text and text.strip():
        text_geom = build_text_polygon(
            text,
            target_height=border_width * text_height_frac,
            max_width=target_diameter_mm * text_max_width_frac,
            font_family=font_family,
            font_weight=font_weight,
        )
        if text_geom is not None and not text_geom.is_empty:
            # Flat-frame bounds of the centered text.
            tminx, tminy, tmaxx, tmaxy = text_geom.bounds
            half_w_flat = (tmaxx - tminx) / 2.0
            pad = border_width * text_padding_frac
            taper_len = border_width * text_taper_frac
            R_mid = disc_radius - border_width / 2.0

            # Build the hexagonal tapered plaque in the flat border frame,
            # then bend both the plaque and the text onto the disc arc so they
            # follow the border's curvature.
            plaque_flat = _build_tapered_plaque(
                half_w_flat + pad, border_width, pad, taper_len,
            )
            plaque_curved = _curve_to_arc(plaque_flat, R_mid, position=text_position,
                                          tolerance=max(border_width / 15.0, 0.1))
            text_geom = _curve_to_arc(text_geom, R_mid, position=text_position,
                                      tolerance=max(border_width / 20.0, 0.08))

            # Clip text to the disc just in case, and carve the plaque out of
            # the border ring so the border tapers into a point at each side
            # of the text.
            text_geom = text_geom.intersection(base)
            if not text_geom.is_empty:
                border = border.difference(plaque_curved)
                if not border.is_valid:
                    border = border.buffer(0)
            else:
                text_geom = None

    mesh = build_token_mesh(
        base, border, art,
        base_thickness=base_thickness,
        relief_height=relief_height,
        extra_raised=text_geom,
    )
    return finalize_mesh(mesh)


def build_text_polygon(
    text: str,
    target_height: float,
    max_width: Optional[float] = None,
    font_family: str = "sans-serif",
    font_weight: str = "bold",
):
    """Build a shapely (Multi)Polygon for the given text, horizontally centered at x=0.

    Letters sit upright (Y-up), scaled so the cap/ascender height equals
    `target_height`. If `max_width` is set and the rendered text exceeds it,
    the whole string is uniformly scaled down to fit.
    """
    if not text:
        return None
    fp = FontProperties(family=font_family, weight=font_weight)
    # Build the whole string as a single path (handles kerning/advances).
    tp = TextPath((0, 0), text, size=100.0, prop=fp)
    polys = tp.to_polygons()
    if not polys:
        return None

    rings = [[(float(x), float(y)) for x, y in arr] for arr in polys]
    geom = _rings_to_polygon(rings)
    if geom is None or geom.is_empty:
        return None

    minx, miny, maxx, maxy = geom.bounds
    h = maxy - miny
    w = maxx - minx
    if h <= 0 or w <= 0:
        return None

    scale = target_height / h
    if max_width is not None and w * scale > max_width:
        scale = max_width / w
    geom = affinity.scale(geom, xfact=scale, yfact=scale, origin=(0, 0))

    # Center horizontally and vertically on origin (caller will translate).
    minx, miny, maxx, maxy = geom.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    geom = affinity.translate(geom, xoff=-cx, yoff=-cy)
    return geom


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a single-color SVG into a circular token STL."
    )
    ap.add_argument("input", type=Path, help="Input SVG file.")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output STL path (default: <input>.stl).")
    ap.add_argument("-s", "--size", type=int, choices=[1, 2, 3], default=1,
                    help="Final disc diameter in inches. Default: 1.")
    ap.add_argument("--base-thickness", type=float, default=DEFAULT_BASE_THICKNESS_MM,
                    help=f"Base disc thickness in mm (default: {DEFAULT_BASE_THICKNESS_MM}).")
    ap.add_argument("--relief-height", type=float, default=DEFAULT_RELIEF_HEIGHT_MM,
                    help=f"Border/art relief height in mm (default: {DEFAULT_RELIEF_HEIGHT_MM}).")
    ap.add_argument("--border-frac", type=float, default=DEFAULT_BORDER_FRAC,
                    help=f"Border width as fraction of disc radius (default: {DEFAULT_BORDER_FRAC}).")
    ap.add_argument("--art-margin-frac", type=float, default=DEFAULT_ART_MARGIN_FRAC,
                    help=f"Gap between border and art as fraction of disc radius "
                         f"(default: {DEFAULT_ART_MARGIN_FRAC}).")
    ap.add_argument("--text", type=str, default=None,
                    help="Optional label text placed as a break in the border ring.")
    ap.add_argument("--text-position", choices=["top", "bottom"], default="bottom",
                    help="Where to place the text on the border (default: bottom).")
    ap.add_argument("--text-height-frac", type=float, default=DEFAULT_TEXT_HEIGHT_FRAC,
                    help=f"Text height as fraction of border width (default: {DEFAULT_TEXT_HEIGHT_FRAC}).")
    ap.add_argument("--text-taper-frac", type=float, default=DEFAULT_TEXT_TAPER_FRAC,
                    help=f"Length of each border taper as multiple of border width (default: {DEFAULT_TEXT_TAPER_FRAC}).")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    output = args.output or args.input.with_suffix(".stl")
    target_diameter_mm = args.size * MM_PER_INCH

    print(f"Loading {args.input}...")
    art_geom = load_svg_as_polygon(args.input)

    print("Building token mesh...")
    mesh = build_token_from_art(
        art_geom,
        size_inches=float(args.size),
        base_thickness=args.base_thickness,
        relief_height=args.relief_height,
        border_frac=args.border_frac,
        art_margin_frac=args.art_margin_frac,
        text=args.text,
        text_position=args.text_position,
        text_height_frac=args.text_height_frac,
        text_taper_frac=args.text_taper_frac,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)

    ex = mesh.extents
    status = "watertight ✅" if mesh.is_watertight else "NOT watertight ⚠️"
    print(
        f"Wrote {output}  "
        f"({len(mesh.faces)} triangles, "
        f"{ex[0]:.2f} x {ex[1]:.2f} x {ex[2]:.2f} mm, "
        f"disc = {args.size}\" / {target_diameter_mm:.2f} mm, {status})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
