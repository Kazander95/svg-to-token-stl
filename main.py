"""Convert a single-color SVG into a circular token STL.

The script wraps the SVG art in a white disc with a black border ring,
extrudes the whole thing into a 3D token, and scales the final mesh so the
disc is exactly N inches across (default 1").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
from svgelements import SVG, Shape, Path as SvgPath

MM_PER_INCH = 25.4

# Defaults (mm). These are final, post-scale dimensions.
DEFAULT_BASE_THICKNESS_MM = 2.0
DEFAULT_RELIEF_HEIGHT_MM = 0.6
# Border width and art margin as a fraction of the disc radius.
DEFAULT_BORDER_FRAC = 0.06
DEFAULT_ART_MARGIN_FRAC = 0.04


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


def build_token_mesh(base, border, art, base_thickness: float, relief_height: float) -> trimesh.Trimesh:
    """Build a single watertight, manifold token mesh.

    Splits the disc into two non-overlapping 2D regions:
      * raised  = border ∪ art    (extruded to base + relief)
      * flat    = disc − raised   (extruded to base only)

    Then boolean-unions the two prisms so the shared vertical wall between
    them is welded into a single 2-manifold surface.
    """
    raised = unary_union([border, art])
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
) -> trimesh.Trimesh:
    """High-level helper: shapely art polygon -> finalized token mesh.

    `art_geom` should already be Y-flipped (as returned by `load_svg_as_polygon`)
    and may be pre-cropped by the caller.
    """
    target_diameter_mm = size_inches * MM_PER_INCH
    disc_radius = target_diameter_mm / 2.0
    border_width = disc_radius * border_frac
    art_margin = disc_radius * art_margin_frac

    base, border, art = build_layers(art_geom, disc_radius, border_width, art_margin)
    mesh = build_token_mesh(base, border, art, base_thickness, relief_height)
    return finalize_mesh(mesh)


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
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    output = args.output or args.input.with_suffix(".stl")

    # Work at final mm scale directly: disc radius in mm.
    target_diameter_mm = args.size * MM_PER_INCH
    disc_radius = target_diameter_mm / 2.0
    border_width = disc_radius * args.border_frac
    art_margin = disc_radius * args.art_margin_frac

    print(f"Loading {args.input}...")
    art_geom = load_svg_as_polygon(args.input)

    print("Composing token layers...")
    base, border, art = build_layers(art_geom, disc_radius, border_width, art_margin)

    print("Extruding to 3D...")
    mesh = build_token_mesh(
        base, border, art,
        base_thickness=args.base_thickness,
        relief_height=args.relief_height,
    )

    mesh.process(validate=True)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces() if hasattr(mesh, "remove_duplicate_faces") else None
    mesh.remove_degenerate_faces() if hasattr(mesh, "remove_degenerate_faces") else None
    mesh.fix_normals()

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
