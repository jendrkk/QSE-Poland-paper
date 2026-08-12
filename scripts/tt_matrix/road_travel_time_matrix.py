#!/usr/bin/env python3
"""
road_travel_time_matrix.py
==========================

Compute a commune-to-commune *road* travel-time matrix (TTM) for Poland from an
arbitrary road-network ``.osm.pbf`` snapshot and a commune GeoPackage produced
by ``commune_centroids.py`` (layers ``communes`` + ``commune_candidates``,
EPSG:3035, id column ``JPT_KOD_JE``).

This is the **baseline** (free-flow, no-congestion, single-mode) matrix used to
calibrate the MRRH-2018 Poland model. It is written so that the two planned
extensions -- endogenous road congestion (Allen-Arkolakis 2022 / BPR) and a
second rail/PT mode from GTFS -- can be bolted on without rewriting it:

  * weight assignment is separated from routing behind ``edge_time()``, whose
    signature already carries a ``load`` argument (a no-op at load == 0), so an
    outer traffic-assignment loop can re-weight edges in place;
  * a per-edge ``capacity`` attribute is populated even in the baseline;
  * centroid selection, snapping, commune aggregation, the diagonal imputation
    and the output format are engine- and mode-agnostic, so a future
    schedule-based rail engine can reuse everything except the graph build and
    the shortest-path call, and emit a matrix in the identical layout for the
    nested-Frechet modal aggregator.

Engine choice
-------------
A self-contained igraph graph is built from the PBF rather than delegating to
OSRM. Rationale: the three project-specific requirements -- Polish statutory
speed defaults with population-grid built-up inference applied *in-engine*, a
self-contained network-trustworthiness verdict, and mutable edge weights for the
future congestion fixed point -- are exactly the things OSRM cannot do without
brittle out-of-band passes (car.lua edits, a PBF rewrite, traffic-CSV
round-trips). A graph we own pays a one-time build cost and gives all three.

Pipeline
--------
1. Parse ways from the PBF (drivable classes only), resolving a free-flow speed
   per way from ``maxspeed`` when present, else the statutory Polish class
   default split rural / built-up, times an optional realistic-speed factor.
2. Build a directed graph (oneway-aware), extract the giant strongly-connected
   component (guarantees every routed OD pair is reachable), and contract
   degree-2 chains (preserving cumulative time and length) into a routing core.
3. Validate the network and print a GOOD / USABLE-WITH-CAVEATS / BAD verdict
   with explicit reasons (connectivity, length vs benchmark, maxspeed coverage,
   snap distances, unreachable pairs).
4. Select origin/destination points per ``--centroid-type`` and snap them to the
   nearest routable node.
5. Many-to-many Dijkstra over all workers, then aggregate to commune x commune
   (for ``multiple`` this is the population-weighted double sum over candidate
   pairs, i.e. T = A D A^T with A the row-normalised commune x node weight
   matrix).
6. Impute the diagonal with the ARSW self-commute and write NPZ + CSV.

Usage
-----
    python road_travel_time_matrix.py --network data/raw/osm_pbf/poland_roads_2021-12-31_optimal.osm.pbf
    python road_travel_time_matrix.py --network ... --centroid-type multiple --n-centroids 3 --workers 32
    python road_travel_time_matrix.py --network ... --centroid-type pop-weighted \
        --builtup-detection grid --population-grid data/raw/pop/poland_bbox_pop_100m.parquet
    python road_travel_time_matrix.py --help

All paths default to the QSE_Poland_paper repository layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy geo/graph deps are imported lazily inside main() so that --help and unit
# imports stay cheap and the failure messages are precise.


# --------------------------------------------------------------------------- #
# Repository-relative default paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMUNES = (
    REPO_ROOT / "data" / "processed" / "shapefiles" / "communes_2021.gpkg"
)
DEFAULT_POP_GRID = (
    REPO_ROOT / "data" / "raw" / "pop" / "poland_bbox_pop_100m.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "tt_matrix"
DEFAULT_ID_COL = "JPT_KOD_JE"
COMMUNES_LAYER = "communes"
CANDIDATE_LAYER = "commune_candidates"

CRS_METRIC = "EPSG:3035"   # matches the commune GeoPackage
CRS_WGS84 = "EPSG:4326"    # OSM node coordinates

LOGGER = logging.getLogger("road_ttm")

MPH_TO_KMH = 1.609344


# --------------------------------------------------------------------------- #
# Speed model
# --------------------------------------------------------------------------- #
# Statutory Polish free-flow defaults (km/h) as (rural, built-up). Used only
# where an OSM ``maxspeed`` tag is absent or unparseable. Overridable via
# --speed-profile (JSON with the same {class: [rural, urban]} shape).
#   motorway (autostrada, A)          140
#   trunk    (ekspresowa, S)          120 dual / 100 single  -> 120 rural default
#   primary/secondary/tertiary        90 rural / 50 built-up  (droga krajowa/woj./pow.)
#   unclassified / road               70 rural / 50 built-up
#   residential                       50
#   living_street                     20
#   service                           30
DEFAULT_SPEED_PROFILE: dict[str, tuple[float, float]] = {
    "motorway": (140, 140),
    "motorway_link": (80, 80),
    "trunk": (120, 100),
    "trunk_link": (80, 70),
    "primary": (90, 50),
    "primary_link": (70, 50),
    "secondary": (90, 50),
    "secondary_link": (60, 50),
    "tertiary": (90, 50),
    "tertiary_link": (60, 50),
    "unclassified": (70, 50),
    "road": (70, 50),
    "residential": (50, 50),
    "living_street": (20, 20),
    "service": (30, 30),
}
# Classes routed for a car. ``track`` and non-car ways (foot/cycle/path/...) are
# excluded. ``service`` is included at low speed for settlement access.
DRIVABLE_CLASSES = set(DEFAULT_SPEED_PROFILE.keys())

# Nominal lane capacities (veh/h/lane) by class -- stored per edge for the
# future congestion extension, NOT used in the baseline weights.
LANE_CAPACITY = {
    "motorway": 2000, "motorway_link": 1500,
    "trunk": 1800, "trunk_link": 1400,
    "primary": 1400, "primary_link": 1000,
    "secondary": 1200, "secondary_link": 900,
    "tertiary": 900, "tertiary_link": 700,
    "unclassified": 600, "road": 600,
    "residential": 600, "living_street": 300, "service": 300,
}

_ONEWAY_YES = {"yes", "true", "1"}
_ONEWAY_REV = {"-1", "reverse"}
_ACCESS_NO = {"no", "private", "agricultural", "forestry", "delivery", "customers"}


def parse_maxspeed(raw) -> tuple[float | None, bool]:
    """Return (speed_kmh, is_explicit).

    ``is_explicit`` is True when the value came from a real numeric/implicit
    ``maxspeed`` tag (used for the coverage statistic). Handles numeric values,
    ``mph``, semicolon/space lists (min taken), and Polish implicit zone tags.
    Returns (None, False) for missing / signal / variable / none.
    """
    if raw is None:
        return None, False
    s = str(raw).strip().lower()
    if not s:
        return None, False

    implicit = {
        "pl:motorway": 140.0, "pl:trunk": 120.0, "pl:expressway": 120.0,
        "pl:rural": 90.0, "pl:urban": 50.0, "pl:living_street": 20.0,
        "pl:zone30": 30.0, "pl:zone:30": 30.0, "pl:bicycle_road": 30.0,
    }
    if s in implicit:
        return implicit[s], True
    if s in ("none", "no", "unlimited", "signals", "variable", "unknown",
             "default", "fixme"):
        return None, False
    if s in ("walk", "foot"):
        return 5.0, True

    # Lists: "50;70" / "50, 70" -> take the minimum sane token.
    tokens = re.split(r"[;,]", s)
    speeds: list[float] = []
    for tok in tokens:
        tok = tok.strip()
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*mph$", tok)
        if m:
            speeds.append(float(m.group(1)) * MPH_TO_KMH)
            continue
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:km/?h|kph)?$", tok)
        if m:
            speeds.append(float(m.group(1)))
            continue
        m = re.search(r"([0-9]{2,3})", tok)
        if m:
            speeds.append(float(m.group(1)))
    speeds = [v for v in speeds if 3.0 <= v <= 150.0]
    if speeds:
        return float(min(speeds)), True
    return None, False


def resolve_way_speed(
    tags: dict,
    hwy: str,
    builtup: bool,
    profile: dict[str, tuple[float, float]],
    speed_factor: float,
) -> tuple[float, bool]:
    """Free-flow speed (km/h) for a way and whether an explicit tag was used.

    Precedence: explicit ``maxspeed`` (as-is) -> class default (rural/built-up)
    times ``speed_factor``. Falls back to a 40 km/h generic if the class is
    unknown (should not happen for DRIVABLE_CLASSES).
    """
    ms, explicit = parse_maxspeed(tags.get("maxspeed"))
    if ms is not None:
        return ms, True
    rural, urban = profile.get(hwy, (40.0, 40.0))
    base = urban if builtup else rural
    return base * speed_factor, False


def edge_time(length_m: float, speed_kmh: float, capacity: float = 0.0,
              load: float = 0.0) -> float:
    """Edge traversal time in seconds.

    Baseline: free-flow ``length / speed``. The ``capacity``/``load`` arguments
    are the seam for the congestion extension -- a BPR term
    ``* (1 + alpha*(load/capacity)**beta)`` would multiply the free-flow time
    here, driven by an outer traffic-assignment loop. They are inert at
    ``load == 0``.
    """
    v = max(speed_kmh, 1.0) / 3.6  # m/s
    t = length_m / v
    return t


# --------------------------------------------------------------------------- #
# Built-up ("obszar zabudowany") inference for ambiguous classes
# --------------------------------------------------------------------------- #
# A class is "ambiguous" when its rural and built-up defaults differ; only then
# does built-up status change the imputed speed (and only when maxspeed is
# missing).
def ambiguous_classes(profile) -> set[str]:
    return {k for k, (r, u) in profile.items() if r != u}


_ZONE_URBAN = re.compile(r"urban|:urban|zone")
_ZONE_RURAL = re.compile(r"rural|:rural")


def builtup_from_tags(tags: dict, hwy: str) -> bool | None:
    """OSM-native built-up signal, or None if the tags say nothing.

    residential / living_street are urban by construction. Otherwise look at
    source:maxspeed / zone:maxspeed / zone:traffic for PL:urban vs PL:rural.
    """
    if hwy in ("residential", "living_street", "service"):
        return True
    for key in ("source:maxspeed", "zone:maxspeed", "zone:traffic",
                "maxspeed:type"):
        v = tags.get(key)
        if not v:
            continue
        v = str(v).lower()
        if _ZONE_RURAL.search(v):
            return False
        if _ZONE_URBAN.search(v):
            return True
    return None


# --------------------------------------------------------------------------- #
# PBF parsing
# --------------------------------------------------------------------------- #
def parse_pbf_ways(network: Path):
    """Read drivable ways from the PBF.

    Returns ``ways`` = list of dicts with keys
    ``nodes`` (list of osm node ids), ``lonlat`` (list of (lon, lat)),
    ``hwy``, ``tags``. Node de-duplication and graph assembly happen later.
    """
    import osmium

    class RoadHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways: list[dict] = []
            self.n_seen = 0

        def way(self, w):
            self.n_seen += 1
            tags = {t.k: t.v for t in w.tags}
            hwy = tags.get("highway")
            if hwy not in DRIVABLE_CLASSES:
                return
            # Access restrictions that forbid general motor traffic.
            acc = tags.get("motor_vehicle") or tags.get("motorcar") or tags.get("access")
            if acc in _ACCESS_NO:
                return
            coords = [(n.ref, n.location.lon, n.location.lat)
                      for n in w.nodes if n.location.valid()]
            if len(coords) < 2:
                return
            self.ways.append({
                "nodes": [c[0] for c in coords],
                "lonlat": [(c[1], c[2]) for c in coords],
                "hwy": hwy,
                "tags": tags,
            })

    LOGGER.info("Parsing PBF (drivable ways): %s", network)
    t0 = time.time()
    h = RoadHandler()
    # locations=True populates node coordinates on the way nodes.
    h.apply_file(str(network), locations=True)
    LOGGER.info(
        "  kept %s drivable ways of %s total ways (%.1fs)",
        f"{len(h.ways):,}", f"{h.n_seen:,}", time.time() - t0,
    )
    return h.ways


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph(ways, profile, speed_factor, builtup_mode,
                pop_tree, pop_cellsize, builtup_ambig_classes,
                transformer_to_metric):
    """Assemble a directed graph from parsed ways.

    Returns a dict with:
      ``n_nodes``            number of distinct nodes
      ``node_lonlat``        (n_nodes, 2) float64 lon/lat
      ``node_xy``            (n_nodes, 2) float64 EPSG:3035 (for snapping)
      ``edges``              list of (u, v, length_m, time_s, capacity)
      ``class_len``          dict class -> total one-way length (m)
      ``class_len_explicit`` dict class -> length with explicit maxspeed (m)
      ``ambiguous_len``      total length of ambiguous classes w/o maxspeed
      ``builtup_len``        of that, length inferred built-up
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    node_id_map: dict[int, int] = {}
    node_lon: list[float] = []
    node_lat: list[float] = []

    def nid(osm_id, lon, lat):
        k = node_id_map.get(osm_id)
        if k is None:
            k = len(node_lon)
            node_id_map[osm_id] = k
            node_lon.append(lon)
            node_lat.append(lat)
        return k

    edges: list[tuple] = []
    class_len: dict[str, float] = defaultdict(float)
    class_len_explicit: dict[str, float] = defaultdict(float)
    ambiguous_len = 0.0
    builtup_len = 0.0

    # Pre-compute built-up decision per way (representative midpoint).
    for w in ways:
        hwy = w["hwy"]
        tags = w["tags"]
        lonlat = w["lonlat"]
        nodes = w["nodes"]

        # Built-up inference (only matters for ambiguous classes w/o maxspeed).
        builtup = False
        is_ambig = hwy in builtup_ambig_classes
        _, explicit = parse_maxspeed(tags.get("maxspeed"))
        if is_ambig and not explicit and builtup_mode != "none":
            b = builtup_from_tags(tags, hwy)
            if b is None and builtup_mode == "grid" and pop_tree is not None:
                mlon = lonlat[len(lonlat) // 2][0]
                mlat = lonlat[len(lonlat) // 2][1]
                mx, my = transformer_to_metric.transform(mlon, mlat)
                dist, _ = pop_tree.query([mx, my], k=1)
                b = bool(dist <= pop_cellsize)
            builtup = bool(b) if b is not None else False
        elif hwy in ("residential", "living_street", "service"):
            builtup = True

        speed, used_explicit = resolve_way_speed(
            tags, hwy, builtup, profile, speed_factor)

        # Oneway handling.
        ow = str(tags.get("oneway", "")).strip().lower()
        junction = str(tags.get("junction", "")).strip().lower()
        forward = True
        backward = True
        if ow in _ONEWAY_YES or junction == "roundabout" or hwy == "motorway":
            backward = False
        if ow in _ONEWAY_REV:
            forward, backward = False, True
        if ow in ("no", "false", "0"):
            forward = backward = True

        lanes = tags.get("lanes")
        try:
            n_lanes = max(1, int(float(str(lanes).split(";")[0])))
        except (TypeError, ValueError):
            n_lanes = 2 if not (backward is False) else 1
        cap = LANE_CAPACITY.get(hwy, 600) * n_lanes

        # Per-segment geodesic lengths.
        lons = [c[0] for c in lonlat]
        lats = [c[1] for c in lonlat]
        seg_len = _segment_lengths(geod, lons, lats)

        for i in range(len(nodes) - 1):
            L = seg_len[i]
            if L <= 0:
                continue
            u = nid(nodes[i], lons[i], lats[i])
            v = nid(nodes[i + 1], lons[i + 1], lats[i + 1])
            if u == v:
                continue
            t = edge_time(L, speed, cap, 0.0)
            if forward:
                edges.append((u, v, L, t, cap))
            if backward:
                edges.append((v, u, L, t, cap))

        # Statistics accumulate on the (undirected) way length once.
        wlen = float(sum(seg_len))
        class_len[hwy] += wlen
        if used_explicit:
            class_len_explicit[hwy] += wlen
        if is_ambig and not used_explicit:
            ambiguous_len += wlen
            if builtup:
                builtup_len += wlen

    node_lon_a = np.asarray(node_lon, dtype=np.float64)
    node_lat_a = np.asarray(node_lat, dtype=np.float64)
    mx, my = transformer_to_metric.transform(node_lon_a, node_lat_a)
    node_xy = np.column_stack([mx, my])

    return {
        "n_nodes": len(node_lon),
        "node_lonlat": np.column_stack([node_lon_a, node_lat_a]),
        "node_xy": node_xy,
        "edges": edges,
        "class_len": dict(class_len),
        "class_len_explicit": dict(class_len_explicit),
        "ambiguous_len": ambiguous_len,
        "builtup_len": builtup_len,
    }


def _segment_lengths(geod, lons, lats) -> np.ndarray:
    """Geodesic length (m) of each consecutive segment along a polyline."""
    if len(lons) < 2:
        return np.zeros(max(len(lons) - 1, 0))
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    _, _, dist = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return np.asarray(dist, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Degree-2 chain contraction (osmnx-style, direction-aware)
# --------------------------------------------------------------------------- #
def simplify_graph(n_nodes: int, edges: list[tuple], protected: set[int]):
    """Contract non-endpoint degree-2 chains, preserving cumulative time/length.

    ``edges`` = list of (u, v, length, time). A node is an endpoint (kept) if it
    is protected, is an intersection / dead-end (undirected degree != 2), or has
    an asymmetric in/out structure. Chains between endpoints are merged per
    direction (a direction survives only if every step of the chain exists in
    that direction -- so oneway chains stay oneway).

    Returns ``(n_new, new_edges, old2new)`` where ``new_edges`` is a list of
    (u, v, length, time) on the reindexed node set and ``old2new`` maps every
    surviving old node id to its new id.
    """
    de: dict[tuple[int, int], tuple[float, float]] = {}
    und: dict[int, set[int]] = defaultdict(set)
    for (u, v, l, t) in edges:
        if u == v:
            continue
        key = (u, v)
        cur = de.get(key)
        if cur is None or t < cur[1]:
            de[key] = (l, t)
        und[u].add(v)
        und[v].add(u)

    outdeg: dict[int, int] = defaultdict(int)
    indeg: dict[int, int] = defaultdict(int)
    for (u, v) in de:
        outdeg[u] += 1
        indeg[v] += 1

    def is_endpoint(n: int) -> bool:
        if n in protected:
            return True
        nb = und[n]
        if len(nb) != 2:
            return True
        i, o = indeg[n], outdeg[n]
        if i != o or i not in (1, 2):
            return True
        return False

    endpoints = {n for n in und if is_endpoint(n)}

    new_edges: list[tuple] = []
    visited_seg: set[frozenset] = set()

    for u in endpoints:
        for v in list(und[u]):
            seg0 = frozenset((u, v))
            if seg0 in visited_seg:
                continue
            visited_seg.add(seg0)
            path = [u, v]
            prev, cur = u, v
            while cur not in endpoints:
                nxts = [w for w in und[cur] if w != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_seg.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == u:  # closed loop back to start
                    break
            F = path[-1]
            # Forward u -> ... -> F.
            fl = ft = 0.0
            fwd_ok = True
            for a, b in zip(path, path[1:]):
                e = de.get((a, b))
                if e is None:
                    fwd_ok = False
                    break
                fl += e[0]
                ft += e[1]
            if fwd_ok:
                new_edges.append((u, F, fl, ft))
            # Backward F -> ... -> u.
            bl = bt = 0.0
            bwd_ok = True
            for a, b in zip(path, path[1:]):
                e = de.get((b, a))
                if e is None:
                    bwd_ok = False
                    break
                bl += e[0]
                bt += e[1]
            if bwd_ok:
                new_edges.append((F, u, bl, bt))

    # Leftover segments (isolated interior cycles): keep uncontracted.
    for (u, v), (l, t) in de.items():
        if frozenset((u, v)) not in visited_seg:
            new_edges.append((u, v, l, t))

    survive: set[int] = set(endpoints)
    for (u, v, _l, _t) in new_edges:
        survive.add(u)
        survive.add(v)
    old2new = {old: i for i, old in enumerate(sorted(survive))}
    remapped = [(old2new[u], old2new[v], l, t) for (u, v, l, t) in new_edges]
    return len(old2new), remapped, old2new


# --------------------------------------------------------------------------- #
# igraph construction + strongly-connected core
# --------------------------------------------------------------------------- #
def giant_scc(n_nodes: int, edges: list[tuple]):
    """Return (members_old_idx, sub_edges) for the largest strongly-connected
    component. ``sub_edges`` uses the ORIGINAL node ids (filtered to members).
    """
    import igraph

    g = igraph.Graph(n=n_nodes,
                     edges=[(u, v) for (u, v, *_r) in edges],
                     directed=True)
    comp = g.connected_components(mode="strong")
    sizes = comp.sizes()
    giant = int(np.argmax(sizes))
    membership = np.asarray(comp.membership)
    keep = membership == giant
    members = np.nonzero(keep)[0]
    member_set = set(members.tolist())
    sub_edges = [(u, v, l, t) for (u, v, l, t, *_c) in edges
                 if u in member_set and v in member_set]
    return members, sub_edges, sizes


def build_igraph(n_nodes: int, edges: list[tuple]):
    import igraph
    g = igraph.Graph(n=n_nodes,
                     edges=[(e[0], e[1]) for e in edges],
                     directed=True)
    g.es["time"] = [float(e[3]) for e in edges]
    g.es["length"] = [float(e[2]) for e in edges]
    return g


# --------------------------------------------------------------------------- #
# Parallel many-to-many shortest paths
# --------------------------------------------------------------------------- #
_WORKER_GRAPH: dict[str, object] = {}


def _load_worker_graph(path: str):
    import igraph
    g = _WORKER_GRAPH.get(path)
    if g is None:
        g = igraph.Graph.Read_Pickle(path)
        _WORKER_GRAPH[path] = g
    return g


def _route_batch(graph_path: str, sources: list[int], targets: list[int]):
    g = _load_worker_graph(graph_path)
    d = g.distances(source=sources, target=targets, weights="time")
    return np.asarray(d, dtype=np.float64)


def route_matrix(graph, source_nodes, target_nodes, workers: int,
                 tmp_dir: Path | None):
    """Seconds matrix (len(source) x len(target)) via parallel igraph Dijkstra.

    The contracted graph is pickled once and lazily loaded per worker (loky
    reuses workers, so it is read at most once each), which avoids pickling the
    graph per task while keeping per-worker memory to a single copy.
    """
    from joblib import Parallel, delayed

    scratch = Path(tempfile.mkdtemp(prefix="road_ttm_g_",
                                    dir=str(tmp_dir) if tmp_dir else None))
    gpath = scratch / "graph.pkl"
    try:
        graph.write_pickle(str(gpath))
        n_jobs = workers if workers > 0 else (os.cpu_count() or 4)
        src = list(source_nodes)
        tgt = list(target_nodes)
        n_batches = max(n_jobs * 4, 1)
        batches = [b.tolist() for b in np.array_split(np.asarray(src), n_batches)
                   if len(b)]
        LOGGER.info(
            "Routing %d x %d (sec) | workers=%d | batches=%d",
            len(src), len(tgt), n_jobs, len(batches),
        )
        t0 = time.time()
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_route_batch)(str(gpath), batch, tgt) for batch in batches
        )
        D = np.vstack(parts)
        LOGGER.info("  routing done in %.1fs", time.time() - t0)
        return D
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Centroid selection
# --------------------------------------------------------------------------- #
def select_points(communes, candidates, id_col, centroid_type, n_centroids):
    """Return an ordered commune id list and, per commune, a list of
    ``(x, y, weight)`` in EPSG:3035 (weights sum to 1 within a commune).

    Modes: unweighted -> geometric centroid; pop-weighted -> weighted_centroid;
    best-pop -> rank-1 population candidate; multiple -> up to N candidates,
    population-weighted.
    """
    import shapely.wkt as wkt

    ids = list(communes[id_col].astype(str))
    per: dict[str, list[tuple[float, float, float]]] = {}

    if centroid_type in ("unweighted", "pop-weighted"):
        col = "centroid" if centroid_type == "unweighted" else "weighted_centroid"
        for cid, w in zip(communes[id_col].astype(str), communes[col]):
            g = wkt.loads(w) if isinstance(w, str) else w
            per[cid] = [(g.x, g.y, 1.0)]
        return ids, per

    # best-pop / multiple use the candidate layer.
    if candidates is None:
        raise ValueError("centroid-type needs the commune_candidates layer")
    cand = candidates.sort_values([id_col, "cand_rank"])
    take = 1 if centroid_type == "best-pop" else max(1, n_centroids)
    grouped = {str(cid): sub for cid, sub in cand.groupby(id_col, sort=False)}

    for cid in ids:
        sub = grouped.get(cid)
        if sub is None or len(sub) == 0:
            # Fallback: geometric centroid.
            g = communes.loc[communes[id_col].astype(str) == cid, "centroid"].iloc[0]
            gg = wkt.loads(g) if isinstance(g, str) else g
            per[cid] = [(gg.x, gg.y, 1.0)]
            continue
        sub = sub.head(take)
        xs = sub.geometry.x.to_numpy()
        ys = sub.geometry.y.to_numpy()
        w = sub["cand_pop"].to_numpy(dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        per[cid] = list(zip(xs.tolist(), ys.tolist(), w.tolist()))
    return ids, per


# --------------------------------------------------------------------------- #
# Aggregation to commune x commune
# --------------------------------------------------------------------------- #
def aggregate(ids, per_points, snap_node, unique_nodes, D_sec):
    """Collapse the node-level seconds matrix to a commune x commune minutes
    matrix via T = A D A^T, A = row-normalised commune x node weight matrix.

    ``snap_node`` maps a point key (cid, k) -> index into ``unique_nodes``.
    ``D_sec`` is (U x U) seconds over ``unique_nodes``.
    """
    import scipy.sparse as sp

    n = len(ids)
    U = len(unique_nodes)
    rows, cols, vals = [], [], []
    for i, cid in enumerate(ids):
        for k, (_x, _y, w) in enumerate(per_points[cid]):
            j = snap_node[(cid, k)]
            rows.append(i)
            cols.append(j)
            vals.append(w)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, U))
    # A already row-normalised (weights sum to 1 per commune).
    T = A @ (D_sec @ A.T)          # (n x n) seconds
    return np.asarray(T) / 60.0    # minutes


# --------------------------------------------------------------------------- #
# Validation verdict
# --------------------------------------------------------------------------- #
def network_verdict(stats: dict, snap_dist: np.ndarray, args) -> tuple[str, list[str]]:
    reasons: list[str] = []
    level = 0  # 0 good, 1 caveat, 2 bad

    lcc = stats["lcc_node_share"]
    if lcc < 0.90:
        level = max(level, 2)
        reasons.append(f"giant SCC holds only {lcc:.1%} of nodes (<90%): network is fragmented")
    elif lcc < 0.98:
        level = max(level, 1)
        reasons.append(f"giant SCC holds {lcc:.1%} of nodes (<98%): minor fragmentation")

    cov = stats["maxspeed_cov_major"]
    if cov < 0.50:
        level = max(level, 2)
        reasons.append(f"explicit maxspeed on only {cov:.1%} of primary+ length: travel times rely heavily on class defaults")
    elif cov < 0.80:
        level = max(level, 1)
        reasons.append(f"explicit maxspeed on {cov:.1%} of primary+ length: some reliance on class defaults")

    far = int((snap_dist > args.snap_max_dist).sum())
    if far > 0:
        level = max(level, 1)
        reasons.append(f"{far} commune centroids snap >{args.snap_max_dist:.0f} m to the routable core (possible islands)")

    if args.benchmark_km is not None:
        ratio = stats["total_km"] / args.benchmark_km
        if ratio < 0.75:
            level = max(level, 2)
            reasons.append(f"total drivable length {stats['total_km']:,.0f} km is {ratio:.0%} of the {args.benchmark_km:,.0f} km benchmark (network incomplete)")
        elif ratio < 0.90:
            level = max(level, 1)
            reasons.append(f"total drivable length is {ratio:.0%} of benchmark")

    unreach = stats.get("unreachable_pairs", 0)
    if unreach > 0:
        level = max(level, 1)
        reasons.append(f"{unreach} OD pairs unreachable after snapping to the SCC")

    verdict = ["GOOD", "USABLE-WITH-CAVEATS", "BAD"][level]
    if not reasons:
        reasons.append("connectivity, maxspeed coverage and snapping all within thresholds")
    return verdict, reasons


def print_verdict(verdict, reasons, stats):
    bar = "=" * 72
    lines = [bar, f"NETWORK VALIDATION VERDICT:  {verdict}", bar,
             f"  nodes (raw)              : {stats['n_nodes_raw']:,}",
             f"  nodes (giant SCC)        : {stats['n_nodes_scc']:,}  ({stats['lcc_node_share']:.1%})",
             f"  nodes (routing core)     : {stats['n_nodes_core']:,}  (after degree-2 contraction)",
             f"  directed edges (core)    : {stats['n_edges_core']:,}",
             f"  total drivable length    : {stats['total_km']:,.0f} km",
             f"  maxspeed coverage (all)  : {stats['maxspeed_cov_all']:.1%} of length",
             f"  maxspeed coverage (P+)   : {stats['maxspeed_cov_major']:.1%} of primary+ length",
             f"  ambiguous unresolved     : {stats['ambiguous_km']:,.0f} km  "
             f"({stats['builtup_km']:,.0f} km inferred built-up)",
             f"  snap dist median / p95   : {stats['snap_med']:.0f} / {stats['snap_p95']:.0f} m",
             "  reasons:"]
    for r in reasons:
        lines.append(f"    - {r}")
    lines.append(bar)
    msg = "\n".join(lines)
    LOGGER.info("Network verdict:\n%s", msg)
    return msg


# --------------------------------------------------------------------------- #
# Diagonal imputation (ARSW self-commute)
# --------------------------------------------------------------------------- #
def diagonal_times(communes, id_col, ids, intra_speed_kmh, min_minutes):
    """t_nn = (2/3) * sqrt(Area/pi) / v * 60, floored, area from EPSG:3035."""
    area = dict(zip(communes[id_col].astype(str),
                    communes.geometry.area.to_numpy()))  # m^2
    out = np.empty(len(ids))
    for i, cid in enumerate(ids):
        a = max(area.get(cid, 0.0), 0.0)
        r_km = math.sqrt(a / math.pi) / 1000.0
        t = (2.0 / 3.0) * r_km / max(intra_speed_kmh, 1.0) * 60.0
        out[i] = max(t, min_minutes)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute a baseline road travel-time matrix between commune "
                    "centroids from an OSM .pbf road network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--network", type=Path, required=True,
                   help="Road-network .osm.pbf (drivable ways).")
    p.add_argument("--communes", type=Path, default=DEFAULT_COMMUNES,
                   help="Commune GeoPackage (communes + commune_candidates).")
    p.add_argument("--communes-layer", default=COMMUNES_LAYER)
    p.add_argument("--candidates-layer", default=CANDIDATE_LAYER)
    p.add_argument("--id-col", default=DEFAULT_ID_COL)
    p.add_argument("--output", type=Path, default=None,
                   help="Output .npz (default: data/processed/tt_matrix/"
                        "ttm_road_<centroid>_<network>.npz).")
    p.add_argument("--centroid-type",
                   choices=["unweighted", "pop-weighted", "best-pop", "multiple"],
                   default="pop-weighted",
                   help="Origin/destination point per commune.")
    p.add_argument("--n-centroids", type=int, default=3,
                   help="For --centroid-type multiple: max candidates per commune.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--speed-profile", type=Path, default=None,
                   help="JSON override for class defaults {class:[rural,urban]}.")
    p.add_argument("--speed-factor", type=float, default=1.0,
                   help="Multiplier on CLASS-DEFAULT speeds only (not on explicit "
                        "maxspeed). <1 approximates realistic average speeds.")
    p.add_argument("--builtup-detection", choices=["none", "osm", "grid"],
                   default="osm",
                   help="How to split rural vs built-up defaults where maxspeed "
                        "is missing on ambiguous classes.")
    p.add_argument("--population-grid", type=Path, default=DEFAULT_POP_GRID,
                   help="100 m population parquet (lon/lat/population) for "
                        "--builtup-detection grid.")
    p.add_argument("--builtup-pop-threshold", type=float, default=25.0,
                   help="Persons per 100 m cell to count a cell as built-up.")
    p.add_argument("--intra-speed", type=float, default=30.0,
                   help="km/h for the intra-commune (diagonal) self-commute.")
    p.add_argument("--min-diagonal-min", type=float, default=3.0)
    p.add_argument("--snap-max-dist", type=float, default=5000.0,
                   help="Warn if a centroid snaps farther than this (m).")
    p.add_argument("--no-simplify", action="store_true",
                   help="Disable degree-2 contraction (slower, more memory).")
    p.add_argument("--benchmark-km", type=float, default=None,
                   help="Reference total drivable km for the completeness check.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if the verdict is BAD.")
    p.add_argument("--no-csv", action="store_true",
                   help="Write only the NPZ, skip the wide CSV.")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="Local scratch dir (needed when --output is on NFS).")
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args(argv)


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def load_speed_profile(path: Path | None) -> dict[str, tuple[float, float]]:
    prof = {k: tuple(v) for k, v in DEFAULT_SPEED_PROFILE.items()}
    if path is not None:
        with open(path) as fh:
            override = json.load(fh)
        for k, v in override.items():
            prof[k] = (float(v[0]), float(v[1]))
    return prof


def build_pop_tree(pop_grid_path: Path, threshold: float, transformer):
    """cKDTree over populated 100 m cell centres in EPSG:3035."""
    from scipy.spatial import cKDTree
    LOGGER.info("Loading population grid for built-up detection: %s", pop_grid_path)
    df = pd.read_parquet(pop_grid_path, columns=["lon", "lat", "population"])
    df = df[df["population"] >= threshold]
    LOGGER.info("  %s populated cells (>= %.0f persons)", f"{len(df):,}", threshold)
    mx, my = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    return cKDTree(np.column_stack([mx, my]))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file)

    if not args.network.exists():
        LOGGER.error("Network PBF not found: %s", args.network)
        return 2
    if not args.communes.exists():
        LOGGER.error("Communes GeoPackage not found: %s", args.communes)
        return 2

    import geopandas as gpd
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    t_start = time.time()
    profile = load_speed_profile(args.speed_profile)
    ambig = ambiguous_classes(profile)
    to_metric = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True)

    # Built-up population tree (optional).
    pop_tree = None
    cellsize = 100.0  # half-diagonal ~71 m; nearest-cell within ~100 m ~ inside
    if args.builtup_detection == "grid":
        if not args.population_grid.exists():
            LOGGER.error("--builtup-detection grid needs --population-grid: %s",
                         args.population_grid)
            return 2
        pop_tree = build_pop_tree(args.population_grid, args.builtup_pop_threshold,
                                  to_metric)

    # 1. Parse + build graph.
    ways = parse_pbf_ways(args.network)
    if not ways:
        LOGGER.error("No drivable ways parsed -- is this a road-filtered PBF?")
        return 3
    G = build_graph(ways, profile, args.speed_factor, args.builtup_detection,
                    pop_tree, cellsize, ambig, to_metric)
    del ways
    LOGGER.info("Graph: %s nodes, %s directed edges",
                f"{G['n_nodes']:,}", f"{len(G['edges']):,}")

    total_km = sum(G["class_len"].values()) / 1000.0
    major = ["motorway", "trunk", "primary", "secondary", "tertiary"]
    major_len = sum(G["class_len"].get(c, 0.0) for c in major)
    major_exp = sum(G["class_len_explicit"].get(c, 0.0) for c in major)
    all_len = sum(G["class_len"].values())
    all_exp = sum(G["class_len_explicit"].values())

    # 2. Giant SCC + contraction.
    members, sub_edges, scc_sizes = giant_scc(G["n_nodes"], G["edges"])
    lcc_share = len(members) / max(G["n_nodes"], 1)
    LOGGER.info("Giant SCC: %s / %s nodes (%.1f%%)",
                f"{len(members):,}", f"{G['n_nodes']:,}", 100 * lcc_share)

    # Reindex SCC nodes to 0..len(members)-1.
    old2scc = {old: i for i, old in enumerate(members.tolist())}
    scc_edges = [(old2scc[u], old2scc[v], l, t) for (u, v, l, t) in sub_edges]
    scc_xy = G["node_xy"][members]

    # 3. Select + snap centroids (to full-resolution SCC nodes) BEFORE contraction.
    communes = gpd.read_file(args.communes, layer=args.communes_layer)
    if str(communes.crs) not in (CRS_METRIC, "epsg:3035", "EPSG:3035"):
        communes = communes.to_crs(CRS_METRIC)
    try:
        candidates = gpd.read_file(args.communes, layer=args.candidates_layer)
        if str(candidates.crs) != CRS_METRIC:
            candidates = candidates.to_crs(CRS_METRIC)
    except Exception:
        candidates = None

    ids, per_points = select_points(communes, candidates, args.id_col,
                                    args.centroid_type, args.n_centroids)

    scc_tree = cKDTree(scc_xy)
    # Snap every distinct point; protect the snapped SCC nodes.
    point_keys: list[tuple[str, int]] = []
    pxy: list[tuple[float, float]] = []
    for cid in ids:
        for k, (x, y, _w) in enumerate(per_points[cid]):
            point_keys.append((cid, k))
            pxy.append((x, y))
    pxy = np.asarray(pxy)
    snap_dist, snap_scc = scc_tree.query(pxy, k=1)
    protected_scc = set(int(s) for s in snap_scc)
    LOGGER.info("Snapped %d points | median %.0f m | p95 %.0f m | max %.0f m",
                len(pxy), np.median(snap_dist),
                np.percentile(snap_dist, 95), snap_dist.max())

    # Contract.
    if args.no_simplify:
        n_core = len(members)
        core_edges = scc_edges
        scc2core = {i: i for i in range(len(members))}
    else:
        n_core, core_edges, scc2core = simplify_graph(
            len(members), scc_edges, protected_scc)
        LOGGER.info("Contraction: %s -> %s nodes, %s -> %s directed edges",
                    f"{len(members):,}", f"{n_core:,}",
                    f"{len(scc_edges):,}", f"{len(core_edges):,}")

    graph = build_igraph(n_core, core_edges)

    # Map each snapped point to its core node index.
    snap_node_core: dict[tuple[str, int], int] = {}
    for key, s in zip(point_keys, snap_scc):
        snap_node_core[key] = scc2core[int(s)]

    # Unique routing nodes.
    unique_nodes = sorted(set(snap_node_core.values()))
    node_to_pos = {nd: i for i, nd in enumerate(unique_nodes)}
    snap_pos = {key: node_to_pos[nd] for key, nd in snap_node_core.items()}

    # 4. Route unique x unique.
    D_sec = route_matrix(graph, unique_nodes, unique_nodes,
                         workers=args.workers, tmp_dir=args.tmp_dir)
    unreachable = int(np.isinf(D_sec).sum())
    if unreachable:
        LOGGER.warning("%d unreachable node pairs (unexpected within an SCC)",
                       unreachable)
        D_sec = np.where(np.isinf(D_sec), np.nan, D_sec)

    # 5. Aggregate to commune x commune (minutes) + diagonal.
    T = aggregate(ids, per_points, snap_pos, unique_nodes, D_sec)
    diag = diagonal_times(communes, args.id_col, ids,
                          args.intra_speed, args.min_diagonal_min)
    np.fill_diagonal(T, diag)

    # Verdict.
    stats = {
        "n_nodes_raw": G["n_nodes"],
        "n_nodes_scc": len(members),
        "n_nodes_core": n_core,
        "n_edges_core": len(core_edges),
        "lcc_node_share": lcc_share,
        "total_km": total_km,
        "maxspeed_cov_all": (all_exp / all_len) if all_len else 0.0,
        "maxspeed_cov_major": (major_exp / major_len) if major_len else 0.0,
        "ambiguous_km": G["ambiguous_len"] / 1000.0,
        "builtup_km": G["builtup_len"] / 1000.0,
        "snap_med": float(np.median(snap_dist)),
        "snap_p95": float(np.percentile(snap_dist, 95)),
        "unreachable_pairs": unreachable,
    }
    verdict, reasons = network_verdict(stats, snap_dist, args)
    verdict_msg = print_verdict(verdict, reasons, stats)

    # 6. Write outputs.
    net_stem = args.network.name.replace(".osm.pbf", "").replace(".pbf", "")
    tag = {"unweighted": "unw", "pop-weighted": "popw",
           "best-pop": "bestpop",
           "multiple": f"mult{args.n_centroids}"}[args.centroid_type]
    output = args.output or (DEFAULT_OUTPUT_DIR / f"ttm_road_{tag}_{net_stem}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "network": str(args.network),
        "communes": str(args.communes),
        "centroid_type": args.centroid_type,
        "n_centroids": args.n_centroids if args.centroid_type == "multiple" else 1,
        "speed_factor": args.speed_factor,
        "builtup_detection": args.builtup_detection,
        "units": "minutes",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "stats": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in stats.items()},
        "class_km": {k: v / 1000.0 for k, v in sorted(G["class_len"].items())},
    }

    ids_arr = np.asarray(ids)
    T32 = T.astype(np.float32)

    scratch = Path(tempfile.mkdtemp(prefix="road_ttm_out_",
                                    dir=str(args.tmp_dir) if args.tmp_dir else None))
    try:
        tmp_npz = scratch / output.name
        np.savez_compressed(tmp_npz, matrix=T32, ids=ids_arr,
                            meta=json.dumps(meta))
        if output.exists():
            output.unlink()
        shutil.move(str(tmp_npz), str(output))

        if not args.no_csv:
            csv_path = output.with_suffix(".csv")
            tmp_csv = scratch / csv_path.name
            pd.DataFrame(T32, index=ids_arr, columns=ids_arr).to_csv(tmp_csv)
            if csv_path.exists():
                csv_path.unlink()
            shutil.move(str(tmp_csv), str(csv_path))

        meta_path = output.with_suffix(".meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    finite = T[np.isfinite(T)]
    LOGGER.info(
        "Done: %d x %d matrix | mean %.1f min | median %.1f min | max %.1f min "
        "| %.1fs total.",
        len(ids), len(ids),
        float(np.nanmean(finite)), float(np.nanmedian(finite)),
        float(np.nanmax(finite)), time.time() - t_start,
    )
    LOGGER.info("Wrote: %s", output)
    print(str(output))

    if args.strict and verdict == "BAD":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
