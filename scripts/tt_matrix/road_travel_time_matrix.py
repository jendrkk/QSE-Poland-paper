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
    signature already carries ``capacity``/``load`` arguments (a no-op at
    load == 0), so an outer traffic-assignment loop can re-weight edges;
  * a per-edge ``capacity`` attribute is populated even in the baseline and is
    carried through contraction as the chain bottleneck (min);
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
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy geo/graph deps are imported lazily inside functions so --help stays cheap
# and failure messages are precise.


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
INF = float("inf")


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

# Nominal lane capacities (veh/h/lane) by class -- carried per edge for the
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


def resolve_speed(ms_raw, hwy, builtup, profile, speed_factor):
    """Free-flow speed (km/h) and whether an explicit maxspeed tag was used.

    Precedence: explicit ``maxspeed`` (as-is) -> class default (rural/built-up)
    times ``speed_factor``.
    """
    ms, explicit = parse_maxspeed(ms_raw)
    if ms is not None:
        return ms, True
    rural, urban = profile.get(hwy, (40.0, 40.0))
    base = urban if builtup else rural
    return base * speed_factor, False


def edge_time(length_m, speed_kmh, capacity=0.0, load=0.0):
    """Edge traversal time (seconds). Baseline: free-flow ``length / speed``.

    ``capacity``/``load`` are the congestion seam -- a BPR term
    ``* (1 + alpha*(load/capacity)**beta)`` would multiply the free-flow time
    here, driven by an outer traffic-assignment loop. Inert at ``load == 0``.
    """
    v = max(speed_kmh, 1.0) / 3.6
    return length_m / v


def ambiguous_classes(profile) -> set[str]:
    """Classes whose rural and built-up defaults differ (built-up matters)."""
    return {k for k, (r, u) in profile.items() if r != u}


_ZONE_URBAN = re.compile(r"urban|zone")
_ZONE_RURAL = re.compile(r"rural")


def builtup_from_zone(zone) -> bool | None:
    """Built-up signal from a source:maxspeed / zone:* tag, or None."""
    if not zone:
        return None
    z = str(zone).lower()
    if _ZONE_RURAL.search(z):
        return False
    if _ZONE_URBAN.search(z):
        return True
    return None


# --------------------------------------------------------------------------- #
# PBF parsing  (compact per-way tuples, only the tags we need)
# --------------------------------------------------------------------------- #
def parse_pbf_ways(network: Path):
    """Read drivable ways. Returns a list of tuples
    ``(refs, lons, lats, hwy, maxspeed, oneway, junction, lanes, zone)``.
    """
    import osmium

    class RoadHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways: list[tuple] = []
            self.n_seen = 0

        def way(self, w):
            self.n_seen += 1
            tags = w.tags
            hwy = tags.get("highway")
            if hwy not in DRIVABLE_CLASSES:
                return
            acc = tags.get("motor_vehicle") or tags.get("motorcar") or tags.get("access")
            if acc in _ACCESS_NO:
                return
            refs, lons, lats = [], [], []
            for n in w.nodes:
                if n.location.valid():
                    refs.append(n.ref)
                    lons.append(n.location.lon)
                    lats.append(n.location.lat)
            if len(refs) < 2:
                return
            zone = (tags.get("source:maxspeed") or tags.get("zone:maxspeed")
                    or tags.get("zone:traffic") or tags.get("maxspeed:type"))
            self.ways.append((refs, lons, lats, hwy,
                              tags.get("maxspeed"),
                              str(tags.get("oneway", "")).strip().lower(),
                              str(tags.get("junction", "")).strip().lower(),
                              tags.get("lanes"), zone))

    LOGGER.info("Parsing PBF (drivable ways): %s", network)
    t0 = time.time()
    h = RoadHandler()
    h.apply_file(str(network), locations=True)
    LOGGER.info("  kept %s drivable ways of %s total ways (%.1fs)",
                f"{len(h.ways):,}", f"{h.n_seen:,}", time.time() - t0)
    return h.ways


def _segment_lengths(geod, lons, lats) -> np.ndarray:
    """Geodesic length (m) of each consecutive segment along a polyline."""
    if len(lons) < 2:
        return np.zeros(max(len(lons) - 1, 0))
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    _, _, dist = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return np.asarray(dist, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Graph assembly (vectorised, compact arrays)
# --------------------------------------------------------------------------- #
def build_graph(ways, profile, speed_factor, builtup_mode,
                pop_tree, pop_cellsize, ambig, to_metric):
    """Assemble a directed graph from parsed ways.

    Returns a dict with node coordinates and directed-edge arrays
    (``eu, ev`` int32 node ids; ``el`` length m; ``et`` time s; ``ec``
    capacity veh/h) plus length/coverage statistics.
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    nmap: dict[int, int] = {}
    nlon = array("d")
    nlat = array("d")

    def nid(o, lon, lat):
        k = nmap.get(o)
        if k is None:
            k = len(nlon)
            nmap[o] = k
            nlon.append(lon)
            nlat.append(lat)
        return k

    # Pass 1: decide built-up per way (batched population-grid query).
    n_ways = len(ways)
    builtup = [False] * n_ways
    ambig_unres = [False] * n_ways
    grid_idx: list[int] = []
    grid_mid: list[tuple[float, float]] = []
    for wi, w in enumerate(ways):
        hwy, ms, zone = w[3], w[4], w[8]
        _, explicit = parse_maxspeed(ms)
        if hwy in ("residential", "living_street", "service"):
            builtup[wi] = True
        if hwy in ambig and not explicit and builtup_mode != "none":
            ambig_unres[wi] = True
            sig = builtup_from_zone(zone)
            if sig is not None:
                builtup[wi] = sig
            elif builtup_mode == "grid" and pop_tree is not None:
                lons, lats = w[1], w[2]
                grid_idx.append(wi)
                grid_mid.append((lons[len(lons) // 2], lats[len(lats) // 2]))
    if grid_idx:
        arr = np.asarray(grid_mid)
        mx, my = to_metric.transform(arr[:, 0], arr[:, 1])
        d, _ = pop_tree.query(np.column_stack([mx, my]), k=1)
        for j, wi in enumerate(grid_idx):
            builtup[wi] = bool(d[j] <= pop_cellsize)

    # Pass 2: emit directed edges.
    eu, ev = array("i"), array("i")
    el, et, ec = array("f"), array("f"), array("f")
    class_len: dict[str, float] = defaultdict(float)
    class_len_exp: dict[str, float] = defaultdict(float)
    ambiguous_len = 0.0
    builtup_len = 0.0

    for wi, w in enumerate(ways):
        refs, lons, lats, hwy, ms, ow, junc, lanes, _zone = w
        bu = builtup[wi]
        speed, used_explicit = resolve_speed(ms, hwy, bu, profile, speed_factor)

        forward = backward = True
        if ow in _ONEWAY_YES or junc == "roundabout" or hwy == "motorway":
            backward = False
        if ow in _ONEWAY_REV:
            forward, backward = False, True
        if ow in ("no", "false", "0"):
            forward = backward = True

        try:
            n_lanes = max(1, int(float(str(lanes).split(";")[0])))
        except (TypeError, ValueError):
            n_lanes = 1 if not backward else 2
        cap = float(LANE_CAPACITY.get(hwy, 600) * n_lanes)

        seg = _segment_lengths(geod, lons, lats)
        for i in range(len(refs) - 1):
            L = float(seg[i])
            if L <= 0:
                continue
            u = nid(refs[i], lons[i], lats[i])
            v = nid(refs[i + 1], lons[i + 1], lats[i + 1])
            if u == v:
                continue
            t = edge_time(L, speed)
            if forward:
                eu.append(u); ev.append(v); el.append(L); et.append(t); ec.append(cap)
            if backward:
                eu.append(v); ev.append(u); el.append(L); et.append(t); ec.append(cap)

        wlen = float(seg.sum())
        class_len[hwy] += wlen
        if used_explicit:
            class_len_exp[hwy] += wlen
        if ambig_unres[wi]:
            ambiguous_len += wlen
            if bu:
                builtup_len += wlen

    eu = np.frombuffer(eu, dtype=np.int32)
    ev = np.frombuffer(ev, dtype=np.int32)
    el = np.frombuffer(el, dtype=np.float32)
    et = np.frombuffer(et, dtype=np.float32)
    ec = np.frombuffer(ec, dtype=np.float32)
    lon_a = np.frombuffer(nlon, dtype=np.float64)
    lat_a = np.frombuffer(nlat, dtype=np.float64)
    mx, my = to_metric.transform(lon_a, lat_a)

    return {
        "n_nodes": len(lon_a),
        "node_xy": np.column_stack([mx, my]),
        "eu": eu, "ev": ev, "el": el, "et": et, "ec": ec,
        "class_len": dict(class_len),
        "class_len_explicit": dict(class_len_exp),
        "ambiguous_len": ambiguous_len,
        "builtup_len": builtup_len,
    }


# --------------------------------------------------------------------------- #
# Giant strongly-connected component (scipy, memory-light)
# --------------------------------------------------------------------------- #
def giant_scc(n, eu, ev, el, et, ec):
    """Return ``members`` (old node ids) and SCC-reindexed edge arrays."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    A = csr_matrix((np.ones(len(eu), dtype=np.int8),
                    (eu.astype(np.int64), ev.astype(np.int64))), shape=(n, n))
    ncomp, labels = connected_components(A, directed=True, connection="strong")
    counts = np.bincount(labels)
    giant = int(np.argmax(counts))
    members = np.nonzero(labels == giant)[0]

    old2scc = np.full(n, -1, dtype=np.int64)
    old2scc[members] = np.arange(len(members))
    keep = (old2scc[eu] >= 0) & (old2scc[ev] >= 0)
    su = old2scc[eu[keep]].astype(np.int32)
    sv = old2scc[ev[keep]].astype(np.int32)
    return members, su, sv, el[keep], et[keep], ec[keep], counts


# --------------------------------------------------------------------------- #
# Degree-2 chain contraction (vectorised CSR, direction- & capacity-aware)
# --------------------------------------------------------------------------- #
def simplify_graph(n, u, v, length, time_s, cap, protected_mask):
    """Contract non-endpoint degree-2 chains, preserving cumulative time/length
    and the bottleneck (min) capacity, keeping oneway chains oneway.

    Returns ``(n_new, (ru, rv, rl, rt, rc), old2new)``. A node is an endpoint if
    it is protected, an intersection / dead-end (undirected degree != 2), or has
    asymmetric in/out structure. old2new maps surviving old ids to new ids.
    """
    m = len(u)
    empty = (np.array([], np.int32), np.array([], np.int32),
             np.array([], np.float64), np.array([], np.float64),
             np.array([], np.float64))
    if m == 0:
        return 0, empty, np.full(n, -1, np.int64)

    u = u.astype(np.int64); v = v.astype(np.int64)
    length = length.astype(np.float64); time_s = time_s.astype(np.float64)
    cap = cap.astype(np.float64)

    # Dedup directed parallels: keep first occurrence per (u, v).
    key = u * n + v
    order = np.argsort(key, kind="stable")
    ks = key[order]
    firstmask = np.empty(len(ks), bool)
    firstmask[0] = True
    firstmask[1:] = ks[1:] != ks[:-1]
    sel = order[firstmask]
    du, dv = u[sel], v[sel]
    dl, dt, dc = length[sel], time_s[sel], cap[sel]

    # Undirected pair ids.
    a = np.minimum(du, dv)
    b = np.maximum(du, dv)
    is_fwd = du < dv
    uk = a * n + b
    ordu = np.argsort(uk, kind="stable")
    uks = uk[ordu]
    newp = np.empty(len(uks), bool)
    newp[0] = True
    newp[1:] = uks[1:] != uks[:-1]
    pid_sorted = np.cumsum(newp) - 1
    pid = np.empty(len(uk), np.int64)
    pid[ordu] = pid_sorted
    P = int(pid_sorted[-1]) + 1

    pa = np.empty(P, np.int64); pb = np.empty(P, np.int64)
    pa[pid] = a; pb[pid] = b
    ft = np.full(P, INF); fl = np.full(P, INF); fc = np.full(P, INF)
    bt = np.full(P, INF); bl = np.full(P, INF); bc = np.full(P, INF)
    fw = is_fwd
    ft[pid[fw]] = dt[fw]; fl[pid[fw]] = dl[fw]; fc[pid[fw]] = dc[fw]
    bw = ~is_fwd
    bt[pid[bw]] = dt[bw]; bl[pid[bw]] = dl[bw]; bc[pid[bw]] = dc[bw]

    # Half-edges (both endpoints) -> CSR adjacency by source node.
    src = np.concatenate([pa, pb])
    dst = np.concatenate([pb, pa])
    hft = np.concatenate([ft, bt]); hfl = np.concatenate([fl, bl]); hfc = np.concatenate([fc, bc])
    hbt = np.concatenate([bt, ft]); hbl = np.concatenate([bl, fl])
    hpid = np.concatenate([np.arange(P), np.arange(P)])

    o = np.argsort(src, kind="stable")
    src_s = src[o]; dst_s = dst[o]
    ft_s = hft[o]; fl_s = hfl[o]; fc_s = hfc[o]
    bt_s = hbt[o]; bl_s = hbl[o]; pid_s = hpid[o]
    del src, dst, hft, hfl, hfc, hbt, hbl, hpid, o
    del key, order, ks, sel, du, dv, dl, dt, dc, a, b, is_fwd, uk, ordu, uks
    del pid_sorted, pid, fw, bw

    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, src_s + 1, 1)
    np.cumsum(indptr, out=indptr)

    und_deg = indptr[1:] - indptr[:-1]
    out_deg = np.zeros(n, np.int64)
    in_deg = np.zeros(n, np.int64)
    np.add.at(out_deg, src_s, np.isfinite(ft_s).astype(np.int64))
    np.add.at(in_deg, src_s, np.isfinite(bt_s).astype(np.int64))

    endpoint = protected_mask.copy()
    endpoint |= (und_deg != 2)
    endpoint |= (in_deg != out_deg)
    endpoint |= ~np.isin(in_deg, (1, 2))

    ep_nodes = np.nonzero(endpoint & (und_deg > 0))[0]

    # Compact C-typed copies for the walk: array.array indexing is fast AND
    # ~40x lighter than list-of-Python-objects (.tolist() on multi-million
    # arrays would add >1 GB). Free the numpy CSR/degree arrays afterwards.
    dst_a = array("i", dst_s.astype(np.int32).tobytes())
    ft_a = array("f", ft_s.astype(np.float32).tobytes())
    fl_a = array("f", fl_s.astype(np.float32).tobytes())
    fc_a = array("f", fc_s.astype(np.float32).tobytes())
    bt_a = array("f", bt_s.astype(np.float32).tobytes())
    bl_a = array("f", bl_s.astype(np.float32).tobytes())
    pid_a = array("i", pid_s.astype(np.int32).tobytes())
    indptr_a = array("q", indptr.astype(np.int64).tobytes())
    endp_a = bytearray(endpoint.astype(np.uint8).tobytes())
    ep_list = ep_nodes.tolist()
    del dst_s, ft_s, fl_s, fc_s, bt_s, bl_s, pid_s
    del und_deg, out_deg, in_deg, indptr, endpoint, ep_nodes

    seg_visited = bytearray(P)
    nu: list[int] = []; nv: list[int] = []
    nl: list[float] = []; nt: list[float] = []; nc: list[float] = []

    for u0 in ep_list:
        for p in range(indptr_a[u0], indptr_a[u0 + 1]):
            pid0 = pid_a[p]
            if seg_visited[pid0]:
                continue
            seg_visited[pid0] = 1
            v0 = dst_a[p]
            fp = ft_a[p]; f_ok = fp != INF; ft_sum = fp; fl_sum = fl_a[p]; fc_min = fc_a[p]
            bp = bt_a[p]; b_ok = bp != INF; bt_sum = bp; bl_sum = bl_a[p]
            prev, cur = u0, v0
            while not endp_a[cur]:
                nxt = -1; pp = -1
                for q in range(indptr_a[cur], indptr_a[cur + 1]):
                    if dst_a[q] != prev:
                        nxt = dst_a[q]; pp = q
                        break
                if nxt == -1:
                    break
                seg_visited[pid_a[pp]] = 1
                fpp = ft_a[pp]
                if fpp != INF:
                    ft_sum += fpp; fl_sum += fl_a[pp]
                    cpp = fc_a[pp]
                    if cpp < fc_min:
                        fc_min = cpp
                else:
                    f_ok = False
                bpp = bt_a[pp]
                if bpp != INF:
                    bt_sum += bpp; bl_sum += bl_a[pp]
                else:
                    b_ok = False
                prev, cur = cur, nxt
                if cur == u0:
                    break
            F = cur
            if f_ok:
                nu.append(u0); nv.append(F); nl.append(fl_sum); nt.append(ft_sum); nc.append(fc_min)
            if b_ok:
                nu.append(F); nv.append(u0); nl.append(bl_sum); nt.append(bt_sum); nc.append(fc_min)

    # Leftover pairs (isolated interior cycles with no endpoint): keep raw.
    left = np.nonzero(np.frombuffer(bytes(seg_visited), dtype=np.uint8) == 0)[0]
    for pidx in left.tolist():
        if ft[pidx] != INF:
            nu.append(int(pa[pidx])); nv.append(int(pb[pidx]))
            nl.append(float(fl[pidx])); nt.append(float(ft[pidx])); nc.append(float(fc[pidx]))
        if bt[pidx] != INF:
            nu.append(int(pb[pidx])); nv.append(int(pa[pidx]))
            nl.append(float(bl[pidx])); nt.append(float(bt[pidx])); nc.append(float(bc[pidx]))

    nu = np.asarray(nu, np.int64); nv = np.asarray(nv, np.int64)
    if len(nu) == 0:
        return 0, empty, np.full(n, -1, np.int64)
    survive = np.unique(np.concatenate([nu, nv]))
    old2new = np.full(n, -1, np.int64)
    old2new[survive] = np.arange(len(survive))
    ru = old2new[nu].astype(np.int32)
    rv = old2new[nv].astype(np.int32)
    return len(survive), (ru, rv, np.asarray(nl), np.asarray(nt), np.asarray(nc)), old2new


def build_igraph(n_nodes, ru, rv, rl, rt, rc):
    import igraph
    g = igraph.Graph(n=n_nodes, edges=list(zip(ru.tolist(), rv.tolist())),
                     directed=True)
    g.es["time"] = rt.tolist()
    g.es["length"] = rl.tolist()
    g.es["capacity"] = rc.tolist()
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


def _route_batch(graph_path, sources, targets):
    g = _load_worker_graph(graph_path)
    d = g.distances(source=sources, target=targets, weights="time")
    return np.asarray(d, dtype=np.float64)


def route_matrix(graph, nodes, workers, tmp_dir):
    """Seconds matrix (len(nodes) x len(nodes)) via parallel igraph Dijkstra.

    The contracted graph is pickled once and lazily loaded per worker (loky
    reuses workers, so it is read at most once each); ``distances`` returns only
    the target columns, so per-worker memory stays small.
    """
    from joblib import Parallel, delayed

    scratch = Path(tempfile.mkdtemp(prefix="road_ttm_g_",
                                    dir=str(tmp_dir) if tmp_dir else None))
    gpath = scratch / "graph.pkl"
    try:
        graph.write_pickle(str(gpath))
        n_jobs = workers if workers > 0 else (os.cpu_count() or 4)
        nodes = list(nodes)
        batches = [b.tolist() for b in np.array_split(np.asarray(nodes),
                                                      max(n_jobs * 4, 1)) if len(b)]
        LOGGER.info("Routing %d x %d (sec) | workers=%d | batches=%d",
                    len(nodes), len(nodes), n_jobs, len(batches))
        t0 = time.time()
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_route_batch)(str(gpath), batch, nodes) for batch in batches)
        D = np.vstack(parts)
        LOGGER.info("  routing done in %.1fs", time.time() - t0)
        return D
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Centroid selection
# --------------------------------------------------------------------------- #
def select_points(communes, candidates, id_col, centroid_type, n_centroids):
    """Return (ordered commune id list, {cid: [(x, y, weight), ...]}) in
    EPSG:3035 with weights summing to 1 within each commune."""
    import shapely.wkt as wkt

    ids = list(communes[id_col].astype(str))
    per: dict[str, list[tuple[float, float, float]]] = {}

    if centroid_type in ("unweighted", "pop-weighted"):
        col = "centroid" if centroid_type == "unweighted" else "weighted_centroid"
        for cid, val in zip(communes[id_col].astype(str), communes[col]):
            g = wkt.loads(val) if isinstance(val, str) else val
            per[cid] = [(g.x, g.y, 1.0)]
        return ids, per

    if candidates is None:
        raise ValueError("--centroid-type needs the commune_candidates layer")
    cand = candidates.sort_values([id_col, "cand_rank"])
    take = 1 if centroid_type == "best-pop" else max(1, n_centroids)
    grouped = {str(cid): sub for cid, sub in cand.groupby(id_col, sort=False)}
    cent_lookup = dict(zip(communes[id_col].astype(str), communes["centroid"]))

    for cid in ids:
        sub = grouped.get(cid)
        if sub is None or len(sub) == 0:
            g = cent_lookup[cid]
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


def aggregate(ids, per_points, snap_pos, n_unique, D_sec):
    """Collapse the node-level seconds matrix to commune x commune minutes via
    T = A D A^T, A = row-normalised commune x node weight matrix."""
    import scipy.sparse as sp

    n = len(ids)
    rows, cols, vals = [], [], []
    for i, cid in enumerate(ids):
        for k, (_x, _y, w) in enumerate(per_points[cid]):
            rows.append(i)
            cols.append(snap_pos[(cid, k)])
            vals.append(w)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n_unique))
    T = A @ (D_sec @ A.T)
    return np.asarray(T) / 60.0


# --------------------------------------------------------------------------- #
# Diagonal imputation (ARSW self-commute)
# --------------------------------------------------------------------------- #
def diagonal_times(communes, id_col, ids, intra_speed_kmh, min_minutes):
    """t_nn = (2/3) * sqrt(Area/pi) / v * 60, floored, area from EPSG:3035."""
    area = dict(zip(communes[id_col].astype(str),
                    communes.geometry.area.to_numpy()))
    out = np.empty(len(ids))
    for i, cid in enumerate(ids):
        r_km = math.sqrt(max(area.get(cid, 0.0), 0.0) / math.pi) / 1000.0
        t = (2.0 / 3.0) * r_km / max(intra_speed_kmh, 1.0) * 60.0
        out[i] = max(t, min_minutes)
    return out


# --------------------------------------------------------------------------- #
# Validation verdict
# --------------------------------------------------------------------------- #
def network_verdict(stats, snap_dist, args):
    reasons: list[str] = []
    level = 0

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
    LOGGER.info("Network verdict:\n%s", "\n".join(lines))


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
                   default="pop-weighted", help="Origin/destination point per commune.")
    p.add_argument("--n-centroids", type=int, default=3,
                   help="For --centroid-type multiple: max candidates per commune.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--speed-profile", type=Path, default=None,
                   help="JSON override for class defaults {class:[rural,urban]}.")
    p.add_argument("--speed-factor", type=float, default=1.0,
                   help="Multiplier on CLASS-DEFAULT speeds only (not explicit "
                        "maxspeed). <1 approximates realistic average speeds.")
    p.add_argument("--builtup-detection", choices=["none", "osm", "grid"],
                   default="osm",
                   help="Split rural vs built-up defaults where maxspeed is "
                        "missing on ambiguous classes.")
    p.add_argument("--population-grid", type=Path, default=DEFAULT_POP_GRID,
                   help="100 m population parquet (lon/lat/population) for grid mode.")
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
    p.add_argument("--no-csv", action="store_true", help="Write only the NPZ.")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="Local scratch dir (needed when --output is on NFS).")
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args(argv)


def setup_logging(log_file):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers)


def load_speed_profile(path):
    prof = {k: tuple(v) for k, v in DEFAULT_SPEED_PROFILE.items()}
    if path is not None:
        with open(path) as fh:
            for k, v in json.load(fh).items():
                prof[k] = (float(v[0]), float(v[1]))
    return prof


def build_pop_tree(pop_grid_path, threshold, transformer):
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

    pop_tree = None
    cellsize = 100.0
    if args.builtup_detection == "grid":
        if not args.population_grid.exists():
            LOGGER.error("--builtup-detection grid needs --population-grid: %s",
                         args.population_grid)
            return 2
        pop_tree = build_pop_tree(args.population_grid, args.builtup_pop_threshold,
                                  to_metric)

    ways = parse_pbf_ways(args.network)
    if not ways:
        LOGGER.error("No drivable ways parsed -- is this a road-filtered PBF?")
        return 3
    G = build_graph(ways, profile, args.speed_factor, args.builtup_detection,
                    pop_tree, cellsize, ambig, to_metric)
    del ways
    LOGGER.info("Graph: %s nodes, %s directed edges",
                f"{G['n_nodes']:,}", f"{len(G['eu']):,}")

    total_km = sum(G["class_len"].values()) / 1000.0
    major = ("motorway", "trunk", "primary", "secondary", "tertiary")
    major_len = sum(G["class_len"].get(c, 0.0) for c in major)
    major_exp = sum(G["class_len_explicit"].get(c, 0.0) for c in major)
    all_len = sum(G["class_len"].values())
    all_exp = sum(G["class_len_explicit"].values())

    members, su, sv, sl, st, sc, _counts = giant_scc(
        G["n_nodes"], G["eu"], G["ev"], G["el"], G["et"], G["ec"])
    lcc_share = len(members) / max(G["n_nodes"], 1)
    LOGGER.info("Giant SCC: %s / %s nodes (%.1f%%)",
                f"{len(members):,}", f"{G['n_nodes']:,}", 100 * lcc_share)
    scc_xy = G["node_xy"][members]
    # Full-graph edge arrays are no longer needed; free them before contraction.
    for _k in ("eu", "ev", "el", "et", "ec", "node_xy"):
        G[_k] = None

    # Read communes + candidates.
    communes = gpd.read_file(args.communes, layer=args.communes_layer)
    if str(communes.crs).upper() != CRS_METRIC:
        communes = communes.to_crs(CRS_METRIC)
    try:
        candidates = gpd.read_file(args.communes, layer=args.candidates_layer)
        if str(candidates.crs).upper() != CRS_METRIC:
            candidates = candidates.to_crs(CRS_METRIC)
    except Exception:
        candidates = None

    ids, per_points = select_points(communes, candidates, args.id_col,
                                    args.centroid_type, args.n_centroids)

    # Snap every distinct point to the nearest full-resolution SCC node.
    scc_tree = cKDTree(scc_xy)
    point_keys, pxy = [], []
    for cid in ids:
        for k, (x, y, _w) in enumerate(per_points[cid]):
            point_keys.append((cid, k))
            pxy.append((x, y))
    pxy = np.asarray(pxy)
    snap_dist, snap_scc = scc_tree.query(pxy, k=1)
    LOGGER.info("Snapped %d points | median %.0f m | p95 %.0f m | max %.0f m",
                len(pxy), np.median(snap_dist), np.percentile(snap_dist, 95),
                snap_dist.max())

    # Contract (protecting snapped nodes so they survive).
    if args.no_simplify:
        n_core = len(members)
        ru = su.astype(np.int32); rv = sv.astype(np.int32)
        rl = sl.astype(np.float64); rt = st.astype(np.float64); rc = sc.astype(np.float64)
        scc2core = np.arange(len(members), dtype=np.int64)
    else:
        protected = np.zeros(len(members), bool)
        protected[np.unique(snap_scc.astype(np.int64))] = True
        n_core, (ru, rv, rl, rt, rc), scc2core = simplify_graph(
            len(members), su, sv, sl, st, sc, protected)
        LOGGER.info("Contraction: %s -> %s nodes, %s -> %s directed edges",
                    f"{len(members):,}", f"{n_core:,}",
                    f"{len(su):,}", f"{len(ru):,}")

    graph = build_igraph(n_core, ru, rv, rl, rt, rc)

    snap_core = scc2core[snap_scc.astype(np.int64)]
    snap_node = {key: int(nd) for key, nd in zip(point_keys, snap_core)}
    unique_nodes = sorted(set(snap_node.values()))
    node_pos = {nd: i for i, nd in enumerate(unique_nodes)}
    snap_pos = {key: node_pos[nd] for key, nd in snap_node.items()}

    D_sec = route_matrix(graph, unique_nodes, args.workers, args.tmp_dir)
    unreachable = int(np.isinf(D_sec).sum())
    if unreachable:
        LOGGER.warning("%d unreachable node pairs (unexpected within an SCC)", unreachable)
        D_sec = np.where(np.isinf(D_sec), np.nan, D_sec)

    T = aggregate(ids, per_points, snap_pos, len(unique_nodes), D_sec)
    diag = diagonal_times(communes, args.id_col, ids, args.intra_speed,
                          args.min_diagonal_min)
    np.fill_diagonal(T, diag)

    stats = {
        "n_nodes_raw": G["n_nodes"], "n_nodes_scc": len(members),
        "n_nodes_core": n_core, "n_edges_core": len(ru),
        "lcc_node_share": lcc_share, "total_km": total_km,
        "maxspeed_cov_all": (all_exp / all_len) if all_len else 0.0,
        "maxspeed_cov_major": (major_exp / major_len) if major_len else 0.0,
        "ambiguous_km": G["ambiguous_len"] / 1000.0,
        "builtup_km": G["builtup_len"] / 1000.0,
        "snap_med": float(np.median(snap_dist)),
        "snap_p95": float(np.percentile(snap_dist, 95)),
        "unreachable_pairs": unreachable,
    }
    verdict, reasons = network_verdict(stats, snap_dist, args)
    print_verdict(verdict, reasons, stats)

    # Write outputs.
    net_stem = args.network.name.replace(".osm.pbf", "").replace(".pbf", "")
    tag = {"unweighted": "unw", "pop-weighted": "popw", "best-pop": "bestpop",
           "multiple": f"mult{args.n_centroids}"}[args.centroid_type]
    output = args.output or (DEFAULT_OUTPUT_DIR / f"ttm_road_{tag}_{net_stem}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "network": str(args.network), "communes": str(args.communes),
        "centroid_type": args.centroid_type,
        "n_centroids": args.n_centroids if args.centroid_type == "multiple" else 1,
        "speed_factor": args.speed_factor, "builtup_detection": args.builtup_detection,
        "units": "minutes", "verdict": verdict, "verdict_reasons": reasons,
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
        np.savez_compressed(tmp_npz, matrix=T32, ids=ids_arr, meta=json.dumps(meta))
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
        with open(output.with_suffix(".meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    finite = T[np.isfinite(T)]
    LOGGER.info(
        "Done: %d x %d | mean %.1f | median %.1f | max %.1f min | %.1fs total.",
        len(ids), len(ids), float(np.nanmean(finite)), float(np.nanmedian(finite)),
        float(np.nanmax(finite)), time.time() - t_start)
    LOGGER.info("Wrote: %s", output)
    print(str(output))

    if args.strict and verdict == "BAD":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
