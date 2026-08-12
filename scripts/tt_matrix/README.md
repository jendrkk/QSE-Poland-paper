# Travel-time matrices (`scripts/tt_matrix`)

Bilateral travel-time matrices between commune (gmina) representative points, for
the MRRH-2018 Poland calibration and its road/rail counterfactuals.

Current contents: **`road_travel_time_matrix.py`** — the baseline (free-flow,
no-congestion, single-mode car) road matrix. A rail/PT engine (`rail_*`, GTFS,
schedule-based) will live alongside it and reuse the same commune-indexed output
layout for the nested-Fréchet modal aggregator.

---

## `road_travel_time_matrix.py`

Input: an arbitrary road-network `.osm.pbf` (e.g.
`data/raw/osm_pbf/poland_roads_2021-12-31_optimal.osm.pbf`) + the commune
GeoPackage from `scripts/geospatial/commune_centroids.py`
(`communes` + `commune_candidates` layers, EPSG:3035, id `JPT_KOD_JE`).

Output: `data/processed/tt_matrix/ttm_road_<centroid>_<network>.npz` (float32
minutes, ordered `JPT_KOD_JE` index, JSON `meta`), a wide `.csv`, and a
`.meta.json` sidecar carrying the run parameters + the validation verdict.

### Quick start

```bash
python road_travel_time_matrix.py \
    --network data/raw/osm_pbf/poland_roads_2021-12-31_optimal.osm.pbf \
    --centroid-type pop-weighted --workers 32
```

### `--centroid-type`
| value | origin/destination point per commune | source column/layer |
|---|---|---|
| `unweighted` | geometric centroid | `communes.centroid` |
| `pop-weighted` | population-weighted centroid | `communes.weighted_centroid` |
| `best-pop` | largest population cluster | `commune_candidates` rank 1 |
| `multiple` (`--n-centroids N`) | up to N clusters, **population-weighted double sum** over origin×destination candidate pairs (`T = A·D·Aᵀ`) | `commune_candidates` ranks 1..N |

`multiple` is the MAUP/Jensen-robust option: it discretises the average
bilateral time between the two population *distributions* rather than between two
single points.

### Speed model
Free-flow edge speed is resolved in strict precedence:
1. explicit `maxspeed` tag (numeric, `mph`, `PL:urban|rural|motorway|trunk`, …);
2. else statutory Polish class default, split rural / built-up, ×`--speed-factor`
   (motorway 140, trunk/S 120, primary/secondary/tertiary 90 rural · 50 built-up,
   unclassified 70·50, residential 50, …; override with `--speed-profile` JSON);
3. `--speed-factor < 1` (e.g. 0.9) turns legal ceilings into achievable averages.

**Built-up ("obszar zabudowany") inference** (`--builtup-detection none|osm|grid`)
only changes speed on *ambiguous* classes (rural≠built-up default) where
`maxspeed` is missing. `osm` uses `source:maxspeed`/`zone:*` tags; `grid` adds a
100 m population-grid midpoint test (`--population-grid`,
`--builtup-pop-threshold`). **It is a second-order, coverage-dependent
correction**: on a well-tagged network (2021) it barely moves times; on the
sparse backdated snapshots (2011/2012, ~2–5% `maxspeed` coverage) it matters more
because the error is spatially correlated with the Polska A/B gradient. The
verdict prints `maxspeed` coverage by class so the choice is data-driven, not a
guess.

### Network validation verdict
Before routing the script prints **GOOD / USABLE-WITH-CAVEATS / BAD** with
reasons: giant-SCC node share (fragmentation), `maxspeed` coverage on primary+
length, snap-distance distribution (island communes), completeness vs an optional
`--benchmark-km`, and post-routing unreachable pairs. `--strict` exits non-zero on
BAD. (Sanity check: the raw 2011 extract scores BAD — 78% SCC, 2.8% coverage, 490
stranded communes — while 2012-optimal scores USABLE-WITH-CAVEATS/BAD driven only
by sparse `maxspeed`.)

### How it works
PBF → directed graph (oneway/`junction`/`access`-aware, geodesic lengths) →
giant strongly-connected component (guarantees reachability) → degree-2 chain
contraction into a routing core (preserves cumulative time/length + bottleneck
capacity) → snap centroids to the nearest core node → parallel many-to-many
igraph Dijkstra (`--workers`) → aggregate to commune×commune → ARSW self-commute
on the diagonal (`t_nn = ⅔·√(Area/π)/v·60`, floored). Memory is ~O(edges) with a
small constant (≈2.7 GB peak for the 5.8 M-edge 2012 network).

### Why a self-contained igraph graph, not OSRM
OSRM (named in `docs/MRRH_multimodal_extension.md`) is faster but cannot do, in
engine, the three things this task needs: Polish speed rules with population-grid
built-up inference, a self-contained trustworthiness verdict, and *mutable* edge
weights for the future congestion fixed point. A graph we own pays a one-time
build cost and gives all three; an engine abstraction still allows an OSRM
backend later.

### Extension seams (already in place)
- **Congestion (Allen–Arkolakis / BPR).** `edge_time(length, speed, capacity,
  load)` is the weight seam (`load=0` in the baseline); per-edge `capacity` is
  populated and carried through contraction as the chain bottleneck. Pair-level
  congestion (doc eq. 22) needs only the free-flow matrix + a per-pair capacity
  summary; link-level Wardrop needs the mutable graph + an outer assignment loop
  and makes `t_ij` an *output* recomputed jointly with the GE.
- **Rail mode (GTFS).** A schedule-based engine (RAPTOR/CSA via r5py/OTP) should
  emit a commune×commune matrix in this same layout, keeping cost components
  (IVT/access/wait/fare) separate; `select_points`, snapping, aggregation,
  diagonal and output are mode-agnostic and reusable. The two matrices feed the
  nested-Fréchet `τ̃` aggregator downstream.
```
python road_travel_time_matrix.py --help   # full option list
```
