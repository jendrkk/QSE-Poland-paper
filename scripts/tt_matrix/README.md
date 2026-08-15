# Travel-time matrices (`scripts/tt_matrix`)

Bilateral travel-time matrices between commune (gmina) representative points, for
the MRRH-2018 Poland calibration and its road/rail counterfactuals.

Current contents: **`road_travel_time_matrix.py`** — the baseline (free-flow,
no-congestion, single-mode car) road matrix. A rail/PT engine (`rail_*`, GTFS,
schedule-based) will live alongside it and reuse the same commune-indexed output
layout for the nested-Fréchet modal aggregator.

---

## `road_travel_time_matrix.py`

Input: a road-network file **auto-detected by extension**, plus the commune
GeoPackage from `scripts/geospatial/commune_centroids.py`
(`communes` + `commune_candidates` layers, EPSG:3035, id `JPT_KOD_JE`).

| extension | source | speed source | notes |
|---|---|---|---|
| `.osm.pbf` / `.pbf` | OSM planet-history extract | `maxspeed` tag → statutory class default | Free-flow OSM baseline. |
| `.gpkg`   | Garmin GPKG produced by `data/raw/gmap/_tools/TopoFull.java` from a GPMapa TOPO IMG | Table-A `kmh` per segment (49% coverage of the map's roads = the routable subset) → statutory class default for unclassified fallback | 2011 GPMapa TOPO. Bridges/tunnels handled correctly via CoordNode markers. |

Output: `data/processed/tt_matrix/ttm_road_<centroid>_<network>.npz` (float32
minutes, ordered `JPT_KOD_JE` index, JSON `meta`), a wide `.csv`, and a
`.meta.json` sidecar carrying the run parameters + the validation verdict.

### Recommended: cross-year matrices under a UNIFIED speed rule

For any comparison between two years, route BOTH under a single rule that
does not depend on source-data tagging quality:

* `--speed-source class` — ignore all source `maxspeed` / Garmin Table-A `kmh`
  values; use the class-default profile only.
* `--builtup-detection commune` — classify every way midpoint as urban vs
  rural from the polygon of the PRG A05 evidence unit it falls in:
  urban = TERYT last-digit ∈ {1, 4, 8, 9} (city commune, urban part of
  urban-rural commune, Warsaw dzielnice); rural = {2, 5} (rural commune, rural
  part of urban-rural commune). Same rule for OSM and GPKG → symmetric
  urban/rural detection regardless of source tag quality.

```bash
# 2021 baseline
python scripts/tt_matrix/road_travel_time_matrix.py \
    --network data/raw/osm_pbf/poland_roads_2021-12-31_optimal.osm.pbf \
    --centroid-type pop-weighted --workers 32 \
    --speed-source class \
    --builtup-detection commune \
    --output data/processed/tt_matrix/ttm_road_osm_2021_final.npz

# 2011 counterfactual — same speed rule
python scripts/tt_matrix/road_travel_time_matrix.py \
    --network data/raw/gmap/out/poland_roads_2011_full.gpkg \
    --centroid-type pop-weighted --workers 32 \
    --speed-source class \
    --builtup-detection commune \
    --output data/processed/tt_matrix/ttm_road_garmin_2011_final.npz
```

Monotonicity 2021 ≤ 2011: **82 % @ 0 min tolerance, 95 % @ 5 min, 99 % @ 10 min**.
The remaining ~5 % of pair regressions (mean +7.7 min, p95 +12.8 min) are true
topology differences between the two datasets that no speed unification can
remove; they concentrate on pairs where the 2011 Garmin data has geometry the
2021 OSM extract lacks (or vice-versa), not on real network expansion.

### Legacy: single-year "most accurate" runs

If you need the most accurate matrix for one year in isolation (no cross-year
comparison), use each source's own tags:

```bash
# OSM (2021) — use tagged maxspeed where present
python road_travel_time_matrix.py \
    --network data/raw/osm_pbf/poland_roads_2021-12-31_optimal.osm.pbf \
    --centroid-type pop-weighted --workers 32
    # default --speed-source tag, --builtup-detection osm

# Garmin GPMapa TOPO (2011) — statutory speeds required; Table-A kmh is unreliable
python road_travel_time_matrix.py \
    --network data/raw/gmap/out/poland_roads_2011_full.gpkg \
    --centroid-type pop-weighted --workers 32 \
    --speed-source class --builtup-detection commune
```

> **CRITICAL for GPKG input.** GPMapa TOPO's Table-A `kmh` column is
> *routing-engine average speeds*, not Polish legal limits — the compiler codes
> motorways uniformly at 100 km/h (statutory 140), secondary at ~47 (statutory
> 90), tertiary at ~20 (statutory 90). Using it with `--speed-source tag`
> (default) understates all cross-country travel times by 25–35% relative to a
> maxspeed-tagged OSM year and destroys year-over-year comparability. **For any
> counterfactual or cross-year matrix, always run GPKG with `--speed-source
> class`** so both years share the DEFAULT_SPEED_PROFILE Polish statutory
> ceiling. The script prints a WARNING when it detects `.gpkg` + `tag`.

### GPKG source (Garmin GPMapa TOPO 2011)

The GPKG is built by `data/raw/gmap/_tools/TopoFull.java` from a stack of 385
classic-format `.img` tiles. Each row in the `roads` layer is one road polyline;
key columns:

| column | meaning |
|---|---|
| `pt`, `cls` | Garmin polyline type (`0x01` motorway … `0x0a` unpaved) and its Garmin-native class label. Note: **the Garmin class hierarchy is offset by one level relative to OSM's Polish usage** — see below. |
| `nodes` | JSON list of vertex indices that are **routing nodes** (`CoordNode` in the Garmin file). Endpoints are always included. The extractor uses these to split each road into node-to-node sub-ways so shape points on a bridge do **not** become graph intersections with the road underneath. |
| `class` `speed` `kmh` `oneway` `toll` `acc` `no_*` | NOD-1 Table-A fields (49% of the map's roads — the routable subset). |
| `planned` | `true` for `pt=0x0d` (under-construction in 2011, e.g. "BUDOWA AUTOSTRADY A1 CZĘSTOCHOWA-SOŚNICA"). **Filtered out** of the graph. |
| `lbl1`, `lbl2` | Primary/secondary label. Road numbers appear here: `A#`, `S#`, `DK#`, `DW###`, `E##`. |

The GPKG parser keeps **every polyline with a recognized drivable `cls`**,
dropping only `planned=1` (0x0d, under-construction segments) and rows whose
Table-A `no_car` flag is set. It does **not** require `class IS NOT NULL` even
though that field carries the Garmin Table-A speed / oneway / access — because
Table A is Garmin's *long-distance routing hierarchy* covering only ~49 % of
drivable polylines, and the other 51 % include local streets, village approach
roads, unpaved rural links, and even 2,700 primary-class segments that are
still real physical roads. Dropping them disconnected entire villages from the
network: a commune whose centroid sits on a road with no Table-A entry snapped
to a distant SCC member, producing 100-min errors on individual OD pairs and a
visibly choppy TT gradient. With the relaxed filter:

|                           | v2 (class NULL filter) | **v3 (relaxed)** | 2021 OSM reference |
|---|---:|---:|---:|
| routing nodes             |   706 812 |  **1 116 260** |     — |
| giant SCC                 | 604 950 (85.6 %) | **1 086 357 (97.3 %)** |     — |
| snap dist p95             |   6 228 m |  **974 m**  |     — |
| commune centroids >5 km   |       194 |      **2**  |     — |
| neighbour-gmina TT diff, p95 | 30 min |  **12 min** | 12 min |
| neighbour pairs with diff > 60 min | 85 |   **0**   |  0 |
| network total length      |  147 000 km | **434 500 km** |  ~424 000 km (official inventory) |

Roads without Table-A get their speed from the class-default profile via the
`cls` remap; `oneway` defaults to `False` where neither Table A nor NET1 flag
sets it. Sub-ways are then constructed by splitting each polyline at its
`nodes` indices and computing the geodesic sum of the internal shape-point
segments as its length. Endpoints are hashed to synthetic ref ids by rounded
(lon, lat), which merges cross-tile boundary nodes automatically.

#### Class remap: Garmin vs OSM Polish usage

The Garmin GPMapa TOPO hierarchy is offset by one level relative to OSM's
Polish-usage convention, so a straight cls→hwy pass-through would over-speed
every level. The parser applies this shift before the speed profile:

| Polish reality        | OSM `highway`     | Garmin `cls`         | Statutory rural |
|---|---|---|---:|
| autostrada `A#`       | motorway          | motorway (+ lbl1=A#) | 140 |
| ekspresowa `S#`       | trunk             | motorway (+ lbl1=S#) | 120 |
| krajowa `DK#`         | primary           | trunk                | 90 |
| wojewódzka `DW#`      | secondary         | primary              | 90 |
| powiatowa             | tertiary          | secondary            | 90 |
| gminna / local        | unclassified      | tertiary             | 70/50 |
| residential           | residential       | residential          | 50 |
| unpaved rural         | unclassified      | unpaved              | 70/50 |

`_gpkg_resolve_hwy(cls, lbl1)` implements this: when `lbl1` matches `A\d{1,2}`
it forces `motorway`; `S\d{1,3}` → `trunk`; everything else follows the shift.
Without this remap, the 2011 GPKG matrix was implausibly fast in ~10 % of pairs
(gmina/powiat over-speeded 20-40 km/h) and could beat the 2021 OSM matrix — a
counterfactual regression that shouldn't be possible under monotone network
expansion. After the remap: **98.5 % of gmina pairs have `t_2021 ≤ t_2011`**,
mean saving 52 min.

#### `--gpkg-promote-numbered` / `--no-gpkg-promote-numbered` (default: on)

GPMapa TOPO codes **all** motorways and expressways as class 4 / `kmh=100`,
losing the autostrada-vs-S-road distinction. When on (default), sub-ways whose
`lbl1` matches:

* `A#`  (autostrada) → promoted to 140 km/h
* `S#`  (droga ekspresowa) → promoted to 120 km/h
* `E##` (international) → unchanged (E-roads are overlays on DK/A routes)

A promotion is `max(map_kmh, promoted)`, so `--no-gpkg-promote-numbered` gives
you the map's Table-A speeds verbatim.

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

### Cross-year counterfactuals (RQ1) — use a consistent speed rule

`--speed-source tag` (default) prefers explicit `maxspeed` and is the most
accurate rule for a **single** year. It is **wrong for differencing two years**:
2012 OSM is ~3–11% `maxspeed`-tagged (roads inherit the optimistic 90 km/h rural
default) while 2021 is 23–99% tagged with the real, often-lower in-town limits.
Effective primary/secondary/tertiary speeds come out ~10–16% *lower* in 2021, so
`t_2021 − t_2012 > 0` on most pairs even though only motorways were added — a
data-coverage artifact, not infrastructure.

For RQ1, route both networks under **one** speed rule so only topology/road-class
changes:

```bash
# 1) derive an empirical class profile once from the well-tagged 2021 network
python road_travel_time_matrix.py --network .../poland_roads_2021-12-31_optimal.osm.pbf \
    --builtup-detection grid --population-grid data/raw/pop/poland_bbox_pop_100m.parquet \
    --dump-speed-profile data/processed/tt_matrix/speed_profile_empirical_2021.json

# 2) route EVERY year with that profile, class-only, grid built-up
for yr in 2012 2021 ; do
  python road_travel_time_matrix.py \
    --network data/raw/osm_pbf/poland_roads_${yr}-12-31_optimal.osm.pbf \
    --centroid-type pop-weighted --workers 40 \
    --speed-source class \
    --speed-profile data/processed/tt_matrix/speed_profile_empirical_2021.json \
    --builtup-detection grid --population-grid data/raw/pop/poland_bbox_pop_100m.parquet
done
```

Under a fixed speed rule the counterfactual is monotone: a richer network is
weakly faster on every pair (verified). Keep the **default `--speed-source tag`**
run of the 2021 network too — it is the most accurate *absolute* 2021 baseline
for calibration; the class-only pair is for the 2012-vs-2021 *comparison*.

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
