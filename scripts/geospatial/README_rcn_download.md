# RCiWN WFS bulk downloader (`budynki`, `lokale`)

Downloads **every feature and every attribute** of the RCiWN buildings
(`budynki`, 6,233,436) and premises (`lokale`, 2,593,413) layers from the GUGiK
WFS and writes one GeoPackage per layer into `data/raw/floorspace/`.

## Why the naive approach fails
The endpoint is MapServer **WFS 2.0.0** (`https://mapy.geoportal.gov.pl/wss/service/rcn`).
It only emits **GML** (no GeoJSON/CSV), `PagingIsTransactionSafe=FALSE`, and has
no guaranteed-unique sortable property, so `STARTINDEX`/`COUNT` paging returns
features in a **non-deterministic order** (duplicates + gaps). That is why
`gpd.read_file("WFS:...")` streamed for an hour and never finished.

## How this works — adaptive BBOX quadtree
Every request is spatially bounded, so ordering is irrelevant.

1. **plan** — recursively subdivide the national extent (EPSG:2180). For each
   tile, `RESULTTYPE=hits` gives the exact count; tiles with ≤ `--max-hits`
   (default 60k) become leaves, denser tiles split into 4. Leaves cached to
   `_wfs_cache/plan_<layer>.json`.
2. **fetch** — download each leaf in one `GetFeature` and immediately save it as
   gzipped GML in `_wfs_cache/tiles_<layer>/`. Completed tiles are marked
   `<id>.done.json` and **skipped on re-run** → progress is never lost (this is
   the paginated-style checkpointing you asked for).
3. **merge** — stream all tiles into `<layer>.gpkg`, **de-duplicating by
   `gml:id`** (WFS BBOX is an *intersects* test, so features on tile borders
   appear in adjacent tiles). Attributes are stored as text to preserve the raw
   registry values exactly; CRS is EPSG:2180.

The three phases are independently resumable; re-running only does missing work.

## Run it
```bash
PY=/Users/jedrek/miniforge3/envs/py314/bin/python   # env with geopandas/pyogrio
cd "scripts/geospatial"

$PY rcn_download.py                 # full run, both layers (plan+fetch+merge)
$PY rcn_download.py --phase plan    # just build the tile plan
$PY rcn_download.py --phase fetch   # just download tiles (resumable)
$PY rcn_download.py --phase merge   # just build the GeoPackages
$PY rcn_download.py --layers budynki --workers 6
$PY rcn_download.py --out-format both   # also emit GeoParquet
```

## Output
```
data/raw/floorspace/
├── budynki.gpkg                 # 1 layer "budynki", EPSG:2180, all attributes
├── lokale.gpkg
├── rcn_download_summary.json    # feature counts, timing
└── _wfs_cache/                  # tile plan + gzipped GML tiles (safe to delete
    ├── plan_budynki.json        #   after a successful merge)
    ├── tiles_budynki/
    └── tiles_lokale/
```

## Notes / expectations
- **Volume:** ~8.8M features. Raw GML is verbose (~2.5 KB/feature); gzipped tile
  cache is a few GB, final GeoPackages several GB. Ensure ~15–20 GB free.
- **Wall time:** transfer-bound; a few hours on a normal connection with 4
  workers. Interrupt any time and re-run — it resumes.
- **Politeness:** default 4 concurrent requests with retry/backoff. Raise
  `--workers` cautiously; it is a public government server.
- **Validated:** on a 20 km test box the merged unique count equals the exact
  `hits` count (2,984), confirming completeness and correct border de-dup.
