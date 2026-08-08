#!/usr/bin/env python3
"""
rcn_download.py
===============
Exhaustive downloader for the Polish RCiWN WFS (Rejestr Cen i Wartosci
Nieruchomosci) served by GUGiK at:

    https://mapy.geoportal.gov.pl/wss/service/rcn

Downloads *all* features and *all* attributes for the ``budynki`` (buildings)
and ``lokale`` (premises/units) layers and writes one GeoPackage per layer to
``data/raw/floorspace``.

WHY NOT PLAIN PAGINATION
------------------------
The service is MapServer WFS 2.0.0. It advertises ImplementsResultPaging=TRUE
but PagingIsTransactionSafe=FALSE and exposes no guaranteed-unique sortable
attribute, so STARTINDEX/COUNT paging over the whole layer returns features in
a non-deterministic order (features get duplicated / skipped across pages).
Deep STARTINDEX paging is also O(n^2) on MapServer. Both make naive pagination
unusable for a full extract of 6.2M + 2.6M features.

STRATEGY -- ADAPTIVE BBOX QUADTREE
----------------------------------
Every request is spatially bounded, which sidesteps ordering entirely:

  plan  : recursively subdivide the national extent (EPSG:2180). For each tile
          ask RESULTTYPE=hits; if the count is <= MAX_HITS the tile is a leaf,
          otherwise split it into four quadrants. Leaves are cached to JSON.
  fetch : download each leaf tile in a single GetFeature request and save it
          immediately as gzipped GML. Completed tiles are skipped on re-run, so
          progress is never lost (this is the "save every page" guarantee).
  merge : stream all cached tiles into one GeoPackage per layer, de-duplicating
          by gml:id (WFS BBOX is an *intersects* test, so features on tile
          borders appear in several tiles and must be de-duplicated).

The three phases are independently resumable. Re-running the script only does
the work that is still missing.

USAGE
-----
    PY=/Users/jedrek/miniforge3/envs/py314/bin/python
    $PY rcn_download.py                 # full run: plan + fetch + merge, both layers
    $PY rcn_download.py --phase plan    # only compute the tile plan
    $PY rcn_download.py --phase fetch   # only download tiles
    $PY rcn_download.py --phase merge   # only build the GeoPackages
    $PY rcn_download.py --layers budynki --workers 6
    $PY rcn_download.py --replan        # discard cached plan and recompute it

Requires: requests, pyogrio, geopandas, pandas, pyproj (present in the
miniforge ``py314`` env).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------------------------------------------------------
# Configuration (overridable via CLI)
# ----------------------------------------------------------------------------
BASE_URL = "https://mapy.geoportal.gov.pl/wss/service/rcn"
WFS_VERSION = "2.0.0"
SRS = "urn:ogc:def:crs:EPSG::2180"          # native CRS of the layers
EPSG = 2180

# Repo layout: this file lives in <repo>/scripts/geospatial/rcn_download.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "floorspace"

LAYERS_DEFAULT = ["budynki", "lokale"]

# Quadtree / request tuning.
MAX_HITS = 60_000        # a tile with <= this many features is downloaded whole
DOWNLOAD_COUNT = 100_000  # page size (server was verified to honour 100k)
MIN_TILE_SIZE = 250.0    # metres; stop subdividing below this (safety floor)
MAX_DEPTH = 18           # hard recursion cap (safety)
WORKERS = 4              # concurrent HTTP requests (be polite to a gov server)
CONNECT_TIMEOUT = 30     # seconds to establish a connection
READ_TIMEOUT = 120       # seconds to wait for the NEXT chunk of data
RETRIES = 20
BACKOFF = 1.5           # exponential backoff factor for retries
BACKOFF_MAX = 60.0      # cap on a single backoff sleep (seconds)
PAGE_SAFETY_CAP = 1000   # max pages per tile before we bail (should never hit)
HEARTBEAT_SECS = 20      # time-based progress line even while big tiles download

# Streaming stall-watchdog: abort a request whose throughput collapses (the
# geoportal throttles heavy IPs to a near-zero trickle). Slow-but-alive tiles
# are tolerated; only near-dead streams are aborted and retried.
STREAM_CHUNK = 1 << 20   # 1 MiB read chunks
STALL_WINDOW = 60        # seconds per throughput-measurement window
STALL_MIN_BPS = 20_000   # < 20 KB/s sustained over a window => abort + retry
PAGE_RETRIES = 5         # in-run retries per page on stall/connection drop
PAGE_RETRY_PAUSE = 30    # seconds to wait before an in-run retry

# Merge: flush accumulated features to the GeoPackage every ~this many rows
# (bounds memory; far fewer SQLite transactions than one-write-per-tile).
FLUSH_EVERY = 400_000

# National extent of Poland in EPSG:2180 with a generous margin. The plan phase
# refines this from the service capabilities, but this static box is a safe
# fallback that fully contains the country.
ROOT_BBOX_2180 = (120_000.0, 90_000.0, 900_000.0, 960_000.0)

BBox = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) = (E,N,E,N)

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_NUM_RETURNED = re.compile(rb'numberReturned="(\d+)"')
_NUM_MATCHED = re.compile(rb'numberMatched="(\d+)"')


def _num_returned(raw: bytes) -> int:
    m = _NUM_RETURNED.search(raw)
    return int(m.group(1)) if m else -1


def _num_matched(raw: bytes) -> Optional[int]:
    m = _NUM_MATCHED.search(raw)
    return int(m.group(1)) if m else None


def quads(b: BBox) -> list[BBox]:
    """Split a bbox into its four equal quadrants."""
    x0, y0, x1, y1 = b
    xm = (x0 + x1) / 2.0
    ym = (y0 + y1) / 2.0
    return [
        (x0, y0, xm, ym), (xm, y0, x1, ym),
        (x0, ym, xm, y1), (xm, ym, x1, y1),
    ]


def tile_id(b: BBox) -> str:
    """Stable, human-readable id from integer-rounded coordinates."""
    return "{:d}_{:d}_{:d}_{:d}".format(*(int(round(v)) for v in b))

# ----------------------------------------------------------------------------
# WFS client
# ----------------------------------------------------------------------------
class WFSError(RuntimeError):
    pass


class StallError(WFSError):
    """Raised when a response streams below the throughput floor (throttled)."""


# Exceptions worth an in-run retry with a fresh connection.
RETRYABLE = (
    StallError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)


class RCNClient:
    def __init__(self, base_url: str = BASE_URL,
                 timeout: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT)):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(
            total=RETRIES, connect=RETRIES, read=RETRIES, status=RETRIES,
            backoff_factor=BACKOFF,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        # Cap per-attempt backoff (attr name differs across urllib3 versions).
        try:
            retry.backoff_max = BACKOFF_MAX
        except Exception:
            pass
        ad = HTTPAdapter(max_retries=retry, pool_connections=WORKERS * 2,
                         pool_maxsize=WORKERS * 2)
        self.session.mount("https://", ad)
        self.session.mount("http://", ad)
        self.session.headers.update({"User-Agent": "QSE-Poland-RCN-downloader/1.0"})

    def _bbox_param(self, b: BBox) -> str:
        return "{:.3f},{:.3f},{:.3f},{:.3f},{}".format(b[0], b[1], b[2], b[3], SRS)

    def _get(self, params: dict, stream: bool = False) -> bytes:
        if not stream:
            r = self.session.get(self.base_url, params=params,
                                 timeout=self.timeout)
            r.raise_for_status()
            raw = r.content
        else:
            raw = self._get_streaming(params)
        if b"ExceptionReport" in raw[:4000] or b"ServiceException" in raw[:4000]:
            raise WFSError(raw[:600].decode("utf-8", "replace"))
        return raw

    def _get_streaming(self, params: dict) -> bytes:
        """GET with a throughput watchdog: raise StallError if the stream drops
        below STALL_MIN_BPS over a STALL_WINDOW-second window."""
        buf = bytearray()
        win_start = time.time()
        win_bytes = 0
        with self.session.get(self.base_url, params=params,
                              timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(STREAM_CHUNK):
                if chunk:
                    buf.extend(chunk)
                    win_bytes += len(chunk)
                elapsed = time.time() - win_start
                if elapsed >= STALL_WINDOW:
                    rate = win_bytes / elapsed
                    if rate < STALL_MIN_BPS:
                        raise StallError(
                            f"throughput {rate:,.0f} B/s < {STALL_MIN_BPS:,} "
                            f"over {int(elapsed)}s ({len(buf):,} B so far)")
                    win_start = time.time()
                    win_bytes = 0
        return bytes(buf)

    def hits(self, layer: str, bbox: Optional[BBox] = None) -> int:
        params = {
            "SERVICE": "WFS", "VERSION": WFS_VERSION, "REQUEST": "GetFeature",
            "TYPENAMES": f"ms:{layer}", "RESULTTYPE": "hits",
        }
        if bbox is not None:
            params["BBOX"] = self._bbox_param(bbox)
        raw = self._get(params)
        n = _num_matched(raw)
        if n is None:
            raise WFSError(f"hits: no numberMatched in response for {layer}")
        return n

    def get_features(self, layer: str, bbox: BBox, count: int,
                     startindex: int = 0) -> bytes:
        params = {
            "SERVICE": "WFS", "VERSION": WFS_VERSION, "REQUEST": "GetFeature",
            "TYPENAMES": f"ms:{layer}", "SRSNAME": SRS,
            "BBOX": self._bbox_param(bbox),
            "COUNT": str(count), "STARTINDEX": str(startindex),
        }
        raw = self._get(params, stream=True)
        if b"</wfs:FeatureCollection>" not in raw[-4000:]:
            raise WFSError("truncated FeatureCollection (no closing tag)")
        return raw

# ----------------------------------------------------------------------------
# Phase 1 -- PLAN (adaptive quadtree over the national extent)
# ----------------------------------------------------------------------------
@dataclass
class Leaf:
    bbox: BBox
    hits: int
    depth: int

    def as_dict(self) -> dict:
        return {"id": tile_id(self.bbox), "bbox": list(self.bbox),
                "hits": self.hits, "depth": self.depth}


def plan_layer(client: RCNClient, layer: str, root: BBox, max_hits: int,
               workers: int) -> list[Leaf]:
    """Breadth-first adaptive subdivision. Returns the list of leaf tiles."""
    leaves: list[Leaf] = []
    frontier: list[tuple[BBox, int]] = [(root, 0)]
    total_root = client.hits(layer, root)
    log(f"plan[{layer}]: root hits = {total_root:,}")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        while frontier:
            hit_vals = list(ex.map(lambda nd: client.hits(layer, nd[0]), frontier))
            nxt: list[tuple[BBox, int]] = []
            for (bbox, depth), h in zip(frontier, hit_vals):
                if h <= 0:
                    continue
                w, ht = bbox[2] - bbox[0], bbox[3] - bbox[1]
                too_small = min(w, ht) <= MIN_TILE_SIZE
                if h <= max_hits or depth >= MAX_DEPTH or too_small:
                    leaves.append(Leaf(bbox, h, depth))
                else:
                    nxt.extend((q, depth + 1) for q in quads(bbox))
            log(f"plan[{layer}]: depth done, leaves={len(leaves):,}, "
                f"frontier next={len(nxt):,}")
            frontier = nxt
    leaves.sort(key=lambda lf: lf.bbox)
    return leaves

def plan_path(cache_dir: Path, layer: str) -> Path:
    return cache_dir / f"plan_{layer}.json"


def load_or_build_plan(client: RCNClient, layer: str, cache_dir: Path,
                       root: BBox, max_hits: int, workers: int,
                       replan: bool) -> list[Leaf]:
    p = plan_path(cache_dir, layer)
    if p.exists() and not replan:
        data = json.loads(p.read_text())
        leaves = [Leaf(tuple(d["bbox"]), d["hits"], d["depth"])
                  for d in data["leaves"]]
        log(f"plan[{layer}]: loaded {len(leaves):,} cached leaf tiles "
            f"(total est. {data.get('total_hits', '?')})")
        return leaves
    t0 = time.time()
    leaves = plan_layer(client, layer, root, max_hits, workers)
    total = sum(lf.hits for lf in leaves)
    payload = {
        "layer": layer, "root_bbox": list(root), "max_hits": max_hits,
        "n_leaves": len(leaves), "total_hits": total,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "leaves": [lf.as_dict() for lf in leaves],
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, p)
    log(f"plan[{layer}]: {len(leaves):,} leaves, sum(hits)={total:,} "
        f"(includes border double-counting) in {time.time()-t0:.0f}s")
    return leaves

# ----------------------------------------------------------------------------
# Phase 2 -- FETCH (download every leaf tile, save immediately, skip existing)
# ----------------------------------------------------------------------------
def _atomic_gzip_write(path: Path, raw: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(raw)
    os.replace(tmp, path)


def tiles_dir(cache_dir: Path, layer: str) -> Path:
    d = cache_dir / f"tiles_{layer}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_leaf(client: RCNClient, layer: str, leaf: Leaf,
               tdir: Path, count: int) -> tuple[str, int]:
    """Download one leaf tile (paging only if the server unexpectedly caps).
    Returns (status, n_features). Idempotent: skips a completed tile."""
    tid = tile_id(leaf.bbox)
    done = tdir / f"{tid}.done.json"
    if done.exists():
        info = json.loads(done.read_text())
        return "skip", info.get("n", 0)

    pages: list[str] = []
    n_total = 0
    start = 0
    k = 0
    while True:
        # In-run retry with a fresh connection on stall/connection drop, so a
        # throttled tile self-heals instead of blocking a worker indefinitely.
        for attempt in range(1, PAGE_RETRIES + 1):
            try:
                raw = client.get_features(layer, leaf.bbox, count, start)
                break
            except RETRYABLE as e:
                if attempt == PAGE_RETRIES:
                    raise
                log(f"tile {tid} p{k} attempt {attempt}/{PAGE_RETRIES} "
                    f"retrying after {PAGE_RETRY_PAUSE}s: {e!r}")
                time.sleep(PAGE_RETRY_PAUSE)
        nret = _num_returned(raw)
        if nret <= 0:
            break
        pfile = tdir / f"{tid}.p{k}.gml.gz"
        _atomic_gzip_write(pfile, raw)
        pages.append(pfile.name)
        n_total += nret
        if nret < count:
            break
        start += count
        k += 1
        if k >= PAGE_SAFETY_CAP:
            raise WFSError(f"{tid}: exceeded page safety cap")
    payload = {"id": tid, "bbox": list(leaf.bbox), "n": n_total,
               "pages": pages, "planned_hits": leaf.hits}
    tmp = done.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, done)
    return "ok", n_total

def fetch_layer(client: RCNClient, layer: str, leaves: list[Leaf],
                cache_dir: Path, count: int, workers: int) -> None:
    tdir = tiles_dir(cache_dir, layer)
    todo = [lf for lf in leaves
            if not (tdir / f"{tile_id(lf.bbox)}.done.json").exists()]
    total = len(leaves)
    log(f"fetch[{layer}]: {total:,} tiles total, {len(todo):,} still to download")
    if not todo:
        log(f"fetch[{layer}]: nothing to do (all tiles already cached)")
        return

    done_ct = total - len(todo)
    feat_ct = 0
    failed: list[tuple[Leaf, str]] = []
    inflight: set[str] = set()
    lock = threading.Lock()
    t0 = time.time()

    # Background heartbeat: prints a status line every HEARTBEAT_SECS even when
    # no tile has completed (large tiles can take 1-2 min each).
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_SECS):
            with lock:
                dc, fc, inf = done_ct, feat_ct, sorted(inflight)
            rate = fc / max(time.time() - t0, 1e-9)
            log(f"fetch[{layer}] ~ {dc:,}/{total:,} tiles done | "
                f"{len(inf)} downloading now {inf[:4]}"
                f"{'...' if len(inf) > 4 else ''} | {fc:,} new feat | "
                f"{rate:,.0f} feat/s | elapsed {int(time.time()-t0)}s")

    def run(lf: Leaf) -> tuple[str, int]:
        tid = tile_id(lf.bbox)
        with lock:
            inflight.add(tid)
        try:
            return fetch_leaf(client, layer, lf, tdir, count)
        finally:
            with lock:
                inflight.discard(tid)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run, lf): lf for lf in todo}
            for fut in cf.as_completed(futs):
                lf = futs[fut]
                tid = tile_id(lf.bbox)
                try:
                    status, n = fut.result()
                except Exception as e:  # keep going; tile retried on re-run
                    with lock:
                        failed.append((lf, repr(e)))
                    log(f"fetch[{layer}]: WARN tile {tid} failed: {e!r} "
                        f"(left for retry)")
                    continue
                with lock:
                    done_ct += 1
                    feat_ct += n
                    dc, fc = done_ct, feat_ct
                rate = fc / max(time.time() - t0, 1e-9)
                log(f"fetch[{layer}]: {dc:,}/{total:,} tiles | +{n:,} feat "
                    f"(tile {tid}) | {fc:,} new feat | {rate:,.0f} feat/s")
    finally:
        stop.set()
        hb.join(timeout=2)

    # One sequential retry pass for tiles that dropped (eases server pressure).
    if failed:
        log(f"fetch[{layer}]: retrying {len(failed)} failed tiles sequentially...")
        still: list[str] = []
        for lf, _ in failed:
            tid = tile_id(lf.bbox)
            try:
                _, n = fetch_leaf(client, layer, lf, tdir, count)
                with lock:
                    done_ct += 1
                    feat_ct += n
                log(f"fetch[{layer}]: retry OK {tid} (+{n:,} feat)")
            except Exception as e:
                still.append(tid)
                log(f"fetch[{layer}]: retry FAILED {tid}: {e!r}")
        if still:
            log(f"fetch[{layer}]: {len(still)} tiles still failing; just re-run "
                f"the script to pick them up: {still[:8]}"
                f"{'...' if len(still) > 8 else ''}")

    log(f"fetch[{layer}]: complete, {feat_ct:,} features fetched this run "
        f"in {time.time()-t0:.0f}s ({done_ct:,}/{total:,} tiles cached)")

# ----------------------------------------------------------------------------
# Phase 3 -- MERGE (stream tiles -> one GeoPackage per layer, de-dup by gml:id)
# ----------------------------------------------------------------------------
# GDAL knobs: expose gml:id as a field, do not resolve xlink refs.
os.environ.setdefault("GML_EXPOSE_GML_ID", "YES")
os.environ.setdefault("GML_SKIP_RESOLVE_ELEMS", "ALL")
os.environ.setdefault("OGR_GEOMETRY_ACCEPT_UNKNOWN_SRS", "YES")


def _read_tile_gml(gz_path: Path):
    """Decompress a gzipped GML tile to a temp file and read it. All non-geom
    columns are cast to pandas string dtype for schema stability across tiles."""
    import tempfile
    import pyogrio
    import pandas as pd

    with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tf:
        tmpname = tf.name
        with gzip.open(gz_path, "rb") as src:
            tf.write(src.read())
    try:
        gdf = pyogrio.read_dataframe(tmpname)
    finally:
        for ext in ("", ".gfs"):
            try:
                os.remove(tmpname + ext)
            except OSError:
                pass
    if gdf is None or len(gdf) == 0:
        return None
    for col in gdf.columns:
        if col == "geometry":
            continue
        gdf[col] = gdf[col].astype("string")
    return gdf

def merge_layer(layer: str, cache_dir: Path, out_dir: Path,
                out_format: str, work_dir: Optional[Path] = None) -> dict:
    import shutil
    import tempfile
    import pyogrio
    import pandas as pd
    import geopandas as gpd

    tdir = tiles_dir(cache_dir, layer)
    page_files = sorted(tdir.glob("*.p*.gml.gz"))
    if not page_files:
        raise SystemExit(f"merge[{layer}]: no tiles in {tdir}; run fetch first")

    # Build the GeoPackage on LOCAL disk first. GPKG is SQLite, which needs
    # POSIX fcntl locks that most network filesystems (NFS) do not provide ->
    # "Failed to start transaction". We assemble locally, then move the
    # finished single-file .gpkg to out_dir (which may live on NFS).
    scratch = Path(tempfile.mkdtemp(prefix=f"rcn_merge_{layer}_",
                                    dir=str(work_dir) if work_dir else None))
    work_gpkg = scratch / f"{layer}.gpkg"
    out_gpkg = out_dir / f"{layer}.gpkg"
    log(f"merge[{layer}]: {len(page_files):,} tile files -> {out_gpkg}")
    log(f"merge[{layer}]: assembling in local scratch {scratch}")

    seen: set[str] = set()
    ref_cols: Optional[list[str]] = None
    written = 0
    buf: list = []
    buf_n = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, buf_n, written
        if not buf:
            return
        gdf = pd.concat(buf, ignore_index=True) if len(buf) > 1 else buf[0]
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=f"EPSG:{EPSG}")
        pyogrio.write_dataframe(gdf, work_gpkg, layer=layer, driver="GPKG",
                                append=(written > 0))
        written += len(gdf)
        buf = []
        buf_n = 0
        rate = written / max(time.time() - t0, 1e-9)
        log(f"merge[{layer}]: flushed -> {written:,} unique features "
            f"| {rate:,.0f} feat/s")

    try:
        for i, pf in enumerate(page_files, 1):
            gdf = _read_tile_gml(pf)
            if gdf is None:
                continue
            if "gml_id" not in gdf.columns:
                raise WFSError(f"{pf.name}: no gml_id column exposed by GDAL")
            gdf = gdf[~gdf["gml_id"].isin(seen)]
            if len(gdf) == 0:
                continue
            seen.update(gdf["gml_id"].tolist())
            if ref_cols is None:
                ref_cols = list(gdf.columns)
            elif list(gdf.columns) != ref_cols:
                gdf = gdf.reindex(columns=ref_cols)
            buf.append(gdf)
            buf_n += len(gdf)
            if buf_n >= FLUSH_EVERY:
                flush()
            if i % 25 == 0 or i == len(page_files):
                log(f"merge[{layer}]: read {i:,}/{len(page_files):,} tiles "
                    f"| {len(seen):,} unique so far | buffer {buf_n:,}")
        flush()

        out_dir.mkdir(parents=True, exist_ok=True)
        if out_gpkg.exists():
            out_gpkg.unlink()
        shutil.move(str(work_gpkg), str(out_gpkg))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # Verify the on-disk feature count matches the de-duplicated total.
    info = pyogrio.read_info(str(out_gpkg))
    n_out = int(info["features"])
    ok = (n_out == written == len(seen))
    stats = {"layer": layer, "unique_features": written,
             "gpkg_features": n_out, "tile_files": len(page_files),
             "verified": bool(ok), "output": str(out_gpkg)}
    if not ok:
        log(f"merge[{layer}]: WARN count mismatch written={written:,} "
            f"gpkg={n_out:,} seen={len(seen):,}")
    if out_format in ("parquet", "both"):
        _write_parquet(out_gpkg, out_dir / f"{layer}.parquet", stats)
    log(f"merge[{layer}]: DONE -> {out_gpkg} "
        f"({written:,} unique features, verified={ok})")
    return stats


def _write_parquet(gpkg_path: Path, parquet_path: Path, stats: dict) -> None:
    """Optional GeoParquet export (reads the finished GeoPackage back)."""
    import geopandas as gpd
    log(f"parquet: exporting {gpkg_path.name} -> {parquet_path.name}")
    gdf = gpd.read_file(gpkg_path)
    gdf.to_parquet(parquet_path, index=False)
    stats["parquet"] = str(parquet_path)


# ----------------------------------------------------------------------------
# Orchestration / CLI
# ----------------------------------------------------------------------------
def run_layer(client: RCNClient, layer: str, cache_dir: Path, out_dir: Path,
              root: BBox, args) -> Optional[dict]:
    if args.phase in ("plan", "fetch", "all"):
        leaves = load_or_build_plan(client, layer, cache_dir, root,
                                    args.max_hits, args.workers, args.replan)
    else:
        leaves = load_or_build_plan(client, layer, cache_dir, root,
                                    args.max_hits, args.workers, False)
    if args.phase in ("fetch", "all"):
        fetch_layer(client, layer, leaves, cache_dir, args.count, args.workers)
    if args.phase in ("merge", "all"):
        return merge_layer(layer, cache_dir, out_dir, args.out_format,
                           work_dir=args.work_dir)
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exhaustive RCiWN WFS downloader (budynki + lokale).")
    p.add_argument("--layers", nargs="+", default=LAYERS_DEFAULT,
                   help="Layers to process (default: budynki lokale).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output directory (default: data/raw/floorspace).")
    p.add_argument("--work-dir", type=Path, default=None, dest="work_dir",
                   help="LOCAL scratch dir for building GeoPackages before "
                        "moving them to --out. Set this to fast local disk if "
                        "--out is on a network filesystem (NFS). Default: "
                        "system temp.")
    p.add_argument("--phase", choices=["plan", "fetch", "merge", "all"],
                   default="all", help="Which phase to run (default: all).")
    p.add_argument("--workers", type=int, default=WORKERS,
                   help=f"Concurrent HTTP requests (default: {WORKERS}).")
    p.add_argument("--max-hits", type=int, default=MAX_HITS, dest="max_hits",
                   help=f"Subdivide tiles above this count (default {MAX_HITS}).")
    p.add_argument("--count", type=int, default=DOWNLOAD_COUNT,
                   help=f"GetFeature page size (default {DOWNLOAD_COUNT}).")
    p.add_argument("--out-format", choices=["gpkg", "parquet", "both"],
                   default="gpkg", dest="out_format",
                   help="Final output format(s) (default: gpkg).")
    p.add_argument("--replan", action="store_true",
                   help="Discard any cached tile plan and recompute it.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_wfs_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = RCNClient()
    root = ROOT_BBOX_2180
    log(f"root bbox (EPSG:{EPSG}) = {root}")
    log(f"layers={args.layers} phase={args.phase} workers={args.workers} "
        f"out={out_dir}")

    all_stats = []
    t0 = time.time()
    for layer in args.layers:
        log(f"===== layer: {layer} =====")
        st = run_layer(client, layer, cache_dir, out_dir, root, args)
        if st:
            all_stats.append(st)

    if all_stats:
        summary = out_dir / "rcn_download_summary.json"
        summary.write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "elapsed_s": round(time.time() - t0, 1),
             "layers": all_stats}, indent=2))
        log(f"summary written -> {summary}")
    log(f"all done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
