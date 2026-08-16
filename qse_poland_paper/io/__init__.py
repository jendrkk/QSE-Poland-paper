"""
qse_poland_paper.io — data-loading layer.

Each submodule loads one external object and returns it aligned to the canonical
`Frame`. Keeping the loaders separate is the main extensibility seam: adding a
rail travel-time matrix, a per-year floor-space index, or a different wage source
means adding/adjusting one loader, not touching the model core.
"""
from __future__ import annotations

from . import labour, ttm, floorspace, trade, flows

__all__ = ["labour", "ttm", "floorspace", "trade", "flows"]
