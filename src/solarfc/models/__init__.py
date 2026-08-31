"""Model implementations, one module per family.

The gradient-boosted trees live here; the recurrent and Transformer
families join them, all consuming :class:`solarfc.dataset.SupervisedSet`
and emitting rows in the schema defined by :mod:`solarfc.results`.
"""

from __future__ import annotations
