"""Model implementations, one module per family.

Phase 2 covers the gradient-boosted trees; Phases 3 and 4 add the recurrent and
Transformer families here alongside them, all consuming
:class:`solarfc.dataset.SupervisedSet` and emitting rows in the schema defined
by :mod:`solarfc.results`.
"""

from __future__ import annotations
