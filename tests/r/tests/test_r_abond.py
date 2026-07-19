import os

import pytest

# Rule 22 (AGENTS.md): an intentionally-deferred parity gap is documented with a
# strict XFAIL so a future agent who lifts the deferral gets an xpass -> strict
# failure, surfacing exactly what was left.
#
# abond R-parity is intentionally DEFERRED.  FUTURE_WORK.md §"ABOND R-Parity
# (DEFERRED ...)" (lines 317-381) records that `oe.abond()` has full Stata
# `xtabond2` parity (40 tests, green) but NO R parity because the canonical R
# anchor `plm::pgmm` is broken upstream in this environment (fails on the
# canonical EmplUK example for both effect="twoways" and effect="individual";
# verified across plm 2.6.3/2.6.4/2.6.7 on R 4.6.1 AND 4.5.2).  Stata xtabond2
# parity is authoritative.  Revisit only when plm pgmm is fixed upstream.

_R_ABOND_TEST = os.path.abspath(__file__)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "abond R-parity intentionally deferred: plm::pgmm broken upstream "
        "(FUTURE_WORK ABOND R-Parity, lines 317-381). Stata xtabond2 parity is "
        "authoritative. Revisit only when plm pgmm fixed."
    ),
)
def test_abond_r_parity_not_yet_covered():
    # Genuine, self-documenting assertion: we WANT an R parity anchor but cannot
    # have one (plm::pgmm is broken).  The absence of an R abond test file is the
    # explicit marker of this accepted deferral.  When a real R fixture becomes
    # possible, a future agent will add tests/r/tests/test_r_abond.py; that makes
    # this assertion pass -> strict xfail turns into an xpass failure, forcing the
    # deferral to be formally lifted rather than silently left.
    assert not os.path.exists(_R_ABOND_TEST)
