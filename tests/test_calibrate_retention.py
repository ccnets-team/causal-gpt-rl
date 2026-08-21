"""The calibration script has to say "no ladder" instead of picking one.

`normalized` returns NaN for a policy that did not beat uniform sampling, and
NaN is silent in both places the summary would use it: `pick` is a min over
`abs(norm - target)`, where every comparison against NaN is False, and the
spread guard compares NaN against a threshold, which is False too. So the run
that most needs the warning is the run that would print tier picks and no
warning at all. These pin the trap as much as the guard — a later reader has to
see why the check is not redundant.
"""

import math

import pytest

from examples.mujoco_collection import calibrate_retention as calib


def _curve(norms) -> list[dict]:
    return [
        {"retention": 16 * 2**i, "norm": norm, "return_mean": float(i)}
        for i, norm in enumerate(norms)
    ]


def test_a_policy_that_lost_to_random_has_no_normalized_score():
    assert math.isnan(calib.normalized(5.0, random_ref=10.0, expert_ref=10.0))
    assert math.isnan(calib.normalized(5.0, random_ref=10.0, expert_ref=4.0))


def test_a_policy_that_beat_random_normalizes_to_the_published_scale():
    assert calib.normalized(50.0, random_ref=0.0, expert_ref=100.0) == 50.0
    assert calib.normalized(100.0, random_ref=0.0, expert_ref=100.0) == 100.0


def test_a_curve_of_nan_is_a_collapsed_ladder():
    assert calib.ladder_collapsed(_curve([float("nan")] * 3))


def test_one_real_score_is_not_a_collapsed_ladder():
    # The grid shares one denominator, so a mixed curve cannot come out of a
    # real run; asserting it anyway keeps the predicate `all`, not `any`.
    assert not calib.ladder_collapsed(_curve([float("nan"), 40.0]))
    assert not calib.ladder_collapsed(_curve([10.0, 40.0, 90.0]))


@pytest.mark.parametrize("target", [40.0, 70.0])
def test_pick_returns_the_first_row_when_every_score_is_nan(target):
    # Not the behavior anyone wants — it is why `ladder_collapsed` gates the
    # call. If `pick` ever learns to refuse on its own, this is the test that
    # says so.
    rows = _curve([float("nan")] * 3)

    assert calib.pick(rows, target) is rows[0]


def test_the_spread_guard_cannot_catch_a_collapsed_ladder():
    # The other half of the same silence: `report_picks` prints its note when
    # the spread falls under the threshold, and a NaN spread never does.
    spread = float("nan")

    assert not spread < calib.MIN_LADDER_SPREAD


def test_the_no_ladder_note_survives_a_legacy_code_page():
    # Every print in this script is ASCII: a console on cp949 raises on a
    # non-ASCII write, which would fail a run whose measurements are done.
    calib.NO_LADDER_NOTE.encode("cp949")
