"""
Validation suite for Component 6 — full engagement loop + persistence.

REDESIGN: the loop no longer tries to recover gun bias and wind error as four
separate parameters from early fall of shot (under-determined on shot 1 — that is
why the old panel showed wind stuck at 0.0). It registers with ONE shot, observes
the FULL miss, and cancels it with a one-step Newton aim correction; the learned
quantity is the TOTAL impact-space correction (gun bias + met error, combined),
which is all that is identifiable from a single impact and all that is needed,
because met is stable over the engagement. A remembered weapon warm-starts the
total so it fires for effect immediately.

Ground truth: the loop must converge to the hidden TOTAL error, persist it, and a
remembered weapon must converge faster next time. Integration only — the frozen
physics is called as a tool, never modified.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.engagement import run_engagement, _load_gun_bias
from engagement_database import EngagementDatabase

# Told MET (imperfect) vs the real atmosphere (hidden): a 3 m/s effective wind
# error plus a gun with a constant bias. The two combine into one total offset.
TOLD = MetMessage.standard_isa(surface_wind=(180.0, 20.0))   # 20 m/s tail (told)
TRUE = MetMessage.standard_isa(surface_wind=(180.0, 23.0))   # 23 m/s tail (real)
GUN = (200.0, 80.0)


@pytest.fixture
def db():
    path = tempfile.mktemp(suffix=".db")
    d = EngagementDatabase(path)
    yield d
    d.close()
    os.remove(path)


def true_conditions(gun=GUN, true_met=TRUE):
    return {"true_met": true_met, "gun_bias": gun}


# ---------------------------------------------------------------------------
# Layer 1 — Full-loop convergence (the core proof)
# ---------------------------------------------------------------------------
def test_layer1_converges_to_hidden_total(db):
    res = run_engagement("HOW-1", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    reg0 = res["phase_misses"]["REGISTRATION"][0]
    # FFE CEP much smaller than the opening registration miss.
    assert res["final_cep"] < 15.0
    assert res["final_cep"] < 0.1 * reg0
    # The learned TOTAL correction matches the hidden true total error (the full
    # miss to be cancelled). We do NOT claim a gun-vs-wind split.
    assert np.allclose(res["total_correction_est"], res["total_correction_true"],
                       atol=12.0), (res["total_correction_est"],
                                    res["total_correction_true"])


def test_layer1_shot2_is_the_kill(db):
    # Shot 1 registers (large miss); shot 2 (first FFE) lands tight in one Newton
    # step — the two-shot-kill premise.
    res = run_engagement("HOW-1", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    reg_first = res["phase_misses"]["REGISTRATION"][0]
    ffe_first = res["phase_misses"]["FFE"][0]
    assert reg_first > 100.0
    assert ffe_first < 15.0
    assert ffe_first < 0.1 * reg_first
    assert res["final_cep"] <= ffe_first + 1e-9


# ---------------------------------------------------------------------------
# Layer 2 — Persistence boundary
# ---------------------------------------------------------------------------
def test_layer2_learned_total_round_trips(db):
    res = run_engagement("HOW-2", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    saved = _load_gun_bias(db, "HOW-2")        # legacy column now holds TOTAL offset
    assert np.allclose(saved, res["total_correction_est"], atol=1e-9)
    # And it survives a fresh read of the same file.
    db._conn.commit()
    row = db._conn.execute(
        "SELECT gun_bias_dr, gun_bias_cr FROM weapon_profiles WHERE weapon_id=?",
        ("HOW-2",)).fetchone()
    assert np.allclose([row[0], row[1]], res["total_correction_est"], atol=1e-9)


def test_layer2_fresh_estimator_starts_from_remembered_total(db):
    run_engagement("HOW-2", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    assert _load_gun_bias(db, "HOW-2") is not None
    assert _load_gun_bias(db, "NEVER-FIRED") is None
    res2 = run_engagement("HOW-2", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    assert res2["warm_started"] is True
    # Warm start: the very first (pre-corrected) registration shot is already tight.
    assert res2["phase_misses"]["REGISTRATION"][0] < 30.0


# ---------------------------------------------------------------------------
# Layer 3 — Fleet memory (the persistence payoff)
# ---------------------------------------------------------------------------
def test_layer3_warm_start_converges_faster(db):
    cold = run_engagement("HOW-3", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    warm = run_engagement("HOW-3", 22000.0, 0.0, TOLD, true_conditions(), db=db)
    assert cold["warm_started"] is False and warm["warm_started"] is True
    # Converges at least as fast, and with a far tighter opening shot.
    assert warm["rounds_to_converge"] <= cold["rounds_to_converge"]
    assert (warm["phase_misses"]["REGISTRATION"][0]
            < cold["phase_misses"]["REGISTRATION"][0])


def test_layer3_separate_weapons_no_cross_contamination(db):
    run_engagement("GUN-A", 22000.0, 0.0, TOLD, true_conditions(gun=(200.0, 80.0)), db=db)
    total_a_before = _load_gun_bias(db, "GUN-A").copy()
    # A different weapon with a very different bias, same weather.
    run_engagement("GUN-B", 22000.0, 0.0, TOLD, true_conditions(gun=(-120.0, 40.0)), db=db)
    total_a_after = _load_gun_bias(db, "GUN-A")
    total_b = _load_gun_bias(db, "GUN-B")
    # GUN-A's profile is untouched by GUN-B's engagement.
    assert np.allclose(total_a_before, total_a_after, atol=1e-9)
    # The two learned totals are distinct, and since the weather is shared their
    # DIFFERENCE recovers the difference in gun bias (the wind part cancels).
    assert np.allclose(total_a_after - total_b, [320.0, 40.0], atol=10.0), \
        (total_a_after, total_b)


# ---------------------------------------------------------------------------
# Layer 4 — End-to-end sanity
# ---------------------------------------------------------------------------
def test_layer4_long_engagement_never_diverges(db):
    # Force many rounds (threshold 0 disables early stop); all stay finite and
    # bounded (the Phase 1 Kalman-gain stability lesson carries to the refiner).
    res = run_engagement("HOW-4", 22000.0, 0.0, TOLD, true_conditions(), db=db,
                         n_register=6, n_adjust=14, n_ffe=12,
                         converge_threshold_m=0.0)
    radials = [h["radial"] for h in res["history"]]
    assert all(np.isfinite(r) for r in radials)
    assert max(radials) < 1000.0          # bounded, never blows up
    assert res["final_cep"] < 15.0        # still converges


def test_layer4_no_false_correction_when_met_correct(db):
    # True conditions == told MET and zero gun bias: nothing to learn. The system
    # must fire accurately from the start without inventing a correction.
    res = run_engagement("HOW-5", 22000.0, 0.0, TOLD,
                         {"true_met": TOLD, "gun_bias": (0.0, 0.0)}, db=db)
    assert res["phase_misses"]["REGISTRATION"][0] < 1.0
    assert res["final_cep"] < 1.0
    assert np.linalg.norm(res["total_correction_est"]) < 5.0


def test_layer4_engagement_record_has_expected_fields(db):
    res = run_engagement("HOW-6", 22000.0, 45.0, TOLD, true_conditions(), db=db)
    row = db._conn.execute("""
        SELECT weapon_id, target_range, target_bearing, corrected_cep,
               rounds_to_converge, total_shots, atmo_dwind_dr, atmo_dwind_cr,
               final_cep, warm_started
        FROM engagements WHERE engagement_id=?""",
        (res["engagement_id"],)).fetchone()
    assert row["weapon_id"] == "HOW-6"
    assert row["target_range"] == 22000.0
    assert row["target_bearing"] == 45.0
    assert row["corrected_cep"] is not None
    assert row["rounds_to_converge"] == res["rounds_to_converge"]
    assert row["total_shots"] == res["total_shots"]
    assert row["final_cep"] is not None
    # atmo_dwind columns now persist the learned TOTAL correction.
    assert abs(row["atmo_dwind_dr"] - res["total_correction_est"][0]) < 1e-9


if __name__ == "__main__":
    path = tempfile.mktemp(suffix=".db")
    d = EngagementDatabase(path)
    print("\n=== Component 6 — full engagement summary (Newton two-shot kill) ===\n")

    r1 = run_engagement("DEMO-HOW", 22000.0, 0.0, TOLD, true_conditions(), db=d)
    print("Engagement 1 (NEW weapon, cold start):")
    for ph in ("REGISTRATION", "FFE"):
        ms = r1["phase_misses"][ph]
        print(f"    {ph:<13} misses (m): {[round(x, 1) for x in ms]}")
    print(f"    -> FFE CEP = {r1['final_cep']:.1f} m   "
          f"(opening miss {r1['phase_misses']['REGISTRATION'][0]:.0f} m)")
    print(f"    learned TOTAL correction = {np.round(r1['total_correction_est'], 1)}  "
          f"(true {np.round(r1['total_correction_true'], 1)})")
    print(f"    rounds to converge = {r1['rounds_to_converge']}")

    saved = _load_gun_bias(d, "DEMO-HOW")
    print(f"\nPersistence round-trip: DB total {np.round(saved, 1)} "
          f"== learned {np.round(r1['total_correction_est'], 1)}")

    r2 = run_engagement("DEMO-HOW", 22000.0, 0.0, TOLD, true_conditions(), db=d)
    print("\nEngagement 2 (SAME weapon, warm start from DB):")
    print(f"    first registration miss = {r2['phase_misses']['REGISTRATION'][0]:.1f} m "
          f"(was {r1['phase_misses']['REGISTRATION'][0]:.0f} m cold)  -> FLEET MEMORY")

    d.close(); os.remove(path)
    print("\nRun 'python -m pytest tests/test_engagement.py -v'.")
