"""
Validation suite for Component 13 — noisy observation of the fall of shot.

The fall of shot is reported by a forward observer / drone / radar, with
measurement error. This component feeds the ESTIMATOR the TRUE miss plus Gaussian
observation noise (a parameter, std dev in metres) and raises the filter's
measurement-noise term to match — while leaving the TRUE physics and the kill
assessment exact.

REDESIGN: the learned quantity is the TOTAL impact-space correction (gun bias +
met error, combined) — not a gun/wind split. Under noise the refiner still
converges to that total (bounded, no divergence), with a wider floor and more
rounds, and it does NOT chase the noise: averaging many observations beats any
single noisy reading.

numpy + stdlib only; everything here is deterministic via noise_seed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.met_message import MetMessage
from physics.engagement import run_engagement, run_engagement_until_destroyed

# Same hidden-truth scenario as Component 6/9: imperfect told MET vs the real
# atmosphere (3 m/s effective wind error) and a gun with a constant bias. The two
# combine into one total impact-space offset.
TOLD = MetMessage.standard_isa(surface_wind=(180.0, 20.0))
TRUE = MetMessage.standard_isa(surface_wind=(180.0, 23.0))
GUN = (200.0, 80.0)
RANGE, BEARING = 22000.0, 0.0


def true_conditions():
    return {"true_met": TRUE, "gun_bias": GUN}


def engage(noise, seed, **kw):
    return run_engagement("W", RANGE, BEARING, TOLD, true_conditions(),
                          observation_noise_m=noise, noise_seed=seed, **kw)


_CACHE = {}


def engage_cached(noise, seed):
    key = (noise, seed)
    if key not in _CACHE:
        _CACHE[key] = engage(noise, seed)
    return _CACHE[key]


def total_err(res):
    """How far the learned TOTAL correction is from the true total error."""
    return float(np.linalg.norm(np.asarray(res["total_correction_est"])
                                - np.asarray(res["total_correction_true"])))


def final_cep(res):
    return res["final_cep"]


def mean_over_seeds(noise, fn, n=5):
    return float(np.mean([fn(engage_cached(noise, s)) for s in range(n)]))


def single_reading_error(noise, n=5000):
    """Expected magnitude of using ONE noisy 2D observation as the estimate."""
    return float(np.mean([np.linalg.norm(np.random.default_rng(s).normal(0.0, noise, 2))
                          for s in range(n)]))


# ---------------------------------------------------------------------------
# Layer 1 — Converges despite noise (core proof)
# ---------------------------------------------------------------------------
def test_layer1_converges_with_realistic_noise():
    res = engage(15.0, seed=7)
    reg0 = res["phase_misses"]["REGISTRATION"][0]
    assert reg0 > 150.0                                   # opening miss really is large
    assert total_err(res) < 45.0, res["total_correction_est"]
    assert res["final_cep"] < 0.5 * reg0                  # far better than opening


def test_layer1_estimator_stays_bounded_no_divergence():
    # Kalman/RLS stability: under noise the estimate never diverges or oscillates
    # unboundedly. Force many rounds (no early stop) at high noise.
    res = engage(40.0, seed=3, n_register=6, n_adjust=8, n_ffe=4,
                 converge_threshold_m=0.0)
    radials = [h["radial"] for h in res["history"]]
    assert all(np.isfinite(r) for r in radials)
    assert max(radials) < 1000.0                          # bounded, no blow-up
    assert np.all(np.isfinite(res["total_correction_est"]))
    assert total_err(res) < 150.0                         # near truth, not chasing noise


def test_layer1_converges_across_many_seeds():
    assert mean_over_seeds(15.0, total_err) < 30.0


# ---------------------------------------------------------------------------
# Layer 2 — Reproduction / boundary (zero noise == perfect observation)
# ---------------------------------------------------------------------------
def test_layer2_zero_noise_reproduces_perfect_observation_bitforbit():
    base = run_engagement("W", RANGE, BEARING, TOLD, true_conditions())
    for seed in (None, 12345):
        r = engage(0.0, seed=seed)
        assert np.array_equal(np.asarray(r["total_correction_est"]),
                              np.asarray(base["total_correction_est"]))
        assert r["final_cep"] == base["final_cep"]
        assert [h["radial"] for h in r["history"]] == \
               [h["radial"] for h in base["history"]]


def test_layer2_until_destroyed_zero_noise_bitforbit():
    def fire(**kw):
        return run_engagement_until_destroyed(
            "W", RANGE, BEARING, TOLD, true_conditions(),
            lethal_radius_m=8.0, max_rounds=20, **kw)
    base = fire()
    z = fire(observation_noise_m=0.0, noise_seed=999)
    assert z["destroyed"] == base["destroyed"]
    assert z["rounds_fired"] == base["rounds_fired"]
    assert z["final_miss"] == base["final_miss"]
    assert [h["radial"] for h in z["history"]] == [h["radial"] for h in base["history"]]


def test_layer2_seed_is_reproducible():
    a = engage(20.0, seed=42)
    b = engage(20.0, seed=42)
    c = engage(20.0, seed=43)
    assert a["final_cep"] == b["final_cep"]
    assert np.array_equal(np.asarray(a["total_correction_est"]),
                          np.asarray(b["total_correction_est"]))
    assert a["final_cep"] != c["final_cep"]


# ---------------------------------------------------------------------------
# Layer 3 — Known-statistics relationship
# ---------------------------------------------------------------------------
def test_layer3_more_noise_widens_the_floor():
    assert mean_over_seeds(40.0, total_err) > mean_over_seeds(3.0, total_err)
    assert mean_over_seeds(40.0, final_cep) > mean_over_seeds(3.0, final_cep)


def test_layer3_floor_scales_monotonically_with_noise():
    errs = [mean_over_seeds(nz, total_err) for nz in (3.0, 15.0, 40.0)]
    assert errs[0] < errs[1] < errs[2], errs


def test_layer3_averaging_beats_a_single_noisy_reading():
    # The refiner AVERAGES: over many effect rounds the noise partially cancels, so
    # the converged total estimate is better than any single noisy reading and
    # tightens as more observations accrue. (Disable early stop so every round
    # fires; one effect round vs twelve.)
    noise, n = 15.0, 6
    one = float(np.mean([total_err(engage(noise, s, n_register=1, n_adjust=0, n_ffe=1,
                                          converge_threshold_m=0.0)) for s in range(n)]))
    many = float(np.mean([total_err(engage(noise, s, n_register=1, n_adjust=0, n_ffe=12,
                                           converge_threshold_m=0.0)) for s in range(n)]))
    ref = single_reading_error(noise)
    assert many < one                       # more observations -> tighter estimate
    assert many < 0.75 * ref                # clearly better than a single reading


# ---------------------------------------------------------------------------
# Layer 4 — Adaptability / robustness
# ---------------------------------------------------------------------------
def test_layer4_parameter_works_across_the_range():
    errs = [mean_over_seeds(nz, total_err) for nz in (3.0, 15.0, 40.0)]
    assert all(errs[i] < errs[i + 1] for i in range(len(errs) - 1)), errs
    assert all(np.isfinite(e) for e in errs)
    assert errs[0] < 12.0                     # a precise sensor learns the total tightly


def test_layer4_extreme_noise_still_bounded():
    for seed in range(5):
        res = engage(120.0, seed=seed)
        assert np.all(np.isfinite(res["total_correction_est"]))
        assert total_err(res) < 300.0           # bounded near the truth's scale
        assert np.isfinite(res["final_cep"])


def test_layer4_extreme_noise_stops_short_of_a_tight_lethal_radius():
    # With heavy observation noise the closed loop can't reliably place a round
    # inside a very tight lethal radius — it converges less tightly (or hits the
    # safety cap) rather than diverging, and never fires forever.
    def kills(noise, lethal, n=3):
        k = 0
        for s in range(n):
            r = run_engagement_until_destroyed(
                "W", RANGE, BEARING, TOLD, true_conditions(),
                lethal_radius_m=lethal, max_rounds=20,
                observation_noise_m=noise, noise_seed=s)
            assert r["rounds_fired"] <= 20            # never fires forever
            k += r["destroyed"]
        return k
    easy = kills(5.0, 30.0)
    hard = kills(60.0, 3.0)
    assert easy >= hard                                # graceful degradation


if __name__ == "__main__":
    print("\n=== Component 13 — noisy observation of fall of shot ===\n")
    base = run_engagement("W", RANGE, BEARING, TOLD, true_conditions())
    z = engage(0.0, seed=7)
    print("Bit-for-bit at zero noise:",
          z["final_cep"] == base["final_cep"])
    r = engage(15.0, seed=7)
    print(f"\n15 m noise (seed 7): learned TOTAL {np.round(r['total_correction_est'],1)} "
          f"(true {np.round(r['total_correction_true'],1)}, err {total_err(r):.1f} m)")
    print(f"    opening miss {r['phase_misses']['REGISTRATION'][0]:.0f} m -> "
          f"FFE CEP {r['final_cep']:.1f} m")
    print("\nRun 'python -m pytest tests/test_observation_noise.py -v'.")
