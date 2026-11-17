"""
Component 6 — Full engagement loop + persistence.

REDESIGN (shot-correction). The old loop treated the first two shots as
registration and ran a forgetting-RLS estimator trying to recover FOUR parameters
(gun bias dr/cr, wind error dr/cr) from fall of shot. That split is
mathematically under-determined from a single impact (which is why the wind term
stayed stuck at 0.0 and shot 2 sometimes regressed). It is replaced by a
two-shot-kill doctrine:

  * physics core (Components 1-4.5): the frozen integrate_trajectory, called with
    met= and use_g7=True for every shot — called as a tool, never modified;
  * Shot 1 = REGISTRATION: fire the nominal told-MET solution, observe the FULL
    miss. That miss is the total impact-space offset (gun bias + met error,
    combined and unrecoverable apart from a single impact);
  * Shot 2 = FIRE FOR EFFECT: a one-step Newton aim correction inverts the
    impact-vs-aim map and lands inside the lethal radius (met is stable over the
    engagement, so the offset recurs);
  * Shot 3+ : a forgetting-RLS refiner (_OffsetRefiner), seeded to agree with the
    shot-2 Newton step, only sharpens the total offset from accumulated impacts;
  * the Phase 1 engagement database (weapon_profiles / engagements / rounds),
    extended via ALTER TABLE for the learned correction (schema never rewritten).

The system only ever sees the miss; the hidden true conditions drive the fall of
shot. numpy + stdlib + sqlite3 only.
"""

import numpy as np

from physics.trajectory import integrate_trajectory
from physics.met_message import MetMessage
from physics.horizontal_met import HorizontalMetField
# _ScalarRLS (the proven scalar forgetting-RLS) backs the shot-3+ offset refiner.
from physics.state_estimator import _ScalarRLS

V0 = 827.0  # standard 155mm muzzle velocity (charge)


# ===========================================================================
# Range/cross <-> world rotation about the firing azimuth.
# ===========================================================================
def _frame(bearing_deg):
    phi = np.radians(bearing_deg)
    u_dr = np.array([np.cos(phi), np.sin(phi)])    # downrange unit (world xy)
    u_cr = np.array([-np.sin(phi), np.cos(phi)])   # cross-range unit
    return u_dr, u_cr


def _rc_to_world(vec_rc, bearing_deg):
    u_dr, u_cr = _frame(bearing_deg)
    return vec_rc[0] * u_dr + vec_rc[1] * u_cr


def _world_to_rc(vec_world, bearing_deg):
    u_dr, u_cr = _frame(bearing_deg)
    return np.array([np.dot(vec_world, u_dr), np.dot(vec_world, u_cr)])


# ===========================================================================
# MET with an added uniform wind (the effective atmospheric correction).
# ===========================================================================
def _met_plus_uniform_wind(base_met, add_world_xy):
    """Return a new MET equal to base_met but with a uniform wind vector
    (world xy) added to every zone. Used to bake the estimated/true effective
    wind correction into the MET that the frozen integrator consumes.

    Component 12: a HorizontalMetField is handled by adding the uniform wind to
    BOTH its gun and target profiles and rebuilding the field (same target range
    and confidence floor), so the estimator's effective-wind correction applies
    to a horizontally-varying field exactly as it does to a plain MetMessage.
    The engagement loop is therefore unchanged whether it is given a MetMessage
    or a HorizontalMetField."""
    if getattr(base_met, "is_horizontal", False):
        return HorizontalMetField(
            _met_plus_uniform_wind(base_met.gun_met, add_world_xy),
            _met_plus_uniform_wind(base_met.target_met, add_world_xy),
            base_met.target_range_m,
            confidence_floor=base_met.confidence_floor)
    ax, ay = float(add_world_xy[0]), float(add_world_xy[1])
    lines = []
    for ln in base_met.lines:
        vec = ln["wind_vector"]
        vx, vy = vec[0] + ax, vec[1] + ay
        speed = float(np.hypot(vx, vy))
        # Invert wind_vector_from_dir_speed: vec = speed*(-cos a, sin a).
        wdir = float(np.degrees(np.arctan2(vy, -vx)) % 360.0) if speed > 1e-12 else 0.0
        lines.append({
            "zone_top_m": ln["zone_top_m"],
            "wind_dir_deg": wdir,
            "wind_speed_ms": speed,
            "temp_C": ln["temp_C"],
            "pressure_Pa": ln["pressure_Pa"],
        })
    return MetMessage(lines)


def _impact_rc(met, elevation_deg, bearing_deg):
    """Impact of one shot through the frozen integrator, in the range/cross
    frame relative to the firing azimuth."""
    r = integrate_trajectory(v0=V0, elevation_deg=elevation_deg,
                             azimuth_deg=bearing_deg, met=met, use_g7=True)
    return _world_to_rc(np.array([r["impact_x"], r["impact_y"]]), bearing_deg)


def _solve_elevation(met, target_range, bearing_deg, target_height_m=0.0,
                     dt=0.01, lo=15.0, hi=48.0):
    """Coarse fire-control elevation (ascending branch) whose downrange under
    `met` ~ target_range, by bisection. Only a SEED for _solve_aim's Newton
    refinement, so a metre-level tolerance and few iterations are plenty.
    Height-aware (target_height_m=0.0 reproduces the former ground-impact solve)."""
    def dr(e):
        p = _aim_impact_rc(met, e, bearing_deg, bearing_deg, target_height_m, dt)
        return -1e9 if p is None else p[0] - target_range  # unreachable -> too short
    if dr(lo) > 0:  # target shorter than even the minimum elevation reaches
        return lo
    if dr(hi) < 0:  # target beyond max range at hi; clamp
        return hi
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        fm = dr(mid)
        if abs(fm) < 5.0:               # ~5 m is ample for a Newton seed
            return mid
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# Newton aim-correction core (REDESIGN — replaces the under-determined
# 4-parameter forgetting-RLS that tried to split gun bias from wind error on
# early shots). The fall of shot disagrees with the told-MET fire-control
# prediction by a near-CONSTANT impact-space offset: the gun's static mechanical
# bias PLUS the (stable over the ~2 h engagement window) met error, combined.
# We do NOT try to decompose that offset from a single impact — it is provably
# unrecoverable from one shot. Instead we cancel the TOTAL offset by correcting
# the AIM (elevation, azimuth) with a Newton step on the trajectory map.
# ===========================================================================
def _aim_impact_rc(met, elev, az, bearing, target_height_m=0.0, dt=0.01):
    """Told-model impact of one shot in the range/cross frame, height-aware.
    Returns None if the target altitude is unreachable at this elevation."""
    r = integrate_trajectory(v0=V0, elevation_deg=elev, azimuth_deg=az,
                             met=met, use_g7=True, target_height_m=target_height_m,
                             dt=dt)
    if r.get("reached") is False:
        return None
    return _world_to_rc(np.array([r["impact_x"], r["impact_y"]]), bearing)


def _aim_jacobian(met, elev, az, bearing, target_height_m=0.0, dt=0.01,
                  d_elev=0.05, d_az=0.05, base=None):
    """2x2 finite-difference Jacobian J = d(impact_dr, impact_cr)/d(elev, az)
    (metres per degree), measured from the frozen integrator — there is no
    analytic form. Columns: [d/d_elev, d/d_az]. This is the same empirical-
    sensitivity technique already used for wind (wind_sensitivity), applied to the
    aim instead. `base` (impact at the un-perturbed aim) is reused if supplied to
    save one integration."""
    if base is None:
        base = _aim_impact_rc(met, elev, az, bearing, target_height_m, dt)
    j_e = (_aim_impact_rc(met, elev + d_elev, az, bearing, target_height_m, dt) - base) / d_elev
    j_a = (_aim_impact_rc(met, elev, az + d_az, bearing, target_height_m, dt) - base) / d_az
    return np.column_stack([j_e, j_a])


def _solve_aim(met, goal_rc, bearing, elev0, az0, target_height_m=0.0, dt=0.01,
               iters=6, tol=0.5):
    """Find the aim (elevation, azimuth) whose TOLD-MODEL impact equals goal_rc
    (no real rounds fired).

    THE NEWTON STEP.  With J the aim->impact Jacobian and err = impact-goal, one
    step is  aim <- aim - J^-1 err.  A SINGLE step leaves a large residual when the
    correction is big (the elevation->range map curves), so we ITERATE on the
    fully-available told model until the model impact sits on goal_rc — this costs
    no real shots and removes the curvature error a one-step Newton leaves. The
    Jacobian is measured ONCE (the map is close to linear over the correction, so a
    fixed-J chord iteration converges) which keeps the integration count low.
    np.linalg.pinv (not inv) makes a near-singular J degrade gracefully instead of
    throwing — J loses rank near maximum range, where d range / d elev -> 0."""
    elev, az = float(elev0), float(az0)
    cur = _aim_impact_rc(met, elev, az, bearing, target_height_m, dt)
    if cur is None:                      # back the elevation off until reachable
        for _ in range(20):
            elev = max(1.0, elev - 1.0)
            cur = _aim_impact_rc(met, elev, az, bearing, target_height_m, dt)
            if cur is not None:
                break
    Jinv = np.linalg.pinv(_aim_jacobian(met, elev, az, bearing, target_height_m,
                                        dt, base=cur))
    for _ in range(iters):
        err = cur - goal_rc
        if np.hypot(*err) < tol:
            break
        step = np.clip(Jinv @ err, -5.0, 5.0)   # fixed-J (chord) Newton step
        elev = float(np.clip(elev - step[0], 1.0, 89.0))
        az = float(az - step[1])
        cur = _aim_impact_rc(met, elev, az, bearing, target_height_m, dt)
        if cur is None:                  # stepped past the apex: undo and stop
            elev = float(np.clip(elev + step[0], 1.0, 89.0))
            az = float(az + step[1])
            break
    return elev, az


def true_total_offset(target_range, target_bearing, told_met, true_met, gun_bias,
                      target_height_m=0.0, dt=0.02):
    """The true total impact-space offset (gun bias + met error, combined) at the
    nominal told-MET aim — i.e. realised fall of shot minus target. This is what a
    PRIOR mission on the same weapon under the same (stable) met would have
    learned, so it is the correct quantity to warm-start a remembered weapon with.
    Used by the sim's warm-start toggle; the live loop learns it from fall of shot."""
    target = np.array([float(target_range), 0.0])
    elev0 = _solve_elevation(told_met, target_range, target_bearing,
                             target_height_m, dt=dt)
    elev, az = _solve_aim(told_met, target, target_bearing, elev0,
                          target_bearing, target_height_m, dt=dt)
    gb = np.asarray(gun_bias, dtype=float)
    return (_aim_impact_rc(true_met, elev, az, target_bearing, target_height_m, dt)
            + gb - target)


class _OffsetRefiner:
    """Two-axis forgetting-RLS estimate of the CONSTANT impact-space offset
    (downrange, cross) between the realised fall of shot and the told-MET model
    — gun bias + met error COMBINED, deliberately not decomposed.

    Demoted from the primary corrector to a shot-3+ refiner. The first observed
    offset SEEDS it exactly, so its correction on shot 2 equals the full shot-1
    miss (it agrees with the one-step Newton solve); thereafter it only SHARPENS
    the estimate from accumulated impacts. lam is near 1.0 because the offset is
    static over the engagement — aggressive forgetting causes covariance windup;
    R damps a single lucky/unlucky impact so it can't whip the aim."""

    def __init__(self, lam=0.99, R=400.0, init_P=1.0e4):
        self._dr = _ScalarRLS(0.0, init_P=init_P, lam=lam, R=R)
        self._cr = _ScalarRLS(0.0, init_P=init_P, lam=lam, R=R)
        self.seeded = False

    def seed(self, offset):
        """Set the estimate to the first observed offset with a tightened
        covariance, so shot 2's Newton correction equals the full shot-1 miss."""
        self._dr.x, self._cr.x = float(offset[0]), float(offset[1])
        self._dr.P = self._dr.R
        self._cr.P = self._cr.R
        self.seeded = True

    def update(self, offset):
        self._dr.update(float(offset[0]))
        self._cr.update(float(offset[1]))

    def state(self):
        return np.array([self._dr.get(), self._cr.get()])


# ===========================================================================
# Maximum achievable range — the honest reachability guard (Option A).
# Calls the frozen integrator (does NOT change it) to find how far this
# projectile can actually shoot, so the engagement can refuse an impossible
# target instead of fabricating a "destroyed" result.
# ===========================================================================
def max_achievable_range(weapon_params=None, met=None, target_height_m=0.0,
                         dt=0.1, return_elev=False, **kw):
    """
    Maximum range this projectile can reach, found by sweeping the firing
    elevation through the integrator and taking the largest range. Uses the
    projectile's ACTUAL parameters (v0, mass, area, drag, ...), so it is correct
    for whatever weapon is configured today (155mm M107) or later — nothing is
    hardcoded. For a drag-limited shell the range peaks broadly in the mid-40s of
    quadrant elevation (below the 45 deg vacuum optimum), so a coarse sweep then a
    refinement near the peak pins it down cheaply. A slightly coarse dt keeps this
    guard fast — it only needs the peak range, not a high-fidelity trajectory.

    weapon_params: optional dict of integrate_trajectory kwargs (e.g. {"v0": 900}).
        Extra **kw override it. met / target_height_m are applied to every probe.
    Returns the max range in metres, or (max_range_m, optimal_elev_deg) when
    return_elev is True.
    """
    params = dict(weapon_params) if weapon_params else {}
    params.update(kw)
    params.setdefault("v0", V0)
    params.setdefault("use_g7", True)
    # the sweep controls these — don't let a caller's dict collide with them
    for owned in ("elevation_deg", "met", "target_height_m", "dt"):
        params.pop(owned, None)

    def reach(elev):
        r = integrate_trajectory(elevation_deg=float(elev), met=met,
                                 target_height_m=target_height_m, dt=dt, **params)
        # an unreachable altitude (target above apex) returns range_m None
        return r["range_m"] if r.get("range_m") is not None else -1.0

    def sweep(lo, hi, step):
        best_e, best_r = lo, -1.0
        e = lo
        while e <= hi + 1e-9:
            r = reach(e)
            if r > best_r:
                best_r, best_e = r, e
            e += step
        return best_e, best_r

    e1, _ = sweep(33.0, 60.0, 3.0)            # coarse: the peak sits in the mid-40s
    e2, _ = sweep(e1 - 3.0, e1 + 3.0, 0.5)    # refine around the coarse peak
    e3, r3 = sweep(e2 - 0.5, e2 + 0.5, 0.1)   # fine
    return (float(r3), float(e3)) if return_elev else float(r3)


# ===========================================================================
# Database schema extension (added columns only — never a rewrite).
# ===========================================================================
def _ensure_phase2_columns(db):
    conn = db._conn
    def cols(table):
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    wp = cols("weapon_profiles")
    for c in ("gun_bias_dr", "gun_bias_cr"):
        if c not in wp:
            conn.execute(f"ALTER TABLE weapon_profiles ADD COLUMN {c} REAL DEFAULT 0.0")
    eng = cols("engagements")
    for c in ("atmo_dwind_dr", "atmo_dwind_cr", "final_cep"):
        if c not in eng:
            conn.execute(f"ALTER TABLE engagements ADD COLUMN {c} REAL")
    conn.commit()


def _load_gun_bias(db, weapon_id):
    """Return remembered (dr, cr) gun bias for a weapon, or None if new."""
    row = db._conn.execute(
        "SELECT gun_bias_dr, gun_bias_cr FROM weapon_profiles WHERE weapon_id=?",
        (weapon_id,)).fetchone()
    if row is None:
        return None
    dr = row[0] if not hasattr(row, "keys") else row["gun_bias_dr"]
    cr = row[1] if not hasattr(row, "keys") else row["gun_bias_cr"]
    if dr is None and cr is None:
        return None
    return np.array([dr or 0.0, cr or 0.0])


def _save_gun_bias(db, weapon_id, weapon_type, gun_bias, n_eng, total_rounds):
    conn = db._conn
    now = db._now()
    conn.execute("""
        INSERT INTO weapon_profiles (weapon_id, weapon_type, gun_bias_dr,
            gun_bias_cr, n_engagements, total_rounds_fired, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(weapon_id) DO UPDATE SET
            weapon_type=excluded.weapon_type,
            gun_bias_dr=excluded.gun_bias_dr,
            gun_bias_cr=excluded.gun_bias_cr,
            n_engagements=excluded.n_engagements,
            total_rounds_fired=excluded.total_rounds_fired,
            updated_at=excluded.updated_at
    """, (weapon_id, weapon_type, float(gun_bias[0]), float(gun_bias[1]),
          int(n_eng), int(total_rounds), now, now))
    conn.commit()


# ===========================================================================
# The engagement loop (REDESIGN — one-step Newton, not 4-parameter RLS).
# ===========================================================================
def run_engagement(weapon_id, target_range, target_bearing,
                   met_message, true_conditions, db=None,
                   n_register=2, n_adjust=4, n_ffe=4,
                   weapon_type="155mm-M107", converge_threshold_m=40.0,
                   observation_noise_m=0.0, noise_seed=None,
                   alpha=1.0, offset_refine_lambda=0.99, dt=0.02):
    """
    Runs a complete Phase 2 fire mission against a target.

    DOCTRINE (redesigned — shoot-and-scoot: minimise rounds, then displace):
      * Shot 1  = REGISTRATION: fire the nominal told-MET fire-control solution
        and observe the FULL miss vector (dr, cr). That miss IS the total impact-
        space offset (gun bias + met error, combined). We do NOT try to split it
        into gun vs wind from one impact — that decomposition is unrecoverable.
      * Shot 2+ = FIRE FOR EFFECT: the aim is re-solved so the told model lands on
        (target - offset) — the one-step Newton correction inverting the impact-
        vs-aim map. Met is stable over the ~2 h engagement, so the offset recurs
        and shot 2 lands on target.
      * Shot 3+ : a forgetting-RLS refiner (seeded to agree with the shot-2 Newton
        step) only SHARPENS the offset from accumulated impacts.

    true_conditions: the HIDDEN truth the system is not told:
        {"true_met": MetMessage, "gun_bias": (dr_m, cr_m)}.
      Real fall of shot = integrate(true_met) + gun_bias; the told `met_message`
      is the imperfect MET the fire-control computer uses.

    n_register/n_adjust/n_ffe: kept for backward compatibility. The redesign
      always fires exactly ONE registration shot; (n_adjust + n_ffe) effect rounds
      follow, early-stopped once the miss is within converge_threshold_m (unless
      the threshold is 0). final_cep is the median of the last n_ffe effect misses.

    alpha: Newton damping in [0, 1] (default 1.0); alpha slightly < 1 guards
      against Jacobian error / overshoot.
    offset_refine_lambda: forgetting factor of the shot-3+ refiner (~1.0; the
      offset is static, so aggressive forgetting only causes covariance windup).
    observation_noise_m / noise_seed: Component 13 — the estimator learns from the
      noisy REPORTED fall of shot; the true physics is untouched. 0.0 reproduces
      perfect observation bit-for-bit.
    """
    true_met = true_conditions["true_met"]
    true_gun = np.asarray(true_conditions["gun_bias"], dtype=float)
    told_met = met_message

    obs_noise = float(observation_noise_m)
    if obs_noise < 0.0:
        raise ValueError("observation_noise_m must be >= 0")
    obs_var = obs_noise ** 2
    _rng = np.random.default_rng(noise_seed) if obs_noise > 0.0 else None

    def observe(true_vec):
        """Noisy sensor report of a true 2-vector (m). noise=0 -> unchanged."""
        v = np.asarray(true_vec, dtype=float)
        return v if _rng is None else v + _rng.normal(0.0, obs_noise, size=2)

    def told_impact(elev, az):
        return _aim_impact_rc(told_met, elev, az, target_bearing, dt=dt)

    def realized(elev, az):
        """Hidden TRUE fall of shot in the range/cross frame."""
        return _aim_impact_rc(true_met, elev, az, target_bearing, dt=dt) + true_gun

    target = np.array([float(target_range), 0.0])

    # Nominal fire-control aim: told model lands on the target in BOTH axes.
    elev0 = _solve_elevation(told_met, target_range, target_bearing, dt=dt)
    elev_nom, az_nom = _solve_aim(told_met, target, target_bearing, elev0,
                                  target_bearing, dt=dt)

    # The demoted RLS refiner; R reflects the model floor plus observation noise.
    refiner = _OffsetRefiner(lam=offset_refine_lambda, R=400.0 + obs_var)

    # Warm start: a remembered weapon seeds the offset so shot 1 is pre-corrected.
    warm = None
    if db is not None:
        _ensure_phase2_columns(db)
        warm = _load_gun_bias(db, weapon_id)   # legacy column name; holds TOTAL offset
    warm_started = warm is not None
    if warm_started:
        refiner.seed(warm)

    history = []
    round_no = 0
    rounds_to_converge = [None]   # boxed so the closure can set it

    def fire(phase):
        """Fire one round at the current offset estimate, log it, learn from the
        noisy report, and track first convergence. Returns the true radial."""
        nonlocal round_no
        if phase == "REGISTRATION" and not warm_started:
            elev, az = elev_nom, az_nom            # cold opener: pure nominal aim
        else:
            goal = target - alpha * refiner.state()
            elev, az = _solve_aim(told_met, goal, target_bearing, elev_nom,
                                  az_nom, dt=dt)
        true_rc = realized(elev, az)               # one integration, reused below
        miss = true_rc - target
        radial = float(np.hypot(*miss))
        round_no += 1
        history.append({"phase": phase, "round": round_no,
                        "miss": miss.copy(), "radial": radial})
        offset_obs = observe(true_rc) - told_impact(elev, az)
        if refiner.seeded:
            refiner.update(offset_obs)
        else:
            refiner.seed(offset_obs)
        if (rounds_to_converge[0] is None and converge_threshold_m > 0.0
                and radial < converge_threshold_m):
            rounds_to_converge[0] = round_no
        return radial

    # --- Shot 1: REGISTRATION (warm: already pre-corrected) -------------------
    fire("REGISTRATION")

    # --- Shots 2+: FIRE FOR EFFECT (Newton correction + refiner) -------------
    n_effect = max(1, int(n_adjust) + int(n_ffe))
    ffe_radials = []
    for _ in range(n_effect):
        ffe_radials.append(fire("FFE"))
        # Once converged, fire just enough effect rounds to form a CEP group.
        if (converge_threshold_m > 0.0 and ffe_radials[-1] < converge_threshold_m
                and len(ffe_radials) >= max(1, int(n_ffe))):
            break
    if rounds_to_converge[0] is None:
        rounds_to_converge[0] = round_no

    grp = ffe_radials[-max(1, int(n_ffe)):] if ffe_radials else []
    final_cep = float(np.median(grp)) if grp else None

    # The honest learned quantity is the TOTAL correction; we do not split it into
    # gun bias vs wind error from these impacts (provably unrecoverable per shot).
    total_correction = refiner.state()
    # The true total offset (for the learned-vs-truth panel): the realised fall of
    # shot at the nominal aim minus the target — i.e. the full error to be cancelled.
    total_correction_true = realized(elev_nom, az_nom) - target

    result = {
        "weapon_id": weapon_id,
        "target_range": target_range,
        "target_bearing": target_bearing,
        "elevation_deg": elev_nom,
        "warm_started": warm_started,
        "history": history,
        "phase_misses": {p: [h["radial"] for h in history if h["phase"] == p]
                         for p in ("REGISTRATION", "ADJUSTMENT", "FFE")},
        "final_cep": final_cep,
        "total_correction_est": total_correction,
        "total_correction_true": total_correction_true,
        # Back-compat keys: gun_bias_est now carries the TOTAL offset; the gun/wind
        # split is no longer claimed, so atmo_correction_est is reported as zero.
        "gun_bias_est": total_correction,
        "atmo_correction_est": np.array([0.0, 0.0]),
        "rounds_to_converge": rounds_to_converge[0],
        "total_shots": round_no,
    }

    # --- PERSIST ------------------------------------------------------------
    if db is not None:
        prof = db.get_weapon_profile(weapon_id)
        n_eng = (prof["n_engagements"] if prof else 0) + 1
        total_rounds = (prof["total_rounds_fired"] if prof else 0) + round_no
        _save_gun_bias(db, weapon_id, weapon_type, total_correction, n_eng, total_rounds)

        reg0 = result["phase_misses"]["REGISTRATION"][0] if result["phase_misses"]["REGISTRATION"] else None
        eid = db.record_engagement(
            weapon_id=weapon_id, target_range=target_range,
            target_bearing=target_bearing, target_height=0.0,
            uncorrected_cep=reg0, corrected_cep=final_cep,
            improvement_pct=(100.0 * (1 - final_cep / reg0) if reg0 and final_cep is not None else None),
            rounds_to_converge=rounds_to_converge[0], total_shots=round_no,
            warm_started=1 if warm_started else 0,
            delta_pitch=None, delta_yaw=None, delta_v0=None)
        db._conn.execute(
            "UPDATE engagements SET atmo_dwind_dr=?, atmo_dwind_cr=?, final_cep=? "
            "WHERE engagement_id=?",
            (float(total_correction[0]), float(total_correction[1]),
             final_cep, int(eid)))
        db._conn.commit()
        for h in history:
            db.record_round(eid, h["round"], h["phase"], elev_nom,
                            target_bearing, V0, None, h["radial"], None, None, None)
        result["engagement_id"] = eid

    return result


# ===========================================================================
# Component 9 — fire until destroyed, then stop (adaptable lethal radius).
# Two-shot-kill doctrine: shot 1 registers, shot 2 fires for effect and lands
# inside the lethal radius; the mission stops the instant a round is lethal.
# Reuses the shared Newton aim core and the frozen integrator as tools.
# ===========================================================================
def run_engagement_until_destroyed(weapon_id, target_range, target_bearing,
                                   met_message, true_conditions, db=None,
                                   lethal_radius_m=8.0, max_rounds=15,
                                   min_learning_rounds=1,
                                   weapon_type="155mm-M107",
                                   target_height_m=0.0,
                                   observation_noise_m=0.0, noise_seed=None,
                                   alpha=1.0, offset_refine_lambda=0.99, dt=0.02):
    """
    Fires, learns from each miss, and STOPS the instant a round lands within
    lethal_radius_m of the target — modelling shoot-and-scoot (minimise rounds,
    then displace). Shot 1 registers (full miss observed); shot 2 is the Newton
    kill shot; shots 3+ (only if shot 2 missed lethal) are refiner rounds.

    lethal_radius_m: kill distance (adaptable; default 8.0 for 155 mm).
    max_rounds: safety cap so the mission can't fire forever.
    min_learning_rounds: a round may be declared a kill only from this round on
      (default 1; the first cold registration shot rarely lands lethal anyway).
    target_height_m: passed through to the integrator (Component 8).
    alpha: Newton damping (default 1.0). offset_refine_lambda: refiner forgetting.
    observation_noise_m / noise_seed: Component 13 — estimator learns from the
      noisy REPORTED miss; the KILL assessment uses the TRUE fall of shot. 0.0
      reproduces perfect observation bit-for-bit.

    A remembered weapon (DB) seeds the total offset so shot 1 is pre-corrected and
    can be lethal on the first round; an unknown weapon fires one registration
    shot first. Either way the loop ends the instant a round falls within lethal.
    """
    if lethal_radius_m <= 0.0:
        raise ValueError("lethal_radius_m must be positive")

    true_met = true_conditions["true_met"]
    true_gun = np.asarray(true_conditions["gun_bias"], dtype=float)
    told_met = met_message

    obs_noise = float(observation_noise_m)
    if obs_noise < 0.0:
        raise ValueError("observation_noise_m must be >= 0")
    obs_var = obs_noise ** 2
    _rng = np.random.default_rng(noise_seed) if obs_noise > 0.0 else None

    def observe(true_vec):
        v = np.asarray(true_vec, dtype=float)
        return v if _rng is None else v + _rng.normal(0.0, obs_noise, size=2)

    # --- Honest reachability guard (Option A): refuse an impossible target
    # instead of fabricating a "destroyed" result. ---------------------------
    max_range_m = max_achievable_range({"v0": V0}, met=told_met,
                                       target_height_m=target_height_m)
    if target_range > max_range_m:
        return {
            "reachable": False,
            "reason": (f"Target range {target_range:.0f} m exceeds weapon "
                       f"maximum range {max_range_m:.0f} m"),
            "max_range_m": float(max_range_m),
            "target_range_m": float(target_range),
            "weapon_id": weapon_id,
            "destroyed": False,
            "rounds_fired": 0,
            "destroying_round": None,
            "final_miss": None,
            "lethal_radius_m": float(lethal_radius_m),
            "max_rounds": max_rounds,
            "warm_started": False,
            "history": [],
        }

    target = np.array([float(target_range), 0.0])

    def told_impact(elev, az):
        return _aim_impact_rc(told_met, elev, az, target_bearing, target_height_m, dt)

    def realized(elev, az):
        ti = _aim_impact_rc(true_met, elev, az, target_bearing, target_height_m, dt)
        return None if ti is None else ti + true_gun

    elev0 = _solve_elevation(told_met, target_range, target_bearing,
                             target_height_m, dt=dt)
    elev_nom, az_nom = _solve_aim(told_met, target, target_bearing,
                                  elev0, target_bearing, target_height_m, dt=dt)

    refiner = _OffsetRefiner(lam=offset_refine_lambda, R=400.0 + obs_var)

    warm = None
    if db is not None:
        _ensure_phase2_columns(db)
        warm = _load_gun_bias(db, weapon_id)   # legacy column name; holds TOTAL offset
    warm_started = warm is not None
    if warm_started:
        refiner.seed(warm)

    if realized(elev_nom, az_nom) is None:
        raise ValueError("target altitude unreachable under true conditions")

    history = []
    round_no = 0
    destroyed = False
    destroying_round = None
    final_miss = None

    while round_no < max_rounds and not destroyed:
        this_round = round_no + 1
        cold_reg = (this_round == 1 and not warm_started)
        if cold_reg:
            elev, az = elev_nom, az_nom                    # cold opener: nominal aim
            phase = "REGISTRATION"
        else:
            goal = target - alpha * refiner.state()        # Newton correction
            elev, az = _solve_aim(told_met, goal, target_bearing,
                                  elev_nom, az_nom, target_height_m, dt=dt)
            phase = "FIRE_FOR_EFFECT"
        true_rc = realized(elev, az)                        # one integration, reused
        miss = true_rc - target                            # TRUE fall of shot
        radial = float(np.hypot(*miss))

        # Kill assessment on the TRUE fall of shot. Stop the instant it's lethal.
        if this_round >= min_learning_rounds and radial <= lethal_radius_m:
            round_no += 1
            history.append({"round": round_no, "phase": phase,
                            "miss": [float(miss[0]), float(miss[1])],
                            "radial": radial, "destroyed_target": True})
            destroyed = True
            destroying_round = round_no
            final_miss = radial
            break

        round_no += 1
        history.append({"round": round_no, "phase": phase,
                        "miss": [float(miss[0]), float(miss[1])],
                        "radial": radial, "destroyed_target": False})
        # Learn the constant offset from the NOISY reported fall of shot.
        offset_obs = observe(true_rc) - told_impact(elev, az)
        if refiner.seeded:
            refiner.update(offset_obs)
        else:
            refiner.seed(offset_obs)

    if final_miss is None and history:
        final_miss = history[-1]["radial"]

    total_correction = refiner.state()
    # True total offset (gun bias + met error) at the nominal aim, for the panel.
    total_correction_true = realized(elev_nom, az_nom) - target

    # Persist the learned TOTAL offset so the next mission warm-starts.
    if db is not None:
        prof = db.get_weapon_profile(weapon_id)
        n_eng = (prof["n_engagements"] if prof else 0) + 1
        total = (prof["total_rounds_fired"] if prof else 0) + round_no
        _save_gun_bias(db, weapon_id, weapon_type, total_correction, n_eng, total)

    return {
        "weapon_id": weapon_id,
        "destroyed": destroyed,
        "rounds_fired": round_no,
        "destroying_round": destroying_round,
        "final_miss": final_miss,
        "lethal_radius_m": float(lethal_radius_m),
        "max_rounds": max_rounds,
        "min_learning_rounds": min_learning_rounds,
        "warm_started": warm_started,
        "total_correction_est": total_correction,
        "total_correction_true": total_correction_true,
        # Back-compat: gun_bias_est now carries the TOTAL offset (gun + met error);
        # the gun/wind split is no longer claimed, so atmo_correction_est is zero.
        "gun_bias_est": total_correction,
        "atmo_correction_est": np.array([0.0, 0.0]),
        "elevation_deg": elev_nom,
        "history": history,
    }
