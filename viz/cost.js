// The cost model of bike_network/network.py, in the browser, so the viewer's
// knobs can be moved live. tests/test_viz.py runs this file under node against
// the Python build to keep the two in step; change both together.
(function (root) {
  "use strict";

  const BIT_STEP = 1 << 4;
  const BIT_VEHICLE_ONLY = 1 << 5;
  const BIT_HIGHWAY = 1 << 6;
  const SEPARATED = 3;
  const PAINTED = 2;
  const SHARED = 1;
  const FACILITY = ["none", "shared", "painted", "separated"];
  const TRAFFIC = ["", "two-way", "one-way", "one-way"];

  function trafDir(p) {
    return p & 3; // 0 no cars, 1 two-way, 2 with geometry (W), 3 against (A)
  }

  function trafficPenalty(p, lanes, speed, k) {
    const traf = trafDir(p);
    if (traf === 0) return 0;
    const perDir = lanes > 0 ? Math.ceil(lanes / (traf === 1 ? 2 : 1)) : k.default_lanes;
    const extra = Math.min(Math.max(perDir - 1, 0), k.max_extra_lanes);
    const mph = speed > 0 ? speed : k.default_speed;
    const over = Math.max((mph - k.default_speed) / 5, 0);
    const truck = [0, k.truck_limited, k.truck_local, k.truck_through][(p >> 2) & 3];
    return k.traffic_base + k.per_extra_lane * extra + k.per_5_mph * over + truck;
  }

  function exposure(fac, k) {
    if (fac === SEPARATED) return 0;
    if (fac === PAINTED) return k.exposure_painted;
    if (fac === SHARED) return k.exposure_shared;
    return 1;
  }

  // One direction of one segment: dir 0 runs along the stored geometry (LION
  // FROM -> TO), dir 1 against it. Returns null when nothing may use it.
  function edge(p, lanes, speed, dir, k) {
    const fac = dir === 0 ? (p >> 7) & 3 : (p >> 9) & 3;
    const traf = trafDir(p);
    const step = (p & BIT_STEP) !== 0;
    const vehicleOnly = (p & BIT_VEHICLE_ONLY) !== 0;
    const highway = (p & BIT_HIGHWAY) !== 0;
    const noRide = step || vehicleOnly || highway;
    const canWalk = !vehicleOnly && !highway;
    const walkFactor = step ? k.step_factor : k.walk_factor;
    const withTraffic = traf === 1 || traf === (dir === 0 ? 2 : 3);
    const canRide = (withTraffic && !noRide) || fac !== 0;
    if (!canRide && !canWalk) return null;
    const penalty = trafficPenalty(p, lanes, speed, k);
    const rideFactor = 1 + exposure(fac, k) * penalty;
    const ride = canRide && (!canWalk || rideFactor <= walkFactor);
    return {
      ride: ride,
      canRide: canRide,
      facility: fac,
      penalty: penalty,
      rideFactor: rideFactor,
      walkFactor: walkFactor,
      factor: ride ? rideFactor : walkFactor,
    };
  }

  const api = { edge, trafficPenalty, FACILITY, TRAFFIC };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BikeCost = api;
})(this);
