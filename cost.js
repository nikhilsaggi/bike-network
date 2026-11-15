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

  // room: feet of moving roadway in the rider's direction, null where the
  // width is unknown (network.room_ft). 1 at the reference, less with more room.
  function roomScale(room, k) {
    const held = room == null ? k.room_ref : Math.min(Math.max(room, k.room_min), k.room_max);
    return Math.pow(k.room_ref / held, k.room_exponent);
  }

  // pave: DOT's 1-10 pavement rating, null where the block is unrated.
  function roughness(pave, k) {
    return pave == null ? 0 : k.pave_per_point * Math.max(k.pave_good - pave, 0);
  }

  // network.room_adjusted: what a rider sharing the lane feels. Room never
  // takes more than room_max_discount off the penalty.
  function roomAdjusted(penalty, room, k) {
    return Math.max(penalty * roomScale(room, k), penalty - k.room_max_discount);
  }

  // vol: vehicles/day the rider shares this direction with (oncoming, in a
  // contraflow lane), already resolved per direction by the export. Before
  // room, which edge applies where the rider shares the lane.
  function trafficPenalty(p, speed, vol, k) {
    if (trafDir(p) === 0) return 0;
    const doublings = Math.max(Math.log2(vol / k.volume_ref), 0);
    const mph = speed > 0 ? speed : k.default_speed;
    const over = Math.max((mph - k.default_speed) / 5, 0);
    const truck = [0, k.truck_limited, k.truck_local, k.truck_through][(p >> 2) & 3];
    return k.traffic_base + k.per_doubling * doublings + k.per_5_mph * over + truck;
  }

  function exposure(fac, k) {
    if (fac === SEPARATED) return 0;
    if (fac === PAINTED) return k.exposure_painted;
    if (fac === SHARED) return k.exposure_shared;
    return 1;
  }

  // One direction of one segment: dir 0 runs along the stored geometry (LION
  // FROM -> TO), dir 1 against it; vf / vr are each direction's volume; room
  // and pave as above. Returns null when nothing may use it.
  function edge(p, speed, vf, vr, dir, k, room, pave) {
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
    const volume = dir === 0 ? vf : vr;
    const raw = trafficPenalty(p, speed, volume, k);
    const sharesLane = fac === 0 || fac === SHARED;
    const penalty = sharesLane ? roomAdjusted(raw, room, k) : raw;
    const rough = roughness(pave, k);
    const rideFactor = 1 + exposure(fac, k) * penalty + rough;
    const ride = canRide && (!canWalk || rideFactor <= walkFactor);
    return {
      ride: ride,
      canRide: canRide,
      facility: fac,
      penalty: penalty,
      volume: volume,
      sharesLane: sharesLane,
      roomScale: raw > 0 ? penalty / raw : 1, // what room did to the penalty
      rough: rough,
      rideFactor: rideFactor,
      walkFactor: walkFactor,
      factor: ride ? rideFactor : walkFactor,
    };
  }

  const api = { edge, trafficPenalty, roomScale, roomAdjusted, roughness, FACILITY, TRAFFIC };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BikeCost = api;
})(this);
