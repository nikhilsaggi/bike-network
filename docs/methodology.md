# Methodology: how the network is costed, and why these weights

The network gives every direction of every NYC block a cost: its length
times a multiplier for how stressful it is to ride. This page explains where
each part of that multiplier comes from: what was **measured**, what was
**calibrated** against observed bike counts, and what is **judgment**. It
also explains why the judgment calls were not calibrated.

All inputs are public. Numbers below are from LION 26C, DOT's bike routes as
fetched in November 2026, DOT traffic counts 2010-2026, DOT pavement ratings
2018-2026, Citi Bike trips for 2025 and DOT bike counters for 2025.

## 1. The network

| Source | What it supplies |
|---|---|
| [LION](https://www.nyc.gov/content/planning/pages/resources/datasets/lion) (NYC Planning), 26C | Every street, path and greenway block; traffic direction; travel lanes; posted speed; truck routes; street width; parking lanes |
| [NYC Bike Routes](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib) (DOT) | The bike facility in each direction of each block, joined on LION's segment id |
| [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) (DOT) | Measured vehicle volumes, per direction, on about 1% of streets |
| [Street Pavement Rating](https://data.cityofnewyork.us/Transportation/Street-Pavement-Rating/6yyb-pb25) (DOT) | A 1-10 pavement rating for each street block, matched to LION by geometry |
| [Citi Bike trip data](https://s3.amazonaws.com/tripdata/index.html) | Every 2025 trip between two stations (44.7M) |
| [Bicycle Counts](https://data.cityofnewyork.us/Transportation/Bicycle-Counts/uczf-rk3c) (DOT) | All bikes past 17 permanent counters, used to calibrate |

Each block becomes two directed edges, one per direction of travel. On each
one, a bike either rides it or is walked along the sidewalk, whichever costs
less:

- **Riding** is legal with the flow of traffic, and in any direction DOT
  records a bike facility for. That brings in contraflow lanes and two-way
  greenways.
- **Walking** is legal on any sidewalk in either direction. So riding the
  wrong way down a one-way street is never possible, but walking it is.
- Vehicle-only roads (highways and some bridges) carry neither. Ferries are
  left out, so Staten Island and Governors Island are separate from the rest.

**Bike facilities come from DOT, with LION filling one kind of gap.** DOT's
bike routes keep a lane's history, so only `Current` records count. The two
datasets have drifted apart in segment ids: 312 current DOT records name
blocks LION 26C does not have, and 862 blocks LION marks as bike network
have no current DOT record. That left stretches of greenway and bridge path
walk-only, among them the Williamsburg Bridge's Manhattan end, Randalls
Island, the East River Esplanade and the Willis Avenue Bridge path. So a
separated path (LION bike class 1, or a combination including it) takes its
directions from LION where DOT has **never** recorded the block: 586 blocks,
35 km.

Blocks DOT has retired are left alone, because LION lags retirements. The
Pulaski Bridge's pedestrian path was a shared path until DOT opened the bike
path beside it in 2016, and LION still marks both. Reviving it split the
bridge's riders across two paths and dropped the counted path from 1,366
trips a day to 59. Painted and signed lanes never fall back to LION, since
those do get removed.

## 2. The cost model

```
cost   = length × factor
riding:  factor = 1 + exposure(facility in this direction) × felt_penalty + roughness
walking: factor = WALK_FACTOR (STEP_FACTOR on a step street)

traffic_penalty = 0                         where no car may go
                = TRAFFIC_BASE
                + TRAFFIC_PER_DOUBLING × max(0, log2(volume / VOLUME_REFERENCE))
                + TRAFFIC_PER_5_MPH × max(0, (speed limit − 25) / 5)
                + TRUCK_PENALTY[truck route]

felt_penalty    = traffic_penalty           in a painted or separated lane
                = max(traffic_penalty × ROOM_REFERENCE / room,
                      traffic_penalty − ROOM_MAX_DISCOUNT)
                                            sharing the lane (no facility, sharrows)
room            = (curb-to-curb width − 8 ft × parking lanes)
                  × (½ on a two-way street), held to [ROOM_MIN, ROOM_MAX]

roughness       = PAVEMENT_PER_POINT × max(0, PAVEMENT_GOOD − DOT rating)
```

`volume` is vehicles a day in the rider's direction. In a contraflow lane it
is the oncoming stream. Every factor is at least 1, so no route can lower its
cost by looping, and Dijkstra's algorithm finds the cheapest route exactly.

**Room to be passed.** The same traffic is easier to share with more road in
your direction. Montrose Ave in Williamsburg is one-way with two travel lanes
on 40 ft: 24 ft of moving roadway for the rider's direction. Bushwick Ave
beside it is 42 ft but two-way, so each direction has 13 ft. A rider on
Montrose is passed with room to spare; on Bushwick, cars squeeze past. Room
only matters where the rider shares the lane. A painted lane's stripe is the
rider's room, and the curb-to-curb width counts the lane itself, so crediting
width there would count paint twice. Room is capped both ways. Past 24 ft
(two travel lanes), more width brings more lanes of traffic, not more room
for the rider. And room takes no more than `ROOM_MAX_DISCOUNT` off the
penalty, so a wide arterial is roomier but its traffic is still an
arterial's: Atlantic Ave in Brooklyn has nearly as much room (a median 22 ft
on each roadbed), with twice Montrose's volume and a through truck route. Width is also the volume model's strongest predictor
(section 3), so a wide street already carries a higher volume estimate; an
uncapped discount mostly cancels that out, which the counters reject
(section 5). A block with no recorded width (about 1% of car-street length)
takes no room term. LION leaves the parking count blank on about a quarter
of car blocks; their widths match blocks with two parking lanes, so blank
is read as two.

**Pavement.** DOT inspects every block and rates it 1-10: 8-10 good, 4-7
fair, 1-3 poor. Each block takes its latest rating since 2018, when DOT moved
to its Pavement Condition Index method. Ratings are keyed by street codes,
not LION's segment id, but their lines are LION's own, so each car block
takes the nearest rated line running the same way within 40 ft (150 ft for
a divided street's roadbeds, which DOT rates on the centerline between
them). That rates 88.6% of car-street length; an unrated block adds nothing.
The rating describes the whole roadway, not the curb lane a rider uses, and
is a coarse signal for that.

### Every weight, and how it was set

| Weight | Value | Set by | Why |
|---|---|---|---|
| Exposure, separated facility | 0 | Definition | No car shares a protected lane, greenway or path. The traffic penalty does not apply |
| Exposure, painted lane | 0.3 | Judgment | Paint is a line drivers mostly keep clear of, though double parking pushes riders out of it. The counters cannot settle this one (section 5) |
| Exposure, sharrows or nothing | 1 | Definition | Sharrows and signed routes separate the rider from nothing |
| `TRAFFIC_BASE` | 0.2 | Judgment | Any street with cars is worse than a car-free path. It makes a quiet street with no lane cost 1.2× |
| `TRAFFIC_PER_DOUBLING` | 1.0 | **Calibrated** | Section 5 |
| `VOLUME_REFERENCE` | 1,600 vehicles/day | Measured | The median estimated volume on one-lane streets, so a typical side street adds nothing for volume |
| `TRAFFIC_PER_5_MPH` | 0.15 | Judgment | Speed is a risk of its own. The volume model found posted speed tells nothing about volume (section 3), so it is priced separately |
| `TRUCK_PENALTY` | 0.2 / 0.3 / 0.5 | Judgment | Limited-local / local / through truck routes. A truck is more dangerous to share a lane with than a car, whatever the count |
| `ROOM_REFERENCE_FT` | 12 ft | Measured | The median room on one-lane-per-direction streets (one-way 14, two-way 9), so a typical street is neither discounted nor penalized |
| `ROOM_MIN_FT` / `ROOM_MAX_FT` | 7 / 24 ft | Judgment | Below 7 ft the width is likely a data error; past two travel lanes more width is more traffic, not more room |
| `ROOM_EXPONENT` | 1 | Checked | Penalty in proportion to room. The counters cannot tell 0.5 from 1.5 (section 5) |
| `ROOM_MAX_DISCOUNT` | 1.25 | Judgment | About a quarter of a busy arterial's penalty. Without a cap the counters score worse (section 5) |
| `PAVEMENT_GOOD` | 8 | Definition | DOT's own threshold for good pavement |
| `PAVEMENT_PER_POINT` | 0.05 | Checked | A poor block (3) costs 1.25× a good one. The counters score 0.05 slightly better than none, and 0.10 far worse (section 5) |
| `WALK_FACTOR` | 8 | **Calibrated** | Section 5 |
| `STEP_FACTOR` | 12 | Judgment | Carrying a bike up steps has to cost more than walking it. Step streets are rare enough that the value barely matters |

Only the three numbers with the largest effect on routes were calibrated:
the traffic weight, the painted-lane exposure and the walking factor.
Seventeen counters, of which eleven are informative, cannot pin down sixteen
numbers. The room and pavement weights were checked against the counters (a
weight the counters reject is not used) but not fitted. The others stay
where judgment put them.

### What a factor means

| Example | Factor |
|---|---|
| Protected lane, greenway, path | 1.0 |
| Quiet street, no lane, good pavement | 1.2 |
| The same on poor pavement (rated 3) | 1.45 |
| Painted lane on a quiet street | about 1.06 |
| Typical one-way side street, no lane (median) | 1.38 |
| Typical two-way side street, no lane (median) | 1.53 |
| Two lanes each way, no lane (median) | about 2.6 |
| Montrose Ave / Bushwick Ave, Brooklyn (median) | 2.25 / 2.31 |
| Atlantic Ave, Brooklyn / Northern Blvd, Queens (median) | 3.8 / 3.9 |
| Busiest 1% of streets with no lane | 4.6 and up |
| Walking the bike | 8 |

A factor of 3 means a rider treats a meter of that street like three meters
of protected lane. Routes trade length for comfort at exactly these rates.

## 3. Traffic volume

DOT's counts cover under 1% of car directions: 1,660 counted directions on
1,447 segments since 2010 (median count year 2016). Volume everywhere else
is estimated.

**Method.** Each count's compass direction (NB/SB/EB/WB) is matched to the
segment's own bearing to give a volume per direction of travel. A
gradient-boosted model then predicts log daily volume from what LION records
on every street: width, travel lanes, one-way, truck route, snow-plough
priority, borough, display level, parking lanes and bridge.

**Evaluation.** Five-fold cross-validation with whole streets held out
together, so a block's neighbor never gives away its answer:

| Predictor of log daily volume | R² | Rank correlation | Typical error |
|---|---|---|---|
| Lanes per direction alone | 0.32 | 0.55 | 1.62× |
| A lane-count rule (an earlier version of this model) | 0.39 | 0.67 | 1.52× |
| Volume model | 0.47 | 0.73 | 1.47× |

Street width is the strongest predictor, then lanes per direction, then the
snow-plough priority route. Posted speed carries no information about volume.
Estimated volume roughly doubles with each travel lane per direction (median
1.6k, 6.4k, 14k, 24k for one to four lanes). One-lane streets alone span
about 7× between their quietest and busiest tenth, which is the distinction
a lane-count rule cannot make.

**Limits.** The counted streets skew busy, so quiet streets are extrapolated
more than learned. And LION lags street redesigns: a road diet keeps the
lanes and width it had before, and with them its old traffic estimate.

**DOT shared streets** are the one redesign handled explicitly. Examples are
Broadway at 27th-32nd and 18th-21st St, and Berry St. DOT codes them
`Shared`, the same name it gives sharrows, while LION still describes the
avenue they replaced. A `Shared` facility offered *against* one-way car
traffic (which sharrows never are) marks one, as does a block between two of
them on the same street. They take the quiet reference volume and no truck
term: 60 segments.

## 4. Routing the trips

Each Citi Bike station is snapped to the nearest intersection of the main
network. The median snap is 16 m, and 94 stations are left off it: Governors
Island, Hoboken, one with no coordinates, and a few over 150 m from any
node. Together they hold 0.08% of trip ends. Trips that return to their own
station are loops with no route, and are dropped.

Every trip takes its cheapest route: all-or-nothing assignment, one Dijkstra
per origin station. Bikes do not congest one another, so one pass is exact.
For comparison, every trip is routed again on the **shortest legal route**:
riding at 1× wherever riding is allowed, walking still at its factor. That
baseline asks what the safety weights add over distance alone.

## 5. Calibration against DOT's bike counters

### The question

DOT's counters see every bike; the routes are Citi Bike's alone. So the
model cannot be asked to match the counts. It is asked whether routed
trips are a *steady share* of counted bikes from place to place. If the
weights send riders where riders actually go, the share should be about the
same at the Brooklyn Bridge as on Columbus Avenue.

**The score** is the spread (standard deviation) of `log(routed / counted)`
across counters. Lower is better.

### Three adjustments, each needed for the score to measure route choice

1. **Citi Bike density.** Citi Bike's share of all riding is higher where
   Citi Bike is denser, whatever the weights. The log ratio is regressed on
   the log of Citi Bike trip ends within 1 km of each counter, and the
   residual spread is the score (raw spread is reported alongside).
2. **Informative counters only.** Counters on outer-borough greenways see
   almost no Citi Bike routes (Concrete Plant Park routes zero, 111th St
   about 15 a day), so their ratios are noise. The scored set is fixed once,
   as the counters the *shortest* routing sends 100+ trips a day through.
   Every weight setting is then judged on the same set.
3. **Bridges are crossings.** A bridge counter sees every crossing. In the
   network, riders join a bridge path at several access points along its
   approach, which in reality all lie before the counter. So a bridge
   counter is compared with the busiest block of the bridge's bike facility.

Counters are tied to the LION street they count, by name, not by nearest
line. DOT gives Columbus Ave at 86th St the same coordinates as Amsterdam
Ave at 86th St, and the Brooklyn Bridge counter sits nearer a car roadbed
than the bike lane it counts.

### The counters

Seventeen counters with 2025 data, one per site, bikes only where DOT splits
bikes from pedestrians. Eleven are scored: the Williamsburg, Queensboro,
Manhattan, Brooklyn, Pulaski and Willis Avenue bridges, Kent Ave, 8th Ave at
50th St, Amsterdam and Columbus Aves at 86th St, and Prospect Park West. Six
are not: High Bridge, Ocean Parkway, Emmons Ave (Shore Parkway Greenway),
Concrete Plant Park, 111th St and Pelham Parkway.

### The grid

116 settings: traffic weight 0-1.0, painted exposure 0-1.0, walking factor
2.5-12, with no room or pavement term. Each one re-routes all 44.7M trips.
The room and pavement terms were then checked around the chosen setting
(below).

| | Score (lower is better) | Raw spread | Rank correlation |
|---|---|---|---|
| Shortest legal route (no safety weights) | 0.405 | 0.620 | 0.86 |
| First settings (traffic 0.3, paint 0.3, walk 4) | 0.316 | 0.475 | 0.82 |
| Traffic 1.0, paint 0.3, walk 8, no room or pavement | 0.271 | 0.452 | 0.86 |
| Best in the grid (traffic 0.75, paint 0.75, walk 12) | 0.267 | 0.469 | 0.85 |
| **Chosen: traffic 1.0, paint 0.3, walk 8, room capped at 1.25, pavement 0.05** | **0.263** | 0.438 | 0.86 |
| Best checked (traffic 0.75, paint 0.3, walk 8, pavement 0.05, no room) | 0.256 | 0.442 | 0.86 |

Every safety-weighted setting in the grid beats distance alone (the worst
scores 0.346), and the chosen one by 35%, within 3% of the best. Rank correlation hardly separates them: the busy bridges
are busy under any routing. It is the size of each counter's share, the
score, that the weights move.

### What the counters decide, and what they don't

**Traffic weight: decided.** The best score at each traffic weight (walk ≤ 6):

| Per doubling | 0 | 0.15 | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| Best score | 0.308 | 0.293 | 0.281 | 0.291 | 0.283 | 0.280 |

A weight below 0.3 is worse; from 0.3 up the surface is flat within noise.
At the higher walking factors the counters prefer the top of that range: at
paint 0.3 and walk 8, a weight of 1.0 scores 0.271 against 0.305 for 0.75.
Riders avoid traffic at least as strongly as 0.3 per doubling says, and
probably more.

**Walking factor: decided.** The best score at each walking factor falls
from 2.5 (0.292) to 4 (0.284) to 6 (0.280) to 8 (0.271), then levels off
(12: 0.267). At the chosen traffic and paint settings: 0.284, 0.280, 0.271
and 0.269 for 4, 6, 8 and 12. Twelve gains 0.002 over 8, which is noise, so
8 is kept. Riders treat walking a bike as far worse than its speed
alone (about 4× slower) suggests. With strong traffic weights, a low walking
factor sends routes onto sidewalks to dodge busy blocks, which the counts do
not show.

**Painted exposure: not decided.** The best score at each value:

| Paint keeps | 0% | 15% | 30% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| Best score | 0.310 | 0.293 | 0.280 | 0.283 | 0.283 | 0.281 |

Paint values of 30% and above are within 1% of each other. Below that,
treating paint as nearly as good as a protected lane, is worse. The surface
runs along a ridge where a stronger traffic weight with more paint relief
fits as well as a weaker one with less, and no counter sits on a painted
lane, so the data barely sees paint at all. 30% is kept as the judgment it
was. It is also the best-scoring value at walking factors up to 6.

**Room and pavement: checked, not fitted.** Both were scored at paint 0.3
and walk 8. Pavement first, with no room term:

| Per point below good | 0 | 0.05 | 0.10 |
|---|---|---|---|
| Traffic 0.75 | 0.305 | 0.256 | 0.757 |
| Traffic 1.0 | 0.271 | 0.269 | 0.756 |

0.05 scores a little better than none. At 0.10 the fit collapses: routes
leave the Willis Avenue Bridge (338 routed trips a day fall to 20) for the
better-paved streets around it. So 0.05 is kept, as a small term the
counters do not reject.

Then several ways of counting room, all with pavement at 0.05:

| Room term | Traffic 0.75 | 1.0 | 1.25 |
|---|---|---|---|
| None | 0.256 | 0.269 | 0.271 |
| Penalty × 12 ft / room, uncapped, every facility | 0.287 | 0.289 | 0.310 |
| The same, but room takes at most 1.25 off | 0.286 | 0.280 | 0.262 |
| **Capped, and painted lanes left out** | | **0.263** | 0.263 |
| Only narrow streets penalized, no discount | 0.258 | 0.275 | 0.274 |
| Volume per travel lane instead of width | 0.291 | 0.279 | 0.282 |

An uncapped discount makes every busy wide street cheaper and the fit
worse. Capping it and leaving paint out recovers the fit at the calibrated
traffic weight, within noise of no room term at all: eleven counters
cannot say whether room matters. It is kept because it does what riders
describe and the counters do not object. Without it, Montrose Ave (one-way,
24 ft) costs 3.49× its length against Bushwick Ave's 2.43× (two-way, 13 ft a
direction), the reverse of how they ride; with it, 2.25× against 2.31×.
Volume per lane would credit Bushwick (one lane each way) over Montrose
(two lanes one way), and it rewards more lanes, which the lane count already
prices through volume. The exponent was checked under the uncapped rule (0.5,
1 and 1.5 within noise of each other) and left at 1.

### The chosen setting, counter by counter

Routed and counted bikes a day, and each counter's residual after the
density adjustment (0 is a perfect fit).

| Counter | Counted | Routed, shortest | Routed, no room or pavement | Routed, chosen | Residual, shortest | Residual, no room or pavement | Residual, chosen |
|---|---|---|---|---|---|---|---|
| Williamsburg Bridge | 6,850 | 3,536 | 3,682 | 3,683 | +0.48 | +0.16 | +0.23 |
| Queensboro Bridge | 5,722 | 1,438 | 1,530 | 1,486 | +0.46 | −0.01 | +0.02 |
| Manhattan Bridge | 4,374 | 1,822 | 2,249 | 2,281 | −0.03 | −0.11 | −0.02 |
| Brooklyn Bridge | 3,817 | 1,087 | 2,203 | 2,160 | −0.25 | +0.12 | +0.18 |
| Kent Ave | 3,438 | 1,216 | 2,155 | 1,666 | +0.01 | +0.24 | +0.06 |
| Pulaski Bridge | 3,247 | 1,281 | 1,366 | 1,355 | +0.68 | +0.27 | +0.32 |
| Willis Ave Bridge | 2,579 | 160 | 344 | 326 | −0.35 | −0.25 | −0.26 |
| 8th Ave at 50th St | 2,946 | 1,278 | 1,145 | 1,017 | −0.10 | −0.48 | −0.51 |
| Amsterdam Ave at 86th St | 2,582 | 364 | 747 | 731 | −0.53 | −0.24 | −0.20 |
| Columbus Ave at 86th St | 1,959 | 344 | 1,027 | 857 | −0.31 | +0.35 | +0.24 |
| Prospect Park West | 2,555 | 579 | 903 | 839 | −0.07 | −0.06 | −0.06 |

"No room or pavement" is traffic 1.0, paint 0.3, walk 8 without those two
terms, the setting chosen before they were added.

The weights mostly fix the bridges: the shortest routing over-sends riders to
Williamsburg, Queensboro and Pulaski, and under-sends them to the Brooklyn
Bridge's protected path. Room and pavement mostly move Kent Ave, whose
over-prediction falls from +0.24 to +0.06, and Columbus Ave. The worst
remaining misses are 8th Ave in Midtown (under-predicted) and the Upper West
Side pair, one-way in opposite directions: Columbus (southbound) is
over-predicted and Amsterdam (northbound) under-predicted. Those two share
one density adjustment, so a directional imbalance in the Citi Bike trips
there would show up the same way.

## 6. What the weights are, and are not

- They describe **route choice**: what riders do, so the routes can say which
  lanes get used and which would be. They are not a crash-risk measure. That
  would need crash data with exposure, which is a different calibration.
- They are fitted to **Citi Bike riders**. People on their own bikes,
  delivery riders and Staten Island are not in the demand.
- **Eleven counters** decide the traffic and walking weights. Rankings built
  on these weights should be checked across the near-best settings (the
  twelve within 10% of the best), not read off one setting.
- There are **no turn or intersection costs**. Risk concentrates at
  intersections, and a per-block cost cannot see a left turn across an
  arterial.

## 7. Where new lanes would help

### One block at a time

`candidates` asks, for each directed block without a protected facility,
what giving it one would save. A trip from `o` to `d` saves
`max(0, C_od − (D_o[u] + c' + R_d[v]))`, where `C_od` is its cost today,
`D_o` the cheapest costs from `o`, `R_d` the cheapest costs to `d`, and `c'`
the block's protected cost (its length). That is exact for one block, and it
counts trips that avoid the block today.

It misses corridors. (The 5th Ave figures here and under the next two
headings illustrate the method, and are from the cost model before room
and pavement were added; 5th Ave's factor has since fallen from about 5.0
to about 3.3, so its numbers would be smaller now.) Midtown 5th Ave has no
lane from 59th St down to 24th St (about 22,000 vehicles a day), and a protected lane from 23rd
St south. Today's cheapest route from 5th Ave at 49th St to Washington
Square goes west to 6th Ave, up to 51st St, down 7th Ave and Broadway, and
back to 5th Ave at 25th St. Protect one block of 5th Ave and the rider still
faces the blocks either side of it, so the detour still wins.

### Stretches

A **stretch** is consecutive blocks of one street in one direction. Each
street is chained into runs, taking the straightest continuation at every
node of the same name, and ending at a turn sharper than 60° or a U-turn.
Both directions are runs, so the wrong way down a one-way street is a
contraflow corridor. The exception is a divided street, which LION draws as
two one-way roadbeds (Park Ave, Queens Blvd): the other roadbed carries that
direction, so a lane for it goes there, never against this one's traffic.
The single-block list follows the same rule. Protect a stretch and a trip can join it at any node
`i` and leave at any later node `j`:

```
via = D_o[n_i] + (P_j − P_i) + R_d[n_j]
saving = max(0, C_od − min over i < j of via)
```

`P` is the running length along the stretch. A single sweep finds the
minimum, keeping the best `D_o[n_i] − P_i` seen so far. Every block costs at
least its length and a protected one exactly that, so leaving a straight
stretch and rejoining it never pays, and the score is exact. On a curving
street it is a lower bound. The test suite checks it against re-routing
every trip, for every stretch of a synthetic grid.

5th Ave southbound, 48th St to 23rd St (2.0 km), on the 2025 trips:

| | Cost saved, cost-km a year | Trips a day using it |
|---|---|---|
| As one stretch | 741,000 | 5,000 |
| Built, and every sampled trip re-routed | 741,500 | 5,000 |
| Its 31 blocks one at a time, summed | 200,000 | |
| Its best single block | 37,000 | |

Most of the value comes from linking up with the existing lane at 23rd St.
From 48th St down to 42nd St saves 37,000 (355 trips a day), to 34th St
282,000, to 26th St 566,000, and to 23rd St 741,000.

### How far a corridor runs

A longer stretch never saves less, so its extent needs a price: extend
while each new meter saves at least that much. Blocks already protected
cost nothing, so a stretch may join gaps across them. On 5th Ave
southbound the south end is 23rd St at almost any price, because that is
where the lane starts. The north end moves with the price:

| Price (cost-m a year saved per new meter) | Best stretch of 5th Ave | New km | Trips a day |
|---|---|---|---|
| 383,000 | 25th St to 23rd St, the Madison Square block | 0.2 | 1,900 |
| 316,000 | 55th St to 23rd St | 2.6 | 6,300 |
| 261,000 | 77th St to 23rd St | 4.3 | 8,000 |
| 215,000 | 91st St to 23rd St, and the painted block at Washington Square | 5.5 | 9,500 |
| 178,000 | 110th St to 23rd St, and that block | 7.1 | 10,300 |

### The build-out

The budget sets the price. Each round scores every stretch of every run
(about 4 million) against the network as built so far, at a grid of
prices. It takes the highest price at which each run's best stretch,
summed over the runs, would fill what is left of the budget. Summing counts
two parallel avenues twice, so the price comes out high, and the next
round sets it again. At that price each run offers one stretch, and the
round builds them in order of saving per new meter. It skips a stretch if
more than 10% of the trips it helps are also helped by one already taken
this round: those two compete for the same riders, so the second waits for
the next round's rescore, which sees the first built. The round stops at
10 km of new lane. Picks on one street that touch, or that join across
blocks already protected, are reported as one corridor.

| Knob | Value | Why |
|---|---|---|
| Budget | 80 km | About 50 miles: NYC's Streets Plan target for protected bike lanes a year |
| Shortest project | 100 m | Every project costs at least about a block. Without a floor, a 9 m sliver at an intersection outranks every corridor on saving per meter |
| New lane per round | 10 km | Picks interact; rescoring every 10 km lets later picks see the earlier ones built |
| Shared riders | 10% | Two stretches that help disjoint sets of trips are each worth what they scored |
| Prices | 10,000 to 10 million cost-m a year per new meter, 12 steps to a factor of ten | Spans the best gap (about 2 million) to far below the price an 80 km budget sets |
| Sharpest turn in a run | 60° | Broadway's own bends, at Madison Square and Columbus Circle, reach 59° between LION nodes |

A trip's distances come from 40,000 station pairs sampled in proportion to
their trips, as in `candidates`. Distances are float32, so a saving under
half a cost-meter is rounding and is not counted.

### What it found

An 80 km build-out on the 2025 trips (`python -m bike_network corridors`)
took 12 rounds, at prices that fell from 147,000 to 83,000 cost-m a year
per new meter. It saves 9.5% of the cost of every Citi Bike trip, and
80,000 trips a day (65%) ride at least one new block. Of the new lane,
26.1 km goes where there is none, 16.3 km makes a painted lane protected,
9.9 km does the same for sharrows or a shared street, and 27.9 km runs
against one-way traffic.

- **Most of the budget fills gaps.** 269 of the 284 corridors are under a
  kilometer (median 150 m), and they hold 57 km and three quarters of the
  saving. Most are missing blocks in lanes that are already busy, such as
  the gaps in 9th Ave's lane below 29th St.
- **Fifteen corridors run a kilometer or more** and hold the other 23 km.
  Built alone, they save 3.4 times what their blocks save one at a time,
  which is why the block list missed them. The largest are a southbound
  contraflow lane on Madison Ave from 99th to 29th St (5.7 km), Lafayette
  Ave westbound in Brooklyn (2.1 km, contraflow), and Grand St westbound
  and 2nd Ave southbound from 43rd St to 2nd St (1.4 km each).
- **The rounds reshape corridors.** Under the cost model before room and
  pavement were added, 5th Ave southbound was worth building from 110th St
  to 23rd St in round 1; by round 10, Park Ave and Park Ave South
  southbound had taken the East Side's riders, and what was left on 5th Ave
  was 44th to 29th St. In this plan Madison Ave's contraflow lane takes
  them instead, and 5th Ave keeps only a 1.2 km stretch above 96th St.
- **Manhattan gets 60 km**, Brooklyn 19, Queens 1.3 and the Bronx 0.2. That
  is where Citi Bike's trips are, not a judgment on where lanes are needed.
- **35% of the new lane runs against one-way traffic** (contraflow). A
  quarter of it is beside a protected lane in the other direction (Grand
  St, Lafayette St, 10th Ave), and over half beside none (Madison Ave).
  With length the only cost, nothing weighs what a contraflow lane on an
  avenue would take space from.

Two checks were run under the cost model before room and pavement were
added, and have not been repeated since. Ranking by saving per new meter,
with no price to size the stretches, spent the same 80 km on 361 fragments
that saved 9.4% (the build-out then saved 10.4%). Its stretches extended
only as far as the densest piece of each street, so long corridors arrived
block by block, and the budget ran out before most of them were complete.

A second build-out on an independent sample of 40,000 station pairs
(`--seed 1`) shared 68% of the new blocks, 76% weighted by saving. Where two
parallel streets serve the same riders, the sample decided which one got
the lane: one plan built Park Ave from 98th to 47th St, the other 5th Ave
the whole way from 110th St to Washington Square. That the East Side's
long corridor has since moved again, to Madison Ave, makes the same point.
Read a corridor as "this street or the next one over".

### What it changes for riders

The build-out scores 40,000 sampled station pairs. `python -m bike_network
effect` routes every one of the year's 44.7 million trips again with the
plan's blocks protected and sets each beside its route today. It agrees with
the sample: 9.3% of the cost saved (9.5% on the sample), and 79,400 trips
a day (65%) gain. The trips are held fixed, the same riders between the same
stations, so this is where they would ride and what it would cost them, not
how many more would ride.

| Share of trip-km | Today | With the plan |
|---|---|---|
| Protected lane or path | 59.9% | 69.6% |
| Painted lane | 19.3% | 13.8% |
| Sharrows or shared street | 3.8% | 2.8% |
| No bike facility | 16.8% | 13.7% |
| Walked | 0.2% | 0.1% |
| Trip-km over the shortest routes | +7.8% | +6.1% |
| Cost per km | 1.28 | 1.18 |
| Km walked a year | 268,000 | 141,000 |

17% of the trip-km would ride the new lane. The detour that safety costs
shrinks by about a fifth, 1.8 million km a year.

- **A map of where trips move misses a third of the riders who gain.** 34%
  of the trips that gain keep today's route to within a meter: the lane goes
  on the street they already ride. They get 18% of the cost saved. The
  facility mix of the trip-km counts them; a map of trips a day with the
  plan minus today does not.
- **Riders move between parallel streets.** The streets that lose the most
  riding are 2nd Ave (5,400 trip-km a day), although the plan fills gaps in
  its lane, the Hudson River Greenway (4,000), 6th Ave (3,800), Central
  Park's West Drive and 9th Ave. Riders gain on Madison Ave (5,900),
  Broadway (4,700), Lafayette Ave in Brooklyn (2,400), Lafayette St, Grand
  St and 8th Ave. Changes of under 20 trips a day hold 4% of the riding
  that moves; the map leaves them faint.
- **The gain is uneven.** Counting a trip at both its stations, the median
  station's trips lose 3.5% of their cost. Of the stations with at least
  100 trips a day, the ten that gain most, 22 to 26%, are on the Upper East
  Side (six of them), in the West Village (two), Tribeca
  and Hudson Yards.

Each corridor gets one example trip: among the 20,000 station pairs that gain
the most (trips times cost saved), the one scoring highest on that gain
times the share of the corridor's new lane its new route rides. 225 trips
cover 278 of the 284 corridors; a trip can be the example for several. The
map draws each one's route today and with the plan. Its list sorts them by
the cost they save (trips times the drop in cost per trip), so a busy,
long trip outranks a short one that rides nothing but new lane. It leaves
out a trip when either route is under 1.2 km; when less than 30% of its
route with the plan is on streets today's route does not ride, since a lane
on the street a trip already takes shows little on a map of both routes;
and when more than 30% of the shorter one's streets, either way and on
either route, are shared with one above it: neighboring corridors on one
avenue often pick much the same trip. That lists 51 of the 225.

### Limits

- **One street per corridor.** A route that turns (5th Ave, then 23rd St)
  is two corridors, built in two rounds if the first pays on its own, and
  missed if neither does.
- **Greedy, not optimal.** Choosing the best set of lanes for a budget is a
  hard combinatorial problem; the rounds account for the biggest
  interactions, not all of them.
- **The only cost of a new lane is its length.** Whether it fits (street
  width, parking, bus lanes, loading) is not in it, so a corridor here is where riders would gain, not
  a design.
- The demand is Citi Bike's, and cost is route choice, not crash risk
  (section 6).

## 8. Reproducing it

```
python -m bike_network build       # network, traffic model, costs
python -m bike_network traffic     # the volume model's hold-out scores
python -m bike_network assign      # route 2025's Citi Bike trips
python -m bike_network calibrate   # the 108-setting grid (about an hour)
python -m bike_network calibrate --per-doubling 0.75,1.0 --painted 0.3,0.75 --walk-factor 6,8,12
python -m bike_network candidates  # single blocks, two samples
python -m bike_network corridors   # the 80 km build-out (--budget-km, --seed), then effect
python -m bike_network effect      # every trip routed again with the plan built
python -m bike_network viz         # export the map
python -m bike_network publish     # push it to GitHub Pages
```

`calibrate` merges each run into `data/build/calibration.parquet`, one row
per setting, with every counter's routed volume.
