# Methodology: how the network is costed, and why these weights

The network gives every direction of every NYC block a cost: its length
times a multiplier for how stressful it is to ride. This page explains where
each part of that multiplier comes from: what was **measured**, what was
**calibrated** against observed bike counts, and what is **judgment**. It
also explains why the judgment calls were not calibrated.

All inputs are public. Numbers below are from LION 26C, DOT's bike routes as
fetched in November 2026, DOT traffic counts 2010-2026, Citi Bike trips for
2025 and DOT bike counters for 2025.

## 1. The network

| Source | What it supplies |
|---|---|
| [LION](https://www.nyc.gov/content/planning/pages/resources/datasets/lion) (NYC Planning), 26C | Every street, path and greenway block; traffic direction; travel lanes; posted speed; truck routes; street width |
| [NYC Bike Routes](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib) (DOT) | The bike facility in each direction of each block, joined on LION's segment id |
| [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) (DOT) | Measured vehicle volumes, per direction, on about 1% of streets |
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
riding:  factor = 1 + exposure(facility in this direction) × traffic_penalty
walking: factor = WALK_FACTOR (STEP_FACTOR on a step street)

traffic_penalty = 0                         where no car may go
                = TRAFFIC_BASE
                + TRAFFIC_PER_DOUBLING × max(0, log2(volume / VOLUME_REFERENCE))
                + TRAFFIC_PER_5_MPH × max(0, (speed limit − 25) / 5)
                + TRUCK_PENALTY[truck route]
```

`volume` is vehicles a day in the rider's direction. In a contraflow lane it
is the oncoming stream. Every factor is at least 1, so no route can lower its
cost by looping, and Dijkstra's algorithm finds the cheapest route exactly.

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
| `WALK_FACTOR` | 8 | **Calibrated** | Section 5 |
| `STEP_FACTOR` | 12 | Judgment | Carrying a bike up steps has to cost more than walking it. Step streets are rare enough that the value barely matters |

Only the three numbers with the largest effect on routes were calibrated:
the traffic weight, the painted-lane exposure and the walking factor.
Seventeen counters, of which eleven are informative, cannot pin down ten
numbers. The others stay where judgment put them.

### What a factor means

| Example | Factor |
|---|---|
| Protected lane, greenway, path | 1.0 |
| Quiet street, no lane | 1.2 |
| Painted lane on a one-lane street | about 1.06 |
| Two lanes each way (about 6,400 vehicles/day), no lane | about 3.2 |
| Busiest arterials, no lane | about 5.5 |
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
2.5-12. Each one re-routes all 44.7M trips.

| | Score (lower is better) | Raw spread | Rank correlation |
|---|---|---|---|
| Shortest legal route (no safety weights) | 0.405 | 0.620 | 0.86 |
| First settings (traffic 0.3, paint 0.3, walk 4) | 0.316 | 0.475 | 0.82 |
| **Chosen (traffic 1.0, paint 0.3, walk 8)** | **0.271** | 0.452 | 0.86 |
| Best in the grid (traffic 0.75, paint 0.75, walk 12) | 0.267 | 0.469 | 0.85 |

Every safety-weighted setting in the grid beats distance alone (the worst
scores 0.346), and the chosen one by 33%, within 2% of the best. Rank correlation hardly separates them: the busy bridges
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

### The chosen setting, counter by counter

Routed and counted bikes a day, and each counter's residual after the
density adjustment (0 is a perfect fit).

| Counter | Counted | Routed, shortest | Routed, chosen | Residual, shortest | Residual, chosen |
|---|---|---|---|---|---|
| Williamsburg Bridge | 6,850 | 3,536 | 3,682 | +0.48 | +0.16 |
| Queensboro Bridge | 5,722 | 1,438 | 1,530 | +0.46 | −0.01 |
| Manhattan Bridge | 4,374 | 1,822 | 2,249 | −0.03 | −0.11 |
| Brooklyn Bridge | 3,817 | 1,087 | 2,203 | −0.25 | +0.12 |
| Kent Ave | 3,438 | 1,216 | 2,155 | +0.01 | +0.24 |
| Pulaski Bridge | 3,247 | 1,281 | 1,366 | +0.68 | +0.27 |
| Willis Ave Bridge | 2,579 | 160 | 344 | −0.35 | −0.25 |
| 8th Ave at 50th St | 2,946 | 1,278 | 1,145 | −0.10 | −0.48 |
| Amsterdam Ave at 86th St | 2,582 | 364 | 747 | −0.53 | −0.24 |
| Columbus Ave at 86th St | 1,959 | 344 | 1,027 | −0.31 | +0.35 |
| Prospect Park West | 2,555 | 579 | 903 | −0.07 | −0.06 |

The weights mostly fix the bridges: the shortest routing over-sends riders to
Williamsburg, Queensboro and Pulaski, and under-sends them to the Brooklyn
Bridge's protected path. The worst remaining misses are 8th Ave in Midtown
(under-predicted) and the Upper West Side pair, one-way in opposite
directions: Columbus (southbound) is over-predicted and Amsterdam
(northbound) under-predicted. Those two share one density adjustment, so a
directional imbalance in the Citi Bike trips there would show up the same
way.

## 6. What the weights are, and are not

- They describe **route choice**: what riders do, so the routes can say which
  lanes get used and which would be. They are not a crash-risk measure. That
  would need crash data with exposure, which is a different calibration.
- They are fitted to **Citi Bike riders**. People on their own bikes,
  delivery riders and Staten Island are not in the demand.
- **Eleven counters** decide the traffic and walking weights. Rankings built
  on these weights should be checked across the near-best settings (the
  seven within 5% of the best), not read off one setting.
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

It misses corridors. Midtown 5th Ave has no lane from 59th St down to 24th
St (factor 5.0, about 22,000 vehicles a day), and a protected lane from 23rd
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

| | Stress saved, cost-km a year | Trips a day using it |
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
took 11 rounds, at prices that fell from 178,000 to 100,000 cost-m a year
per new meter. It saves 10.4% of the stress on every Citi Bike trip, and
79,000 trips a day (65%) ride at least one new block. Of the new lane,
27.5 km goes where there is none, 16.5 km makes a painted lane protected,
11.2 km does the same for sharrows or a shared street, and 25.0 km runs
against one-way traffic.

- **Most of the budget fills gaps.** 253 of the 269 corridors are under a
  kilometer (median 150 m), and they hold 54 km and three quarters of the
  saving. Most are missing blocks in lanes that are already busy, such as
  short gaps in 2nd Ave's lane.
- **Sixteen corridors run a kilometer or more** and hold the other 26 km.
  Built alone, they save 3.3 times what their blocks save one at a time,
  which is why the block list missed them. The largest are Park Ave
  southbound from 98th to 47th St (4.1 km), DeKalb Ave westbound in
  Brooklyn (3.1 km, paint made protected), Nostrand Ave southbound (2.3 km)
  and a southbound contraflow lane on 8th Ave from 45th to 20th St
  (2.0 km).
- **The rounds reshape corridors.** In round 1, 5th Ave southbound was worth
  building from 110th St to 23rd St. By round 10, Park Ave and Park Ave
  South southbound had taken the East Side's riders, and what was left on
  5th Ave was 44th to 29th St (1.2 km, 2,600 trips a day). Madison Ave
  northbound from 39th to 103rd St saves 781,000 on its own today, 151,000
  per new meter; with the rest of the plan built, 85,000 per new meter,
  under the last round's price, so it is not built.
- **Manhattan gets 62 km**, Brooklyn 16, Queens 1.4 and the Bronx 0.1. That
  is where Citi Bike's trips are, not a judgment on where lanes are needed.
- **31% of the new lane runs against one-way traffic** (contraflow), often
  beside a protected lane in the other direction: 8th Ave, 6th Ave and
  Grand St each get one. With length the only cost, nothing weighs what a
  contraflow lane on an avenue would take space from.

Ranking by saving per new meter, with no price to size the stretches,
spent the same 80 km on 361 fragments that saved 9.4%. Its stretches
extended only as far as the densest piece of each street, so long
corridors arrived block by block, and the budget ran out before most of
them were complete.

A second build-out on an independent sample of 40,000 station pairs
(`--seed 1`) shares 68% of the new blocks, 76% weighted by saving. Both
build DeKalb Ave westbound, East 72nd St, Lafayette St, Bleecker St, Park
Ave South, the Grand St contraflow lane and 5th Ave northbound beside the
park. Where two parallel streets serve the same riders, the sample decides
which one gets the lane. The clearest case is the East Side's long
southbound corridor. This plan builds Park Ave from 98th to 47th St and 5th
Ave from 44th to 29th St; the other builds 5th Ave the whole way from 110th
St to Washington Square (7.2 km) and Park Ave only from 42nd to 32nd St.
The 8th Ave and 6th Ave contraflow lanes are mostly in one plan only. Read
a corridor as "this street or the next one over".

### Limits

- **One street per corridor.** A route that turns (5th Ave, then 23rd St)
  is two corridors, built in two rounds if the first pays on its own, and
  missed if neither does.
- **Greedy, not optimal.** Choosing the best set of lanes for a budget is a
  hard combinatorial problem; the rounds account for the biggest
  interactions, not all of them.
- **The only cost is length.** Street width, parking, bus lanes and
  loading are not in it, so a corridor here is where riders would gain, not
  a design.
- The demand is Citi Bike's, and stress is route choice, not crash risk
  (section 6).

## 8. Reproducing it

```
python -m bike_network build       # network, traffic model, costs
python -m bike_network traffic     # the volume model's hold-out scores
python -m bike_network assign      # route 2025's Citi Bike trips
python -m bike_network calibrate   # the 108-setting grid (about an hour)
python -m bike_network calibrate --per-doubling 0.75,1.0 --painted 0.3,0.75 --walk-factor 6,8,12
python -m bike_network candidates  # single blocks, two samples
python -m bike_network corridors   # the 80 km build-out (--budget-km, --seed)
python -m bike_network viz         # export the map
python -m bike_network publish     # push it to GitHub Pages
```

`calibrate` merges each run into `data/build/calibration.parquet`, one row
per setting, with every counter's routed volume.
