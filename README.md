# NYC Bike Route Network

Converting NYC into a graph where edges are the roads, weighted by how stressful
it is to ride them. Then, using bike traffic data, we determine where riders
are going and where new bike infrastructure would help the most.

[Explore the network](https://nikhilsaggi.github.io/bike-network/)

## Data

All inputs are public:

| Source | Used for |
|---|---|
| [LION](https://www.nyc.gov/content/planning/pages/resources/datasets/lion) (NYC DCP), release `26c` | streets, paths, traffic direction, travel lanes, posted speed, truck routes, street width and parking lanes |
| [NYC Bike Routes](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib) (NYC DOT) | bike facility in each direction, joined on LION `SegmentID` |
| [Street Pavement Ratings](https://data.cityofnewyork.us/d/6yyb-pb25) (NYC DOT) | each block's latest 1-10 pavement rating since 2018, matched to LION by geometry |
| [Citi Bike System Data](https://citibikenyc.com/system-data) (Citi Bike) | start and end locations for each Citi Bike ride |
| [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) (NYC DOT) | measured car volumes on about 1% of streets, to train the traffic model |
| [Bicycle Counts](https://data.cityofnewyork.us/Transportation/Bicycle-Counts/uczf-rk3c) (NYC DOT) | bikes past DOT's permanent counters, to calibrate the weights |

## How it works

1. **Give every street a cost.** Each block, in each direction, costs its
   length times how stressful it is to ride. A protected lane or greenway
   costs just its length. A painted lane costs a little more. Riding in
   traffic costs more the busier and faster the street is and if it is a
   truck route, and less if there is room for cars to pass. Bad pavement
   adds a little. Where riding is not allowed, such as the wrong way on a
   one-way street, you can walk the bike on the sidewalk at 8 times the
   length. A quiet side street works out to about 1.2x, a busy avenue
   3-4x.
2. **Estimate car traffic.** DOT has counted traffic on fewer than 1% of
   streets, so a model trained on those counts estimates the rest from
   what each street looks like: its width, lanes, one-way or not, truck
   route.
3. **Route the rides.** Each of 2025's 44.7 million Citi Bike trips takes
   the lowest-cost route between its two stations. Adding up the routes
   gives estimates of rider traffic on every block.
4. **Tune cost parameters.** The weight on traffic and the cost of
   walking were tuned so the routed riders line up with DOT's automatic
   bike counters on the bridges and avenues.
5. **Find where new lanes help most.** Try a protected lane on every
   stretch of every street and measure how much it would lower the cost of
   all the trips. Build the best per meter first, re-route, and score
   again, since each new lane changes what the next one is worth. Stop at
   50 mi of new lanes, which is NYC's yearly target for protected lanes.

The details, and the evidence for every weight, are in
[docs/methodology.md](docs/methodology.md).

## Limitations

* **Routing**: affects how accurately cycling traffic is portrayed.
    * Cycling traffic data is approximated by Citi Bike ride data, which is not
      representative of all cycling traffic. For example, the outer boroughs are
      notably lacking in traffic since Citi Bike riding is more popular in Manhattan.
    * The actual paths taken by riders are unknown. The Citi Bike dataset only gives
      us the start and end points, and we assume the lowest-cost paths are taken.
    * No accounting for turns or intersections in the route calculation.
* **Traffic**: most traffic numbers are approximated. Less than 1% of streets have
  reported counts, actual numbers may vary. This affects the cost calculations.
  The counted streets skew busy and the median count is from 2016, so
  estimates rank streets better than they measure any one of them. Volumes
  are daily totals, so rush hour and 3am cost the same.
* **Cost model**: only the traffic weight and the walking cost are tuned,
  against just 11 informative bike counters. The other weights are
  judgment calls.
* **Network**: inaccuracies in network edges compared to NYC's actual roads.
    * DOT's bike routes and LION have drifted apart in segment ids. Separated
      paths DOT has never recorded take their directions from LION (586 blocks);
      see [docs/methodology.md](docs/methodology.md).
    * About 30 DOT lane records are attached to the generic centerline of a
      divided street (LION layer `G`) rather than its carriageways, so they do not
      reach the network. Another ~270 current records name a segment LION does
      not have.
    * LION lags street redesigns. Lane counts, width and truck routes can
      describe a street as it was before DOT rebuilt it.
* **Planning**: limits on the new-lane recommendations.
    * Ridership is held fixed: the plan shows where today's Citi Bike riders
      would move, not the new riders a lane would attract.
    * The only cost of a new lane is its length. Whether it fits the street
      (width, parking, bus lanes, loading) is not considered, so a corridor
      is where riders would gain, not a design.
    * Which of two parallel streets gets a lane can depend on the sample of
      trips used. Read a corridor as "this street or the next one over".
    * Lanes are picked greedily, one street at a time, so the plan is good
      but not provably the best for its budget.

## Setup

```
uv venv && uv pip install -e . --group dev
python -m bike_network build      # the street network and its costs
python -m bike_network assign     # route the Citi Bike trips (downloads ~8 GB once)
python -m bike_network corridors  # plan the new lanes
python -m bike_network viz        # export the map
python -m http.server -d data/build/viz
```

The other commands, what each step writes, testing and publishing the site
are in [docs/development.md](docs/development.md).
