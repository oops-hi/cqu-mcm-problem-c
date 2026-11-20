"""Capacity-constrained geofence design for high-volume pilot cells."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .demand import TRAIN, haversine, is_weekend


@dataclass
class Layout:
    active: np.ndarray         # chosen candidate indices, indexed into the fine cells
    capacity: np.ndarray
    assignment: np.ndarray     # site position in active, or -1
    budget: int
    gap: float
    status: str


def fine_profiles(rides, zones, zone_id, days):
    cells = np.flatnonzero(zones.code_zone == zone_id)
    index = np.full(len(rides.codes), -1, dtype="int32")
    index[cells] = np.arange(len(cells))
    n = len(cells)
    flows = {}
    for day in days:
        day_mask = rides.day == np.datetime64(day)
        out = day_mask & (zones.origin == zone_id)
        back = day_mask & (zones.destination == zone_id)
        departed = rides.minute[out] // 60
        duration = np.maximum(3, rides.distances[back] * 1.3 * 60 / 12)
        finish = ((rides.minute[back] + duration) // 60).astype(int)
        kept = finish < 24
        departures = np.bincount(departed * n + index[rides.origin[out]], minlength=24*n).reshape(24, n)
        arrivals = np.bincount(finish[kept] * n + index[rides.destination[back][kept]],
                               minlength=24*n).reshape(24, n)
        flows[day] = (departures, arrivals)
    return cells, flows


def design_layout(distance, occupancy, weight, sites, walk_limit=0.35, max_sites=8):
    """Allocate cells to individual facilities and size them against hourly occupancy."""
    candidates = len(sites)
    n, hours = occupancy.shape[1], occupancy.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(candidates)
             if distance[i, sites[j]] <= walk_limit]
    a_count = len(pairs)
    omitted = a_count
    opened = omitted + n
    capacity = opened + candidates
    spill = capacity + candidates
    size = spill + candidates * hours
    by_cell = [[] for _ in range(n)]
    by_site = [[] for _ in range(candidates)]
    for a, (i, j) in enumerate(pairs):
        by_cell[i].append(a)
        by_site[j].append((a, i))

    cost = np.zeros(size)
    for a, (i, j) in enumerate(pairs):
        cost[a] = 0.9 * weight[i] * distance[i, sites[j]]
    cost[omitted:opened] = 3.0 * weight + occupancy.mean(axis=0) + 0.01
    cost[opened:capacity] = 20.0
    cost[capacity:spill] = 0.05
    cost[spill:] = 1.2
    upper = np.full(size, np.inf)
    upper[:capacity] = 1
    site_limit = 220
    upper[capacity:spill] = site_limit
    budget = min(max_sites * site_limit, max(100, int(np.ceil(0.85 * occupancy.sum(axis=1).max()))))
    integrality = np.zeros(size, dtype="int8")
    integrality[:spill] = 1

    row, col, values = [], [], []
    lower_bounds, upper_bounds = [], []

    def constraint(terms, lower=-np.inf, upper=np.inf):
        k = len(lower_bounds)
        for key, value in terms:
            if value:
                row.append(k)
                col.append(key)
                values.append(value)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    for i, indices in enumerate(by_cell):
        constraint([(a, 1) for a in indices] + [(omitted+i, 1)], 1, 1)
    constraint([(opened+j, 1) for j in range(candidates)], 1, max_sites)
    constraint([(capacity+j, 1) for j in range(candidates)], upper=budget)
    for j, assigned in enumerate(by_site):
        constraint([(capacity+j, 1), (opened+j, -site_limit)], upper=0)
        constraint([(capacity+j, -1), (opened+j, 10)], upper=0)
        for a, _ in assigned:
            constraint([(a, 1), (opened+j, -1)], upper=0)
        for t in range(hours):
            constraint([(a, occupancy[t, i]) for a, i in assigned] +
                       [(capacity+j, -1), (spill + j*hours+t, -1)], upper=0)

    matrix = coo_matrix((values, (row, col)), shape=(len(lower_bounds), size)).tocsc()
    result = milp(cost, integrality=integrality, bounds=Bounds(np.zeros(size), upper),
                  constraints=LinearConstraint(matrix, lower_bounds, upper_bounds),
                  options={"time_limit": 15.0, "mip_rel_gap": 0.05})
    if result.x is None:
        raise RuntimeError(f"No feasible fence layout found: {result.message}")
    chosen = np.flatnonzero(result.x[opened:capacity] > 0.5)
    positions = {site: loc for loc, site in enumerate(chosen)}
    assignment = np.full(n, -1, dtype=int)
    for fitted, (i, j) in zip(result.x[:a_count], pairs):
        if fitted > 0.5:
            assignment[i] = positions[j]
    return Layout(sites[chosen], np.rint(result.x[capacity+chosen]).astype(int),
                  assignment, budget, float(result.mip_gap), str(result.message))


def popularity_layout(distance, weight, count, spaces, walk_limit=0.35):
    hot = np.argsort(-weight, kind="stable")[:count]
    closest = np.argmin(distance[:, hot], axis=1)
    closest[np.min(distance[:, hot], axis=1) > walk_limit] = -1
    base, remainder = divmod(spaces, count)
    capacity = np.full(count, base, dtype=int)
    capacity[:remainder] += 1
    return Layout(hot, capacity, closest, spaces, 0.0, "baseline")


def evaluate(layout, distance, occupancy, weight, competitor=0.0, shared=True):
    reached = layout.assignment >= 0
    sites = layout.assignment[reached]
    load = np.zeros((len(occupancy), len(layout.active)))
    for j in range(len(layout.active)):
        load[:, j] = occupancy[:, layout.assignment == j].sum(axis=1)
    rival = competitor * np.roll(load, 1, axis=0)
    total = load + rival
    if shared or not competitor:
        combined_over = np.maximum(0, total - layout.capacity)
        mobike_over = combined_over * np.divide(load, total, out=np.zeros_like(load), where=total > 0)
    else:
        quota = np.floor(layout.capacity * 0.75)
        mobike_over = np.maximum(0, load - quota)
        combined_over = mobike_over + np.maximum(0, rival - (layout.capacity - quota))
    uncovered = occupancy[:, ~reached].sum()
    walk = distance[np.flatnonzero(reached), layout.active[sites]]
    assigned_weight = weight[reached].sum()
    return {
        "spaces": int(layout.capacity.sum()), "sites": len(layout.active),
        "covered_return_share": float(assigned_weight / max(1, weight.sum())),
        "mean_walk_proxy_m": float(np.sum(walk * weight[reached]) * 1000 / max(1, assigned_weight)),
        "peak_assigned_occupancy_proxy": float(load.sum(axis=1).max()),
        "uncovered_bike_hours": float(uncovered),
        "on_site_overflow_bike_hours_all": float(combined_over.sum()),
        "mobike_uncovered_or_overflow_bike_hour_rate": float(
            (uncovered + mobike_over.sum()) / max(1, occupancy.sum())
        ),
    }


def plan_pilots(rides, zones, tests=("2017-05-19", "2017-05-21"), count=3):
    fine = [i for i, key in enumerate(zones.labels) if len(key) == 6]
    pilots = sorted(fine, key=lambda i: -zones.trips[i])[:count]
    layout_rows, evaluations, assignments = [], [], []
    for zone in pilots:
        weekdays = [d for d in TRAIN if not is_weekend(d)]
        cells, flows = fine_profiles(rides, zones, zone, (*weekdays, *tests))
        departure, arrival = np.mean([flows[d] for d in weekdays], axis=0)
        net = (arrival-departure).cumsum(axis=0)
        start = np.maximum(0, -net.min(axis=0))
        nominal = start + net
        weight = arrival.sum(axis=0)
        xy = rides.coordinates[cells]
        distance = haversine(xy[:, None, :], xy[None, :, :])
        candidates = np.argsort(-weight, kind="stable")[:min(16, np.count_nonzero(weight))]
        optimum = design_layout(distance, nominal, weight, candidates)
        baseline = popularity_layout(distance, weight, len(optimum.active), int(optimum.capacity.sum()))
        for key, design in [("popular_equal", baseline), ("optimized", optimum)]:
            for loc, j in enumerate(design.active):
                layout_rows.append({
                    "pilot_zone": zones.labels[zone], "layout": key,
                    "site_geohash": rides.codes[cells[j]], "latitude": xy[j, 0], "longitude": xy[j, 1],
                    "slots": int(design.capacity[loc]), "pilot_budget_slots": design.budget,
                    "mip_gap": design.gap, "solver_status": design.status,
                })
            for pos, cell in enumerate(cells):
                j = design.assignment[pos]
                assignments.append({
                    "pilot_zone": zones.labels[zone], "layout": key,
                    "demand_geohash": rides.codes[cell], "latitude": xy[pos, 0], "longitude": xy[pos, 1],
                    "assigned_to": rides.codes[cells[design.active[j]]] if j >= 0 else "",
                    "training_mean_daily_returns": weight[pos],
                })
            for date in ("train_weekday_average", *tests):
                if date == "train_weekday_average":
                    occupancy, daily_weight = nominal, weight
                else:
                    dep, arr = flows[date]
                    occupancy = np.maximum(0, start + (arr-dep).cumsum(axis=0))
                    daily_weight = arr.sum(axis=0)
                for rival, shared in [(0.0, True), (0.25, True), (0.25, False), (0.50, True), (0.50, False)]:
                    evaluations.append({
                        "pilot_zone": zones.labels[zone], "date_or_profile": date, "layout": key,
                        "competitor_relative_load": rival, "shared": shared,
                        **evaluate(design, distance, occupancy, daily_weight, rival, shared),
                    })
    return pd.DataFrame(layout_rows), pd.DataFrame(evaluations), pd.DataFrame(assignments)
