"""Choose physically timed truck jobs over a short, uncertain demand horizon."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .demand import STEP_MINUTES


SCENARIO_FACTORS = np.array([0.85, 1.00, 1.15])
SCENARIO_WEIGHTS = np.array([0.25, 0.50, 0.25])


@dataclass(frozen=True)
class Trip:
    start: int
    pickup: int
    dropoff: int
    quantity: int
    pickup_minute: float
    arrival_slot: int
    empty_km: float
    loaded_km: float

    @property
    def km(self):
        return self.empty_km + self.loaded_km


def timed_trip(distance, cfg, slot, start, pickup, dropoff, quantity):
    empty_km = float(cfg.route_factor * distance[start, pickup])
    loaded_km = float(cfg.route_factor * distance[pickup, dropoff])
    pickup_minute = slot * STEP_MINUTES + empty_km / cfg.truck_speed * 60
    finish = pickup_minute + loaded_km / cfg.truck_speed * 60
    finish += 2 * cfg.handling_minutes_per_bike * quantity
    arrival_slot = max(slot+1, int(np.ceil(finish / STEP_MINUTES)))
    return Trip(int(start), int(pickup), int(dropoff), int(quantity), pickup_minute,
                arrival_slot, empty_km, loaded_km)


def recent_order_scale(actual, expected, slot, window=8, prior_orders=1600):
    """Shrink the recent observed/forecast ratio toward one using only past slots."""
    left = max(0, slot-window)
    seen = actual[left:slot].sum()
    predicted = expected[left:slot].sum()
    return float(np.clip((prior_orders+seen) / (prior_orders+predicted), 0.25, 1.75))


def cumulative_shortage(departures, returns, inbound, stock, scale):
    """Prefix deficits with current-slot arrivals already accounted for in stock."""
    net = (departures - returns).copy()
    net[0] = departures[0]
    planned = inbound.copy()
    planned[0] = 0
    return np.cumsum(scale*SCENARIO_FACTORS[:, None, None]*net - planned, axis=1) - stock


def candidate_trips(distance, cfg, slot, stock, gaps, truck_groups, km_penalty):
    """Screen nearby jobs with a positive stand-alone benefit under the same scenarios."""
    horizon = gaps.shape[1]
    shortage = np.maximum(0, gaps.max(axis=1))
    surplus = np.minimum(stock, np.maximum(0, np.floor(-gaps.max(axis=(0, 1))))).astype(int)
    mean_shortage = SCENARIO_WEIGHTS @ shortage
    # The last node is an aggregate outside the truck service area.
    receivers = np.flatnonzero(mean_shortage[:-1] > 0.5)
    receivers = receivers[np.argsort(-mean_shortage[receivers], kind="stable")[:14]]
    donors = np.flatnonzero(surplus[:-1] > 0)
    if not len(donors):
        return surplus, []
    trips = []
    for j in receivers:
        nearby = donors[np.argsort(distance[donors, j], kind="stable")[:5]]
        for i in nearby:
            if cfg.route_factor*distance[i, j] > cfg.max_loaded_km:
                continue
            maximum = min(int(surplus[i]), cfg.truck_capacity, int(np.ceil(shortage[:, j].max())))
            for start in truck_groups:
                for count in sorted({maximum, max(1, maximum//2)}):
                    trip = timed_trip(distance, cfg, slot, start, i, j, count)
                    arrival = trip.arrival_slot-slot
                    if arrival >= horizon:
                        continue
                    early = np.maximum(0, gaps[:, :arrival, j].max(axis=1))
                    late = gaps[:, arrival:, j].max(axis=1)
                    benefit = SCENARIO_WEIGHTS @ (shortage[:, j] - np.maximum(early, late-count))
                    if benefit > km_penalty*trip.km + cfg.rolling_trip_penalty:
                        trips.append(trip)
    return surplus, trips


def solve_dispatch(distance, cfg, slot, stock, departures, returns, inbound,
                   truck_groups, scale, km_penalty):
    """Solve truck assignment and cargo together, then return feasible jobs for this slot."""
    gaps = cumulative_shortage(departures, returns, inbound, stock, scale)
    donors, trips = candidate_trips(distance, cfg, slot, stock, gaps, truck_groups, km_penalty)
    if not trips:
        return [], {"rolling_status": "no_profitable_job", "rolling_candidates": 0}

    by_start, by_pickup, by_dropoff = defaultdict(list), defaultdict(list), defaultdict(list)
    for idx, trip in enumerate(trips):
        by_start[trip.start].append(idx)
        by_pickup[trip.pickup].append(idx)
        by_dropoff[trip.dropoff].append(idx)

    n_trips = len(trips)
    receivers = list(by_dropoff)
    horizon = gaps.shape[1]
    n_receivers = len(receivers)
    objective = np.r_[
        [km_penalty*trip.km + cfg.rolling_trip_penalty for trip in trips],
        np.tile(SCENARIO_WEIGHTS, n_receivers),
    ]
    upper = np.r_[[len(truck_groups[trip.start]) for trip in trips],
                  np.full(n_receivers*len(SCENARIO_WEIGHTS), np.inf)]
    integrality = np.r_[np.ones(n_trips, dtype=np.int8),
                       np.zeros(n_receivers*len(SCENARIO_WEIGHTS), dtype=np.int8)]
    row, col, values, rhs = [], [], [], []

    def constrain(terms, bound):
        for idx, coefficient in terms:
            row.append(len(rhs))
            col.append(idx)
            values.append(coefficient)
        rhs.append(bound)

    for start, indices in by_start.items():
        constrain([(i, 1) for i in indices], len(truck_groups[start]))
    for origin, indices in by_pickup.items():
        constrain([(i, trips[i].quantity) for i in indices], donors[origin])
    for j_idx, destination in enumerate(receivers):
        indices = by_dropoff[destination]
        # Avoid persistent extra stock, while allowing a few rounded fractional units.
        max_delivery = max(1, int(np.ceil(np.maximum(0, gaps[:, :, destination]).max())))
        constrain([(i, trips[i].quantity) for i in indices], max_delivery)
        for s in range(len(SCENARIO_WEIGHTS)):
            unmet = n_trips + j_idx*len(SCENARIO_WEIGHTS) + s
            for h in range(horizon):
                deficit = gaps[s, h, destination]
                if deficit > 0:
                    arrived = [(i, -trips[i].quantity) for i in indices
                               if trips[i].arrival_slot <= slot+h]
                    constrain(arrived + [(unmet, -1)], -deficit)

    constraints = LinearConstraint(coo_matrix((values, (row, col)),
                                               shape=(len(rhs), len(objective))).tocsc(),
                                   -np.inf, rhs)
    result = milp(objective, integrality=integrality, bounds=Bounds(np.zeros(len(objective)), upper),
                  constraints=constraints, options={"time_limit": cfg.rolling_solver_seconds,
                                                    "mip_rel_gap": 0.03})
    if result.x is None:
        return [], {"rolling_status": f"fallback_{result.status}", "rolling_candidates": n_trips}
    count = np.rint(result.x[:n_trips]).astype(int)
    chosen = result.x.copy()
    chosen[:n_trips] = count
    if np.any(chosen < -1e-6) or np.any(chosen > upper + 1e-6) or np.any(
        constraints.A @ chosen > np.asarray(rhs) + 1e-5
    ):
        raise RuntimeError("Rolling dispatch returned an infeasible solution.")
    baseline = float(np.sum(np.maximum(0, gaps[:, :, receivers].max(axis=1)) *
                            SCENARIO_WEIGHTS[:, None]))
    objective_value = float(objective @ chosen)
    details = {"rolling_status": f"solver_{result.status}", "rolling_candidates": n_trips,
               "rolling_mip_gap": getattr(result, "mip_gap", None),
               "rolling_baseline_shortage": baseline, "rolling_objective": objective_value}
    # The no-dispatch solution is always feasible and is better than a poor incumbent.
    if objective_value > baseline + 1e-6:
        details["rolling_status"] = "fallback_baseline"
        return [], details
    return [(trip, int(q)) for trip, q in zip(trips, count) if q > 0], details
