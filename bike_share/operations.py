"""Dynamic dispatch, user incentives and vehicle-conserving replay."""

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .demand import SLOTS, STEP_MINUTES, arrivals_from_od, haversine, opening_stock, ride_lag
from .rolling import recent_order_scale, solve_dispatch, timed_trip


@dataclass(frozen=True)
class Settings:
    trucks: int = 18
    truck_capacity: int = 45
    route_factor: float = 1.3
    truck_speed: float = 25.0
    handling_minutes_per_bike: float = 0.20  # at each end
    max_loaded_km: float = 13.0
    incentive_budget: float = 600.0
    incentive_walk_km: float = 1.0
    lookahead: int = 8
    reference_stock_factor: float = 0.25
    dispatch_ranges: tuple = ((5, 9), (11, 14), (15, 19))
    rolling_km_penalty: float = 0.50
    rolling_trip_penalty: float = 0.75
    rolling_solver_seconds: float = 2.0


class DayModel:
    def __init__(self, rides, zones, day, forecast, od_distance, settings):
        self.rides = rides
        self.zones = zones
        self.day = day
        self.forecast = forecast
        self.od_distance = od_distance
        self.settings = settings
        self.n = len(zones.labels)
        self.distance = haversine(zones.coordinates[:, None, :], zones.coordinates[None, :, :])
        self.distance[~np.isfinite(self.distance)] = np.inf
        selection = np.flatnonzero(rides.day == np.datetime64(day))
        hour_slot = rides.minute[selection] // STEP_MINUTES
        order = np.lexsort((rides.second[selection], rides.minute[selection],
                            zones.origin[selection], hour_slot))
        self.indices = selection[order]
        self.bounds = np.searchsorted(hour_slot[order], np.arange(SLOTS + 1))
        self.origin = zones.origin[self.indices]
        self.destination = zones.destination[self.indices]
        self.distance_of_ride = rides.distances[self.indices]
        self.rank = np.empty(len(self.indices), dtype=np.int32)
        for slot in range(SLOTS):
            a, b = self.bounds[slot:slot+2]
            origin = self.origin[a:b]
            if len(origin):
                first = np.flatnonzero(np.r_[True, origin[1:] != origin[:-1]])
                self.rank[a:b] = np.arange(len(origin)) - np.repeat(first, np.diff(np.r_[first, len(origin)]))
        self.random = np.random.default_rng(2026).random(len(self.indices))
        self._states = {}
        self.departures = forecast.sum(axis=2)
        self.core = self.n - 1
        self.depot = int(np.argmax(zones.trips[:self.core]))

    def setup(self, speed, stock_factor):
        key = (speed, stock_factor)
        if key not in self._states:
            start_stock, carry = opening_stock(self.rides, self.zones, self.day, stock_factor, speed)
            expected_arrivals = arrivals_from_od(self.forecast, self.od_distance, speed)
            net = self.departures - expected_arrivals
            target = np.empty_like(net)
            for t in range(SLOTS):
                window = net[t:t + self.settings.lookahead]
                target[t] = np.ceil(np.maximum(
                    0, np.max(np.cumsum(window, axis=0), axis=0)
                ) + 0.25 * self.departures[t:min(SLOTS, t+4)].sum(axis=0))
            minute = self.rides.minute[self.indices] % STEP_MINUTES
            offset = minute + self.rides.second[self.indices] / 60
            lag = ride_lag(self.distance_of_ride, speed, offset)
            self._states[key] = (start_stock, carry, target.astype("int32"), lag, expected_arrivals)
        return self._states[key]

    def record_dispatch(self, truck, slot, trip, stock, locations, available, pending, log):
        minute = slot * STEP_MINUTES
        if (locations[truck] != trip.start or available[truck] > minute or
                not 0 < trip.quantity <= min(stock[trip.pickup], self.settings.truck_capacity) or
                not slot < trip.arrival_slot < SLOTS):
            raise RuntimeError("Infeasible truck job in replay.")
        stock[trip.pickup] -= trip.quantity  # reserve bikes as soon as the truck is committed
        pending[trip.arrival_slot, trip.dropoff] += trip.quantity
        locations[truck] = trip.dropoff
        available[truck] = trip.arrival_slot * STEP_MINUTES
        log.append({
            "date": self.day, "truck": truck+1, "dispatch_minute": minute,
            "truck_start": self.zones.labels[trip.start], "pickup": self.zones.labels[trip.pickup],
            "dropoff": self.zones.labels[trip.dropoff], "bikes": trip.quantity,
            "pickup_minute": round(trip.pickup_minute, 2),
            "arrival_minute": trip.arrival_slot * STEP_MINUTES, "empty_km": trip.empty_km,
            "loaded_km": trip.loaded_km, "estimated_km": trip.km,
        })

    def dispatch_rolling(self, t, stock, truck_location, truck_until, pending, log,
                         arrivals, scale, km_penalty):
        cfg = self.settings
        minute = t * STEP_MINUTES
        if not any(a <= minute / 60 < b for a, b in cfg.dispatch_ranges):
            return {}
        groups = defaultdict(list)
        for k in np.flatnonzero(truck_until <= minute):
            groups[int(truck_location[k])].append(int(k))
        if not groups:
            return {"rolling_scale": scale, "rolling_status": "no_idle_truck"}
        window = slice(t, min(SLOTS, t+cfg.lookahead))
        jobs, details = solve_dispatch(self.distance, cfg, t, stock, self.departures[window],
                                       arrivals[window], pending[window], groups, scale, km_penalty)
        for trip, quantity in jobs:
            for _ in range(quantity):
                truck = groups[trip.start].pop()
                self.record_dispatch(truck, t, trip, stock, truck_location, truck_until, pending, log)
        return {"rolling_scale": scale, "rolling_jobs": sum(q for _, q in jobs), **details}

    def dispatch(self, policy, t, stock, target, truck_location, truck_until, pending, log):
        cfg = self.settings
        minute = t * STEP_MINUTES
        ready = list(np.flatnonzero(truck_until <= minute))
        if not ready or not any(a <= minute / 60 < b for a, b in cfg.dispatch_ranges):
            return
        projected = stock + pending[t+1:min(len(pending), t+cfg.lookahead+1)].sum(axis=0)
        surplus = np.maximum(0, stock - target[t]).astype(int)
        deficit = np.maximum(0, target[t] - projected).astype(int)
        surplus[self.core] = deficit[self.core] = 0
        pairs = []
        for j in np.flatnonzero(deficit):
            nearest = np.argsort(self.distance[j] + np.where(surplus > 0, 0, np.inf))
            for i in nearest[:10]:
                km = cfg.route_factor * self.distance[i, j]
                if surplus[i] > 0 and km <= cfg.max_loaded_km:
                    pairs.append((int(i), int(j), float(km)))
        if not pairs:
            return
        delivery_limit = len(ready) * cfg.truck_capacity
        if policy == "optimized":
            graph = nx.DiGraph()
            graph.add_edge("source", "allocation", capacity=delivery_limit, weight=0)
            graph.add_node("sink")
            for i, j, km in pairs:
                graph.add_edge("allocation", ("pickup", i), capacity=int(surplus[i]), weight=0)
                graph.add_edge(("delivery", j), "sink", capacity=int(deficit[j]), weight=0)
                graph.add_edge(("pickup", i), ("delivery", j),
                               capacity=int(min(surplus[i], deficit[j])), weight=int(100*km))
            flow = nx.max_flow_min_cost(graph, "source", "sink")
            jobs = [(i, j, km, flow[("pickup", i)][("delivery", j)]) for i, j, km in pairs]
            jobs = sorted((job for job in jobs if job[3]), key=lambda row: (-row[3], row[2]))
        else:
            jobs = [(i, j, km, min(surplus[i], deficit[j]))
                    for i, j, km in sorted(pairs, key=lambda row: row[2])]
        for i, j, _, amount in jobs:
            remaining = int(min(amount, surplus[i], deficit[j]))
            while remaining > 0 and ready:
                k = min(ready, key=lambda k: self.distance[truck_location[k], i])
                quantity = min(remaining, cfg.truck_capacity)
                trip = timed_trip(self.distance, cfg, t, truck_location[k], i, j, quantity)
                if trip.arrival_slot >= SLOTS:
                    break
                ready.remove(k)
                self.record_dispatch(k, t, trip, stock, truck_location, truck_until, pending, log)
                surplus[i] -= quantity
                deficit[j] -= quantity
                remaining -= quantity
            if not ready:
                break

    def reroute(self, mode, response, t, destination, random, stock, target, arrivals, budget):
        if mode == "none" or budget < 0.4:
            return destination, 0.0, 0, 0, 0.0
        revised = destination.copy()
        expected = stock + arrivals[t+1:min(len(arrivals), t+5)].sum(axis=0)
        horizon = min(SLOTS-1, t+3)
        shortage = np.maximum(0, target[horizon] - expected)
        # Avoid diverting returns from zones that need their near-term stock.
        margin = np.maximum(10, (target[horizon] * 0.12).astype(int))
        excess = np.maximum(0, expected - target[horizon] - margin)
        excess[-1] = shortage[-1] = 0
        spent = 0.0
        offered = accepted = 0
        extra_walking = 0.0
        for old in np.unique(destination[excess[destination] > 0]):
            near = np.flatnonzero((self.distance[old] <= self.settings.incentive_walk_km) &
                                  (shortage > 0) & (np.arange(self.n) != old))
            if len(near) == 0:
                continue
            new = int(near[np.argmax(shortage[near] / (0.25 + self.distance[old, near]))])
            distance = self.distance[old, new]
            if mode == "fixed":
                reward = 0.8
            else:
                reward = 1.2 if shortage[new] >= 90 and distance > 0.4 else (
                    0.4 if distance < 0.35 or shortage[new] < 20 else 0.8
                )
            eligible = np.flatnonzero(destination == old)
            quota = max(0, min(int(excess[old]), int(shortage[new]),
                               int(np.floor((budget-spent+1e-8)/reward))))
            eligible = eligible[:quota]
            if len(eligible) == 0:
                continue
            offered += len(eligible)
            # Coefficients are scenario assumptions, not fitted effects.
            logit = {"low": -2.4, "mid": -1.4, "high": -0.4}[response]
            logit += 2.0 * reward - 2.8 * distance
            selected = eligible[random[eligible] < 1 / (1 + np.exp(-logit))]
            revised[selected] = new
            changed = len(selected)
            accepted += changed
            spent += reward * changed
            shortage[new] -= changed
            extra_walking += distance * changed
        return revised, spent, offered, accepted, extra_walking

    def run(self, dispatch="none", incentive="none", response="mid", speed=12.0, stock_factor=None,
            reward_budget=None, calibrate=True, km_penalty=None):
        if stock_factor is None:
            stock_factor = self.settings.reference_stock_factor
        if reward_budget is None:
            reward_budget = self.settings.incentive_budget
        if km_penalty is None:
            km_penalty = self.settings.rolling_km_penalty
        stock, carry, target, lag, forecast_arrivals = self.setup(speed, stock_factor)
        stock = stock.copy()
        pending = np.zeros((SLOTS + int(lag.max()) + 2, self.n), dtype=np.int32)
        for slot, cells in enumerate(carry):
            if cells:
                pending[slot] = np.bincount(cells, minlength=self.n)
        start_fleet = int(stock.sum() + pending.sum())
        truck_arrival = np.zeros_like(pending)
        truck_location = np.full(self.settings.trucks, self.depot, dtype=int)
        truck_until = np.zeros(self.settings.trucks, dtype=int)
        route_log, slot_log = [], []
        cost = walk = 0.0
        offers = accepts = missed = missed_peak = demand_peak = 0
        internal_requests = 0
        recent_served, recent_expected = np.zeros(SLOTS), np.zeros(SLOTS)
        for t in range(SLOTS):
            stock += pending[t] + truck_arrival[t]
            decisions = {}
            if dispatch == "rolling":
                scale = recent_order_scale(recent_served, recent_expected, t) if calibrate else 1.0
                decisions = self.dispatch_rolling(t, stock, truck_location, truck_until, truck_arrival,
                                                  route_log, forecast_arrivals, scale, km_penalty)
            elif dispatch in ("greedy", "optimized"):
                self.dispatch(dispatch, t, stock, target, truck_location, truck_until, truck_arrival, route_log)
            a, b = self.bounds[t:t+2]
            origins = self.origin[a:b]
            is_core = origins != self.core
            internal_requests += int(is_core.sum())
            can_serve = self.rank[a:b] < stock[origins]
            rejected = int((~can_serve & is_core).sum())
            missed += rejected
            hour = t // 4
            if 7 <= hour < 10 or 17 <= hour < 20:
                demand_peak += int(is_core.sum())
                missed_peak += rejected
            ctr = np.bincount(origins[can_serve], minlength=self.n)
            stock -= ctr.astype("int32")
            if dispatch == "rolling" and calibrate:
                # Use only zones whose inventory did not run out in this completed slot.
                usable = stock[:self.core] > 0
                recent_served[t] = ctr[:self.core][usable].sum()
                recent_expected[t] = self.departures[t, :self.core][usable].sum()
            dst = self.destination[a:b][can_serve]
            arrivals = pending + truck_arrival if incentive != "none" else pending
            dst, spent, shown, taken, distance = self.reroute(
                incentive, response, t, dst, self.random[a:b][can_serve], stock, target,
                arrivals, reward_budget - cost,
            )
            cost += spent
            offers += shown
            accepts += taken
            walk += distance
            # Keep assigned ride duration in the incentive counterfactual.
            np.add.at(pending, (t + lag[a:b][can_serve], dst), 1)
            expected_fleet = int(stock.sum() + pending[t+1:].sum() + truck_arrival[t+1:].sum())
            if expected_fleet != start_fleet or np.any(stock < 0):
                raise RuntimeError(f"Vehicle balance violated in slot {t}.")
            slot_log.append({"slot": t, "observed_in_area": int(is_core.sum()),
                             "simulated_missed": rejected, "parked_in_area": int(stock[:self.core].sum()),
                             **decisions})
        km = float(sum(route["estimated_km"] for route in route_log))
        metrics = {
            "date": self.day, "dispatch": dispatch, "incentive": incentive, "response": response,
            "assumed_speed_kmh": speed, "stock_factor": stock_factor, "opening_fleet_all": start_fleet,
            "in_area_orders": internal_requests, "simulated_missed": missed,
            "simulated_missed_rate": missed / internal_requests,
            "peak_missed_rate": missed_peak / demand_peak,
            "loaded_trips": len(route_log), "bikes_trucked": sum(r["bikes"] for r in route_log),
            "estimated_truck_km": km, "offers": offers, "accepted": accepts,
            "reward_budget": reward_budget,
            "reward_spend": round(cost, 2),
            "rolling_km_penalty": km_penalty if dispatch == "rolling" else None,
            "online_scale": calibrate if dispatch == "rolling" else None,
            "extra_walk_km_per_acceptance": (walk / accepts if accepts else 0.0),
            "reward_per_redirect": (cost / accepts if accepts else 0.0),
            "conservation_check": "pass",
        }
        return metrics, pd.DataFrame(route_log), pd.DataFrame(slot_log)
