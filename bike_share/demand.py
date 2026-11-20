"""Read rides, define service zones and estimate time-dependent OD flows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SLOTS = 96
STEP_MINUTES = 15
TRAIN = tuple(f"2017-05-{day:02d}" for day in range(10, 17))
VALIDATE = "2017-05-18"
TEST = ("2017-05-19", "2017-05-21", "2017-05-22")
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def is_weekend(day):
    return pd.Timestamp(day).dayofweek >= 5


def geohash_center(code):
    """Decode a geohash to the center of its latitude-longitude rectangle."""
    bounds = [[-180.0, 180.0], [-90.0, 90.0]]
    axis = 0
    for digit in code:
        value = BASE32.index(digit)
        for bit in (16, 8, 4, 2, 1):
            interval = bounds[axis]
            middle = sum(interval) / 2
            interval[0 if value & bit else 1] = middle
            axis ^= 1
    return tuple(sum(b) / 2 for b in reversed(bounds))


def haversine(left, right):
    lat1, lon1 = np.moveaxis(np.radians(left), -1, 0)
    lat2, lon2 = np.moveaxis(np.radians(right), -1, 0)
    a = np.sin((lat2 - lat1) / 2)**2
    a += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 12_742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class Rides:
    day: np.ndarray
    minute: np.ndarray
    second: np.ndarray
    bike: np.ndarray
    origin: np.ndarray
    destination: np.ndarray
    codes: np.ndarray
    coordinates: np.ndarray
    distances: np.ndarray


def read_rides(path):
    fields = ["starttime", "bikeid", "geohashed_start_loc", "geohashed_end_loc"]
    raw = pd.read_csv(path, usecols=fields, dtype={
        "bikeid": "int32", fields[2]: "category", fields[3]: "category",
    })
    timestamp = pd.to_datetime(raw["starttime"], format="%Y-%m-%d %H:%M:%S")
    o = raw[fields[2]]
    d = raw[fields[3]]
    codes = np.union1d(o.cat.categories, d.cat.categories)
    origin = np.searchsorted(codes, o.cat.categories)[o.cat.codes.to_numpy()].astype("int32")
    destination = np.searchsorted(codes, d.cat.categories)[d.cat.codes.to_numpy()].astype("int32")
    if o.isna().any() or d.isna().any():
        raise ValueError("There are orders with a missing location.")
    coordinates = np.array([geohash_center(code) for code in codes])
    distances = haversine(coordinates[origin], coordinates[destination]).astype("float32")
    return Rides(
        day=timestamp.to_numpy(dtype="datetime64[D]"),
        minute=(timestamp.dt.hour * 60 + timestamp.dt.minute).to_numpy(dtype="int16"),
        second=timestamp.dt.second.to_numpy(dtype="int8"),
        bike=raw["bikeid"].to_numpy(), origin=origin, destination=destination,
        codes=codes, coordinates=coordinates, distances=distances,
    )


def coverage_audit(rides):
    counts = []
    for day in np.arange(rides.day.min(), rides.day.max() + 1):
        observed = rides.day == day
        by_hour = np.bincount(rides.minute[observed] // 60, minlength=24)
        counts.append({
            "date": str(day), "orders": int(by_hour.sum()),
            "hours_with_orders": int(np.count_nonzero(by_hour)),
            "empty_hours": ",".join(map(str, np.flatnonzero(by_hour == 0))),
        })
    return pd.DataFrame(counts)


@dataclass
class Zones:
    labels: list
    coordinates: np.ndarray
    origin: np.ndarray
    destination: np.ndarray
    code_zone: np.ndarray
    trips: np.ndarray


def make_zones(rides, parents=55, split_parents=7, children=12):
    """Fit a mixed-resolution zone system using training endpoints only."""
    mask = np.isin(rides.day, np.array(TRAIN, dtype="datetime64[D]"))
    counts = np.bincount(rides.origin[mask], minlength=len(rides.codes))
    counts += np.bincount(rides.destination[mask], minlength=len(rides.codes))
    observed_codes = np.char.array(rides.codes, unicode=True)
    prefix5 = np.array([c[:5] for c in observed_codes])
    prefix6 = np.array([c[:6] for c in observed_codes])
    weights = pd.DataFrame({"p": prefix5, "c": prefix6, "n": counts})
    parent_volume = weights.groupby("p")["n"].sum().sort_values(ascending=False)
    kept_parents = parent_volume.head(parents).index
    children_labels = []
    for parent in kept_parents[:split_parents]:
        top = (weights[weights["p"] == parent].groupby("c")["n"].sum()
               .sort_values(ascending=False).head(children))
        children_labels.extend(top[top > 0].index)
    labels = [*kept_parents, *children_labels, "EXTERNAL"]
    lookup = {key: index for index, key in enumerate(labels)}
    external = len(labels) - 1
    code_zone = np.array([
        lookup.get(fine, lookup.get(coarse, external)) for fine, coarse in zip(prefix6, prefix5)
    ], dtype="int32")

    # Use the order-weighted centroid of the residual area when a parent is split.
    xy = np.zeros((len(labels), 2))
    total = np.bincount(code_zone, weights=counts, minlength=len(labels))
    for index in range(len(labels) - 1):
        mask = (code_zone == index) & (counts > 0)
        xy[index] = (np.average(rides.coordinates[mask], weights=counts[mask], axis=0)
                     if np.any(mask) else geohash_center(labels[index]))
    xy[external] = np.nan
    return Zones(labels, xy, code_zone[rides.origin], code_zone[rides.destination], code_zone, total)


def od_by_date(rides, zones, dates):
    n = len(zones.labels)
    result = {}
    for date in dates:
        mask = rides.day == np.datetime64(date)
        keys = (rides.minute[mask] // STEP_MINUTES).astype("int32") * n * n
        keys += zones.origin[mask] * n + zones.destination[mask]
        result[date] = np.bincount(keys, minlength=SLOTS * n * n).reshape(SLOTS, n, n)
    return result


def type_average(profiles, day):
    peers = [values for date, values in profiles.items() if is_weekend(date) == is_weekend(day)]
    return np.mean(peers or list(profiles.values()), axis=0)


def forecast_models(profiles, dates):
    """Fit observed departure counts and reconcile them to the historical OD distribution."""
    training = {date: profiles[date] for date in TRAIN}
    hourly = {d: a.sum(axis=2).reshape(24, 4, -1).sum(axis=1) for d, a in training.items()}
    n = next(iter(hourly.values())).shape[1]
    zone = np.tile(np.arange(n), 24)
    hour = np.repeat(np.arange(24), n)

    def features(date, history):
        baseline = type_average(history, date)
        return np.column_stack((zone, hour, np.full(len(hour), is_weekend(date)),
                                np.log1p(baseline.ravel())))

    # The historical feature for each training day excludes that day's own label.
    x = np.concatenate([features(d, {k: arr for k, arr in hourly.items() if k != d}) for d in TRAIN])
    y = np.concatenate([hourly[d].ravel() for d in TRAIN])
    tree = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.07, max_iter=100, max_leaf_nodes=15,
        min_samples_leaf=28, l2_regularization=0.8, early_stopping=False,
        categorical_features=[0, 2], random_state=123,
    )
    results = {}
    # Keep the small count model serial so replay works in restricted Windows sessions.
    with threadpool_limits(limits=1):
        tree.fit(x, y)
        for day in dates:
            base = type_average(training, day)
            total = base.sum(axis=2).reshape(24, 4, n).sum(axis=1)
            fitted = tree.predict(features(day, hourly)).reshape(24, n)
            factor = np.divide(fitted, total, out=np.ones_like(fitted), where=(total > 0))
            tree_od = base * np.repeat(factor, 4, axis=0)[:, :, None]
            results[day] = {"historical": base, "poisson_tree": tree_od}
    return results


def forecast_scores(day, forecast, actual):
    def hourly(flows, axis):
        return flows.sum(axis=axis).reshape(24, 4, -1).sum(axis=1)
    predicted = hourly(forecast, 2)
    observed = hourly(actual, 2)
    # This auxiliary metric uses the start-time cohort's OD, not observed finish times.
    predicted_net = hourly(forecast, 1) - predicted
    actual_net = hourly(actual, 1) - observed
    error = np.abs(predicted - observed)
    return {
        "date": day, "observed_orders": int(observed.sum()), "forecast_orders": float(predicted.sum()),
        "mae": float(error.mean()), "wape": float(error.sum() / observed.sum()),
        "cumulative_od_net_mae": float(np.abs((predicted_net - actual_net).cumsum(axis=0)).mean()),
    }


def ride_lag(distance, speed=12.0, departure_offset=STEP_MINUTES / 2):
    """Convert straight-line distances to positive 15-minute travel lags."""
    minutes = np.maximum(3.0, distance * 1.3 * 60 / speed)
    return np.maximum(1, np.ceil((departure_offset + minutes) / STEP_MINUTES)).astype("int16")


def od_distances(rides, zones, training_days):
    n = len(zones.labels)
    train = np.isin(rides.day, np.array(training_days, dtype="datetime64[D]"))
    key = zones.origin[train] * n + zones.destination[train]
    count = np.bincount(key, minlength=n*n).reshape(n, n)
    sums = np.bincount(key, weights=rides.distances[train], minlength=n*n).reshape(n, n)
    fallback = haversine(zones.coordinates[:, None, :], zones.coordinates[None, :, :])
    fallback = np.nan_to_num(fallback, nan=float(np.median(rides.distances[train])))
    return np.divide(sums, count, out=fallback, where=count > 0)


def arrivals_from_od(forecast, distance, speed):
    arr = np.zeros(forecast.shape[:2])
    lags = ride_lag(distance, speed)
    for lag in np.unique(lags):
        if lag < SLOTS:
            arr[lag:] += (forecast[:-lag] * (lags == lag)).sum(axis=1)
    return arr


def opening_stock(rides, zones, day, coverage=1.0, speed=12.0):
    """Use the latest pre-day order per active bike as a noisy location proxy."""
    target = np.datetime64(day)
    mask = (rides.day < target) & (rides.day >= target - np.timedelta64(3, "D"))
    idx = np.flatnonzero(mask)
    delta = (rides.day[idx] - target).astype("timedelta64[D]").astype("int64")
    start_absolute = delta * 1440 + rides.minute[idx] + rides.second[idx] / 60
    sorted_indices = np.lexsort((-start_absolute, rides.bike[idx]))
    _, first = np.unique(rides.bike[idx][sorted_indices], return_index=True)
    chosen = sorted_indices[first]
    indices = idx[chosen]
    observed = np.bincount(zones.destination[indices], minlength=len(zones.labels))
    stock = np.rint(observed * coverage).astype("int32")
    # Remove the scaled share of recent riders whose simulated arrival is after midnight.
    travel = np.maximum(3.0, rides.distances[indices] * 1.3 * 60 / speed)
    arrival = start_absolute[chosen] + travel
    carry = [[] for _ in range(SLOTS)]
    for j in range(len(stock)):
        late = np.flatnonzero((arrival > 0) & (zones.destination[indices] == j))
        expected = min(int(stock[j]), int(round(len(late) * coverage)))
        if not expected:
            continue
        stock[j] -= expected
        # We carry counts, not exact bike identities, when coverage is varied.
        chosen_late = np.resize(late, expected)
        for ix in chosen_late:
            t = min(SLOTS - 1, int(np.ceil(arrival[ix] / STEP_MINUTES)))
            carry[t].append(j)
    return stock, carry
