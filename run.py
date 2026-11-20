"""Reproduce the Problem C results from the original CSV."""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bike_share.demand import (STEP_MINUTES, TEST, TRAIN, VALIDATE, coverage_audit, forecast_models,
                              forecast_scores, make_zones, od_by_date, od_distances, read_rides)
from bike_share.fences import plan_pilots
from bike_share.operations import DayModel, Settings

SOURCE_NAME = "2026_CQU_MCM_Problem_C_DATA.csv"

def guess_source():
    project_data = HERE / "data" / SOURCE_NAME
    if project_data.is_file():
        return project_data
    return next(HERE.parent.glob(f"*/{SOURCE_NAME}"), None)


def save(frame, out, name):
    frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")


def model_tables(out, rides, zones, od, predicted, metrics, schedules, slots,
                 fences, fence_evaluation, assignment, selected, settings):
    save(pd.DataFrame({
        "zone": zones.labels, "latitude": zones.coordinates[:, 0], "longitude": zones.coordinates[:, 1],
        "training_endpoints": zones.trips.astype(int),
    }), out, "zones")
    score_table = pd.DataFrame(metrics)
    save(score_table, out, "policy_metrics")
    sensitivity = score_table[score_table["scenario"].str.startswith("sensitivity_")]
    save(sensitivity, out, "sensitivity")
    route_columns = ["date", "truck", "dispatch_minute", "truck_start", "pickup", "dropoff",
                     "bikes", "pickup_minute", "arrival_minute", "empty_km", "loaded_km",
                     "estimated_km", "scenario"]
    routes = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame(columns=route_columns)
    save(routes, out, "truck_schedule")
    save(pd.concat(slots, ignore_index=True), out, "slot_metrics")
    save(fences, out, "fence_sites")
    save(fence_evaluation, out, "fence_evaluation")
    save(assignment, out, "fence_assignments")

    mean_flows = np.mean([od[d] for d in TRAIN], axis=0).sum(axis=0)
    pairs = np.argwhere(mean_flows > 0)
    # Include all retained pairs, including transfers to/from the external aggregate.
    save(pd.DataFrame({
        "origin": [zones.labels[i] for i in pairs[:, 0]],
        "destination": [zones.labels[i] for i in pairs[:, 1]],
        "mean_orders_per_training_day": mean_flows[pairs[:, 0], pairs[:, 1]],
    }).sort_values("mean_orders_per_training_day", ascending=False), out, "od_summary")

    rows = []
    for date in TEST:
        for t in range(len(od[date])):
            observed = od[date][t].sum(axis=1)
            forecast = predicted[date][selected][t].sum(axis=1)
            for i, label in enumerate(zones.labels):
                rows.append((date, t*STEP_MINUTES, label, observed[i], forecast[i]))
    save(pd.DataFrame(rows, columns=["date", "minute", "zone", "observed_departures", "forecast_departures"]),
         out, "departure_forecasts")
    report = {
        "source_orders": len(rides.day),
        "core_zones": len(zones.labels)-1,
        "training_dates": list(TRAIN),
        "validation_date": VALIDATE,
        "test_dates": list(TEST),
        "retained_training_endpoint_share": float(zones.trips[:-1].sum()/zones.trips.sum()),
        "selected_forecast": selected,
        "solver": {"dispatch": "max flow/min-cost + greedy truck assignment",
                   "fence": "scipy.optimize.milp / HiGHS"},
        "settings": settings.__dict__,
        "analysis_scope": {"dispatch": "mixed-resolution city zones, no dispatch to/from EXTERNAL",
                           "geofences": "three high-volume six-character pilot cells"},
        "observational_limits": ["completed rides only", "no measured ride end time or stock",
                                 "no observed incentive response", "fence site points have no field survey"],
    }
    (out / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_results(out, od, predictions, chosen, fences, assignments, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    palette = ["#284b63", "#329589", "#e07a51"]
    for (day, color) in zip(TEST, palette):
        actual = od[day].sum(axis=(1, 2)).reshape(24, 4).sum(axis=1)
        fit = predictions[day][chosen].sum(axis=(1, 2)).reshape(24, 4).sum(axis=1)
        axes[0].plot(np.arange(24), actual, label=day[5:] + " observed", color=color, linewidth=2)
        axes[0].plot(np.arange(24), fit, linestyle=":", linewidth=1.5, color=color)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set(xlabel="Hour (dotted = forecast)", ylabel="Ride starts", title="Held-out demand profiles")
    labels = ["no_dispatch", "greedy", "optimized", "optimized_dynamic"]
    x = np.arange(len(labels))
    table = pd.DataFrame(metrics)
    for index, (day, color) in enumerate(zip(TEST, palette)):
        sample = table[table["date"] == day].set_index("scenario").loc[labels]
        axes[1].bar(x+(index-1)*0.24, sample["simulated_missed_rate"]*100, width=0.22,
                    color=color, label=day[5:])
    axes[1].set_xticks(x, ["No dispatch", "Greedy", "Optimized", "Opt. + rewards"], rotation=15)
    axes[1].set(ylabel="Simulated unserved (%)", title="Same requests and initial stock")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(out / "demand_and_policies.png", dpi=180)
    plt.close(fig)

    pilots = list(fences["pilot_zone"].unique())
    fig, axes = plt.subplots(1, len(pilots), figsize=(4.05*len(pilots), 4.1),
                             squeeze=False, layout="constrained")
    for ax, pilot in zip(axes[0], pilots):
        points = assignments[(assignments["pilot_zone"] == pilot) & (assignments["layout"] == "optimized")]
        fit = fences[(fences["pilot_zone"] == pilot) & (fences["layout"] == "optimized")]
        covered = points["assigned_to"] != ""
        ax.scatter(points["longitude"], points["latitude"], c=np.where(covered, "#90c9c3", "#c8c6c2"),
                   s=10 + 8*np.sqrt(points["training_mean_daily_returns"]), edgecolors="white", linewidth=0.6)
        ax.scatter(fit["longitude"], fit["latitude"], marker="D", s=20 + 0.35*fit["slots"],
                   color="#1f526a", edgecolor="white", linewidth=0.6)
        for row in fit.itertuples():
            ax.annotate(str(row.slots), (row.longitude, row.latitude), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
        ax.set(title=f"Pilot {pilot}", xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect(1/np.cos(np.radians(points["latitude"].mean())))
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.locator_params(nbins=4)
    fig.suptitle("Proposed geofence cells (labels: parking spaces)")
    fig.savefig(out / "fence_pilots.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=guess_source())
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.data is None or not args.data.is_file():
        parser.error("pass the contest CSV with --data")
    args.out.mkdir(parents=True, exist_ok=True)

    print("Reading order data and auditing hours...", flush=True)
    rides = read_rides(args.data)
    audit = coverage_audit(rides)
    audit.to_csv(args.out / "data_audit.csv", index=False, encoding="utf-8-sig")
    dates = (*TRAIN, VALIDATE, *TEST)
    included = audit[audit["date"].isin(dates)]
    if len(included) != len(dates) or not (included["hours_with_orders"] == 24).all():
        raise ValueError("The configured train/validation/test dates must all contain 24 observed hours.")

    print("Fitting zone system, OD profiles and count models...", flush=True)
    zones = make_zones(rides)
    od = od_by_date(rides, zones, dates)
    predicted = forecast_models(od, (VALIDATE, *TEST))
    scores = pd.DataFrame([
        {"model": model, **forecast_scores(date, fit, od[date])}
        for date, models in predicted.items() for model, fit in models.items()
    ])
    validation = scores[scores["date"] == VALIDATE]
    selected = validation.sort_values(["cumulative_od_net_mae", "wape"]).iloc[0]["model"]
    scores["selected"] = scores["model"] == selected
    save(scores, args.out, "forecast_metrics")
    distance = od_distances(rides, zones, TRAIN)

    print(f"Replaying dispatch and reward policies with {selected} forecasts...", flush=True)
    settings = Settings()
    metrics, schedules, slots = [], [], []
    primary = {
        "no_dispatch": {},
        "greedy": {"dispatch": "greedy"},
        "optimized": {"dispatch": "optimized"},
        "optimized_fixed": {"dispatch": "optimized", "incentive": "fixed"},
        "optimized_dynamic": {"dispatch": "optimized", "incentive": "dynamic"},
        "dynamic_low": {"dispatch": "optimized", "incentive": "dynamic", "response": "low"},
        "dynamic_high": {"dispatch": "optimized", "incentive": "dynamic", "response": "high"},
    }
    for day in TEST:
        day_model = DayModel(rides, zones, day, predicted[day][selected], distance, settings)
        cases = dict(primary)
        if day == TEST[0]:
            for key, variant in [("stock_015", {"stock_factor": 0.15}),
                                 ("stock_040", {"stock_factor": 0.40}),
                                 ("stock_075", {"stock_factor": 0.75}),
                                 ("stock_100", {"stock_factor": 1.00}),
                                 ("speed_09", {"speed": 9.0}), ("speed_16", {"speed": 16.0})]:
                cases["sensitivity_" + key + "_none"] = variant
                cases["sensitivity_" + key + "_optimized"] = {"dispatch": "optimized", **variant}
        for name, params in cases.items():
            result, route, series = day_model.run(**params)
            metrics.append({"scenario": name, **result})
            if not route.empty:
                schedules.append(route.assign(scenario=name))
            slots.append(series.assign(date=day, scenario=name))
        print(f"  {day}: {len(cases)} scenarios, conservation checked at each time step", flush=True)

    print("Sizing pilot-area geofences and evaluating platform competition...", flush=True)
    fences, fence_scores, assignments = plan_pilots(rides, zones)
    model_tables(args.out, rides, zones, od, predicted, metrics, schedules, slots, fences,
                 fence_scores, assignments, selected, settings)
    plot_results(args.out, od, predicted, selected, fences, assignments, metrics)
    print(f"Results written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
