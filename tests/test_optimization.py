import unittest

import numpy as np

from bike_share.fences import design_layout
from bike_share.operations import Settings
from bike_share.rolling import recent_order_scale, solve_dispatch


class RollingDispatchTests(unittest.TestCase):
    def setUp(self):
        self.distance = np.array([[0, 1, np.inf], [1, 0, np.inf], [np.inf, np.inf, np.inf]])
        self.stock = np.array([10, 0, 0])
        self.cfg = Settings()
        self.departures = np.zeros((3, 3))
        self.returns = np.zeros((3, 3))

    def solve(self):
        return solve_dispatch(self.distance, self.cfg, 10, self.stock, self.departures,
                              self.returns, np.zeros_like(self.departures),
                              {0: [0]}, 1.0, 0.50)

    def test_future_orders_do_not_change_current_scale(self):
        actual = np.array([10, 20, 30, 4000])
        expected = np.array([12, 19, 40, 0])
        self.assertEqual(recent_order_scale(actual, expected, 3),
                         recent_order_scale(actual[:3], expected[:3], 3))

    def test_late_delivery_does_not_recover_earlier_lost_orders(self):
        self.departures[0, 1] = 8
        self.returns[0, 1] = 20  # stock at decision time already reflects arrivals in this slot
        jobs, _ = self.solve()
        self.assertEqual(jobs, [])

    def test_timed_assignment_can_avoid_a_later_shortage(self):
        self.departures[2, 1] = 8
        jobs, details = self.solve()
        self.assertEqual(len(jobs), 1)
        trip, trucks_used = jobs[0]
        self.assertEqual((trip.start, trip.pickup, trip.dropoff, trucks_used), (0, 0, 1, 1))
        self.assertLessEqual(trip.arrival_slot, 12)
        self.assertLessEqual(trip.quantity, 10)
        self.assertLess(details["rolling_objective"], details["rolling_baseline_shortage"])


class FenceLayoutTests(unittest.TestCase):
    def test_weekend_risk_can_change_location_at_same_budget(self):
        distance = np.array([[0, 0.5], [0.5, 0]])
        weekday = np.array([[20.0, 0.0], [20.0, 0.0]])
        weekend = np.array([[0.0, 100.0], [0.0, 100.0]])
        sites = np.arange(2)
        weight = np.ones(2)
        ordinary = design_layout(distance, weekday, weight, sites, fixed_resources=(1, 30))
        mixed = design_layout(distance, np.stack([weekday, weekend]), weight, sites,
                              fixed_resources=(1, 30), worst_weight=0.35)
        self.assertEqual(ordinary.active.tolist(), [0])
        self.assertEqual(mixed.active.tolist(), [1])
        self.assertEqual(mixed.capacity.tolist(), [30])


if __name__ == "__main__":
    unittest.main()
