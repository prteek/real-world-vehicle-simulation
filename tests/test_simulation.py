import numpy as np

from vehicle_sim.drive_cycle import drive_cycle, load
from vehicle_sim.simulation import CHARGING_EFFICIENCY, REGEN_MIN_SPEED, battery_power, simulate

CYCLE = drive_cycle(load())  # the first 3,830 s of driving


def test_small_battery_runs_out_and_a_bigger_one_gets_through():
    _, small = simulate(CYCLE, 70, 70 / 8)
    _, big = simulate(CYCLE, 120, 120 / 8)
    assert small["ran_out_at_hours"] is not None and 0 < small["electric_distance_km"] < small["logged_distance_km"]
    assert big["ran_out_at_hours"] is None and np.isclose(big["electric_distance_km"], big["logged_distance_km"])


def test_no_charger_means_no_charging():
    trace, s = simulate(CYCLE, 300, 0)
    assert s["charge_events"] == 0 and s["charge_energy_kwh"] == 0
    assert (trace["charge_time"] == 0).all()


def test_each_stop_charges_only_for_its_own_length():
    """The original code charged each stop for the cumulative charging time of all stops so far."""
    trace, _ = simulate(CYCLE, 120, 1, min_park_minutes=5)  # a slow charger
    charged = trace[trace["charge_time"] > 0]
    assert (charged["charge_time"] <= charged["stop_time"]).all()
    # where the battery is not full afterwards, it charged for the whole stop and no longer
    not_full = charged[charged["soc"] < 0.99]
    assert len(not_full) >= 3
    assert np.allclose(not_full["charge_time"], not_full["stop_time"])
    assert np.allclose(not_full["charge_energy"], 1000 * not_full["stop_time"] * CHARGING_EFFICIENCY)


def test_a_long_enough_stop_fills_the_battery():
    """Filling takes missing energy / (charger power x charging efficiency); the original code left out the efficiency,
    so a stop never got the battery past about 90% of the way to full."""
    trace, _ = simulate(CYCLE, 120, 15, initial_soc=0.5)
    overnight = trace[trace["stop_time"] > 5 * 3600]  # 9.7 hours at 15 kW fills 120 kWh
    assert len(overnight) == 1
    assert np.isclose(overnight["soc"].iloc[0], 1, atol=0.01)
    assert (overnight["charge_time"] < overnight["stop_time"]).all()  # and stops when full


def test_regenerative_braking_needs_speed_and_room_in_the_battery():
    assert battery_power(-10_000, -5_000, REGEN_MIN_SPEED + 1, 0.5) > 0
    assert battery_power(-10_000, -5_000, REGEN_MIN_SPEED - 1, 0.5) == 0  # too slow
    assert battery_power(-10_000, -5_000, REGEN_MIN_SPEED + 1, 1) == 0  # full
    assert battery_power(20_000, 0, 10, 0.5) < 0  # propulsion draws from the battery
    assert battery_power(20_000, 0, 10, 0) == 0  # not from an empty one
