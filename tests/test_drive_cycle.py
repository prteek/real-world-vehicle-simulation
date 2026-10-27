import numpy as np

from vehicle_sim.drive_cycle import drive_cycle, first_order_filter, load


def test_drive_cycle_is_one_row_per_second_of_driving():
    cycle = drive_cycle(load())
    assert len(cycle) == 3831 and cycle["drive_time"].iloc[-1] == 3830
    assert (cycle["timestamp"].diff().dropna() >= 1).all()


def test_published_data_is_only_the_stretch_the_app_shows():
    data = load()
    assert len(data) == 3831 and data["time_s"].is_monotonic_increasing
    assert list(data.columns) == [
        "time_s", "speed_kph", "engine_torque_nm", "overall_gear_ratio", "transmission_output_speed_rpm",
        "accelerator_pedal_pct", "vehicle_mass_kg", "road_grade_pct",
    ]


def test_brake_power_only_absorbs_energy():
    assert (drive_cycle(load())["brake_power"] <= 0).all()


def test_first_order_filter_follows_a_step():
    out = first_order_filter(np.r_[0.0, np.ones(20)], tc=2)
    assert out[0] == 0 and 0 < out[1] < out[5] < out[-1] < 1
    assert np.isclose(out[1], 1 - np.exp(-1 / 2))
