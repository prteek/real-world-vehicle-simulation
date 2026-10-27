# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Real world vehicle origins simulation of electric vehicle
#
# Replays a logged drive cycle second by second as if the vehicle were electric. The demand at the wheels goes through
# the driveline and the electric machine to the battery: propulsion draws from it, regenerative braking feeds it, and
# its state of charge follows. Whenever the vehicle was parked for long enough between two log entries, it is plugged
# in and charged at the charger's power until full or until it drives off. Power limits are not modelled.

import numpy as np
import pandas as pd

DRIVELINE_EFFICIENCY = 0.98
MACHINE_EFFICIENCY = 0.8  # motoring and generating
CHARGING_EFFICIENCY = 0.9  # battery, charging (from the machine or the plug)
DISCHARGING_EFFICIENCY = 0.9  # battery, discharging
REGEN_MIN_SPEED = 10 / 3.6  # [m/s] no regenerative braking below this
TERMINAL_VOLTAGE = 200  # [V] battery current = power / voltage


def battery_power(wheel_power, brake_power, speed, soc):
    """Electrical power at the battery terminals [W] (negative: discharging) for one second's demand at the wheels.

    Propulsion when the driver asks for net positive power; regenerative braking of the coasting and braking power when
    the driver is slowing down, the vehicle is fast enough and the battery is not full.
    """
    total = wheel_power + brake_power
    if total > 0 and soc > 0:
        return -total / DRIVELINE_EFFICIENCY / MACHINE_EFFICIENCY
    if total < 0 and speed > REGEN_MIN_SPEED and soc < 1:
        return -(min(wheel_power, 0) + brake_power) * DRIVELINE_EFFICIENCY * MACHINE_EFFICIENCY
    return 0.0


def stored(power):
    """Change in stored energy for a power at the terminals: the battery loses some on the way in and out."""
    return power * CHARGING_EFFICIENCY if power >= 0 else power / DISCHARGING_EFFICIENCY


def simulate(
    cycle: pd.DataFrame,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    min_park_minutes: float = 30,
    initial_soc: float = 1,
) -> tuple[pd.DataFrame, dict]:
    """Simulate the battery over `cycle` (from `drive_cycle.drive_cycle`).

    Returns a second-by-second trace (timestamp, drive_time, speed, soc, battery_power [W, into the battery], and for
    a row that follows a stop long enough to plug in: stop_time, charge_time [s] and charge_energy [J]) and a summary.
    The simulation stops when the battery is empty; the rest of the trace is left at 0.
    """
    capacity = battery_capacity_kwh * 3.6e6  # [J]
    charger = charger_power_kw * 1000  # [W]
    timestamp = cycle["timestamp"].to_numpy()
    speed = cycle["speed"].to_numpy()
    wheel_power = cycle["power_at_wheels"].to_numpy() * (speed > 0)  # no tractive power at standstill
    brake_power = cycle["brake_power"].to_numpy() * (speed > 0)
    dt = cycle["drive_time"].iloc[1] - cycle["drive_time"].iloc[0]

    n = len(cycle)
    soc = np.zeros(n)
    power = np.zeros(n)
    stop_time = np.zeros(n)  # length of a stop long enough to plug in, just before this second [s]
    charge_time = np.zeros(n)  # how long the vehicle charged at that stop [s]
    charge_energy = np.zeros(n)  # energy put into the battery at that stop [J]

    level = initial_soc  # charged overnight before the first trip
    for i in range(n):
        parked = timestamp[i] - timestamp[i - 1] if i else 0
        if parked > min_park_minutes * 60 and charger > 0:
            stop_time[i] = parked
            # charge until full or until the vehicle drives off, whichever comes first
            charge_time[i] = min((1 - level) * capacity / (charger * CHARGING_EFFICIENCY), parked)
            charge_energy[i] = charger * CHARGING_EFFICIENCY * charge_time[i]
            level = min(level + charge_energy[i] / capacity, 1)

        power[i] = battery_power(wheel_power[i], brake_power[i], speed[i], level)
        level = min(level + stored(power[i]) * dt / capacity, 1)
        soc[i] = level
        if level <= 0:
            break

    trace = pd.DataFrame({
        "timestamp": timestamp, "drive_time": cycle["drive_time"].to_numpy(), "speed": speed, "soc": soc,
        "battery_power": power, "stop_time": stop_time, "charge_time": charge_time, "charge_energy": charge_energy,
    })
    on_battery = soc > 0
    drive_time = trace["drive_time"]
    stops = stop_time > 0
    summary = {
        "logged_distance_km": np.trapezoid(speed, drive_time) / 1000,
        "electric_distance_km": np.trapezoid(speed * on_battery, drive_time) / 1000,
        "ran_out_at_hours": (timestamp[np.argmin(on_battery)] - timestamp[0]) / 3600 if not on_battery.all() else None,
        "end_soc": max(level, 0),
        "min_soc": max(soc.min(), 0),
        "charge_events": int((charge_time > 0).sum()),
        "potential_charge_events": int(stops.sum()),
        "charge_hours": charge_time.sum() / 3600,
        "charge_energy_kwh": charge_energy.sum() / 3.6e6,
        "propulsion_energy_kwh": -np.trapezoid(np.minimum(power, 0), drive_time) / 3.6e6,
        "regen_energy_kwh": np.trapezoid(np.maximum(power, 0), drive_time) / 3.6e6,
        "max_propulsion_power_kw": -power.min() / 1000,
    }
    return trace, summary

