# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : The drive cycle from the logged data: speed, power at the wheels and brake power
#
# Brake power is not logged: it is estimated as what is left of the power at the wheels after road load (rolling
# resistance and aerodynamic drag), grade and inertia, whenever the driver is off the accelerator and slowing down.

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data.csv"  # not in git: downloaded from the dataset when missing
DATASET = "prteek/drive-cycle"

G = 9.8  # [m/s^2]
SPEED_FILTER_WINDOW = 10  # [samples] moving average before differentiating speed
POWER_FILTER_TC = 2  # [samples] first-order filter on power at the wheels (engine torque is much more dynamic)
# Road load [W] = ROLLING * mass * g * cos(grade) * v + AERO * v^3, fitted by least squares on the moments the truck
# was driving rather than braking, over the whole 428 km log it came from (R^2 0.74 on a held-out 30%). The published
# data is only the first 3,830 s of driving, which is too uniform in speed to separate the two terms.
ROLLING = 0.011932077183171142
AERO = 2.7376895505031347  # [kg/m]


def load_from_hub() -> pd.DataFrame:
    """The logged driving data from the Hugging Face dataset (what the app uses)."""
    from huggingface_hub import hf_hub_download

    return pd.read_csv(hf_hub_download(DATASET, "data.csv", repo_type="dataset"))


def load(path=DATA) -> pd.DataFrame:
    """The logged driving data: one row per second of driving, `time_s` in seconds since the start of the log.

    Read from `path` if it exists (a local copy for publishing the dataset), otherwise from the Hugging Face dataset.
    """
    return pd.read_csv(path) if Path(path).exists() else load_from_hub()


def first_order_filter(x, tc):
    gain = 1 - np.exp(-1 / tc)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) * gain
    return out


def drive_cycle(data: pd.DataFrame) -> pd.DataFrame:
    """Everything the simulation needs from the log, one row per second of driving.

    Columns: timestamp (seconds since the start of the log, with gaps where the vehicle was parked), drive_time
    (seconds of driving), speed [m/s], power_at_wheels [W] and brake_power [W, <= 0].
    """
    kph = data["speed_kph"].to_numpy(float)
    kph = kph * (kph < 250)  # 255: no reading
    speed = kph / 3.6
    dt = data["time_s"].iloc[1] - data["time_s"].iloc[0]
    smooth_speed = np.convolve(kph, np.ones(SPEED_FILTER_WINDOW) / SPEED_FILTER_WINDOW, mode="same") * (kph != 0) / 3.6
    acceleration = np.append(np.diff(smooth_speed) / dt, 0)
    raw_power = (data["engine_torque_nm"] * data["overall_gear_ratio"]
                 * data["transmission_output_speed_rpm"] / 30 * np.pi).to_numpy(float)
    power_at_wheels = first_order_filter(raw_power, POWER_FILTER_TC) * (raw_power != 0)

    mass = data["vehicle_mass_kg"].to_numpy(float)
    grade = data["road_grade_pct"].to_numpy(float) / 100
    road_load = ROLLING * mass * G * np.cos(grade) * speed + AERO * speed**3
    gradient = mass * G * np.sin(grade) * speed
    inertia = mass * acceleration * speed
    brake = road_load + gradient + inertia - power_at_wheels  # what the brakes must have absorbed
    # only with the accelerator released, absorbing energy, while the driver means to slow down
    off_pedal = data["accelerator_pedal_pct"].to_numpy(float) == 0
    brake = brake * ((power_at_wheels + brake) <= 0) * off_pedal * (brake <= 0)

    return pd.DataFrame({
        "timestamp": data["time_s"].to_numpy(float),
        "drive_time": np.arange(len(data)) * dt,
        "speed": speed,
        "power_at_wheels": power_at_wheels,
        "brake_power": brake,
    })

