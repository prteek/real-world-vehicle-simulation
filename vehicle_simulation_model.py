# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Real world vehicle drigins simulation of electric vehicle

import os

os.system('pip install -U -r ßrequirements.txt')

import time as TT
import numpy as np
import pandas as pd
from brake_power_determination import *
from utility_functions import *
from vehicle_controller_models import simple_vehicle_controller
from electric_machine_models import constant_parameter_electric_machine
from battery_models import constant_parameter_battery


t = TT.time()

column_names = [
    "coefficients",
    "scaled_coefficients",
    "train_rmse",
    "train_gof",
    "test_rmse",
    "test_gof",
    "logged_distance",
    "logged_driving_time",
    "logged_total_time",
    "logged_fuel_consumption",
    "logged_propulsion_energy",
    "logged_brake_energy",
    "logged_start_time_of_first_trip",
    "logged_end_time_of_last_trip",
    "logged_number_of_trips",
    "ev_distance",
    "ev_driving_time",
    "ev_total_time",
    "ev_opportunistic_charge_total_events",
    "ev_opportunistic_charge_time",
    "ev_opportunistic_charge_energy",
    "ev_opportunistic_potential_charge_events",
    "ev_opportunistic_potential_charge_time",
    "ev_battery_propulsion_energy",
    "ev_battery_regen_energy",
    "ev_battery_max_propulsion_power",
    "ev_battery_max_regen_power",
    "ev_min_soc",
    "ev_average_soc",
    "ev_end_of_day_soc",
    "ev_fuel_saved",
    "avit_name",
    "file_name",
]

results = pd.DataFrame(columns=column_names)


data = pd.read_csv("./datalog.csv")
timestamp = np.array(data["timestamp"])

(
    dt,
    time,
    speed_meters_per_second,
    acceleration,
    power_at_wheels,
    mass,
    grade,
    transmission_output_speed,
    overall_ratio,
    engine_speed,
    accelerator_pedal,
) = get_required_signals(data)

# -------------------------------------------------------------------------------------- #

# Prepare data for training the ML model
x_model_input, y_to_fit, _, _ = prepare_data_for_ml_model_training(data)

# Train and evaluate the ML model
(
    rl_model,
    coefficients,
    scaled_coefficients,
    train_rmse,
    train_gof,
    test_rmse,
    test_gof,
) = train_road_load_model(x_model_input, y_to_fit)

# Calculating brake power using the ML model predictions
brake_power = calculate_brake_power(rl_model, data)

# --------------------------------------------------------------------------------- #

# Initialise system models and parametrize them
vsc = simple_vehicle_controller()

electric_machine = constant_parameter_electric_machine()
electric_machine.generating_electrical_power_limit = -np.Inf
electric_machine.generating_mechanical_power_limit = -np.Inf

battery = constant_parameter_battery(
    initial_soc=1
)  # 100% charged at the start of the day
battery.capacity = 70 * 1000 * 3600  # [J] 70 kWh

opportunistic_charging_threshold = 1800  # [sec]

# ---------------------------------------------------------------------------------- #

# Prepate inputs for the system model
wheel_speed = speed_meters_per_second / vsc.wheel_radius
wheel_torque = power_at_wheels / (wheel_speed + np.finfo(float).eps)  # safe divide
brake_torque = brake_power / (wheel_speed + np.finfo(float).eps)  # safe divide

# Properties of driving day
current_file_name = "datalog.csv"
avit_name = "1234"
logged_distance = np.trapz(speed_meters_per_second, time)
logged_driving_time = time[-1]
logged_total_time = timestamp[-1] - timestamp[0]
logged_fuel_consumption = np.trapz(data["Fuel_Rate"], time) / 3600 / 1000  # [m^3]
logged_propulsion_energy = np.trapz(
    power_at_wheels * ((power_at_wheels + brake_power) > 0), time
)
logged_brake_energy = np.trapz(brake_power, time)
logged_start_time_of_first_trip = TT.strftime("%H:%M:%S", TT.gmtime(timestamp[0]))
logged_end_time_of_last_trip = TT.strftime("%H:%M:%S", TT.gmtime(timestamp[-1]))
logged_number_of_trips = (
    np.sum((np.diff(data["Ign_Voltage"] < 2) > 0)) - 1
) / 2 + 1  # pair wise on-off and 2 off for 1st trip

# ------------------------------------------------------------------------------------ #

# Signals to log
battery_soc = np.zeros(len(time))
electric_machine_delivered_power = np.zeros(len(time))
battery_current = np.zeros(len(time))
battery_voltage = np.zeros(len(time))
vsc_mode = np.zeros(len(time))

# Initial values and variables for simulation
time_previous = timestamp[0]
plug_in_charge_power = (
    battery.capacity
    * (battery.maximum_battery_soc - battery.minimum_battery_soc)
    / battery.plug_full_charge_time
)
plug_in_charge_current = plug_in_charge_power / battery.plug_charge_voltage
ev_opportunistic_charge_total_events = 0
ev_opportunistic_charge_time = 0
ev_opportunistic_charge_energy = 0
ev_opportunistic_potential_charge_time = 0
ev_opportunistic_potential_charge_events = 0

# -------------------------------------------------------------------------------------- #

# second by second simulation to calculate soc

for i in range(len(timestamp)):

    if (timestamp[i] - time_previous) > opportunistic_charging_threshold:
        ev_opportunistic_potential_charge_time = (
            ev_opportunistic_potential_charge_time + (timestamp[i] - time_previous)
        )
        ev_opportunistic_potential_charge_events += 1

        # Opportunistic charging
        battery_capacity_buffer = battery.capacity * (
            battery.maximum_battery_soc - battery.soc
        )
        battery_charging_time_valid = min(
            battery_capacity_buffer / plug_in_charge_power,
            (timestamp[i] - time_previous),
        )
        ev_opportunistic_charge_time = (
            ev_opportunistic_charge_time + battery_charging_time_valid
        )
        ev_opportunistic_charge_energy = (
            ev_opportunistic_charge_energy
            + plug_in_charge_power * battery_charging_time_valid
        )
        ev_opportunistic_charge_total_events += 1 * (ev_opportunistic_charge_time > 0)
        battery._calculate_soc(plug_in_charge_current, ev_opportunistic_charge_time)

    else:
        pass

    # normal operation
    vsc.calculate_electric_machine_power_demand(
        wheel_torque[i], brake_torque[i], wheel_speed[i], battery, electric_machine
    )
    electric_machine.calculate_battery_current_demand(
        vsc.electric_machine_power_demand, battery
    )
    battery.update_battery_state(
        electric_machine.battery_current_demand, electric_machine, dt
    )

    battery_soc[i] = battery.soc
    electric_machine_delivered_power[i] = electric_machine.delivered_power
    battery_current[i] = battery.current
    battery_voltage[i] = battery.voltage
    vsc_mode[i] = vsc.mode
    time_previous = timestamp[i]
    ev_total_time = timestamp[i] - timestamp[0]

    if battery.soc <= 0:
        break

# ------------------------------------------------------------------------------------- #

# Post processing simulation results

ev_distance = np.trapz(speed_meters_per_second * (battery_soc > 0), time)
ev_driving_time = np.sum((battery_soc > 0)) * dt  # active trip time
ev_battery_propulsion_energy = np.trapz(
    battery_current * battery_voltage * (vsc_mode == 1), time
)
ev_battery_regen_energy = np.trapz(
    battery_current * battery_voltage * (vsc_mode == -1), time
)
ev_battery_max_propulsion_power = np.min(
    battery_current * battery_voltage * (vsc_mode == 1)
)
ev_battery_max_regen_power = np.max(
    battery_current * battery_voltage * (vsc_mode == -1)
)
ev_end_of_day_soc = max(battery.soc, 0)
ev_min_soc = np.min(battery_soc)
ev_average_soc = np.mean(battery_soc[battery_soc > 0])
ev_fuel_saved = (
    np.trapz(data["Fuel_Rate"] * (battery_soc > 0), time) / 3600 / 1000
)  # [m^3/sec]

file_results = pd.DataFrame(
    [
        [
            coefficients,
            scaled_coefficients,
            train_rmse,
            train_gof,
            test_rmse,
            test_gof,
            logged_distance,
            logged_driving_time,
            logged_total_time,
            logged_fuel_consumption,
            logged_propulsion_energy,
            logged_brake_energy,
            logged_start_time_of_first_trip,
            logged_end_time_of_last_trip,
            logged_number_of_trips,
            ev_distance,
            ev_driving_time,
            ev_total_time,
            ev_opportunistic_charge_total_events,
            ev_opportunistic_charge_time,
            ev_opportunistic_charge_energy,
            ev_opportunistic_potential_charge_events,
            ev_opportunistic_potential_charge_time,
            ev_battery_propulsion_energy,
            ev_battery_regen_energy,
            ev_battery_max_propulsion_power,
            ev_battery_max_regen_power,
            ev_min_soc,
            ev_average_soc,
            ev_end_of_day_soc,
            ev_fuel_saved,
            avit_name,
            current_file_name,
        ]
    ],
    columns=column_names,
)

results = results.append(file_results, ignore_index=True)

# --------------------------------  End of simulation ------------------------------------ #


results.to_csv("ev_requirements.csv")

elapsed_time = TT.time() - t
print(TT.strftime("%H:%M:%S", TT.gmtime(elapsed_time)))
