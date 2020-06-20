# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Brake power estimation library

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def get_required_signals(data):
    speed_dl = np.array(data["KPH_DL"], dtype="float")
    speed = speed_dl * (speed_dl < 250)

    time = data["time"]
    engine_torque = np.array(
        data["Net_Teng"], dtype="float"
    )  # Use net torque for positive and negative values
    overall_ratio = np.array(data["OverallTorqueRatio"], dtype="float")
    engine_speed = np.array(data["Engine_Speed"], dtype="float")
    transmission_output_speed = np.array(data["Output_Speed"], dtype="float")
    accelerator_pedal = data["Pedal%_DL"] / 100
    # Todo: Check if mass is in pounds -> is in kgs: Andrej
    mass = data["Vehicle_Mass"]
    grade = data["%GradeAvg_new"] / 100

    ## Calculate additional signals
    power_at_wheels_raw = (
        engine_torque * overall_ratio * transmission_output_speed / 30 * np.pi
    )

    # Filter some signals to make them appropriate for training
    moving_average_window = 10
    low_pass_tc = 2

    # Calculate acceleration on filtered speed
    dt = time[1] - time[0]
    time = np.arange(len(time)) * dt
    filter_mov_mean = np.ones(moving_average_window) / moving_average_window
    speed_meters_per_second_filtered = (
        np.convolve(speed, filter_mov_mean, mode="same") * (speed != 0) / 3.6
    )
    acceleration = np.append(np.diff(speed_meters_per_second_filtered) / dt, 0)

    speed_meters_per_second = (
        speed / 3.6
    )  # if not converted to float already, use /3.6 since speed is uint8

    # Filter power at wheels since it is calculated from engine torque which has much higher dynamics
    pos_power_at_wheels = fo_lp_filter(power_at_wheels_raw, low_pass_tc) * (
        power_at_wheels_raw > 0
    )
    neg_power_at_wheels = fo_lp_filter(power_at_wheels_raw, low_pass_tc) * (
        power_at_wheels_raw < 0
    )

    power_at_wheels = pos_power_at_wheels + neg_power_at_wheels

    return (
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
    )


def fo_lp_filter(data, tc=1):
    """first order filter"""
    samples = len(data)
    previous_actual = data[0]
    filtered_data = data[0]
    for i in range(samples - 1):
        target_value = data[i + 1]
        actual_value = previous_actual + (target_value - previous_actual) * (
            1 - np.exp(-1 / tc)
        )
        previous_actual = actual_value
        filtered_data = np.append(filtered_data, actual_value)

    return filtered_data


def transform_data(X, model="lin_reg"):
    """transform input data and generate input and output that will feed into machine learning model,
    for either training or prediction"""

    speed_meters_per_second = X[:, 0]
    acceleration = X[:, 1]
    power_at_wheels = X[:, 2]
    mass = X[:, 3]
    grade = X[:, 4]

    aero_power = speed_meters_per_second ** 3
    rolling_resistance_power = mass * 9.8 * np.cos(grade) * speed_meters_per_second
    gradient_power = mass * 9.8 * np.sin(grade) * speed_meters_per_second

    # Todo: Add a speed dependent resistance term -> Does not has much influence
    #         other_resistance_power   = speed_meters_per_second**2

    # Todo: Correct for rotating inertia -> Does not has much influence
    inertia_power = mass * acceleration * speed_meters_per_second
    #         weight_correction        = mass*(overall_ratio**2)*acceleration*speed_meters_per_second

    if model == "lin_reg":
        power_to_fit = power_at_wheels - inertia_power - gradient_power
        x_values = np.c_[rolling_resistance_power, aero_power]

    else:
        power_to_fit = power_at_wheels - inertia_power
        x_values = np.c_[speed_meters_per_second, mass, grade]

    return x_values, power_to_fit, inertia_power, gradient_power


def prepare_data_for_ml_model_training(data):
    """Prepare input data for linear regression model RL power = a*v + b*v^3 for only propulsion part,
    where brakes should possibly not be active and powertrain power = RL power + grade power + inertia power"""
    (
        _,
        _,
        speed_meters_per_second,
        acceleration,
        power_at_wheels,
        mass,
        grade,
        transmission_output_speed,
        overall_ratio,
        engine_speed,
        _,
    ) = get_required_signals(data)

    X = np.c_[speed_meters_per_second, acceleration, power_at_wheels, mass, grade]
    slip = (
        transmission_output_speed * overall_ratio / (engine_speed + np.finfo(float).eps)
    )  # safe divide

    positive_power_index = np.where(
        (acceleration >= 0) & (speed_meters_per_second > 0) & (slip <= 1.0)
    )[0]
    X_positive_power = X[positive_power_index, :]
    x_model_input, y_to_fit, inertia_power, gradient_power = transform_data(
        X_positive_power, model="lin_reg"
    )

    return x_model_input, y_to_fit, inertia_power, gradient_power


def train_road_load_model(x_model_input, y_to_fit):
    """Train and evaluate performace of linear regression Road load model on train-test data"""

    x_train, x_test, y_train, y_test = train_test_split(
        x_model_input, y_to_fit, test_size=0.30
    )
    lin_reg = LinearRegression(
        fit_intercept=False
    )  # set this to false as we don't want any constant term to be fit
    lin_reg.fit(x_train, y_train)

    model = lin_reg
    coefficients = lin_reg.coef_
    scaled_coefficients = coefficients * np.std(x_train, axis=0)

    train_predict = lin_reg.predict(x_train)
    train_rmse = np.round(np.sqrt(mean_squared_error(train_predict, y_train)), 0)
    train_gof = np.round(r2_score(train_predict, y_train), 2)

    test_predict = lin_reg.predict(x_test)
    test_rmse = np.round(np.sqrt(mean_squared_error(test_predict, y_test)), 0)
    test_gof = np.round(r2_score(test_predict, y_test), 2)

    return (
        model,
        coefficients,
        scaled_coefficients,
        train_rmse,
        train_gof,
        test_rmse,
        test_gof,
    )


def calculate_brake_power(rl_model, data):
    """Brake power is left over of Powertrain - grade - RL - inertia, considered only during following scenarios:
    Brake power can only be observed if accelerator pedal is released
    Brake power can only absorb energy from vehicle (so it is always -ve)
    Brake power only exists if there is intent of deceleration i.e. powertrain power + brake power <= 0"""
    (
        _,
        _,
        speed_meters_per_second,
        acceleration,
        power_at_wheels,
        mass,
        grade,
        _,
        _,
        _,
        accelerator_pedal,
    ) = get_required_signals(data)

    X = np.c_[speed_meters_per_second, acceleration, power_at_wheels, mass, grade]

    x_values, power_to_fit, inertia_power, gradient_power = transform_data(
        X, model="lin_reg"
    )
    brake_power_raw = -power_to_fit + rl_model.predict(x_values)
    brake_power = (
        brake_power_raw
        * ((power_at_wheels + brake_power_raw) <= 0)
        * (accelerator_pedal == 0)
        * (brake_power_raw <= 0)
    )

    return brake_power
