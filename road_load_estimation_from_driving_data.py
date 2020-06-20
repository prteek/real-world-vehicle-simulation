# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Modelling road load behaviour based on real-time driving data

import os

os.system("pip install -U -r requirements.txt")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utility_functions import *
import time as TT


### Get data
data = pd.read_csv("./datalog.csv")

### Get required signals
speed_dl = np.array(data["KPH_DL"], dtype="float")
speed = speed_dl * (speed_dl < 250)

time = data["time"]
dt = time[1] - time[0]
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


### Functions for filtering and transforming data for Road Load estimation


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
        power_to_fit = power_at_wheels - inertia_power - gradient_power
        x_values = np.c_[speed_meters_per_second, mass]

    return x_values, power_to_fit, inertia_power, gradient_power


### Calculate signals that will be useful for modelling Road Load / Brake power

power_at_wheels_raw = (
    engine_torque * overall_ratio * transmission_output_speed / 30 * np.pi
)

# Filter some signals to make them appropriate for training
moving_average_window = 10
low_pass_tc = 2

# Calculate acceleration on filtered speed
dt = time[1] - time[0]
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


### Training the model
""" For training the model it is imperative that data is used only from when vehicle is being driven by engine power, 

i.e. acceleration >= 0. 
Note: also don't use vehicle speed = 0 since most residuals are zero for those and GOF is over estimated 

Since, only under this scneario 
engine_power = road_load_power + aero_power + gradient_power + inertia power

Also,  Only take into account slip <= 1 since we are working with wheel power (after TC loss) """

X = np.c_[
    speed_meters_per_second, acceleration, power_at_wheels, mass, grade, overall_ratio
]

slip = transmission_output_speed * overall_ratio / (engine_speed + np.finfo(float).eps)

positive_power_index = np.where((acceleration >= 0) & (speed > 0) & (slip <= 1.0))[0]

X_positive_power = X[positive_power_index, :]


### Linear regression model
x_model_input, y_to_fit, ip, gp = transform_data(X_positive_power, model="lin_reg")

x_train, x_test, y_train, y_test = train_test_split(
    x_model_input, y_to_fit, test_size=0.20
)

lin_reg = LinearRegression(
    fit_intercept=False
)  # set this to false as we don't want any constant term to be fit
scores = cross_val_score(
    lin_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=30
)

lin_reg.fit(x_train, y_train)

print("fitted coefficients:", lin_reg.coef_)
print("scaled fitted coefficients:", lin_reg.coef_ * np.std(x_train, axis=0), "\n")

train_predict = lin_reg.predict(x_train)
print(
    "RMSE train lr:",
    np.round(np.sqrt(mean_squared_error(train_predict, y_train)), 0),
    " \t GOF:",
    np.round(r2_score(train_predict, y_train), 2),
)

test_predict = lin_reg.predict(x_test)
print(
    "RMSE test lr:",
    np.round(np.sqrt(mean_squared_error(test_predict, y_test)), 0),
    " \t GOF:",
    np.round(r2_score(test_predict, y_test), 2),
)


fig = plt.figure()
plt.hist(x=np.sqrt(-scores))
plt.title("Distribution of cross validation errors")
plt.xlabel("errors [W]")
plt.show()
print("\n standard deviation of cross val errors:", np.std(np.sqrt(-scores)))


### Model fit examination
"""
Check the following: <br/>
- [x] If residuals (errors) are distributed approximately normally
- [x] If residuals show Homoscedasticity (errors have constant variance w.r.t. x)
- [x] RMSE is acceptable (in this case RMSE means 65% of predictions will have error below x kW in total RL power)
- [x] Goodness of fit is acceptable
"""

residuals = y_train - train_predict

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.hist(x=residuals / 1000, normed=True, bins=50)
plt.title("Probability distribution of residuals")
plt.xlabel("residuals")
plt.ylabel("probability")

plt.subplot(1, 2, 2)
plt.plot(train_predict, residuals / 1000, ".")
plt.title("Check for homoscedasity")
plt.xlabel("fitted values")
plt.ylabel("residuals")
plt.show()

fig = plt.figure()
plt.plot(y_to_fit / 1000, label="rl power to fit")
plt.plot(lin_reg.predict(x_model_input) / 1000, label="predicted rl")
plt.plot(X_positive_power[:, 0] * 18 / 5, label="speed")
plt.xlabel("time [s]")
plt.ylabel("values [kw, kw, kph]")
plt.legend()
plt.grid()
plt.show()


### Random forest regression model
"""
Notes:
* Performs reasonable well
* Tries to overfit so need to constraint the tree growth
* The fitted RL plotted against speed do not make much physical sense
* The fitted RL (only RL no grade) show negative values due to noisy training data (effect of sudden large acceleration values). Model learns this and predicts negative RL values when there is sudden high value of acceleration in the data.
* Being non parametric the restriction of max depth becomes subjective on training data. So to train the model with cross validation to find max depth will take a lot of time, (when there are no parallel workers to run).

**At this stage using RF is not recommended**
"""

tic = TT.time()
x_model_input, y_to_fit, _, _ = transform_data(X_positive_power, model="rfr")

x_train, x_test, y_train, y_test = train_test_split(
    x_model_input, y_to_fit, test_size=0.20, random_state=42
)

pca = PCA(n_components=x_model_input.shape[1])
x_train = pca.fit_transform(x_train)

rfr = RandomForestRegressor(bootstrap=True, min_samples_leaf=10)

param_grid = [{"max_depth": [1, 2]}, {"n_estimators": [4, 8, 16]}]

grid_search = GridSearchCV(
    rfr,
    param_grid,
    cv=8,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=4,
)
grid_search.fit(x_train, y_train)

print("best parameters:", grid_search.best_params_)

rfr = grid_search.best_estimator_

train_predict = rfr.predict(x_train)
print(
    "RMSE train rf:",
    np.round(np.sqrt(mean_squared_error(train_predict, y_train)), 0),
    " \t GOF:",
    np.round(r2_score(train_predict, y_train), 2),
)

x_test = pca.transform(x_test)
test_predict = rfr.predict(x_test)
print(
    "RMSE test rf:",
    np.round(np.sqrt(mean_squared_error(test_predict, y_test)), 0),
    " \t GOF:",
    np.round(r2_score(test_predict, y_test), 2),
)


elapsed = TT.time() - tic
print("Grid search:", elapsed)


### Cross Validation

tic = TT.time()
scores = cross_val_score(
    rfr, x_train, y_train, scoring="neg_mean_squared_error", cv=32, n_jobs=4
)

fig = plt.figure()
plt.hist(x=np.sqrt(-scores))
plt.title("Distribution of cross validation errors")
plt.xlabel("errors [W]")
plt.show()
print("\n standard deviation of cross val errors:", np.std(np.sqrt(-scores)))

elapsed = TT.time() - tic
print("Cross Val:", elapsed)


### Model fit examination
"""
Check the following: <br/>
- [x] If residuals (errors) are distributed approximately normally
- [x] RMSE is acceptable (in this case RMSE means 65% of predictions will have error below x kW in total RL power)
- [x] Goodness of fit is acceptable
"""

residuals = y_train - train_predict


fig = plt.figure()
plt.subplot(1, 2, 1)
plt.hist(x=residuals / 1000, normed=True, bins=50)
plt.title("Probability distribution of residuals")
plt.xlabel("residuals")
plt.ylabel("probability")


fig = plt.figure()
plt.plot(y_to_fit / 1000, label="rl power")
plt.plot(rfr.predict(pca.transform(x_model_input)) / 1000, label="rl predicted")
plt.plot(X_positive_power[:, 0] * 18 / 5, label="speed")
plt.xlabel("time [s]")
plt.ylabel("values [kw, kw, kph]")
plt.legend()
plt.grid()
plt.show()


### Estimating brake power
x_values, power_to_fit, inertia_power, gradient_power = transform_data(
    X, model="lin_reg"
)

brake_power_raw = power_to_fit - lin_reg.predict(x_values)
brake_power = (
    brake_power_raw
    * ((power_at_wheels - brake_power_raw) <= 0)
    * (accelerator_pedal == 0)
    * (brake_power_raw >= 0)
)

print("Total brake energy [MJ]: ", np.sum(brake_power / 1000000))

fig = plt.figure()
plt.plot(power_at_wheels / 1000, label="wheelpow")
plt.plot(lin_reg.predict(x_values) / 1000 + 0 * gradient_power / 1000, label="rl")
plt.plot(brake_power / 1000, label="brake power")
plt.plot(speed, label="speed")
plt.legend()
plt.grid()
plt.show()


### Feature importance in Random forest
importances = rfr.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfr.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
fig = plt.figure()
plt.title("feature importances")
plt.bar(np.arange(x_train.shape[1]), importances[indices])
plt.xlim([-1, x_train.shape[1]])
plt.xticks(np.arange(x_train.shape[1]))
plt.show()


### Partial dependence plot for Random Forest

from dask import delayed, compute, visualize
import dask.array as da

x_values_rfr, power_to_fit_rfr, inertia_power, gradient_power = transform_data(
    X, model="rfr"
)

x_values_rfr = pca.transform(x_values_rfr)

n_samples = 200
random_data_points = np.random.choice(x_values_rfr.shape[0], n_samples, replace=True)
yy = []
for i in range(len(random_data_points)):
    xx = np.hstack(
        (
            x_values_rfr[:, 0].reshape(-1, 1),
            np.ones((len(x_values_rfr), 1)) * x_values_rfr[random_data_points[i], 1],
        )
    )
    pp = delayed(rfr.predict)(xx)
    yy.append(pp)


yc = np.array(compute(yy)[0]).T

fig = plt.figure()
plt.plot(x_values_rfr[:, 0], yc[:, :30], ".")
plt.plot(x_values_rfr[:, 0], np.mean(yc, axis=1), "ko", label="partial dependence")
plt.legend()
plt.grid()
plt.show()

### Inferring physical relationship between speed and RL from Random forest model
fig = plt.figure()
plt.plot(
    speed_meters_per_second,
    rfr.predict(x_values_rfr),
    ".",
    alpha=0.1,
    label="random forest model",
)
plt.plot(
    speed_meters_per_second,
    lin_reg.predict(x_values),
    ".",
    alpha=0.1,
    label="linear regression",
)
plt.legend()
plt.grid()
plt.show()
