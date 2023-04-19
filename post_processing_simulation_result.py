# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Post processing simulation results

import matplotlib.pyplot as plt
import numpy as np
from vehicle_simulation_model import simulation

# Scenario 1
battery_capacity_kwh = 70
plug_full_charge_time_hrs = 8
out = simulation(battery_capacity_kwh, plug_full_charge_time_hrs)
time, speed_meters_per_second, battery_soc, battery_current, timestamp = (out['time'],
                                                                          out['speed_meters_per_second'],
                                                                          out['battery_soc'],
                                                                          out['battery_current'],
                                                                          out['timestamp']
                                                                          )
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, speed_meters_per_second * 3.6, label="speed [km/h]")
ax.plot(time, battery_soc * 100, label="battery soc [%]")
ax.plot(time, battery_current / 10, label="battery current [A]")
ax.plot(
    time,
    (np.append(0, np.diff(timestamp)) > 1800) * np.append(0, np.diff(timestamp)) / 60,
    label="charging window [minutes]",
)
ax.set_title("Example day of driving (Scenario 1: 70kWh battery)")
ax.set_xlabel("driving duration [sec]")
ax.set_ylabel("signal values")
ax.set_xlim([0, 3830])
ax.set_ylim([0, 150])
ax.legend()
ax.grid()
plt.show()
fig.savefig("simulation_output_scenario_1.png")


# Scenario 2
battery_capacity_kwh = 120
plug_full_charge_time_hrs = 8
out = simulation(battery_capacity_kwh, plug_full_charge_time_hrs)
time, speed_meters_per_second, battery_soc, battery_current, timestamp = (out['time'],
                                                                          out['speed_meters_per_second'],
                                                                          out['battery_soc'],
                                                                          out['battery_current'],
                                                                          out['timestamp']
                                                                          )
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, speed_meters_per_second * 3.6, label="speed [km/h]")
ax.plot(time, battery_soc * 100, label="battery soc [%]")
ax.plot(time, battery_current / 10, label="battery current [A]")
ax.plot(
    time,
    (np.append(0, np.diff(timestamp)) > 1800) * np.append(0, np.diff(timestamp)) / 60,
    label="charging window [minutes]",
    )
ax.set_title("Example day of driving (Scenario 2: 120 kWh battery)")
ax.set_xlabel("driving duration [sec]")
ax.set_ylabel("signal values")
ax.set_xlim([0, 3830])
ax.set_ylim([0, 150])
ax.legend()
ax.grid()
plt.show()
fig.savefig("simulation_output_scenario_2.png")


