# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Post processing simulation results

import matplotlib.pyplot as plt

from vehicle_simulation_model import *  # This will run the simulation and import all the variables generated in the process

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, speed_meters_per_second * 3.6, label="speed [km/h]")
ax.plot(time, battery_soc * 100, label="battery soc [%]")
ax.plot(time, battery_current / 10, label="battery current [A]")
ax.plot(
    time,
    (np.append(0, np.diff(timestamp)) > 1800) * np.append(0, np.diff(timestamp)) / 60,
    label="charging window [minutes]",
)
ax.set_title("Example day of driving")
ax.set_xlabel("time [sec]")
ax.set_ylabel("signal values")
ax.set_xlim([0, 3830])
ax.set_ylim([0, 150])
ax.legend()
ax.grid()
plt.show()
fig.savefig("simulation_output.png")
