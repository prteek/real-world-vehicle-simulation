# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Post processing simulation results

import matplotlib.pyplot as plt

from vehicle_simulation_model import *  # This will run the simulation and import all the variables generated in the process

x = time
plt.plot(x, speed_meters_per_second * 3.6, label="speed [km/h]")
plt.plot(x, battery_soc * 100, label="battery soc [%]")
plt.plot(x, battery_current / 10, label="battery current [A]")
plt.plot(x, (np.append(0, np.diff(timestamp)) > 1800) * np.append(0, np.diff(timestamp)) / 60,
    label="charging window [minutes]",
)
plt.title("Example day of driving")
plt.xlabel("time [sec]")
plt.ylabel("signal values")
plt.xlim([0, 3830])
plt.ylim([0, 150])
plt.legend()
plt.show()
plt.savefig("simulation_output.png")
