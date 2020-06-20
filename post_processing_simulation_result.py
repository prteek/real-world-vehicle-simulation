import os
os.system('pip install -U -r requirements.txt')

import plotly.graph_objects as go
import numpy as np
from utility_functions import *

from vehicle_simulation_model import *  # This will run the simulation and import all the variables generated in the process

x = time
fig = plt_plot(x=x, y=speed_meters_per_second * 3.6, name="speed [km/h]")
fig.add_scatter(x=x, y=battery_soc * 100, name="battery soc [%]")
fig.add_scatter(x=x, y=battery_current / 10, name="battery current [A]")
fig.add_scatter(
    x=x,
    y=(np.append(0, np.diff(timestamp)) > 1800) * np.append(0, np.diff(timestamp)) / 60,
    name="charging window [minutes]",
)
fig.update_layout(
    title="Example day of driving",
    xaxis_title="time [sec]",
    xaxis_range=[0, 3830],
    yaxis_title="signal values",
    yaxis_range=[0, 150],
)
fig.show()
