import plotly.graph_objects as go
import numpy as np
from utility_functions import *

import vehicle_simulation_model as vs # This will run the simulation

x = vs.time
fig = plt_plot(x=x, y=vs.speed_meters_per_second*3.6, name='speed [km/h]')
fig.add_scatter(x=x,y=vs.battery_soc*100, name='battery soc [%]')
fig.add_scatter(x=x,y=vs.battery_current/10, name='battery current [A]')
fig.add_scatter(x=x,y=(np.append(0, np.diff(vs.timestamp))>1800)*np.append(0, np.diff(vs.timestamp))/60,  name='charging window [minutes]')
fig.update_layout(title='Example day of driving',
                  xaxis_title='time [sec]',
                  xaxis_range=[0, 3830],
                  yaxis_title='signal values',
                  yaxis_range=[0, 150])
fig.show()
