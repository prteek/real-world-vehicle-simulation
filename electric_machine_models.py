import numpy as np

class constant_parameter_electric_machine():
    # Sign conventions -ve current battery discharging, +ve current battery charging
    # Sign conventions +ve Torque machine motoring (propulsion), -ve torque machine generating power 
    
    # Motoring parameters
    motoring_efficiency                    = 0.8 # [frac]
    motoring_electrical_power_limit        = np.Inf # [W]
    motoring_mechanical_power_limit        = np.Inf # [W] : Could be changed to torque limit 
    
    # generating parameters
    generating_efficiency                  = 0.8 # [frac]
    generating_electrical_power_limit      = -np.Inf # [W]
    generating_mechanical_power_limit      = -np.Inf # [W] : Could be changed to torque limit 
    generating_set_point_voltage           = 200    # [V]
    
    def __init__(self):
        return None
    
    def _electric_machine_voltage(self, battery):
        if self.delivered_power >= 0:
            self.voltage  = battery.terminal_voltage
        else:
            self.voltage  = self.generating_set_point_voltage
    
    def _electric_machine_delivered_power(self, power_demand):
        self.delivered_power  = power_demand
        
    def _electric_machine_electrical_power(self):
        if self.delivered_power >= 0:
            self.electrical_power = self.delivered_power/self.motoring_efficiency
        else:
            self.electrical_power = self.delivered_power*self.generating_efficiency
        
    def calculate_battery_current_demand(self, power_demand, battery):
        self._electric_machine_delivered_power(power_demand)
        self._electric_machine_voltage(battery)
        self._electric_machine_electrical_power()
        self.battery_current_demand = -self.electrical_power/self.voltage # Make sure to flip the sign to follow sign conventions
                
