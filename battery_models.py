import numpy as np

class constant_parameter_battery():
    # Sign conventions - discharging , + charging
    # Charging parameters
    charging_efficiency      = 0.9 # [frac]
    charging_power_limit     = np.Inf # [W]
    
    # discharging parameters
    discharging_efficiency   = 0.9 # [frac]
    discharging_power_limit  = -np.Inf # [W]
    terminal_voltage         = 200    # [V]
    
    # capacity
    capacity                 = 70000*3600 # [J]
    
    # plug charging
    plug_full_charge_time    = 8*3600 # [sec]
    plug_charge_voltage      = 200    # [V]
    
    # soc limits
    maximum_battery_soc      = 1 # [frac]
    minimum_battery_soc      = 0 # [frac]
    
    # initial soc
    init_soc                 = 0.5 # [frac]
    
    def __init__(self, initial_soc=init_soc):
        self.soc     = initial_soc
        self.voltage = self.terminal_voltage
        
    def _calculate_soc(self, current_demand, dt):
        # +ve current charging, -ve current discharging
        
        """completely un-realistic but useful energy loss during charging and discharging"""
        if current_demand >= 0:
            current_demand = current_demand*self.charging_efficiency
        else:
            current_demand = current_demand/self.discharging_efficiency
            
        self.soc = self.soc + current_demand*dt/(self.capacity/self.voltage)
        self.soc = min(self.soc, self.maximum_battery_soc)  # To avoid > 100% in opportunisitic charging
        
    def _calculate_voltage(self, current_demand, electric_machine):
        if current_demand >= 0:
            self.voltage = electric_machine.voltage
        else:
            self.voltage = self.terminal_voltage
    
    def update_battery_state(self, current_demand, electric_machine, dt):
        self._calculate_soc(current_demand, dt)
        self._calculate_voltage(current_demand, electric_machine)
        
        self.current = current_demand

