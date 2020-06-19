import numpy as np


class simple_vehicle_controller():
    driveline_efficiency                        = 0.98 # [frac]
    wheel_radius                                = 1.013/2 # [m]
    final_drive_ratio                           = 4.33 # [ratio]
    regen_electric_to_friction_braking_ratio    = 1 # [ratio]
    regen_speed_limit                           = 10*5/18 # [m/s]
    
    def __init__(self):
        return None
        
    def _power_demand_at_wheels(self, powertrain_torque, brake_torque, speed):
        self._powertrain_power_demand = powertrain_torque*speed
        self._brake_power_demand      = brake_torque*speed
    
    def _wheel_power_limit_calculation(self, battery, electric_machine):
        _battery_discharging_mechanical_limit = -battery.discharging_power_limit*electric_machine.motoring_efficiency
        _machine_motoring_mechanical_limit    = min(electric_machine.motoring_mechanical_power_limit, 
                                                      (electric_machine.motoring_electrical_power_limit*electric_machine.motoring_efficiency))
        self._propulsion_wheel_power_limit    = min(_battery_discharging_mechanical_limit*self.driveline_efficiency, 
                                                      _machine_motoring_mechanical_limit*self.driveline_efficiency)
        
        _battery_charging_mechanical_limit    = -battery.charging_power_limit/electric_machine.generating_efficiency
        _machine_generating_mechanical_limit  = max(electric_machine.generating_mechanical_power_limit, 
                                                      electric_machine.generating_electrical_power_limit/electric_machine.generating_efficiency)
        self._generation_wheel_power_limit    = max(_battery_charging_mechanical_limit/self.driveline_efficiency, 
                                                      _machine_generating_mechanical_limit/self.driveline_efficiency)
    
    def _regen_power_demand_electric_machine(self, speed, battery):
        _battery_soc_limit                    = battery.soc >= battery.maximum_battery_soc
        _vehicle_speed_limit                  = speed <= self.regen_speed_limit
        if _battery_soc_limit or _vehicle_speed_limit:
            _regen_coasting_power_limited     = 0*self.driveline_efficiency
            _regen_braking_power_limited      = 0*self.driveline_efficiency
            
        else:
            if self._powertrain_power_demand  <= 0:
                _regen_coasting_power_raw         = self._powertrain_power_demand
                _regen_braking_power_raw          = self._brake_power_demand*self.regen_electric_to_friction_braking_ratio
                _regen_coasting_power_limited     = max(self._generation_wheel_power_limit, _regen_coasting_power_raw)*self.driveline_efficiency
                _residual_limit_after_coasting    = self._generation_wheel_power_limit - _regen_coasting_power_limited/self.driveline_efficiency
                _regen_braking_power_limited      = max(_residual_limit_after_coasting, _regen_braking_power_raw)*self.driveline_efficiency 

            elif (self._brake_power_demand + self._powertrain_power_demand) < 0: # Driver wants to decelerate but filtered powertrain demand may not be 0
                _regen_coasting_power_raw         = 0
                _regen_braking_power_raw          = self._brake_power_demand*self.regen_electric_to_friction_braking_ratio
                _regen_coasting_power_limited     = max(self._generation_wheel_power_limit, _regen_coasting_power_raw)*self.driveline_efficiency
                _residual_limit_after_coasting    = self._generation_wheel_power_limit - _regen_coasting_power_limited/self.driveline_efficiency
                _regen_braking_power_limited      = max(_residual_limit_after_coasting, _regen_braking_power_raw)*self.driveline_efficiency 
                
            else:
                _regen_coasting_power_limited     = 0*self.driveline_efficiency
                _regen_braking_power_limited      = 0*self.driveline_efficiency
                
        self._regen_power_demand          = _regen_coasting_power_limited + _regen_braking_power_limited
                
    def _propulsion_power_demand_electric_machine(self, battery):
        _battery_soc_limit                        = battery.soc <= battery.minimum_battery_soc
        
        if _battery_soc_limit:
            self._propulsion_power_demand = 0
            
        else:
            if (self._powertrain_power_demand + self._brake_power_demand) >= 0: # driver wants to accelerate overall at constrained speed
                self._propulsion_power_demand = min(self._propulsion_wheel_power_limit, 
                                                       (self._powertrain_power_demand + self._brake_power_demand))/self.driveline_efficiency
            else:
                self._propulsion_power_demand = 0
                
    def _power_mode_arbitration(self):
        if self._propulsion_power_demand > 0:
            self.mode = 1
            self.electric_machine_power_demand = self._propulsion_power_demand
        elif (self._propulsion_power_demand == 0) and (self._regen_power_demand < 0):
            self.mode = -1
            self.electric_machine_power_demand = self._regen_power_demand
        else:
            self.mode = 0
            self.electric_machine_power_demand = 0
    
    def calculate_electric_machine_power_demand(self, powertrain_torque, brake_torque, speed, battery, electric_machine):
        # Sign conventions +ve Torque machine motoring (propulsion), -ve torque machine generating power 
        self._wheel_power_limit_calculation(battery, electric_machine)
        self._power_demand_at_wheels(powertrain_torque, brake_torque, speed)
        self._regen_power_demand_electric_machine(speed, battery)
        self._propulsion_power_demand_electric_machine(battery)
        self._power_mode_arbitration()            
