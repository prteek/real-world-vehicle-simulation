# author            : Prateek
# email             : prateekpatel.in@gmail.com
# description       : Unit test for battery models

import os

os.system('pip install -U -r requirements.txt')

from battery_models import constant_parameter_battery


batt = constant_parameter_battery(0.5)


def test_increase_soc():
    batt_soc_prev = batt.soc
    batt._calculate_soc(10, 100)
    assert batt.soc > batt_soc_prev


def test_decrease_soc():
    batt_soc_prev = batt.soc
    batt._calculate_soc(-10, 100)
    assert batt.soc < batt_soc_prev
