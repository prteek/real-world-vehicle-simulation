---
license: cc-by-4.0
language:
  - en
pretty_name: Heavy truck drive cycle
tags:
  - time-series
  - automotive
  - electric-vehicles
  - energy
  - tabular
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files: data.csv
---

# Heavy truck drive cycle

3,831 seconds of driving logged once a second from a diesel heavy truck (15 to 32 tonnes): 58 km over about 14 hours,
with three stops of half an hour or more in between. It is the data behind the
[electric truck battery sizing app](https://huggingface.co/spaces/prteek/electric-truck-battery-sizing), which replays
it as if the truck were electric to see what battery and charger it would need.
Code: [github.com/prteek/real-world-vehicle-simulation](https://github.com/prteek/real-world-vehicle-simulation).

Rows are logged only while the truck is running: a jump in `time_s` larger than a second is a stop.

| column | content |
|---|---|
| `time_s` | seconds since the start of the log (gaps are stops) |
| `speed_kph` | vehicle speed [km/h]; 255 means no reading |
| `engine_torque_nm` | net engine torque [N·m]; negative when the engine is driven |
| `overall_gear_ratio` | overall torque ratio from the engine to the transmission output |
| `transmission_output_speed_rpm` | transmission output shaft speed [rpm] |
| `accelerator_pedal_pct` | accelerator pedal position [%] |
| `vehicle_mass_kg` | estimated vehicle mass [kg] |
| `road_grade_pct` | road grade [%], averaged |

Power at the wheels is engine torque × overall gear ratio × transmission output speed. Brake power is not logged; the
code estimates it as what is left after road load, grade and inertia whenever the driver is off the accelerator and
slowing down.

## Road load (`road_load.json`)

The road-load model, frozen: power [W] = `rolling` × mass × g × cos(grade) × speed + `aero` × speed³ (speed in m/s,
grade = `road_grade_pct` / 100). It was fitted by least squares on the whole 428 km log this stretch came from, on the
moments the truck was driving rather than braking (R² 0.74 on a held-out 30%); the stretch alone is too uniform in
speed to separate the two terms.

```python
import json, pandas as pd
from huggingface_hub import hf_hub_download

log = pd.read_csv(hf_hub_download("prteek/drive-cycle", "data.csv", repo_type="dataset"))
road_load = json.load(open(hf_hub_download("prteek/drive-cycle", "road_load.json", repo_type="dataset")))
```
