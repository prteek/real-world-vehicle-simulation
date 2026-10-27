# real-world-vehicle-simulation

Could an electric truck do the job of a diesel one? A stretch of a diesel heavy truck's real driving (the first 3,830 s
of driving: 58 km over 14 hours, 15 to 32 tonnes) is replayed second by second as if the truck were electric. The log
has no battery signals, so the simulation supplies them: battery current, state of charge, and the effect of
opportunistic charging, where the truck is plugged in whenever it is parked long enough.

**Live app:** https://huggingface.co/spaces/prteek/electric-truck-battery-sizing · **Data:** https://huggingface.co/datasets/prteek/drive-cycle

## Layout

```
vehicle_sim/            the model
  drive_cycle.py        loads the data; power at the wheels and brake power from a road-load model
  simulation.py         simulate(cycle, battery kWh, charger kW, shortest stop) -> trace and summary
space/                  Gradio app: app.py, Space card, requirements
dataset/README.md       the dataset card
publish.sh              upload the dataset and/or the Space
wait_for_space.py       wait until the Space runs its latest commit (used by CI)
tests/                  drive cycle, simulation and app
```

## Common tasks

```bash
uv sync                                       # package + app + dev groups
uv run pytest -q

uv run python space/app.py                    # app on http://127.0.0.1:7860
./publish.sh "message" dataset space          # publish (targets: dataset, dataset-card, space; default dataset space)
```

## Publishing from CI

Pushes to `master` publish to Hugging Face with the `HF_TOKEN` repository secret (both workflows can also be run by
hand):

- `publish-space.yml`: when the app or `vehicle_sim/` changes; runs the tests, uploads, and waits until the Space runs
  the new commit
- `publish-dataset.yml`: when the dataset card changes; uploads the card only. The data is not in git, so it is
  published by hand (`./publish.sh "message" dataset`, from a local `data/data.csv`)

## Notes

- **How it works.** Power at the wheels comes from engine torque, gear ratio and transmission output speed. Brake power
  isn't logged, so it is estimated as what is left after road load (rolling resistance and aerodynamic drag), grade
  and inertia. The road-load coefficients are constants in `drive_cycle.py`, fitted by least squares on the whole
  428 km log the stretch came from; the stretch alone is too uniform in speed to separate the two terms. The simulation passes the
  demand through the driveline (98%), the electric machine (80%) and the battery (90%); regenerative braking works
  above 10 km/h. At every stop long enough to plug in, the battery charges at the charger's power until full or until
  the truck drives off. Power limits are not modelled: the stretch asks the battery for up to 351 kW.
- **What it shows.** A 70 kWh battery with a charger that fills it in 8 hours runs out after 30 of the 58 km; 120 kWh
  gets through with 3% left. The first two stops add little because the battery is still nearly full; the 57-minute
  stop later in the day is the one that matters.
- **Simplified and corrected (2026-10-27).** The component classes (controller, electric machine, battery) collapsed
  into `simulation.battery_power`, since every power limit was infinite and every voltage 200 V; the results are
  unchanged to machine precision. The road-load fit became fixed coefficients, which removed scikit-learn. Three
  bugs were fixed: each stop charged for the *cumulative* charging time of all stops so far; the time to fill the
  battery left out the charging efficiency, so no stop could take it past about 90% of the way to full; and the regen
  cut-off compared wheel speed in rad/s with 10 km/h in m/s. Charger power is now an input rather than capacity / 8 h.
- **The data** is the Hugging Face dataset `prteek/drive-cycle`, not part of the repository: no CSV is kept in git.
  `vehicle_sim.drive_cycle.load()` reads `data/data.csv` if it is there and downloads it from the dataset otherwise.
  It is only the stretch the app shows (3,831 seconds of driving), with only the columns the model uses and readable
  names (described in the dataset card). The rest of the log, and the raw log with its GPS trace, are not published.
