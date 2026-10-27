---
title: Electric Truck Battery Sizing
emoji: 🔋
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Could an electric truck do this real driving? Battery sizing
datasets:
  - prteek/drive-cycle
---

A stretch of a diesel heavy truck's real driving (58 km over 14 hours, 15 to 32 tonnes) replayed second by second as if it were electric. Choose the battery capacity, the charger power and how long a stop has to be before the truck is plugged in, and see whether it gets through on electricity, how low the battery gets, and how much each stop tops it up.

Code: [github.com/prteek/real-world-vehicle-simulation](https://github.com/prteek/real-world-vehicle-simulation) · Data: [prteek/drive-cycle](https://huggingface.co/datasets/prteek/drive-cycle).
