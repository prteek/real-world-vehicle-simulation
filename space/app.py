"""Gradio app: could an electric truck do this stretch of real driving, and what battery and charger would it need?

The first 3,830 s of a heavy truck's driving log (58 km over 14 hours, the dataset prteek/drive-cycle) are replayed
second by second as if it were electric. The battery starts full; whenever the truck was parked long
enough it is plugged in and charges at the charger's power until full or until it drives off.
"""
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from vehicle_sim.drive_cycle import drive_cycle, load_from_hub
from vehicle_sim.simulation import TERMINAL_VOLTAGE, simulate

pio.templates.default = "ggplot2"

CODE_URL = "https://github.com/prteek/real-world-vehicle-simulation"
DATA_URL = "https://huggingface.co/datasets/prteek/drive-cycle"
CYCLE = drive_cycle(load_from_hub())  # the data is read from the Hugging Face dataset, never a local copy
DEFAULTS = dict(battery_kwh=70, charger_kw=8.75, min_park_minutes=30)  # a charger that fills 70 kWh in 8 hours
Y_MAX = 200  # signal values; longer stops are cut off at the top and labelled with their length
# ggplot2's default hues
SIGNALS = {
    "speed [km/h]": dict(color="#7F7F7F", width=1),
    "battery state of charge [%]": dict(color="#00BA38", width=3),
    "battery discharge current [A ÷ 10]": dict(color="#F8766D", width=1),
}
STOP_COLOUR, EMPTY_COLOUR = "#619CFF", "#C77CFF"


def overlay(trace: pd.DataFrame, min_park_minutes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The signals on one scale against driving time, and the stops long enough to plug in.

    Battery current is shown as discharge current (positive while driving) so it is visible on the same scale. Stops
    come from the drive cycle, so those after the battery ran out still show.
    """
    signals = pd.DataFrame({
        "drive_time": trace["drive_time"],
        "speed [km/h]": trace["speed"] * 3.6,
        "battery state of charge [%]": trace["soc"] * 100,
        "battery discharge current [A ÷ 10]": np.maximum(-trace["battery_power"] / TERMINAL_VOLTAGE, 0) / 10,
    })
    gap = trace["timestamp"].diff().fillna(0)
    stops = pd.DataFrame({
        "drive_time": trace["drive_time"],
        "minutes": gap / 60,
        "charge_energy": trace["charge_energy"],
    })[gap > min_park_minutes * 60]
    return signals, stops


def stop_label(minutes, charge_energy, battery_kwh) -> str:
    length = f"{minutes / 60:.1f} h stop" if minutes >= 120 else f"{minutes:.0f} min stop"
    charged = f"+{charge_energy / 3.6e6 / battery_kwh:.0%} charge" if charge_energy > 0 else "not charged"
    return f"{length}<br>{charged}"


def figure(trace, s, battery_kwh, min_park_minutes) -> go.Figure:
    signals, stops = overlay(trace, min_park_minutes)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stops["drive_time"], y=np.minimum(stops["minutes"], Y_MAX), width=30, name="stop, long "
                         "enough to plug in [minutes]", marker_color=STOP_COLOUR, opacity=0.7,
                         customdata=stops["minutes"], hovertemplate="%{customdata:.0f} min stop<extra></extra>"))
    for name, line in SIGNALS.items():
        fig.add_trace(go.Scatter(x=signals["drive_time"], y=signals[name], mode="lines", name=name, line=line,
                                 opacity=0.6 if "current" in name else 1,
                                 hovertemplate="%{y:.0f}<extra>" + name + "</extra>"))
    next_stop = stops["drive_time"].shift(-1, fill_value=np.inf)
    for (_, stop), following in zip(stops.iterrows(), next_stop):
        left = following - stop["drive_time"] < 300  # label on the left when the next stop is close
        fig.add_annotation(x=stop["drive_time"], y=min(stop["minutes"], Y_MAX), yanchor="top",
                           xanchor="right" if left else "left", xshift=-8 if left else 8, showarrow=False,
                           align="right" if left else "left", font=dict(color=STOP_COLOUR, size=11),
                           text=stop_label(stop["minutes"], stop["charge_energy"], battery_kwh))
    if s["ran_out_at_hours"] is not None:
        empty = trace["drive_time"][trace["soc"] <= 0].iloc[0]
        fig.add_trace(go.Scatter(x=[empty, empty], y=[0, Y_MAX], mode="lines", name="battery empty",
                                 line=dict(color=EMPTY_COLOUR, width=2, dash="dash"), hoverinfo="name"))
    fig.update_xaxes(title_text="driving time [s]", range=[0, trace["drive_time"].max()])
    fig.update_yaxes(title_text="signal values", range=[0, Y_MAX])
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18, x=0))
    return fig


def verdict(s) -> str:
    total = s["logged_distance_km"]
    if s["ran_out_at_hours"] is None:
        head = f"**The battery lasts**: all {total:,.0f} km on electricity, never below {s['min_soc']:.0%} charge."
    else:
        head = (f"**The battery runs out** after {s['electric_distance_km']:,.0f} of the {total:,.0f} km "
                f"({s['electric_distance_km'] / total:.0%}).")
    rows = [
        "| | |", "|---|---:|",
        f"| distance on battery | {s['electric_distance_km']:,.0f} of {total:,.0f} km |",
        f"| lowest charge | {s['min_soc']:.0%} |",
        f"| stops where it charged | {s['charge_events']} of {s['potential_charge_events']} long enough to plug in |",
        f"| energy charged at stops | {s['charge_energy_kwh']:,.1f} kWh |",
        f"| energy drawn for driving | {s['propulsion_energy_kwh']:,.1f} kWh "
        f"({s['regen_energy_kwh']:,.1f} kWh back from braking) |",
        f"| peak battery power | {s['max_propulsion_power_kw']:,.0f} kW |",
    ]
    return head + "\n\n" + "\n".join(rows)


def run(battery_kwh, charger_kw, min_park_minutes):
    if not battery_kwh or battery_kwh <= 0:
        return "Battery capacity must be above 0 kWh.", None
    trace, s = simulate(CYCLE, float(battery_kwh), float(charger_kw or 0), float(min_park_minutes))
    return verdict(s), figure(trace, s, float(battery_kwh), float(min_park_minutes))


with gr.Blocks(title="Electric truck battery sizing") as demo:
    gr.Markdown(
        "# Could an electric truck do this job?\n"
        "A stretch of a diesel truck's real driving (58 km over 14 hours, 15 to 32 tonnes) replayed as if it were "
        "electric. The battery starts full; whenever the truck is parked long enough it is plugged in and charges "
        "until it is full or drives off. Choose the battery and the charger and see whether it gets through: the "
        "stops are marked with how long they were and how much charge they added."
    )
    with gr.Row():
        battery = gr.Slider(20, 300, value=DEFAULTS["battery_kwh"], step=5, label="Battery capacity [kWh]")
        charger = gr.Slider(0, 100, value=DEFAULTS["charger_kw"], step=0.25,
                            label="Charger power [kW] (0: no charging)")
        park = gr.Slider(5, 120, value=DEFAULTS["min_park_minutes"], step=5,
                         label="Shortest stop worth plugging in for [minutes]")
    result = gr.Markdown()
    plot = gr.Plot(show_label=False)
    gr.Markdown(
        "The simple model treats the motor and battery as 80% and 90% efficient, recovers braking energy above "
        "10 km/h, puts no limit on power, and estimates brake power from a road-load model fitted to the log. "
        f"Code: [{CODE_URL.removeprefix('https://')}]({CODE_URL}) · "
        f"Data: [{DATA_URL.removeprefix('https://huggingface.co/datasets/')}]({DATA_URL})."
    )
    inputs = [battery, charger, park]
    gr.on([c.change for c in inputs], run, inputs, [result, plot], trigger_mode="always_last")
    demo.load(run, inputs, [result, plot])

if __name__ == "__main__":
    demo.launch()
