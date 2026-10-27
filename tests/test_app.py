import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("app", Path(__file__).parents[1] / "space/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_default_run_gives_a_verdict_and_a_figure():
    text, fig = app.run(**app.DEFAULTS)
    assert text.startswith("**The battery runs out**") and fig is not None


def test_verdicts():
    assert app.run(120, 15, 30)[0].startswith("**The battery lasts**")
    assert app.run(0, 50, 30)[0].startswith("Battery capacity must be")


def test_stops_are_labelled_with_length_and_charge():
    trace, s = app.simulate(app.CYCLE, 120, 15, 30)
    _, stops = app.overlay(trace, 30)
    assert len(stops) == 3
    labels = [app.stop_label(m, e, 120) for m, e in zip(stops["minutes"], stops["charge_energy"])]
    assert labels[1].startswith("9.7 h stop") and labels[2].endswith("% charge")


def test_app_reads_the_data_from_the_hub():
    from vehicle_sim.drive_cycle import DATASET

    source = (Path(__file__).parents[1] / "space/app.py").read_text()
    assert "drive_cycle(load_from_hub())" in source and DATASET == "prteek/drive-cycle"
    assert len(app.CYCLE) == 3831
