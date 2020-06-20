# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : utilities library

import plotly.graph_objects as go

def plt_plot(**kwargs):
    fig = go.Figure()
    fig.add_scatter(**kwargs)
    return fig


def plt_hist(**kwargs):
    """histnorm:{'density', 'probability', 'percent'}
       cumulative_enabled:{True, False}"""
    fig = go.Figure()
    fig.add_histogram(**kwargs)
    return fig
