"""Inference on the RUM distance: Hong and Li's bootstrap p-value and confidence interval.

    boot = rum.bootstrap(panel=panel)              # solver.rum_solver.RUM.bootstrap; DSR style, n = 520
    boot = rum.bootstrap(replies=replies)          # AI style, n = 100
    boot.p_value(), boot.confidence_interval(), boot.distance_interval()

    hong_li     the formulas, from certified brackets (pure numpy)
    draws       the choices behind p_hat, as a panel (respondent, menu, chosen) or as replies
                (menu, chosen), and the draws made from them
    bootstrap   Bootstrap and its results, PValue and Interval
"""
from . import hong_li
from .bootstrap import Bootstrap, DistanceInterval, Interval, PValue
from .draws import WEIGHTS, Draws, MenuDraws, cells_from_panel, read_panel

__all__ = ["hong_li", "Bootstrap", "DistanceInterval", "Interval", "PValue", "WEIGHTS", "Draws", "MenuDraws", "cells_from_panel",
           "read_panel"]
