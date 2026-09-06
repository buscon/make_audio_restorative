"""Published ARAUSD results from Versuemer et al. (2025), Tables V and VI.

Phase 1 acceptance criterion: our numbers should land close to these. We use
six of their seven indicators (no Relative Approach), so a small shortfall is
expected and is not a bug.
"""

# (target, outer-inner splitting) -> {model: value}
R2 = {
    ("ISOPl", "sgkf-gkf"):  {"LR": 0.1675, "RF": 0.1914, "XGBoost": 0.1872},
    ("ISOEv", "sgkf-gkf"):  {"LR": 0.2276, "RF": 0.2775, "XGBoost": 0.2711},
    ("ISOPl", "cskf-cskf"): {"LR": 0.1700, "RF": 0.1931, "XGBoost": 0.2027},
    ("ISOEv", "cskf-cskf"): {"LR": 0.2312, "RF": 0.2807, "XGBoost": 0.2910},
}

MSE = {
    ("ISOPl", "sgkf-gkf"):  {"LR": 0.5176, "RF": 0.5026, "XGBoost": 0.5052},
    ("ISOEv", "sgkf-gkf"):  {"LR": 0.5045, "RF": 0.4718, "XGBoost": 0.4760},
    ("ISOPl", "cskf-cskf"): {"LR": 0.5163, "RF": 0.5019, "XGBoost": 0.4960},
    ("ISOEv", "cskf-cskf"): {"LR": 0.5021, "RF": 0.4698, "XGBoost": 0.4631},
}


def lookup(table, target, outer, inner, model):
    return table.get((target, f"{outer}-{inner}"), {}).get(model)
