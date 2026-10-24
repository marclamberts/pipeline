"""Refit the xG model on every played match: python -m analysis.fit_xg"""
import json

from analysis.footy.data import squad_names
from analysis.footy.xg import fit

if __name__ == "__main__":
    squad_names(refresh=True)
    print(json.dumps(fit(), indent=2))
