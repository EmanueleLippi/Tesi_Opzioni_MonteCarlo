# Tesi Monte Carlo – Option Pricing

## Obiettivi
- Pricing di opzioni (europee, barrier, jump-diffusion) via Monte Carlo.
- Confronto con soluzioni in forma chiusa (BSM) e analisi VR.

## Struttura
- `src/tesi_montecarlo`: codice del pacchetto (processi, prodotti, MC, VR).
- `notebooks`: analisi e presentazioni ordinate.
- `experiments`: run riproducibili con config salvate.
- `tests`: unit test (MC vs BSM ecc.).
- `reports`: figure e tabelle pronte per la tesi.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pre-commit install
pytest
