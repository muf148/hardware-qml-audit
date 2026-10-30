# hardware-qml-audit

Code for *A null-circuit protocol for testing hardware quantum machine learning claims* (M. Faryad, 2026).

The paper tests a hardware quantum machine learning claim: that features measured on an IBM quantum processor
improve fill-probability models far beyond classical features (Ciceri et al., arXiv:2509.17715). It proposes a
five-step protocol for this kind of claim and runs it on 129 qubits of `ibm_fez`:

1. **Null circuit.** Run the same circuit with every data angle set to zero, in the same jobs as the data.
2. **Batch identifiability.** Check whether a classifier can tell which job a row came from.
3. **Controlled assignment.** Compare random job assignment with the natural pipeline, on the same events.
4. **Replay.** Give the null slots to the events in other submission orders and rerun the backtest.
5. **Job-purged backtest.** Keep each test block's job out of its training window.

The design, predictions and decision rules were frozen before the first hardware run (`outputs/prereg/`,
SHA-256 `3eb1668a2cbd0ff3bd698c025657aedf137ed9cba82e75747970668648e85a8e`).

## What is here

| Path | Contents |
|---|---|
| `hsbcrep/` | Feature map, Pauli-propagation simulator, public task, backtest, design and hardware helpers |
| `notebooks/` | H1 to H6, run in order (below) |
| `outputs/prereg/` | The frozen pre-registration: schedule, plan, predictions, baselines, fingerprint |
| `outputs/results_*` | Results of H5 (pre-registered) and H6 (exploratory) |

The data are on Zenodo (DOI below): the event and feature tables, the noiseless features, every hardware job's
output and the device calibration at each session. Download the two archives and unzip both in the repository folder; they fill `outputs/`.

## Notebooks

| Notebook | What it does | Needs |
|---|---|---|
| `H1_data` | Builds the public task from Binance futures data: 4,000 events, touch-fill labels, 216 features | Zenodo tables, or `RECOMPUTE = True` to rebuild from Binance |
| `H2_circuits_noiseless` | The circuits on a 129-qubit chain and their exact noiseless features | Zenodo tables |
| `H3_design_preregistration` | Schedule, emulator predictions, endpoints; verifies the frozen fingerprint | Zenodo tables |
| `H4_hardware` | Runs the sessions on IBM hardware (already done; kept for the record) | an IBM Quantum account |
| `H5_analysis` | The pre-registered analysis | Zenodo hardware outputs |
| `H6_exploratory` | Batch identifiability, job-purged replay, shot-noise ceiling, sensitivity checks | Zenodo hardware outputs |

Do not edit `hsbcrep/` or `outputs/prereg/`: H3 and H4 check the fingerprint and stop if it changes.

## Setup

```
pip install -r requirements.txt
```

Python 3.12 or later. Tested with Qiskit 2.5.1 and qiskit-ibm-runtime 0.48.0. H4 needs a saved IBM Quantum account;
set `ACCOUNT` in its first cell to yours.

## Citation

See `CITATION.cff`. Data: Zenodo, DOI to be added.

## License

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`LICENSE-DATA`). The market data come from Binance public data
(data.binance.vision).
