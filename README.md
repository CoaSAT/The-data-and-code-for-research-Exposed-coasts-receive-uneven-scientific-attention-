# Coastal scientific attention: reproducible figure code and processed data


This archive reproduces main-text Figures 1, 2 and 3 of the submitted manuscript. 

Python 3.12.14 is the validation environment. Windows was used for validation; a graphical desktop, GPU, network connection and external services are not needed to run the figures. Install Python dependencies in an isolated environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions found in the successful local validation environment. Installation in a fresh environment has not been independently tested. Arial is used for manuscript typography; where it is unavailable, font substitution can change text placement. Figure 1 also uses Windows Arial font paths with its existing font fallback.

## Reproduce the figures

From this directory:

```bash
python reproduce.py
python reproduce.py --figure 1
python reproduce.py --figure 2
python reproduce.py --figure 3
```

The default runs all three figures. Each figure also has a `plot.py` entry point. All input paths resolve relative to their code directory. Outputs are written to each figure's `outputs/` directory; subsequent runs overwrite outputs with the same filenames. Execution logs and the machine-readable run report are written to `validation/`. The processed-data rendering is deterministic; no fresh stochastic analysis is performed. Rendering usually takes seconds to a few minutes per figure, depending on the machine.

## Archive structure

- `Figure1/`: global attention distribution, attention-class shares and Lorenz curve.
- `Figure2/`: exposure/attention flows, blind-spot burden map and IPCC regional totals.
- `Figure3/`: population representation, GDP association, network-site map and 1,000 km window associations.
- Each figure contains its code, `data/` and `outputs/`.
- `DATA_DICTIONARY.csv`: input tables, columns and original project paths.
- `PROVENANCE.md`: input processing, interpretation and limitations.
- `validation/`: actual execution records and input/output checksums.

Figure 1 produces PNG/PDF. Figure 2 produces PNG (200/600 dpi), PDF and compressed TIFF. Figure 3 produces PNG, PDF, SVG and 600 dpi TIFF, including a separate c/d panel version. Figure 1's PDF embeds its raster rendering; it is not an editable vector reconstruction.


