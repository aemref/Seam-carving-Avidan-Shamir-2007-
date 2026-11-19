# Seam Carving (Avidan & Shamir, 2007)

A from-scratch NumPy implementation of *Seam Carving for Content-Aware Image Resizing*
(S. Avidan and A. Shamir, ACM Trans. Graph. 26(3), 2007, [doi:10.1145/1276377.1276390](https://doi.org/10.1145/1276377.1276390)):
energy map, minimum seams with dynamic programming, shrinking, enlarging and mask-based object removal.
Final project for INF605 Fundamentals of Image Processing (Turkish-German University, Fall 2026).

> **Status:** skeleton. Every module ships with its final signatures and raises
> `NotImplementedError` until its owner fills it in.

## Setup

```
git clone https://github.com/aemref/Seam-carving-Avidan-Shamir-2007-.git
cd Seam-carving-Avidan-Shamir-2007-
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

Python 3.10 or newer. Before opening a pull request, `ruff check .` and `pytest -q` must be clean.

## Team

| Name | GitHub | Core module | Also owns |
|---|---|---|---|
| Emre | [@aemref](https://github.com/aemref) | `seamcarving/seam.py` | skeleton and CI, `resize.py`, `experiments/common.py`, `run_all.py` |
| Ahsen | [@ahsenayvaz](https://github.com/ahsenayvaz) | `seamcarving/filters.py`, `energy.py` | report assembly |
| Mirza | [@Marzi77](https://github.com/Marzi77) | `seamcarving/carve.py` | `baselines.py`, `cli.py`, integration tests |
| Furkan | [@Freigeist3](https://github.com/Freigeist3) | `seamcarving/enlarge.py` | `tools/prepare_dataset.py`, `tools/check_dataset.py` |
| Berkay | [@berkayaysel](https://github.com/berkayaysel) | `seamcarving/object_removal.py` | `viz.py`, `tools/mask_painter.py`, `tools/make_demo_gif.py` |

## Project documents (Turkish)

- [`PROJE_HARITASI.md`](PROJE_HARITASI.md): project map with the algorithm, pitfalls, course links and experiment plan.
- [`gorevler/`](gorevler/): one task file per team member, including the binding interface contract (section 2.5).
