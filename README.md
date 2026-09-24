# SC4061 / CE4003 / CZ4003 Computer Vision — Lab 1

Point Processing + Spatial Filtering + Frequency Filtering + Imaging Geometry.

## Layout

| Path | Purpose |
|---|---|
| `images/` | Input images supplied by the course (NTULearn → Laboratory Material/Images) |
| `matlab/Lab1.m` | **Graded submission item:** all sections in ONE `.m` file (report req. 3b) |
| `python/` | Executable Python mirror used to generate every figure in this container |
| `results/secNN/` | Generated figures, one folder per lab section |
| `creativity/` | **Graded submission item:** self-contained program for §2.8 (report req. 3c) |
| `report/` | **Graded submission item:** the written report (req. 3a) |
| `docs/PLAN.md` | Delivery plan and status tracker |

## Reproducing the results

```bash
pip install numpy scipy matplotlib pillow
python3 python/lab1.py --all          # writes every figure into results/
python3 python/lab1.py --section 2.3  # single section
```

`matlab/Lab1.m` is the canonical submission and is written for MATLAB + Image
Processing Toolbox. It produces the same figures; it is kept byte-for-byte in
step with the Python mirror so the numbers quoted in the report match either
implementation.
