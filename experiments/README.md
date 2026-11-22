# experiments_2 — multi-PC SRA-FL plan

```
experiments_2/
  run_pc1.py   # Core i7 — baselines + integration
  run_pc2.py   # Core i5 — DS-ANE negotiation
  run_pc3.py   # AAVL + communication
  run_pc4.py   # RAG + privacy
  common.py
  merge_results.py
  COMMANDS.md
  results/pc1..pc4/
```

PC-1 / PC-2 are **CPU-only** (Core i7 / Core i5). No GPU required.

Experiments are independent: run in parallel, copy `results/pcN`, then `python experiments_2/merge_results.py`.

See **COMMANDS.md**.
