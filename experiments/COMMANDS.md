
# experiments — 4-PC parallel commands

Shared settings: Fashion-MNIST, **10 clients**, **10 rounds**, 1 local epoch, non-IID α=0.5, seed **42**.

| PC             | Hardware               | Role                    |
| -------------- | ---------------------- | ----------------------- |
| **PC-1** | **Core i7**      | Baselines + integration |
| **PC-2** | **Core i5**      | DS-ANE negotiation      |
| PC-3           | Core i7 (or spare CPU) | AAVL + communication    |
| PC-4           | Core i5 (or spare CPU) | RAG + privacy           |

Results: `experiments_2/results/pcN/<experiment_id>/` (`summary.json`, `summary.csv`, `history.csv`).

---

## On every PC (once)

```powershell
cd C:\Users\nooru\Documents\SRA-FL
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Smoke test:

```powershell
python experiments/run_pc1.py --pilot --only fedavg
```

---

## Run at the same time

### PC-1 — Core i7 (baselines + integration)

```powershell
cd C:\Users\nooru\Documents\SRA-FL
.\.venv\Scripts\Activate.ps1
python experiments/run_pc1.py --device cpu
```

### PC-2 — Core i5 (DS-ANE)

```powershell
cd C:\Users\nooru\Documents\SRA-FL
.\.venv\Scripts\Activate.ps1
# Optional: GROQ_API_KEY in .env for N3_dsane_llm
python experiments/run_pc2.py --device cpu
```

Optional heavy LLM complete stack on i5 (only if time left):

```powershell
python experiments/run_pc2.py --device cpu --only integ_complete_llm
```

### PC-3 — AAVL + communication

```powershell
python experiments/run_pc3.py --device cpu
```

### PC-4 — RAG + privacy

```powershell
python experiments/run_pc4.py --device cpu
```

---

## What each PC produces

| PC       | Experiment IDs                                                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pc1 (i7) | `B1_fedavg`, `B2_fedprox`, `I2_fedavg_aavl_dsane`, `I3_complete_sra_fl`                                                                                      |
| pc2 (i5) | `N1_fixed_config`, `N2_rule_based`, `N3_dsane_llm` (+ optional `I3_complete_sra_fl_llm`)                                                                     |
| pc3      | `A1_no_verification`, `A2_threshold_verification`, `A3_aavl_historical`, `C1_comm_fixed`, `C2_comm_threshold`, `C3_comm_learning`                        |
| pc4      | `R0_retrieval_probe`, `R1_no_rag`, `R2_flat_rag`, `R3_hierarchical_rag`, `P1_resource_only`, `P2_privacy_aware`, `P3_epsilon_high`, `P4_epsilon_low` |

---

## Copy to one PC and merge

```powershell
Compress-Archive -Path experiments\results\pc1 -DestinationPath pc1_results.zip
# same for pc2, pc3, pc4
```

On the consolidation PC:

```powershell
Expand-Archive -Path pc1_results.zip -DestinationPath experiments\results -Force
Expand-Archive -Path pc2_results.zip -DestinationPath experiments\results -Force
Expand-Archive -Path pc3_results.zip -DestinationPath experiments\results -Force
Expand-Archive -Path pc4_results.zip -DestinationPath experiments\results -Force

python experiments/merge_results.py
```

---

## Partial re-runs

```powershell
python experiments/run_pc1.py --device cpu --only baselines
python experiments/run_pc1.py --device cpu --only integration
python experiments/run_pc2.py --device cpu --only dsane
python experiments/run_pc3.py --device cpu --only aavl
python experiments/run_pc4.py --device cpu --only rag
```
