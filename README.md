# SRA-FL

**Strategic-Robust Agentic Federated Learning**

SRA-FL is a Python implementation of an agent-assisted federated-learning system. It combines FedAvg/FedProx training with:

- dishonest-client report verification;
- EMA-based client reputation;
- client-server workload negotiation;
- adaptive upload/compress/wait/skip communication;
- hierarchical retrieval for client profiles;
- and simulated privacy-budget accounting.

The implementation uses Fashion-MNIST, 10 clients, 10 communication rounds, a Dirichlet non-IID partition, FedScale-style device traces, MobiPerf network traces, and a fixed random seed.

## Project Overview

The system has three main parts:

1. **FL training:** A shared CNN is trained with FedAvg or FedProx.
2. **Agentic modules:** AAVL audits clients, DS-ANE negotiates workloads, Q-learning chooses communication actions, RAG retrieves client profiles, and a privacy manager tracks resource budgets.
3. **Experiments:** A four-PC runner suite executes and compares the baselines and module variants.

## Requirements

- Windows 10/11 or another Python-supported operating system
- Python 3.10
- CPU is sufficient
- Optional NVIDIA GPU and CUDA-enabled PyTorch
- Approximately 5 GB of free storage

Python packages used by the project:

- `torch`
- `torchvision`
- `flwr`
- `numpy`
- `pandas`
- `Pillow`
- `PyYAML`
- `matplotlib`
- `faiss-cpu`
- `python-dotenv`
- `groq` only for the optional LLM
- `pytest` for tests

## Setup on Windows

Open PowerShell in the project folder.

```powershell
cd C:\Users\nooru\Documents\SRA_FL
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the required packages:

```powershell
pip install torch torchvision flwr numpy pandas Pillow PyYAML matplotlib faiss-cpu python-dotenv pytest groq
```

The project source is under `src/`. Make it visible to Python:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

To keep this setting in the current PowerShell session, run it again whenever a new terminal is opened.

## Dataset and Traces

The required data is already included:

```text
data/raw/Fashionmnist/
data/raw/cifar-10/
data/raw/device_profiles/
data/raw/network_traces/
data/raw/availability_traces/
```

No internet download is required for the default Fashion-MNIST experiment.

To use the optional Groq language-model integration:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set:

```text
GROQ_API_KEY=your_key_here
GROQ_MODEL=your_model_name
```

The project also provides a fallback when no API key or model service is available.

## Quick Start

Run a single complete experiment:

```powershell
$env:PYTHONPATH = "$PWD\src"
python experiments\run_pc1.py --device cpu --only fedavg
```

Run a short pilot:

```powershell
python experiments\run_pc1.py --device cpu --pilot --only fedavg
```

Check the command parser for the main simulation:

```powershell
python -m sra_fl --help
```

Run a default Fashion-MNIST simulation:

```powershell
python -m sra_fl --dataset fashion_mnist --strategy fedavg --num-clients 10 --num-rounds 3
```

## Complete Experiment Suite

All experiments use Fashion-MNIST, 10 clients, 10 rounds, seed 42, and a non-IID Dirichlet partition with $\alpha=0.5$.

### PC-1: baselines and integration

```powershell
python experiments\run_pc1.py --device cpu
```

Produces:

- `B1_fedavg`
- `B2_fedprox`
- `I2_fedavg_aavl_dsane`
- `I3_complete_sra_fl`

### PC-2: negotiation

```powershell
python experiments\run_pc2.py --device cpu
```

Produces:

- `N1_fixed_config`
- `N2_rule_based`
- `N3_dsane_llm`

Optional complete LLM run:

```powershell
python experiments\run_pc2.py --device cpu --only integ_complete_llm
```

### PC-3: audit and communication

```powershell
python experiments\run_pc3.py --device cpu
```

Produces:

- `A1_no_verification`
- `A2_threshold_verification`
- `A3_aavl_historical`
- `C1_comm_fixed`
- `C2_comm_threshold`
- `C3_comm_learning`

### PC-4: RAG and privacy

```powershell
python experiments\run_pc4.py --device cpu
```

Produces:

- `R0_retrieval_probe`
- `R1_no_rag`
- `R2_flat_rag`
- `R3_hierarchical_rag`
- `P1_resource_only`
- `P2_privacy_aware`
- `P3_epsilon_high`
- `P4_epsilon_low`

## Partial Runs

```powershell
# PC-1
python experiments\run_pc1.py --device cpu --only baselines
python experiments\run_pc1.py --device cpu --only integration

# PC-2
python experiments\run_pc2.py --device cpu --only dsane

# PC-3
python experiments\run_pc3.py --device cpu --only aavl
python experiments\run_pc3.py --device cpu --only comm

# PC-4
python experiments\run_pc4.py --device cpu --only rag
python experiments\run_pc4.py --device cpu --only privacy
```

Use `--dry-run` to validate an experiment without training:

```powershell
python experiments\run_pc1.py --dry-run --only fedavg
```

## Configuration

Default settings are in [configs/base.yaml](/C:/Users/nooru/Documents/SRA_FL/configs/base.yaml).

Other settings are split across:

```text
configs/fashion_mnist.yaml
configs/cifar10.yaml
configs/attack_config.yaml
configs/negotiation_config.yaml
configs/rag_config.yaml
configs/communication_config.yaml
configs/privacy_config.yaml
```

Important settings include:

| Setting             |    Default | Meaning                     |
| ------------------- | ---------: | --------------------------- |
| `num_clients`     |         10 | Number of simulated clients |
| `num_rounds`      |         10 | Number of FL rounds         |
| `local_epochs`    |          1 | Fixed local-training epochs |
| `batch_size`      |         32 | Client batch size           |
| `learning_rate`   |       0.01 | Local SGD learning rate     |
| `momentum`        |        0.9 | SGD momentum                |
| `strategy`        | `fedavg` | `fedavg` or `fedprox`   |
| `dirichlet_alpha` |        0.5 | Non-IID data concentration  |
| `seed`            |         42 | Random seed                 |

## Results

Each experiment writes a result folder under:

```text
experiments/results/<pc>/<experiment-id>/
```

Typical files include:

- `summary.json`: final performance and module summary;
- `summary.csv`: flat summary table;
- `history.csv`: per-round accuracy and loss;
- `run_config.json`: the settings used;
- `metrics/`: detailed per-round and module metrics;
- `logs/sra_fl.log`: execution log.

Merge all multi-PC summaries:

```powershell
python experiments\merge_results.py
```

The merged output is written to:

```text
experiments/results/all_summaries.json
experiments/results/all_summaries.csv
```

## Generate the Paper Figures

After the experiment results are available:

```powershell
python experiments\plot_paper_figures.py
```

This creates the eight figures used in the paper under:

```text
results/figures/
```

It generates:

- `architecture_horizontal.png`
- `fig_baseline_final_accuracy.png`
- `fig_gap1_audit_accuracy.png`
- `fig_gap2_negotiation_accuracy.png`
- `fig_gap3_comm_actions.png`
- `fig_gap4_retrieval_accuracy.png`
- `fig_gap5_privacy_accuracy.png`
- `fig_convergence_accuracy.png`

## Tests

Run the complete test suite:

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest -q
```

Run a targeted integration test:

```powershell
pytest -q tests/test_fl_integration.py
```

## Recorded Experimental Results

The current single-seed results include:

| Configuration            | Final accuracy |
| ------------------------ | -------------: |
| FedAvg                   |         72.65% |
| FedProx                  |         71.94% |
| Complete SRA-FL          |         76.15% |
| Q-learning communication |         77.13% |

Important limitations:

- These runs use one random seed.
- They report observations, not statistical significance.
- Few clients were online in many rounds.
- The privacy account is simulated and is not formal differential privacy.
- RAG and reputation mechanisms did not consistently change client selection in this setup.

## Common Problems

### `ModuleNotFoundError: No module named 'sra_fl'`

Set the source folder before running:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### Missing Fashion-MNIST files

Confirm that these files exist:

```text
data/raw/Fashionmnist/train-images-idx3-ubyte
data/raw/Fashionmnist/train-labels-idx1-ubyte
data/raw/Fashionmnist/t10k-images-idx3-ubyte
data/raw/Fashionmnist/t10k-labels-idx1-ubyte
```

### Groq or LLM calls fail

Disable LLM use or continue with the built-in heuristic fallback:

```powershell
python -m sra_fl --no-llm
```

### Flower or training import errors

Verify that packages are installed in the active virtual environment:

```powershell
python -m pip install flwr torch torchvision faiss-cpu python-dotenv groq
```

## Authors

- Noor Ul Huda
- Malaika Zainab

FAST National University of Computer and Emerging Sciences, Islamabad.
