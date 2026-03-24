# ReasonForge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyCoding8/MCP/blob/main/ReasonForge_Colab.ipynb)

**Deterministic math tools for small language models.**

ReasonForge gives small LLMs (8B–32B) access to a verified SymPy computation backend via tool calling.
Instead of relying on the model to compute, all math is delegated to deterministic tools — the model only reasons about *what* to compute and *how* to present results.

## Architecture

```
User Question → LLM (Qwen3) → Tool Calls → SymPy Backend → Verified Results → LLM → Final Answer
```

**Multi-Turn Agentic Loop:**
1. **Reason:** The model uses `<think>` tags to analyze the problem and decide on a strategy.
2. **Execute:** The model delegates computation to a deterministic tool (SymPy or Python sandbox).
3. **Iterate:** The model observes the verified tool output and either concludes the answer or calls another tool until solved (up to `MAX_ROUNDS`).

## Tools

| Tool | Operations | Backend |
|------|-----------|---------| 
| `math_tool` | compute, solve, simplify, factor, expand, gcd, lcm, prime_factors, divisors, mod_inverse, nsolve, crt + SymPy builtins (totient, fibonacci, isprime...) | SymPy |
| `calculus_tool` | differentiate, integrate, limit, series, summation, partial_fraction, trigsimp, ode_solve, laplace | SymPy |
| `matrix_tool` | determinant, inverse, eigenvalues, eigenvectors, rank, rref, transpose, multiply, add, trace, nullspace, columnspace, charpoly, norm, adjugate, solve (Ax=b) | SymPy |
| `statistics_tool` | describe, mean, median, mode, std, variance, correlation, regression, percentile, zscore, skewness, kurtosis, geometric_mean, harmonic_mean | Python stdlib |
| `code_tool` | run, check, ast_inspect — sandboxed Python code execution, syntax checking, and structure analysis | subprocess |

## Project Structure

```
MCP/
├── core.py                    # Shared LLM request logic, expert definitions, tool schemas
├── experts/
│   ├── math/
│   │   ├── server.py          # MCP server entry point (math tools)
│   │   └── tools/
│   │       ├── preprocess.py  # Expression parser (^ → **, implicit multiplication)
│   │       ├── algebra.py     # algebra + number theory
│   │       ├── calculus.py    # derivatives, integrals, ODEs
│   │       ├── matrix.py      # linear algebra
│   │       └── statistics.py  # descriptive & inferential stats
│   └── code/
│       ├── server.py          # MCP server entry point (code execution)
│       └── tools/
│           └── code.py        # Sandboxed Python runner & syntax checker
├── tests/
│   ├── sanity.py              # Tool unit tests (16 checks)
│   ├── math_benchmark.py      # A/B math benchmark (MATH-500 dataset)
│   ├── code_benchmark.py      # A/B code benchmark (HumanEval)
│   └── results/               # Local benchmark outputs 
├── ui/
│   ├── app.py                 # Gradio chat interface with intermediate thinking steps
│   └── style.css              # Custom UI styles (dark mode, thinking blocks)
├── ReasonForge_Colab.ipynb    # One-click Colab deployment notebook
├── pyproject.toml
├── requirements.txt
├── run_tests.bat              # Local tests launcher (Windows)
└── run_ui.bat                 # Local UI launcher (Windows)
```

## Quick Start (Local)

```bash
# Requires: Ollama running with a supported model (qwen3:8b, qwen3:32b, etc.)
uv sync
uv run python -m ui.app
# Open at http://localhost:7861
```

### Endpoint Defaults (Basic Robustness)

- Outbound model endpoints default to localhost-only.
- Allow remote endpoints explicitly with `RF_ALLOW_REMOTE_ENDPOINTS=1`.
- Extend allowed hosts with `RF_ENDPOINT_ALLOWLIST`.

Examples:

```bash
export RF_ENDPOINT_ALLOWLIST="localhost,127.0.0.1,::1,api.mycompany.com"
export RF_ALLOW_REMOTE_ENDPOINTS=1
```

### Code Tool Docker Option (Basic Sandbox)

`code_tool` supports optional Docker isolation with safe fallback:

- `RF_CODE_TOOL_ISOLATION=auto` (default): use Docker if available, else process mode
- `RF_CODE_TOOL_ISOLATION=docker`: prefer Docker, fallback to process if unavailable
- `RF_CODE_TOOL_ISOLATION=process`: force process mode

Optional image override:

```bash
export RF_CODE_TOOL_DOCKER_IMAGE=python:3.11-alpine
```

## Colab Deployment (GPU)

Open `ReasonForge_Colab.ipynb` in Google Colab Pro with an A100 GPU.
It clones this repo, installs Ollama + `qwen3:32b`, and launches the UI with a public Gradio link.

## Benchmarking

```bash
# Math benchmark — MATH-500 (requires Ollama running)
uv run python -m tests.math_benchmark --model llama3.2:3b --n 10
uv run python -m tests.math_benchmark --model qwen3:32b --n 50 --think

# Code benchmark — HumanEval (requires Ollama running)
uv run python -m tests.code_benchmark --model qwen3:8b --n 20
uv run python -m tests.code_benchmark --model qwen3:32b --n 164 --think
```

## Running Sanity Tests

```bash
uv run python -m tests.sanity
```

## Running All Unit Tests

```bash
uv run python -m tests.test_all
```

## Running Release Gate

```bash
uv run python -m tests.release_gate
```

## Benchmark Results

### MATH-500 (`qwen3:8b`, 50 problems)

| Metric | Baseline | ReasonForge |
|---|---|---|
| **Correct** | 43/50 | **45/50** |
| **Uniform Accuracy** | 86.0% | **90.0%** (▲ +4.0%) |
| **Weighted Score**  | 144/176 | **154/176** |
| **Weighted Accuracy** | 81.8% | **87.5%** (▲ +5.7%) |

- **Delegation:** 40.0% (20/50) of tasks used tools
- **Avg Rounds:** 1.5 
- **Avg Time:** Baseline 46.3s vs ReasonForge 31.0s (Δ -15.2s)

#### By Difficulty
```text
Level 1      5/5   100%  ████████████████████
Level 2      7/7   100%  ████████████████████
Level 3      8/9   89%   █████████████████
Level 4     14/15  93%   ██████████████████
Level 5     11/14  79%   ███████████████  (+14%)
```

#### By Category
```text
Algebra                   10/12  83%   ████████████████
Counting & Probability     4/4   100%  ████████████████████
Geometry                   4/4   100%  ████████████████████
Intermediate Algebra      11/13  85%   ████████████████  (+8%)
Number Theory              2/2   100%  ████████████████████
Prealgebra                 7/7   100%  ████████████████████
Precalculus                7/8   88%   █████████████████  (+12%)
```

### HumanEval (Code: `qwen3:8b`, 160 problems)

| Metric | Baseline | ReasonForge |
|---|---|---|
| **Pass@1** | 4/160 | **102/160** |
| **Accuracy** | 2.5% | **63.7%** (▲ +61.2%) |

- **Delegation:** 31.2% (50/160) of tasks used tools
- **Avg Rounds:** 1.5 
- **Avg Time:** Baseline 23.9s vs ReasonForge **24.8s** (Δ +0.9s)
- **Wins vs Losses:** ReasonForge successfully solved 100 problems that the Baseline failed on, while only losing 2.

### Key Takeaways

Testing the **8-billion parameter** `qwen3` model reveals exactly why deterministic tool-delegation is crucial for smaller models:

1. **Math (MATH-500):** While both models achieved incredibly high baseline accuracy, giving the model access to the SymPy backend **massively reduced latency** (cutting the average computation time from `46.3s` down to `31.0s`), all while squeezing out an extra `~5%` in weighted grading accuracy.
2. **Code (HumanEval):** Without sandboxed execution tools, the 8B model almost entirely collapsed on HumanEval, only passing a dismal `4/160` (2.5%) of the problems. However, the simple addition of the ReasonForge Python runtime tools allowed the exact same model to safely hypothesize, test, and iteratably structure its code, propelling its accuracy to **102/160 (63.7%)**—a gigantic **+61.2% improvement** with zero fine-tuning required.

## Tech Stack

- **LLM Backend:** [Ollama](https://ollama.com) (local) or any OpenAI-compatible API
- **Math Engine:** [SymPy](https://sympy.org) — symbolic computation
- **Math Grading:** [math-verify](https://github.com/huggingface/Math-Verify) — deterministic LaTeX parser (Linux/Colab)
- **Code Grading:** Self-contained HumanEval harness (inspired by [openai/human-eval](https://github.com/openai/human-eval))
- **UI:** [Gradio](https://gradio.app) — chat interface with LaTeX rendering
- **Protocol:** [MCP](https://modelcontextprotocol.io) (Model Context Protocol) compatible
