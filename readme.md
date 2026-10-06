
# ⚡ PagedAttention & vLLM Inference Benchmarks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-v0.6.0+-orange.svg)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end benchmarking suite for evaluating Key-Value (KV) cache memory efficiency, time-to-first-token (TTFT), and throughput under heavy concurrent workloads using **vLLM** and **PagedAttention**.

---

## 📌 Overview

Large Language Model (LLM) serving is heavily constrained by GPU memory, specifically memory fragmented by dynamically sized Key-Value (KV) caches. This repository provides custom profiling scripts, benchmark runners, and evaluation utilities to compare standard continuous memory allocation against virtual paged memory management via **PagedAttention**.

### Key Features

- **Paged Memory Management:** Evaluates non-contiguous block allocation for KV pairs.
- **High-Throughput Benchmarking:** Load testing under variable batch sizes and prompt lengths.
- **Latency Analytics:** Breakdown of Time to First Token (TTFT), Time per Output Token (TPOT), and P95/P99 end-to-end latencies.
- **Memory Profiling:** Tracks peak VRAM usage, KV cache block allocation, and GPU compute utilization.

---

## 🛠️ Requirements & Installation

### Prerequisites

- Linux OS (Ubuntu 22.04+ recommended)
- NVIDIA GPU (Ampere/Hopper architecture with CUDA 12.1+)
- Python 3.10+

### Setup

1. **Clone the repository:**

    ```bash
    git clone [https://github.com/adriennie/vllm-paged-attention-benchmarks.git](https://github.com/adriennie/vllm-paged-attention-benchmarks.git)
    cd vllm-paged-attention-benchmarks
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## 🚀 Quickstart

### Running the Inference Benchmark

Execute a benchmark test on `meta-llama/Llama-2-7b-hf` or any HuggingFace model:

```bash
python bench.py \
    --model meta-llama/Llama-2-7b-hf \
    --num-prompts 100 \
    --block-size 16 \
    --max-model-len 2048 \
    --output-json results.json

```

### Options & Flags

| Flag | Description | Default |
| --- | --- | --- |
| `--model` | HuggingFace model path or name | `meta-llama/Llama-2-7b-hf` |
| `--block-size` | Number of tokens per KV cache block | `16` |
| `--gpu-memory-utilization` | Fraction of GPU memory allocated to vLLM | `0.90` |
| `--max-num-seqs` | Maximum number of concurrent sequences | `256` |

---

## 📊 Directory Structure

```text
.
├── benchmarks/
│   ├── benchmark_throughput.py    # Throughput & request-rate evaluation
│   ├── benchmark_latency.py       # TTFT and TPOT metric capture
│   └── profile_memory.py          # VRAM & KV-cache allocation profiler
├── scripts/
│   └── run_all.sh                 # Complete automated benchmark pipeline
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md                      # Project documentation

```

---

## 📜 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
