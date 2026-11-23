# Correcting the Inference Performance Index (IPI)

Experiment harness and outputs for a review of Epoch AI's
[Inference Performance Index](https://epoch.ai/publications/inference-performance-index) (IPI) v1.
It re-runs IPI's own search with corrections to how the simulator prices standard inference
optimizations, and reports how the index changes per chip.

The IPI code is **not modified**. Every correction is an in-process patch applied while IPI's
unmodified `reproduce.py sweep` runs, so each configuration is fully re-searched (layouts and
batch sizes re-optimize under the corrected costs).

## Headline results

Central estimate = all fixes + BF16 attention (IPI's stated policy) + measured cache misses.
Values in GB300e (GB300 NVL72 = 1). Full data: `outputs/final_chart_matrix.csv`.

| Chip | Published | Corrected (central) | Range across scenarios |
|---|---|---|---|
| AMD MI455X | 2.89 | 2.09 | 2.09 – 2.60 |
| NVIDIA VR200 | 2.18 | 1.57 | 1.57 – 1.94 |
| Microsoft Maia 300 | 1.59 | 0.86 | 0.86 – 1.29 |
| NVIDIA GB300 | 1.00 | 1.00 | — |
| Google TPU v7 | 0.74 | 0.85 | 0.83 – 0.85 |
| NVIDIA B300 | 0.54 | 0.87 | 0.81 – 0.87 |
| AMD MI355X | 0.55 | 0.87 | 0.74 – 0.87 |
| NVIDIA H100 | 0.040 | 0.26 | 0.20 – 0.26 |

The spread between the best and worst chip shrinks from 73× to about 10×.

## Corrections (harness switches)

| # | Issue in IPI v1 | Switch |
|---|---|---|
| 1 | Spec-decode verification reads the KV cache once per verify position (4×), not once per request | `kfix=C1` |
| 1b | Related re-reads: Kimi draft KV (3×), GLM sparse-attention indexer keys (4×), minor GLM/Kimi accounting | `kfix=C2,C4,C5,C6,C7` |
| 2 | Fixed parallelism rules (8-GPU systems limited to node-local EP, serial PP) instead of a search | `free_layouts` |
| 3 | Pipeline parallelism counted with one microbatch in flight | `pp_pipelined` |
| 4 | Expert weights stored at compute precision (FP8) on chips without FP4 math | `w4a8_experts` |
| 5 | Cache misses excluded from the workload ratio (measured from IPI's own trace) | scoring: `r = 8.56` (`scripts/ratio_at.py`) |
| 6 | MoE dispatch/combine priced as an all-gather instead of an all-to-all | `kfix=C3` |
| 7 | Attention precision inconsistent across phases (BF16 prefill, FP8 decode) | `attn_prec=bf16` / `attn_prec=fp8` |

Tested and found small (not in the charts): TPU torus bisection (`tpu_bisection`), EP load
imbalance (`ep_imbalance`), compute/communication overlap (`comm_overlap`, `fuse=...`),
uniform derates (`derate`), chip-budget caps (`budget`), prefill-in-decode serving (`scripts/agg/`).

## Layout

```
scripts/
  harness.py            stackable in-process patches; runs IPI's reproduce.py sweep
  pool.py               parallel runner (shards each job by chip; CPU/RAM-bounded)
  jobs_final.txt        every run the outputs/ charts use
  make_overlay.py       reference overlay: IPI's frozen inputs + VR200 Rubin revision + Trainium2/3
  trainium_ext.py       port of IPI's Trainium research extension (reproduces published 0.128 / 0.349)
  fastyaml.py           memoizes AISim's per-call YAML parsing (~6x faster sweeps; identical results)
  kernel_audit/         kernel-accounting corrections (kfix=C1..C7)
  precision_audit/      precision / fusion re-timing (prec=..., fuse=...)
  sweep_kv_once.py      standalone reference implementation of correction 1
  score.py, ratio_at.py scoring against the published index; cache-miss re-scoring
  make_final.py, make_heatmap.py, make_scenarios.py   charts in outputs/
  cache_research/       miss-rate measurement on the SyFI Coding Trace
  agg/                  aggregated (chunked prefill-in-decode) serving experiment
  op_breakdown.py, prefill_ops.py, step_roofline.py, verify/probe_fix.py   single-point engine probes
extra_hardware/         generated Trainium profiles (scripts/trainium_ext.py write)
outputs/                final charts (PNG/SVG) and data (CSV/JSON)
```

## Reproducing

Requirements: Linux x86-64, Python 3.12, about 8 cores. A full run of `jobs_final.txt` takes
a few hours.

1. **Clone IPI at the reviewed commit** into `./ipi-repo`:
   ```bash
   git clone https://github.com/jasonlizhengjian/inference-performance-index ipi-repo
   git -C ipi-repo checkout 9299d1f2c4ec736fb7c82df72443b4e6729cacfc
   ```
2. **Install IPI's pinned AISim environment** (AISim 0.13.0, commit `2f0f4cd`; see IPI's README):
   ```bash
   mkdir -p ipi-repo/vendor && curl -L -o ipi-repo/vendor/aisimulate-0.13.0-cp311-abi3-linux_x86_64.whl \
     https://github.com/jasonlizhengjian/inference-performance-index/releases/download/v1.0.0/aisimulate-0.13.0-cp311-abi3-linux_x86_64.whl
   (cd ipi-repo && python3 environment/setup_current.py && .venv-current/bin/python reproduce.py doctor)
   ```
3. **Rebuild the published Rubin revision** (the VR200 baseline):
   `(cd ipi-repo && python3 reproduce.py rubin-update)`
4. **Build the reference overlay:** `python3 scripts/make_overlay.py`
5. **Run the sweeps:** `python3 scripts/pool.py scripts/jobs_final.txt` (writes `results/<run>/parts/`)
6. **Make the charts:**
   ```bash
   PY=ipi-repo/.venv-current/bin/python
   $PY scripts/make_final.py ipi-repo outputs/final_chart
   $PY scripts/make_heatmap.py ipi-repo outputs/heatmap_fixes
   $PY scripts/make_scenarios.py ipi-repo outputs/scenario_charts
   ```

With no switches, the harness reproduces the published frontiers exactly. That includes the
VR200 Rubin revision and Trainium2/3.

**Cache-miss measurement:** download `syfi_coding_trace.jsonl.gz` from
[TraceLab v0.0.2](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.2). Its database hash
matches IPI's `workload.json`. Then run
`python3 -I scripts/cache_research/analyze_syfi.py <path-to-jsonl.gz> results.json`.

## Caveats

- All numbers are speed-of-light estimates (no per-chip software efficiency), as in IPI.
- Corrections 1, 1b and 6 adjust AISim's per-op timings. The proper fix is upstream in AISim
  (for example, a query-length axis for MLA/DSA attention).
- Cache misses are charged a full re-prefill at IPI's per-token prefill rate, which is an
  upper-ish cost. The measured rate comes from one lab's interactive trace.
- Unreleased chips use IPI's pre-release hardware profiles. For example, TPU v8i's modelled
  BF16 rate drives its ranking.
- A100 (published at 0.006) has no profile in the IPI repository and is not reproduced.
