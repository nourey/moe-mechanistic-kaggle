# OLMoE physical capacity experiment — v0.11.0

This is an executable **single-host, native AsyncLLM** experiment, not a claim that
new GPU measurements have already passed. It reuses the verified model, workload,
ownership/KV hooks and bounded MoE observer. No BLIS calibration, Qwen/DeepEP,
disaggregated NIXL, KEDA or Real EPP experiment is performed here.

## Comparisons and physical scope

| Arm | Native groups | Full physical GPUs | Main offer routing |
|---|---|---:|---|
| A | D1: TP1/PP1/DP1/EP1 | 1 | All offers to D1 rank 0 |
| B | D1: TP1/PP1/DP2/EP2 | 2 | All offers to D1, alternating rank ownership |
| C0 | Independent D1 and D2, each TP1/PP1/DP2/EP2 | 4 | D2 resident throughout; all offers to D1 |
| C1 | Same resident groups as C0 | 4 | Before midpoint D1; afterwards alternate D1/D2, then alternate ranks within each group |

Each group is a separate process/session with disjoint CUDA UUID visibility and its
own native DP rendezvous. Startup and fully drained shutdown collect an actual EP
all-gather membership witness containing the group incarnation, GPU UUID and rank.
All expected peers must match; configuration names alone do not prove isolation.
The witness is an experiment identity check, not cryptographic attestation.

**A→B is a composite DP/EP/layout comparison:** scheduler/KV owner count changes too.
It cannot establish an isolated expert-parallel speedup. B→C1 also changes resident
hardware. **C0→C1 is the preferred additional-group contrast**, holding both groups
and four GPUs resident in both arms. D2 is primed identically even in C0.

Routing is one frozen client assignment plan, not an EPP eligibility authority or
command transport. Only new offers after its explicit midpoint can use D2; existing
request assignments never migrate. A late actual submission crossing that boundary
invalidates the matched run. The single global offered cohort is partitioned, never
duplicated across groups. These engines perform **both prefill and decode**; the D1/D2
names do not make this a pure decode-capacity experiment.

## Preserved pins and feasibility

* Model: `allenai/OLMoE-1B-7B-0924`, revision
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`; complete file hashes in `model-pin.json`.
* Runtime: the existing 199-wheel `runtime-lock.txt`, Linux x86_64 CPython 3.13,
  vLLM 0.31.0, torch 2.13.0+cu130, Triton 3.7.1. Exact wheel hashes and installed
  versions are checked before native startup. Selected vLLM Python source hashes
  are checked against the existing observation contracts; this does not assert that
  a wheel includes unrelated local vLLM instrumentation.
* FP16, eager, TP1/PP1, Triton MoE, `allgather_reducescatter`, native async setting,
  no APC/DBO/EPLB, max context 512, max sequences **4 per rank**, token budget 512,
  memory utilization **0.8**, temperature 0, seed 1525, output cap 8.
* FP16 tensor floor: EP1 **13,838,323,712 bytes**, EP2 **7,395,872,768 bytes per
  GPU**. This includes replicated non-expert parameters; it excludes runtime,
  workspace and KV overhead. Native load/primer must still pass.
* **A is infeasible on the retained T4 allocation at unchanged 0.8 utilization:**
  its actual CUDA total was 15,636,037,632 bytes, so even weights exceed the target.
  No silent quantization, offload or utilization increase is allowed. A larger-memory
  GPU is a candidate, not a certified fit. A/B must use the same GPU SKU/settings.

The future host needs four exclusively allocated, homogeneous, full NVIDIA GPUs
(distinct UUIDs and PCI devices, no MIG), a compatible driver for the locked CUDA
runtime, functioning native NCCL/AGRS, sufficient shared memory/host RAM and local
storage for the locked runtime, 13.84 GB model payload, compiler cache and evidence.
Actual driver/kernel/architecture compatibility remains a physical preflight. No
particular GPU SKU, container digest, fabric or infrastructure allocation is invented.
This first runner supports one Linux host, not cross-node scheduling or Kubernetes.

## Frozen workload, warmup and timing

`freeze` reuses the historical **85-offer / 100-second** calendar from
`load_replay_trace.json`; it does not remeasure capacity. Intensity uniformly scales
integer offsets and duration (0.25–4 operational range). Prompt token IDs come from
`load_protocol.json`: narrative, identifiers or a fixed mixed pattern. All arms and
repetitions consume the **same saved workload SHA**. Content, input token IDs and
output cap are fixed; realized generated lengths remain observations.

Every group first replays the complete cohort as a retained **primer**, drains, then
waits at a shared monotonic-clock barrier before the measured pass. Primer is excluded
from main metrics. This improves shape preparation without guaranteeing all JIT/cache
states are identical. All native processes are recreated between independent trials.
Arm order reverses on alternating repetitions; requests within a trial are not treated
as independent statistical repetitions.

Main offers are open-loop tasks. Ledgers distinguish planned deadline, actual frontend
submission, native acceptance/owner, grants, progress and terminal receipt. Client
lateness is reported, not relabelled as server waiting. TTFT and E2E begin at actual
submission; the planned-time ledger remains available for offered-to-response analysis.
TPOT uses frontend receipt times, not GPU kernel time. Coalesced token increments make
strict per-token TPOT **UNKNOWN**; a separately labelled chunk proxy is retained.
Zero-token and abnormal completions cannot become successful SLO requests.

SLO thresholds default to `null`: goodput is **UNDEFINED**, not zero. Supply an
owner-chosen JSON with exactly `ttft_seconds`, `mean_tpot_seconds`,
`end_to_end_seconds` (positive seconds or null). Freeze it before inspecting compared
outcomes. All planned offers remain in the denominator; missing/failed/censored
requests are not dropped. Unknown required token timing produces a goodput interval;
a known violated bound is a failure even if another metric is unknown. Follow-up is
fixed at 60 seconds after offers stop. These are finite-cohort results, not a measured
steady-state saturation knee or statistical superiority claim.

## Measurement modes and causal limits

**reference** retains source/device/membership receipts, native rank ownership and
frontend per-request chunk/terminal accounting. Heavy scheduler/KV/routing hooks are
off; it is a lightweight reference, not zero-overhead execution. Use ≥2 independent
repetitions for a non-smoke performance comparison.

**diagnostic** adds the existing rank-local scheduler population/limits/grants,
allocation refusal/free and per-request KV conservation checks. For EP2 it also uses
the verified multi-request InputBatch joins, bounded layer/top-k routing, placement,
active-expert and padding/timing records. Sampling limits and dropped/unknown coverage
are retained; there is no full tensor/profiler collection. **EP1 expert routing/timing
remains UNKNOWN** because the existing observer requires paired AGRS ranks; generic
ownership/scheduler/KV evidence is still supported. EP1 collectives are not applicable.

Keep diagnostic and reference results separate; matched workload hashes allow an
overhead comparison without assuming instrumentation is free. No numerical negligible-
overhead threshold is invented. CUDA event spans include their stated dependencies;
they are not automatically pure network cost or critical-path proof. Active experts,
padding and queue growth are observations, not bottleneck labels.

Pre-midpoint R/K snapshots carry rank-local publication age and are not an atomic
group snapshot. GPU routing arrays are exported only after drain: **retrospective,
unavailable as online predictor inputs**. No future output lengths are supplied.
The earlier ~300 ms content/routing difference stays an unresolved causal question.

## Manual Kaggle B smoke (two T4s only)

Use a fresh authorized T4×2 session, Internet on. Run only this cell, not old notebook
qualification cells. The new directory is intentionally never overwritten. Existing
verified runtime/model caches are reused when valid; an expired session may need the
same locked runtime/model restore. There is no repeated model/kernel/NCCL qualification
suite. This cell executes only B, one diagnostic primer and one eight-offer main pass.

```python
import json, subprocess, sys
from pathlib import Path
from IPython.display import FileLink, display

repo = Path('/kaggle/working/moe-capacity-v0.11.0')
if repo.exists():
    raise RuntimeError('Bu klasör zaten var; eski kanıtı silmeden yeni bir çalışma klasörü seç.')
subprocess.run(['git', 'clone', '--branch', 'capacity-v0.11.0', '--depth', '1',
    'https://github.com/nourey/moe-mechanistic-kaggle.git', str(repo)], check=True)
print('SOURCE_COMMIT:', subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
p = subprocess.Popen([sys.executable, '-u', '-B', str(repo/'capacity_kaggle.py')],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
receipt = None
for line in p.stdout:
    print(line, end='', flush=True)
    if line.startswith('EVIDENCE_JSON:'):
        receipt = json.loads(line.split(':', 1)[1])
code = p.wait()
if receipt and receipt.get('archive'):
    display(FileLink(receipt['archive']))
if code or not receipt:
    raise RuntimeError('Çalıştırıcı tamamlanmadı; üstteki hata ve oluşan ZIP korunmalı.')
```

Check the printed commit against the reviewed delivery SHA. Return the new
`moe-capacity-smoke-*.zip` and its receipt, including failed runs. A Kaggle B PASS is
not evidence for A fit, independent four-GPU residency or C0/C1 intervention.

## Future Linux four-GPU procedure

Run only after hardware access, resource budget and experiment authorization. Clone
the reviewed commit on the GPU host. Never run on someone else's active allocation.
The following uses a new external directory and the existing locks; do not install
into system Python. Base Python 3.13 pip must support `--python`.

```bash
export MOE_CAPACITY_ROOT="$HOME/moe-capacity-run-01"
mkdir "$MOE_CAPACITY_ROOT"
python3.13 -m venv --without-pip "$MOE_CAPACITY_ROOT/venv"
python3.13 -c 'import json,sys; from pathlib import Path; c=json.loads(Path("model-qualification.json").read_text()); Path(sys.argv[1]).write_text("pip=="+c["seed_pip_version"]+" --hash=sha256:"+c["seed_pip_sha256"]+"\n")' "$MOE_CAPACITY_ROOT/seed-pip.txt"
python3.13 -m pip --python "$MOE_CAPACITY_ROOT/venv/bin/python" install --only-binary=:all: --require-hashes --index-url https://pypi.org/simple -r "$MOE_CAPACITY_ROOT/seed-pip.txt"
"$MOE_CAPACITY_ROOT/venv/bin/python" -m pip install --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --report "$MOE_CAPACITY_ROOT/pip-install-report.json" -r runtime-lock.txt
mkdir "$MOE_CAPACITY_ROOT/model-receipts"
"$MOE_CAPACITY_ROOT/venv/bin/python" -B model_qualify.py --acquire --cache "$MOE_CAPACITY_ROOT/cache" --output "$MOE_CAPACITY_ROOT/model-receipts"
export MOE_MODEL_PATH="$(python3.13 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$MOE_CAPACITY_ROOT/model-receipts/model-acquisition.json")"
python3.13 -B capacity_run.py plan
python3.13 -B capacity_run.py freeze --smoke --output "$MOE_CAPACITY_ROOT/smoke.json"
python3.13 -B capacity_run.py run --python "$MOE_CAPACITY_ROOT/venv/bin/python" --model "$MOE_MODEL_PATH" --install-report "$MOE_CAPACITY_ROOT/pip-install-report.json" --devices 0,1,2,3 --workload "$MOE_CAPACITY_ROOT/smoke.json" --arms B,C0,C1 --mode diagnostic --repetitions 1 --output "$MOE_CAPACITY_ROOT/evidence-smoke"
```

Inspect this archive before a load comparison: correct four physical identities,
two isolated EP worlds, normal terminals, exact native owners, request/KV conservation,
boundary integrity, source hashes and full shutdown. Do not proceed after PARTIAL,
missing routing coverage or forced cleanup. A remains optional until memory fit.

Freeze one main workload and owner SLO file outside the checkout, then use the exact
same files in the performance and diagnostic invocations:

```bash
python3.13 -B capacity_run.py freeze --family narrative --intensity 1 --output "$MOE_CAPACITY_ROOT/workload.json"
python3.13 -B capacity_run.py run --python "$MOE_CAPACITY_ROOT/venv/bin/python" --model "$MOE_MODEL_PATH" --install-report "$MOE_CAPACITY_ROOT/pip-install-report.json" --devices 0,1,2,3 --workload "$MOE_CAPACITY_ROOT/workload.json" --arms B,C0,C1 --mode reference --repetitions 2 --output "$MOE_CAPACITY_ROOT/evidence-reference"
python3.13 -B capacity_run.py run --python "$MOE_CAPACITY_ROOT/venv/bin/python" --model "$MOE_MODEL_PATH" --install-report "$MOE_CAPACITY_ROOT/pip-install-report.json" --devices 0,1,2,3 --workload "$MOE_CAPACITY_ROOT/workload.json" --arms B,C0,C1 --mode diagnostic --repetitions 1 --output "$MOE_CAPACITY_ROOT/evidence-diagnostic"
```

Add `--slo /absolute/path/to/frozen-slo.json` identically to both commands when chosen.
On suitable identical larger GPUs add A as `--arms A,B,C0,C1`; do not change its model
or memory settings to force a fit. A first-run device/port/native incompatibility is
preserved as BLOCKED, not patched by a fallback. Review one intensity before selecting
another from the bounded range. No guaranteed overload, hardware utilization or D2
benefit is assumed. Do not select favorable intensities after viewing treatment.

## Bounds, failures and evidence

Per pass: ≤128 offers, ≤64 pending requests per group, ≤8 output tokens, duration
25–400 seconds for the full trace (4 seconds for smoke), fixed 60-second follow-up.
Startup timeout 600 seconds per trial; whole-trial budget is checked before starting
another trial (`--max-wall-seconds`, default four hours, not an expenditure approval).
Disk below 2 GiB free stops the next trial. Any dirty source, wheel/model/hash drift,
wrong owner, duplicate request, ambiguous GPU, invalid collective membership, KV
conservation failure, boundary crossing or incomplete drain stops later trials.
Only owned process sessions may be terminated; existing GPU jobs cause refusal.
Forced cleanup is a failure, not a successful result. Log/ZIP storage beyond the
initial free-space guard is not a filesystem quota guarantee.

`PASS_COLLECTION_NOT_CAUSAL_VALIDATION` means this collection completed. It does not
mean a bottleneck mechanism, statistical benefit, prediction accuracy or GPU parity
claim passed. Main comparison includes every prespecified arm/repetition; failures
and UNKNOWN measurements stay visible. Native API/collective startup under the new
multi-group harness still needs its first physical run; CPU fixtures cannot prove it.

Expected outer archive (`moe-capacity-<UTC>-<id>.zip`, plus `.zip.sha256`):

```text
SOURCE.json, source/                 # run-wide Git identity and exact source copies
WORKLOAD.json, SLO.json, PREFLIGHT.json
model-receipts.json, pip-install-report.json, *-command.json
status.json, COMPARISON.json         # descriptive fixed contrasts, no fitting
rep1-B-reference/                    # similarly C0/C1 and independent repeats
  ROUTING_PLAN.json, TRIAL.json      # or TRIAL_FAILURE.json
  control/GO_primer.json, GO_main.json, STOP.json
  D1/                               # D2/ exists for C0/C1
    spec.json, native.log, configuration.json, native-sources.json
    READY.json, DONE.json, loaded-dp*.json, after-dp*.json
    identity-loaded-dp*.json, identity-after-dp*.json
    owned-before.json, owned-after.json, request-identity-joins.json
    primer-plan.json, main-plan.json, *-analysis.json
    client-dp-1.jsonl, core-dp*.jsonl
    *-load-worker-dp*.json            # EP2 diagnostic only
    watchdog.json                    # only if forced/uncertain cleanup
SHA256SUMS
```

The Kaggle outer archive additionally contains preparation ZIP, command receipt and
the nested capacity archive. ZIP CRC is checked during sealing; verify every retained
file against SHA256SUMS before analysis. Model weights and venvs are excluded.

The llm-d benchmark stack remains appropriate for later HTTP/Kubernetes experiments.
It is not imported here: its standup/HTTP workload path does not provide this pinned
direct-AsyncLLM rank-owner/InputBatch/KV join. Reusing the existing small native
experiment avoids adding a second deployment framework or pretending that this
client routing plan is production llm-d routing.
