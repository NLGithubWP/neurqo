# Reproduction Guide

This guide reproduces the paper experiments using released experience buffers and checkpoints. It does not replace saved runtimes with fresh SQL timings. Use the [README](../README.md) for the separate end-to-end Docker benchmark and [new-dataset training](../README.md#benchmark-new-workloads).

## 1. Setup

The image already contains the patched PostgreSQL engine, extensions, Python environment, databases, and pretrained policies. There is no need to apply the patch, compile PostgreSQL, install Python dependencies, or start the policy runtime manually.

| Component | Bundled configuration |
|---|---|
| Database engine | PostgreSQL 16.3-based NeurDB with NeurQO optimization primitives, `pg_hint_plan`, and `pg_lip_bloom` |
| Database connection | Inside the container: `127.0.0.1:5432`, user `neurdb` |
| PostgreSQL configuration | `/opt/neurqo/postgresql.conf`; `shared_buffers=32GB`, `effective_cache_size=96GB`, `work_mem=128MB`; parallel query execution and JIT disabled |
| Python runtime | Python 3.11, CPU PyTorch, NumPy, and psycopg2 in `/code/neurdb-dev/aiengine/ai_for_db/query_opt/.venv`; `python` already uses this environment inside the container |
| Policy runtime | JOB: port `8088`; STACK: `8089`; TPC-H: `8090`, all on container loopback |
| Active checkpoints | `/opt/neurqo/models/<dataset>/<protocol>/<fold>/best.pt` |
| Database source | `/code/neurdb-dev/dbengine` |
| Research source | `/opt/neurqo/source` |

The container entrypoint starts PostgreSQL, exports database catalog snapshots to `/opt/neurqo/runtime/catalogs/`, and starts one policy service per dataset. Each service initially loads its Random-a checkpoint; the benchmark client loads the matching checkpoint for every requested protocol and fold. Training does not start automatically.

The host-side `neurqo-client` Conda environment only runs the benchmark client. Database execution and model inference run inside Docker. No database port publishing or host directory mounts are needed for the main benchmark.

## 2. Datasets and Container Access

### Included workloads

| Dataset | Main database | Additional database scales | SQL directory | Test protocols |
|---|---|---|---|---|
| JOB | `imdb_ori` | `imdb_scale_25`, `imdb_scale_50`, `imdb_scale_75` | `workloads/query_job`, 113 SQLs | Base-query, Leave-one-out, Random; folds a/b/c |
| STACK | `so` | `so_scale_50` | `workloads/query_stack`, 112 SQLs | Base-query, Leave-one-out, Random; folds a/b/c |
| TPC-H | `tpch` | None | `workloads/query_tpch`, 22 SQLs | Random; test folds of 7/7/8 SQLs |

Fold definitions are in [`workloads/train_test.py`](../workloads/train_test.py). TPC-H experiments disable query decomposition. The Docker benchmark client reads SQL from the cloned repository and checks it against the bundled workload.

To inspect a database directly, run this command on the host:

```bash
docker exec -it neurqo psql -U neurdb -d imdb_ori
```

### Container shell

The source code, training scripts, and SQL workloads are already included. Open a shell in the bundled source directory using the existing Python environment:

```bash
docker exec -it -w /opt/neurqo/source -e PYTHONPATH=/opt/neurqo/source/src:/opt/neurqo/source neurqo bash
```

Only the optional manual SQL examples use this shell. Buffer-replay and analysis commands below run on the host, from the cloned repository.

## 3. Reproducing NeurQO Results

### Released artifacts

The reproduction workflow uses the released checkpoints and SQLite experience buffers. The runtime for a matching SQL/action trajectory comes from the buffer, not from a new SQL execution. PostgreSQL reference times come from the released measurements as well.

| Repository path | Contents |
|---|---|
| `results/buffers/` | Seven read-only SQLite buffers, about 169 MiB; JOB/STACK/TPC-H have 11,045/12,090/266 execution records |
| `results/models/` | 176 checkpoints, including main, learning-curve, transfer, and ablation policies |
| `results/benchmark/nqo/` | PostgreSQL reference times, released result CSVs, and analysis programs |
| `scripts/reproduce/neurqo/` | Original experiment producers and action configurations |

The historical `.sql` buffer files are SQLite databases. Their schema is documented in [results/buffers/README.md](../results/buffers/README.md). Keep released buffers and models unchanged; write replay outputs to a new directory.

Keep these artifacts in the cloned repository on the host. The replay client reads buffers locally and transfers only the requested checkpoints to a temporary inference process in the container, using its existing Python environment and source. No server-code update, bind mount, port publishing, or image rebuild is required. In the host's `neurqo-client` environment:

```bash
conda activate neurqo-client
python -m pip install psycopg2-binary
set -euo pipefail
```

Check the released checkpoint and buffer references without executing benchmark SQL:

```bash
python scripts/reproduce/neurqo/verify_artifacts.py
```

### Buffer replay

There are two distinct operations: **model replay** predicts actions from saved states and looks up a matching complete trajectory; **analysis** summarizes already-produced CSVs and recorded events. Analysis alone is not a replacement for model replay.

`scripts/reproduce/docker_replay.py` invokes the original experiment producers with the release-container connection adapter. It keeps checkpoint selection, state/action matching, cached runtimes, and fold aggregation unchanged. The default container is `neurqo`; use `--container NAME` before the experiment name to select another container. Temporary inference processes exit after evaluation and do not replace the bundled services.

Replay uses `--cache-miss error`: missing trajectories are reported, never filled by an implicit SQL execution. The adapter refuses cache-miss execution and uses recorded PostgreSQL reference rows. The README's `docker_benchmark.py` remains the separate fresh end-to-end benchmark.

Released CSVs/logs are under `results/benchmark/nqo/`. Save replay outputs separately. The original development-environment entry points remain available under `scripts/reproduce/neurqo/`; the commands below use their release adapter.

### Overall performance

**Purpose.** Evaluate NeurQO against PostgreSQL over every workload--split combination, both end to end and per query.

**Buffer replay.** Evaluate every protocol and fold using the original core runner. The adapter seeds the output with the released PostgreSQL rows, loads each fold's model, and looks up matching trajectories in its dataset buffer:

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"
for dataset in job stack tpch; do
  protocols="base_query leave_one_out random"
  if [ "$dataset" = tpch ]; then protocols="random"; fi
  for protocol in $protocols; do
    for fold in a b c; do
      python scripts/reproduce/docker_replay.py run --dataset "$dataset" --method neurqo --protocol "$protocol" --fold "$fold" --output "$OUT/nqo_runs.csv"
    done
  done
done
```

The runner predicts actions on saved states and reuses a runtime only when a complete trajectory matches. A cache miss is an error; it does not execute SQL or substitute a different action's runtime. Overall WS is the sum of PostgreSQL times across the three test folds divided by the corresponding sum of NeurQO times, not the mean of fold WS values.

**Outputs and analysis.** The released measurements are in `results/benchmark/nqo/nqo_runs.csv`. `analyze_nqo.py` prints overall, per-query, and action-frequency summaries. Its released output is `analyze_nqo.log`.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"
python results/benchmark/nqo/analyze_nqo.py all --input "$OUT/nqo_runs.csv" \
  2>&1 | tee "$OUT/logs/analyze_nqo.log"
python scripts/reproduce/common/build_overall_performance_comparison.py --output "$OUT/overall_performance_comparison.csv"
```

The unified NeurQO/baseline table is written to `$OUT/overall_performance_comparison.csv`. Its builder reads the released measurements; it does not automatically consume the new replay output.

### Learning efficiency

**Purpose.** Measure how held-out workload performance changes with training iterations, wall-clock training time, and accumulated execution feedback.

**Buffer replay.** Evaluate the retained checkpoint inventory with `run_learning_trace.py --cache-miss error --devices cpu --workers 1`. Reuse recorded execution labels; do not restart training to reproduce the released learning curve. `export_learning_time.py` is only needed after an explicitly requested fresh training run.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"

# Evaluate all retained checkpoints.
python scripts/reproduce/docker_replay.py learning \
  --output "$OUT/nqo_learning_trace.csv" \
  2>&1 | tee "$OUT/logs/learning_trace.log"

# Reuse the original measured training times; do not retrain.
cp results/benchmark/nqo/nqo_learning_time.csv "$OUT/nqo_learning_time.csv"
```

**Outputs and analysis.** Released checkpoint evaluations and measured training times are `nqo_learning_trace.csv` and `nqo_learning_time.csv`. Analyze them together with `analyze_nqo_learning.py`; the released printed output is `analyze_nqo_learning.log`.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"
python results/benchmark/nqo/analyze_nqo_learning.py --input "$OUT/nqo_learning_trace.csv" --time-input "$OUT/nqo_learning_time.csv" \
  2>&1 | tee "$OUT/logs/analyze_nqo_learning.log"
```

### Transferability

**Purpose.** Evaluate whether NeurQO policies transfer across workloads without retraining and remain effective when database scale changes.

**Buffer replay.** Evaluate the saved mixed-workload and zero-shot checkpoints, then the reduced-scale databases using their matching buffers and PostgreSQL measurements. No retraining is required.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"

# Replay mixed-workload and zero-shot checkpoints; no training.
python scripts/reproduce/docker_replay.py transfer \
  --experiment all --output "$OUT/nqo_transfer_run.csv" \
  2>&1 | tee "$OUT/logs/transfer.log"

for scale in 25 50 75; do
  for fold in a b c; do
    python scripts/reproduce/docker_replay.py run --dataset job --protocol random --fold "$fold" --database "imdb_scale_${scale}" --cache "results/buffers/job_scale_${scale}.sql" --pg-reference "results/benchmark/nqo/nqo_job_scale_${scale}_runs.csv" --output "$OUT/nqo_job_scale_${scale}_runs.csv"
  done
done
for fold in a b c; do
  python scripts/reproduce/docker_replay.py run --dataset stack --protocol random --fold "$fold" --database so_scale_50 --cache results/buffers/stack_scale_50.sql --pg-reference results/benchmark/nqo/nqo_stack_scale_50_runs.csv --output "$OUT/nqo_stack_scale_50_runs.csv"
done
```

**Outputs and analysis.** Released cross-workload results are in `nqo_transfer_run.csv`; scale results are in `nqo_job_scale_{25,50,75}_runs.csv` and `nqo_stack_scale_50_runs.csv`. Analyze them with `analyze_nqo_transfer.py` and `analyze_nqo_data_scale.py`. Their released printed outputs are the adjacent `.log` files.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"
python results/benchmark/nqo/analyze_nqo_transfer.py "$OUT/nqo_transfer_run.csv" \
  2>&1 | tee "$OUT/logs/analyze_nqo_transfer.log"
python results/benchmark/nqo/analyze_nqo_data_scale.py --input-dir "$OUT" \
  2>&1 | tee "$OUT/logs/analyze_nqo_data_scale.log"
```

### Component analysis

**Purpose.** Quantify action frequency and importance, compare RL and state representations, and study decomposition depth and scheduling sensitivity.

**Buffer replay.** Replay retained ablation checkpoints and fixed-alpha policies with strict cache-miss handling. Decomposition-depth results are already released in `nqo_decomposition_depth.csv`; `run_decomposition_depth.py` executes SQL and writes a fixed output path, so do not run it for buffer-only reproduction.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"

# Action importance: evaluate checkpoints retrained with one action disabled.
python scripts/reproduce/docker_replay.py abl-action \
  --output "$OUT/nqo_abl_action_run.csv" \
  2>&1 | tee "$OUT/logs/abl_action.log"
# RL formulation: evaluate hierarchical SMDP and one-step RL checkpoints.
python scripts/reproduce/docker_replay.py abl-rl \
  --output "$OUT/nqo_abl_rl_run.csv" \
  2>&1 | tee "$OUT/logs/abl_rl.log"
# State representation: evaluate checkpoints trained without query or plan topology.
python scripts/reproduce/docker_replay.py abl-state \
  --output "$OUT/nqo_abl_sate_run.csv" \
  2>&1 | tee "$OUT/logs/abl_state.log"
# Scheduling sensitivity: override learned scheduling with fixed alpha values.
python scripts/reproduce/docker_replay.py alpha \
  --output "$OUT/nqo_alpha_sensitivity.csv" \
  2>&1 | tee "$OUT/logs/alpha_sensitivity.log"
```

**Outputs and analysis.** Action frequency comes from `nqo_runs.csv`. The released component results are `nqo_abl_action_run.csv`, `nqo_abl_rl_run.csv`, `nqo_abl_sate_run.csv` (original spelling), `nqo_decomposition_depth.csv`, and `nqo_alpha_sensitivity.csv`. Print the corresponding analyses with:

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"
# Action frequencies from the main NeurQO evaluation.
python results/benchmark/nqo/analyze_nqo.py all --input "$OUT/nqo_runs.csv" \
  2>&1 | tee "$OUT/logs/action_frequency.log"
# Action, RL-formulation, and state-representation ablations.
python results/benchmark/nqo/analyze_nqo_abl.py --action "$OUT/nqo_abl_action_run.csv" --rl "$OUT/nqo_abl_rl_run.csv" --state "$OUT/nqo_abl_sate_run.csv" \
  2>&1 | tee "$OUT/logs/analyze_nqo_abl.log"
# Fixed decomposition-depth experiment.
python results/benchmark/nqo/analyze_nqo_decomposition_depth.py \
  2>&1 | tee "$OUT/logs/analyze_nqo_decomposition_depth.log"
# Fixed versus learned scheduling alpha.
python results/benchmark/nqo/analyze_nqo_alpha_sensitivity.py --input "$OUT/nqo_alpha_sensitivity.csv" \
  2>&1 | tee "$OUT/logs/analyze_nqo_alpha_sensitivity.log"
```

### Overhead analysis

**Purpose.** Measure NeurQO training and inference costs, plus the runtime and materialization overhead of its individual optimization actions.

**Recorded measurements.** Read the saved database/policy events and standalone-action CSVs. This does not rebuild Bloom filters or execute materialization queries. `run_independent_actions.py` is the original fresh-measurement collector and is not needed for buffer-based overhead reproduction.

**Outputs and analysis.** Released action measurements are in `nqo_independent_action_runs.csv`; inference measurements come from `nqo_runs.csv`; and training-time measurements are in `nqo_learning_time.csv`. `nqo_training_time_comparison.csv` assembles cross-system training times from the method-specific measurements and estimates recorded in its evidence column. Print the released overhead tables with `analyze_nqo_overhead.py`; its captured output is `analyze_nqo_overhead.log`.

```bash
OUT=results/benchmark/temp_neurqo_reproduction
mkdir -p "$OUT/logs"
python results/benchmark/nqo/analyze_nqo_overhead.py \
  2>&1 | tee "$OUT/logs/analyze_nqo_overhead.log"
```

### Non-learned baselines and individual actions

**Buffer reproduction.** Use the original standalone-record lookup through the same adapter; for example, `python scripts/reproduce/docker_replay.py run --dataset job --method query_split --output "$OUT/nqo_standalone.csv"`. Other methods include `top5`, `lip_selective`, and `aja_conservative`. PostgreSQL reference rows are reused automatically. `analyze_nqo.py` and the overhead analysis also summarize the released standalone measurements without running SQL.

**Optional fresh SQL execution.** The examples below are only for manually testing an individual action, not for reproducing saved buffer labels.

`SET neurqo = on` enables the pipeline; the policy runtime chooses its actions. To run an independent non-learned action, start a fixed policy on a separate port. In the **container shell**, run **one** of the following commands:

```bash
# QuerySplit: continue decomposition with fixed alpha=0.5.
NEURQO_FIXED_DEC=apply NEURQO_FIXED_SCHED_ALPHA=0.5 python -m runtime.action_server --port 18090 --model-module runtime.policies.fixed:predict --require-model

# TOP-5 search only.
NEURQO_FIXED_ENUM=top5 python -m runtime.action_server --port 18090 --model-module runtime.policies.fixed:predict --require-model

# Selective LIP only.
NEURQO_FIXED_FILTER=selective python -m runtime.action_server --port 18090 --model-module runtime.policies.fixed:predict --require-model

# Conservative AJA only.
NEURQO_FIXED_AJOIN=conservative python -m runtime.action_server --port 18090 --model-module runtime.policies.fixed:predict --require-model
```

Unspecified actions stay disabled/native. The fixed QuerySplit baseline uses `alpha=0.5`; learned scheduling still predicts alpha. For the other released standalone variants, use `NEURQO_FIXED_ENUM=top10 NEURQO_FIXED_ENUM_K=10`, `NEURQO_FIXED_FILTER=full`, or `NEURQO_FIXED_AJOIN=aggressive`. QuerySplit is evaluated only on JOB and STACK.

Open another container shell using the `docker exec` command from Section 2. The following JOB example applies the recorded action thresholds and compares one SQL against PostgreSQL; set `method` to match the fixed server you started:

```bash
python - <<'PY'
from argparse import Namespace
from pathlib import Path
from scripts.reproduce.neurqo.run import POSTGRES_PROFILE, execute_sql_once, profile_for

method = "query_split"
args = Namespace(workload="JOB", database="imdb_ori", host="127.0.0.1", port=5432, user="neurdb")
sql = Path("workloads/query_job/29a.sql").read_text()
results = {}
for name, profile in [("postgres", POSTGRES_PROFILE), (method, profile_for("JOB", method))]:
    results[name] = execute_sql_once(
        args, sql=sql, profile=profile, timeout_ms=60000,
        db_trace_container="/opt/neurqo/runtime/logs/standalone.db.jsonl",
        server_url="http://127.0.0.1:18090/action",
    )
    print(name, results[name])
assert all(row["status"] == "ok" for row in results.values()), "Check timeout/error output"
assert results["postgres"]["result_hash"] == results[method]["result_hash"]
assert results["postgres"]["result_rows"] == results[method]["result_rows"]
PY
```

This is a single-query execution example, not a whole-workload WS measurement. `profile_for` reads dataset-specific thresholds from `scripts/reproduce/neurqo/action_config.json`; choosing a fixed policy alone does not set these PostgreSQL parameters. Eligible plans use the selected primitive; unsupported operator/query shapes may have no applicable transformation. Stop only this extra fixed-policy process with Ctrl-C when finished; the bundled services remain running.

## 4. Reproducing Learned Baselines

These are the original third-party training/collection recipes, not NeurQO buffer replay. They require the external implementations and their development database/connection setup; they are not bundled one-command jobs in the release container. Skip them when reproducing NeurQO from its released experience.

The official source repositories are listed in [thrid_party/README.md](../thrid_party/README.md). All commands below execute the systems locally and write fresh artifacts to `results/benchmark/temp_baseline_reproduction/`. Redirecting stdout and stderr with `tee` provides a complete run log alongside those artifacts.

First collect one PostgreSQL execution per query:

```bash
OUT=results/benchmark/temp_baseline_reproduction
mkdir -p "$OUT/logs"
python -m scripts.reproduce.postgresql.measure_baseline_postgres \
  --workload all --output-root "$OUT" 2>&1 | tee "$OUT/logs/postgresql.log"
```



### FASTgres

We use the official FASTgres release in `thrid_party/FASTgres-PVLDBv16`. Our adapters enumerate the 64 hint configurations for JOB, STACK, and TPC-H, construct the upstream features, train each paper split, predict a hint, and execute it once:

```bash
python -m scripts.reproduce.fastgres.measure_fastgres_labels \
  --workload all --output-root "$OUT" 2>&1 | tee "$OUT/logs/fastgres-labels.log"
python -m scripts.reproduce.fastgres.run_fastgres_full \
  --workload all --output-root "$OUT" 2>&1 | tee "$OUT/logs/fastgres.log"
```

Fresh labels, model predictions, executions, and timing JSON files are written under `$OUT/{job,stack,tpch}/fastgres/`. The released compact results are in `results/benchmark/fastgres/`. Each workload directory contains:

- `hint_configuration_executions.csv`: measured runtimes for the 64 candidate
  hint configurations, which provide the labels used to train FASTgres.
- `predicted_hints.json`: predicted hints for every protocol and fold, together
  with the corresponding training, prediction, and fold-audit metadata.
- `predicted_hint_executions.csv`: one final execution of each predicted hint,
  including the PostgreSQL runtime, timeout status, and inference time.

### TONIC

We use the official TONIC release in `thrid_party/TONIC`. JOB uses the authors' released feedback after locally extracting query skeletons; STACK and TPC-H collect feedback in our database. Training is isolated by split, and each predicted join assignment is executed once:

```bash
python -m scripts.reproduce.tonic.measure_tonic_feedback \
  --workload JOB --skeletons-only --output-root "$OUT"
python -m scripts.reproduce.tonic.measure_tonic_feedback \
  --workload STACK --output-root "$OUT"
python -m scripts.reproduce.tonic.measure_tonic_feedback \
  --workload TPCH --output-root "$OUT"
python -m scripts.reproduce.tonic.run_tonic_full \
  --workload JOB --feedback-source original_job --output-root "$OUT"
python -m scripts.reproduce.tonic.run_tonic_full \
  --workload STACK --output-root "$OUT"
python -m scripts.reproduce.tonic.run_tonic_full \
  --workload TPCH --output-root "$OUT"
```

Fresh skeletons, feedback, predictions, executions, and timing JSON files are under `$OUT/{job,stack,tpch}/tonic*/`. The released compact results are in `results/benchmark/tonic/`. The workload directories contain:

- `query_skeletons.json`: the extracted join skeleton and controllable join
  steps for each query.
- `released_feedback/*.sql` (JOB): released join-assignment feedback represented
  as candidate hints and their measured runtimes.
- `join_assignment_feedback_executions.csv` (STACK and TPC-H): locally measured
  runtimes for candidate join assignments used as training feedback.
- `predicted_join_assignments.json`: predicted assignments for every protocol
  and fold, together with training, prediction, and fold-audit metadata.
- `predicted_join_assignment_executions.csv`: one final execution of each
  predicted assignment, including the generated hint, PostgreSQL runtime,
  timeout status, and inference time.

### GenJoin, HybridQO, and AutoSteer

These adapters follow the source and benchmark process in `thrid_party/genjoin` while retaining one final execution of PostgreSQL and one final execution of the selected system plan for comparison. For example, the TPC-H experiments are run directly as follows:

```bash
python -m scripts.reproduce.genjoin.run_genjoin_tpch all \
  --models 1 --output-root "$OUT"
for fold in random_a random_b random_c; do
  python -m scripts.reproduce.hybridqo.run_hybridqo_tpch \
    --fold "$fold" --output-root "$OUT"
done
python -m scripts.reproduce.autosteer.run_autosteer_tpch \
  --phase all --output-root "$OUT"
```

These commands collect, train, predict, and execute locally. Released results are in `results/benchmark/genjoin_baselines/`.
