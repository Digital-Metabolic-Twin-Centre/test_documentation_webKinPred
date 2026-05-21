# GPU Embedding Service

Minimal FastAPI service for remote GPU embedding offload.

## Endpoints

- `GET /health`
- `POST /embed/jobs`
- `GET /embed/jobs/{job_id}`
- `GET /embed/jobs/{job_id}/logs?tail=200`

## Request Shape

`POST /embed/jobs`

```json
{
  "method_key": "KinForm-H",
  "target": "kcat",
  "profile": "kinform_full",
  "step_work": {
    "kinform_t5_full": ["seq_1", "seq_2"]
  },
  "seq_id_to_seq": {
    "seq_1": "MPE...",
    "seq_2": "MQA..."
  }
}
```

## Optional Environment Variables

- `GPU_EMBED_SERVICE_TOKEN`: optional bearer token expected by clients.
- `GPU_NAME`: value surfaced by `/health`.
- `GPU_EMBED_STEP_CMD_<STEP_KEY>`: optional shell command per step.
  - Example key: `GPU_EMBED_STEP_CMD_KINFORM_ESM2_LAYERS`
  - Command template supports:
    - `{step_key}`
    - `{seq_ids}` (comma-separated)
    - `{seq_count}`
    - `{seq_id_to_seq_file}` (JSON file path with seq_id -> sequence map)
    - `{job_id}`
- `GPU_EMBED_JOB_LOG_DIR`: directory for dedicated per-job worker logs.
  - Default: `/tmp/webkinpred-gpu-embed/jobs`
- `GPU_EMBED_ERROR_LOG_TAIL_LINES`: number of worker-log lines included in API errors.
  - Default: `120`
- `KINFORM_PARALLEL_STREAM_ENABLE`: enable KinForm in-memory streaming pipeline.
  - Default: `0` (guarded rollout)
- `KINFORM_PARALLEL_STREAM_ALLOW_LEGACY_FALLBACK`: fallback to legacy file-polling pipeline if stream mode errors.
  - Default: `1`
- `KINFORM_PARALLEL_MAX_GPU_WORKERS`: cap concurrently running KinForm GPU embedding workers (`t5`, `esm2`, `esmc`).
  - Default: `3`
- `KINFORM_PARALLEL_INCLUDE_PSEQ_IN_GPU_CAP`: when `1`, count `pseq2sites` toward `KINFORM_PARALLEL_MAX_GPU_WORKERS`.
  - Default: `1`
- `KINFORM_PARALLEL_T5_BATCH_SIZE`: batch size for KinForm T5 worker (file-polling mode, and fallback default for stream mode if stream-specific value is unset).
  - Default: `1` in file-polling mode; `4` stream fallback default is used when stream-specific value is not set
- `KINFORM_PARALLEL_ESM2_BATCH_SIZE`: batch size for KinForm ESM2 worker (file-polling mode, and fallback default for stream mode if stream-specific value is unset).
  - Default: `1` in file-polling mode; `4` stream fallback default is used when stream-specific value is not set
- `KINFORM_PARALLEL_ESMC_BATCH_SIZE`: batch size for KinForm ESMC worker (file-polling mode, and fallback default for stream mode if stream-specific value is unset).
  - Default: `1` in file-polling mode; `4` stream fallback default is used when stream-specific value is not set
- `KINFORM_PARALLEL_STREAM_T5_BATCH_SIZE`: stream-mode override for KinForm T5 worker batch size.
  - Default: fallback to `KINFORM_PARALLEL_T5_BATCH_SIZE`, then `4`
- `KINFORM_PARALLEL_STREAM_ESM2_BATCH_SIZE`: stream-mode override for KinForm ESM2 worker batch size.
  - Default: fallback to `KINFORM_PARALLEL_ESM2_BATCH_SIZE`, then `4`
- `KINFORM_PARALLEL_STREAM_ESMC_BATCH_SIZE`: stream-mode override for KinForm ESMC worker batch size.
  - Default: fallback to `KINFORM_PARALLEL_ESMC_BATCH_SIZE`, then `4`
- `KINFORM_PARALLEL_RESIDUE_CACHE_GB`: GPU-side residue cache budget in GB.
  - Default: `4`
- `KINFORM_PARALLEL_SPILL_DIR`: first spill target for overflow residue matrices.
  - Default: `/dev/shm/webkinpred-kinform`
- `KINFORM_PARALLEL_SPILL_FALLBACK_DIR`: fallback spill target when primary spill directory cannot be used.
  - Default: `/tmp/webkinpred-kinform`
- `KINFORM_PARALLEL_PSEQ_STREAM_BATCH_SIZE`: Pseq2Sites streaming micro-batch size.
  - Default: `8`
- `KINFORM_PARALLEL_PSEQ_STREAM_QUEUE_SIZE`: max in-memory queue items for streamed T5 residues inside pseq worker.
  - Default: `max(32, 8 * batch_size)` (typically `64` with batch size `8`)
- `KINFORM_PARALLEL_PSEQ_STREAM_IDLE_FLUSH_SECONDS`: flush partial pseq batches when stream goes idle.
  - Default: `0.2`
- `KINFORM_PARALLEL_PSEQ_STREAM_PERSIST_EVERY_ROWS`: flush pseq binding-site TSV merge after this many streamed updates.
  - Default: `64`
- `KINFORM_PARALLEL_PSEQ_STREAM_PERSIST_EVERY_SECONDS`: max seconds between pseq TSV persistence flushes.
  - Default: `15`
- `KINFORM_PARALLEL_PSEQ_STREAM_READ_EXISTING_ON_START`: when `1`, pseq worker scans existing TSV on startup to skip known IDs.
  - Default: `0` (orchestrator already launches unresolved IDs)
- `KINFORM_PARALLEL_PSEQ_SENDS_PER_TICK`: max T5 residue payloads sent to pseq worker per orchestrator loop.
  - Default: `4`
- `KINFORM_PARALLEL_PSEQ_SEND_QUEUE_SIZE`: max queued T5-last payloads waiting on async pseq sender thread.
  - Default: `16`
- `KINFORM_PARALLEL_TSV_REFRESH_SECONDS`: minimum seconds between orchestrator refreshes from shared binding-sites TSV.
  - Default: `30`
- `KINFORM_PARALLEL_WORKER_DONE_WAIT_SECONDS`: grace period before treating `rc=0` worker exit as complete when `WORKER_DONE` stream event is delayed.
  - Default: `30`
- `KINFORM_PARALLEL_STREAM_SOCKET_DIR`: Unix socket directory for orchestrator-worker stream IPC.
  - Default: `/tmp/webkinpred-gpu-embed/kinform`

If no step command is configured for a step, it is treated as a no-op.

### Real Step Commands (GPU host)

Use the bundled step runner so jobs write real cache artifacts:

```bash
export GPU_EMBED_STEP_CMD_KINFORM_T5_FULL="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_t5_full --seq-ids '{seq_ids}' --seq-id-to-seq-file '{seq_id_to_seq_file}' --job-id '{job_id}'"
export GPU_EMBED_STEP_CMD_KINFORM_ESM2_LAYERS="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_esm2_layers --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_KINFORM_ESMC_LAYERS="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_esmc_layers --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_PROT_T5_MEAN="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step prot_t5_mean --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_TURNUP_ESM1B="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step turnup_esm1b --seq-ids '{seq_ids}'"
```

## Worker Logs

Worker stdout/stderr are streamed in real time to per-job log files (not mixed with access logs).

You can read the path from:

- `GET /embed/jobs/{job_id}` → `worker_log_path`

or fetch tail lines directly:

- `GET /embed/jobs/{job_id}/logs?tail=500`

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```
