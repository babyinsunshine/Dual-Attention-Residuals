# Parallel FineWeb-Edu 100BT Preprocessing Design

## Goal

Add a local, resumable preprocessing path that uses a many-core server to
convert an existing FineWeb-Edu `sample-100BT` Parquet mirror into the single
`int32` token files consumed by this repository:

- `data/processed/train.bin`
- `data/processed/val.bin`
- `data/processed/metadata.json`

The target machine has 96 CPU cores, 2 TiB of RAM, and at least 800 GB of free
disk. The raw Parquet files already exist locally. Downloading data and changing
the training data loader are outside this change.

## Selected approach

Add `data/preprocess_parallel.py` and keep `data/preprocess.py` unchanged. The
new command reads local Parquet files with PyArrow, distributes files across a
spawn-based process pool, writes restartable token shards, and then performs a
deterministic rolling merge into the two files expected by the current
`PackedTokenDataset`.

This is preferred over DataTrove because it avoids a new preprocessing
framework, and over teaching the training loader about multiple shards because
it leaves training order, checkpoint behavior, and configuration unchanged.

## Command-line interface

The primary invocation will be:

```bash
python data/preprocess_parallel.py \
  --input-glob '/gemini/space/private/liyining/datasets/fineweb-100B/sample/100BT/*.parquet' \
  --tokenizer tokenizer/tokenizer.model \
  --output-dir data/processed \
  --num-workers 64 \
  --validation-tokens 10000000 \
  --add-eos \
  --resume
```

Supported arguments:

- `--input-glob`: required glob for local Parquet files.
- `--tokenizer`: SentencePiece model; defaults to
  `tokenizer/tokenizer.model`.
- `--output-dir`: final output and state directory; defaults to
  `data/processed`.
- `--text-column`: Parquet text column; defaults to `text`.
- `--num-workers`: worker process count; defaults to the smaller of 64 and the
  detected CPU count.
- `--batch-size`: rows read from Parquet per batch; defaults to 1,024.
- `--validation-tokens`: exact number of tokens assigned to `val.bin`; defaults
  to 10,000,000.
- `--add-bos` and `--add-eos`: tokenizer boundary controls matching the existing
  preprocessor.
- `--resume`: reuse valid completed shards and merge state.
- `--max-files`: process only the first N sorted inputs for smoke tests.
- `--overwrite`: discard output/state owned by this preprocessing run and start
  again. Without it, existing final outputs cause a safe failure.

The input glob must resolve to at least one regular file. Inputs are sorted by
their full normalized path, giving deterministic output independent of worker
completion order.

## Worker pipeline

The parent creates a `ProcessPoolExecutor` with the `spawn` multiprocessing
context. Each process initializes one `SentencePieceTokenizer`; tokenizer
objects are not sent between processes.

Each task processes one Parquet file:

1. Open it with `pyarrow.parquet.ParquetFile`.
2. Iterate only the configured text column with `iter_batches`, with PyArrow
   internal threading disabled to avoid multiplying 64 processes by another
   thread pool.
3. Ignore null, non-string, and whitespace-only values, matching the existing
   preprocessor.
4. Tokenize each accepted document and append its IDs explicitly as
   little-endian `int32` (`<i4`) values to a per-input `.bin.partial` file.
5. Flush and close the file, atomically rename it to `.bin`, and atomically
   write a JSON sidecar containing the source path, source size and mtime,
   accepted document count, token count, tokenizer fingerprint, and tokenizer
   boundary options.

Worker files live under `OUTPUT/.preprocess/shards/`. A shard is reusable only
when both its binary and sidecar exist, its byte size is exactly four times its
recorded token count, and the recorded source/tokenizer settings still match.
Invalid or incomplete `.partial` files are rebuilt.

The parent reports each completed file and prints a heartbeat at least every 30
seconds containing completed/total files, accumulated documents and tokens,
elapsed time, and average token throughput. This prevents a long-running job
from appearing stuck even when no large shard has just finished.

After every requested shard is valid, the parent atomically writes
`OUTPUT/.preprocess/run-manifest.json`. It contains the ordered source identity
and per-shard statistics needed for final metadata. The merge state records the
hash of this manifest, so temporary shard sidecars can be removed without
losing provenance or document counts.

## Deterministic validation split and rolling merge

After all requested token shards are valid, the parent merges them in sorted
input order. The first exactly `--validation-tokens` tokens go to
`val.bin.partial`; every later token goes to `train.bin.partial`. Splitting may
occur inside one token shard, but never inside an `int32` value. With `--add-eos`
enabled, document boundaries remain represented in the stream.

The rolling merge keeps peak token storage close to one copy:

1. Append one shard to the appropriate partial final file(s).
2. Flush and `fsync` both partial final files.
3. Atomically update `OUTPUT/.preprocess/merge-state.json` with the run-manifest
   hash, next shard index, committed byte lengths, and cumulative counts.
4. Delete the now-committed temporary token shard and its sidecar.

If the process stops before step 3, resume truncates the partial final files to
the byte lengths recorded by the previous merge state and replays that shard.
If it stops after step 3 but before step 4, resume removes the already-committed
shard and continues. Raw Parquet inputs are never modified or deleted.

When the merge is complete, the program verifies that validation has the
requested token count and training is non-empty, atomically renames the two
partial outputs to their final names, and atomically writes `metadata.json`
last. Metadata is the completion marker. If interruption occurs between the two
binary renames or before metadata is written, resume validates the final and
partial file sizes against merge state, completes any remaining rename, and
writes metadata without re-tokenizing. A completed output with matching
metadata is a successful no-op under `--resume`.

## Metadata

`metadata.json` retains the fields produced by the sequential preprocessor where
they apply and adds provenance required for reproduction:

- format/version identifier for the parallel preprocessor;
- sorted source file count and a hash of their path/size/mtime manifest;
- accepted and skipped document counts;
- train, validation, and total token counts;
- `int32` dtype and little-endian byte order;
- tokenizer path, vocabulary size, and SHA-256;
- BOS/EOS settings;
- text column, worker count, and Parquet batch size.

Worker count and batch size are informational and do not affect output order or
contents.

## Dependencies

Declare `pyarrow==21.0.0` as a direct project dependency because the new script
imports it directly. This satisfies the `pyarrow>=21.0.0` requirement of the
already pinned `datasets==5.0.0` without relying on an undeclared transitive
installation. No ModelScope SDK dependency is needed because the raw data is
already local.

## Safety and error handling

- Final outputs are never overwritten unless `--overwrite` is supplied.
- `--overwrite` is restricted to files beneath the resolved output directory
  that are owned by this preprocessor; it never touches input Parquet files.
- A worker exception stops scheduling new merge work, preserves valid completed
  shards, and exits nonzero with the source filename and traceback.
- Insufficient validation tokens, an empty training split, tokenizer vocabulary
  mismatch, malformed Parquet, and token IDs outside the `int32` range are hard
  errors.
- `SIGINT`/`SIGTERM` stops workers cleanly where possible and leaves only
  resumable state or disposable `.partial` files.
- Available space on the output filesystem is printed before processing. A full
  run (no `--max-files`) warns when less than 450 GiB is available because 100B
  `int32` tokens are approximately 400 GB; disk-full errors remain resumable.

## Verification

Automated `unittest` tests use small synthetic Parquet files and verify:

1. One worker and multiple workers produce byte-identical outputs.
2. Input completion order does not change final file order.
3. The validation file contains exactly the requested number of tokens.
4. Null/empty rows and BOS/EOS behavior match `data/preprocess.py`.
5. A completed shard is skipped on resume, while an invalid or partial shard is
   rebuilt.
6. An interrupted merge truncates to the last committed byte lengths and
   resumes without duplication or loss.
7. Existing final outputs are refused without `--overwrite`.

A real-data smoke test processes two sorted FineWeb-Edu files with 2 workers,
then checks metadata, file sizes, and direct `np.memmap` readability. The full
100BT run starts at 64 workers; 32- and 48-worker smoke measurements can be used
if shared-storage throughput makes 64 slower.

## Non-goals

- Downloading or updating FineWeb-Edu.
- Deleting or reorganizing the existing raw Parquet mirror.
- Distributed preprocessing across multiple machines.
- GPU tokenization.
- Changing `PackedTokenDataset` to consume multiple files.
- Reproducing the source dataset's original GPT-2 token count; output counts are
  based on this repository's 131,072-piece SentencePiece tokenizer.
