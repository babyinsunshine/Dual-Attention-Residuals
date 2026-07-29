# Anonymous Single-Node Review Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the repository to an anonymous reviewer package with three 7B training entry points and a complete Hugging Face FineWeb-Edu-100BT download and preprocessing workflow.

**Architecture:** Add one narrowly scoped downloader and promote the tested parallel Parquet tokenizer to the sole preprocessing entry point. Remove cluster orchestration and its tests, then replace the README with a relative-path, link-free workflow guarded by repository-hygiene tests.

**Tech Stack:** Python 3.10, Bash, PyTorch distributed launcher, Megatron-Core, Transformer Engine, Triton, `huggingface_hub`, PyArrow, NumPy, SentencePiece, pytest/unittest.

## Global Constraints

- The final public training surface contains exactly the Baseline, Attention Residual, and Dual Attention Residual 7B configurations and one launcher for each.
- Training commands operate on one machine through `torch.distributed.run --standalone`.
- The README must not discuss unsupported launch topologies or contain Markdown links, URLs, author identities, IP addresses, or private absolute paths.
- The default dataset is repository `HuggingFaceFW/fineweb-edu`, pattern `sample/100BT/*.parquet`.
- The default raw-data directory is `data/fineweb_edu_100BT`; the default processed directory is `data/fineweb_edu_100BT_pretok`.
- Preserve deterministic merge order, resumability, atomic finalization, input identity checks, and `.partial` recovery semantics from the parallel preprocessor.
- Delete the three untracked 1B benchmark configurations as explicitly approved.
- Do not modify or stage the untracked `main.pdf`.
- Do not modify or stage the existing user change in `dual_attentions_residuals/mcore_compat.py`.
- Delete legacy `docs/superpowers` content, including this implementation plan and its design spec, before the final reviewer-package commit.

---

## File Map

- Create `data/download.py`: Hugging Face dataset snapshot downloader.
- Create `tests/test_download.py`: downloader defaults and delegation tests.
- Replace `data/preprocess.py`: sole public parallel local-Parquet tokenizer, moved from `data/preprocess_parallel.py` with reviewer-facing defaults.
- Rename `tests/test_preprocess_parallel.py` to `tests/test_preprocess.py`: retain the complete preprocessing regression suite and add default-path coverage.
- Create `tests/test_review_package.py`: enforce the final scripts/configs and anonymous README surface.
- Modify `tests/test_attnres_kernel_mode.py`: retain single-machine kernel-mode coverage and remove deleted launcher coverage.
- Replace `README.md`: concise installation, data preparation, training, testing, and license instructions.
- Delete cluster launchers, cluster-only tests, the unused overview asset, obsolete preprocessing code, untracked 1B configs, and legacy planning documents.

---

### Task 1: Hugging Face FineWeb-Edu Downloader

**Files:**
- Create: `data/download.py`
- Create: `tests/test_download.py`

**Interfaces:**
- Consumes: `huggingface_hub.snapshot_download(**kwargs) -> str`
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Defaults: repository `HuggingFaceFW/fineweb-edu`, include pattern `sample/100BT/*.parquet`, output `data/fineweb_edu_100BT`, workers `8`

- [ ] **Step 1: Write failing downloader tests**

Create `tests/test_download.py` with tests equivalent to:

```python
from pathlib import Path
from unittest import mock

from data import download


def test_default_arguments() -> None:
    args = download.parse_args([])
    assert args.output_dir == Path("data/fineweb_edu_100BT")
    assert args.num_workers == 8


def test_main_downloads_only_fineweb_edu_100bt(tmp_path: Path) -> None:
    destination = tmp_path / "raw"
    with mock.patch.object(
        download, "snapshot_download", return_value=str(destination)
    ) as snapshot_download:
        assert download.main(["--output-dir", str(destination), "--num-workers", "3"]) == 0
    snapshot_download.assert_called_once_with(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        allow_patterns=["sample/100BT/*.parquet"],
        local_dir=destination,
        max_workers=3,
    )
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `python -m pytest tests/test_download.py -q`

Expected: collection fails because `data.download` does not exist.

- [ ] **Step 3: Implement the downloader**

Create `data/download.py` with constants, a positive-worker parser, and this delegation shape:

```python
DEFAULT_REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_PATTERN = "sample/100BT/*.parquet"
DEFAULT_OUTPUT_DIR = Path("data/fineweb_edu_100BT")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu-100BT Parquet shards.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=positive_int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    destination = snapshot_download(
        repo_id=DEFAULT_REPO_ID,
        repo_type="dataset",
        allow_patterns=[DEFAULT_PATTERN],
        local_dir=args.output_dir,
        max_workers=args.num_workers,
    )
    print(f"downloaded FineWeb-Edu-100BT to {destination}")
    return 0
```

End the module with `raise SystemExit(main())` under the standard `__main__` guard.

- [ ] **Step 4: Run downloader tests and CLI help**

Run:

```bash
python -m pytest tests/test_download.py -q
python data/download.py --help
```

Expected: tests pass; help lists only `--output-dir` and `--num-workers` in addition to standard help.

- [ ] **Step 5: Commit the downloader**

```bash
git add data/download.py tests/test_download.py
git commit -m "feat: add FineWeb-Edu-100BT downloader"
```

---

### Task 2: Consolidate Preprocessing into One Public Entry Point

**Files:**
- Replace: `data/preprocess.py`
- Delete: `data/preprocess_parallel.py`
- Rename: `tests/test_preprocess_parallel.py` to `tests/test_preprocess.py`

**Interfaces:**
- Consumes: raw Parquet matching `data/fineweb_edu_100BT/sample/100BT/*.parquet`
- Produces: `data/fineweb_edu_100BT_pretok/train.bin`, `val.bin`, and `metadata.json`
- Preserves: all public functions and dataclasses currently imported from `data.preprocess_parallel`, under module name `data.preprocess`

- [ ] **Step 1: Add a failing defaults test to the existing preprocessing suite**

Before moving files, add this assertion using the current module alias:

```python
def test_review_package_defaults(self) -> None:
    args = pp.parse_args([])
    self.assertEqual(
        args.input_glob,
        "data/fineweb_edu_100BT/sample/100BT/*.parquet",
    )
    self.assertEqual(args.output_dir, "data/fineweb_edu_100BT_pretok")
```

- [ ] **Step 2: Run the new test and verify it fails**

Run: `python -m pytest tests/test_preprocess_parallel.py -q`

Expected: the new test fails because `--input-glob` is required and the old output default is `data/processed`.

- [ ] **Step 3: Promote the parallel implementation and update imports/paths**

Perform these mechanical moves:

```bash
git rm data/preprocess.py
git mv data/preprocess_parallel.py data/preprocess.py
git mv tests/test_preprocess_parallel.py tests/test_preprocess.py
```

Then replace:

```python
from data import preprocess_parallel as pp
```

with:

```python
from data import preprocess as pp
```

Replace subprocess strings `data/preprocess_parallel.py` with `data/preprocess.py`.

- [ ] **Step 4: Set the reviewer-facing defaults**

In `data/preprocess.py`, change only the parser surface:

```python
parser = argparse.ArgumentParser(description="Parallel FineWeb-Edu Parquet tokenizer.")
parser.add_argument(
    "--input-glob",
    default="data/fineweb_edu_100BT/sample/100BT/*.parquet",
)
parser.add_argument("--output-dir", default="data/fineweb_edu_100BT_pretok")
```

Keep every existing resume, identity, worker, merge, finalization, and error-handling function unchanged unless a renamed module path requires adjustment.

- [ ] **Step 5: Run preprocessing tests**

Run:

```bash
python -m pytest tests/test_preprocess.py -q
python data/preprocess.py --help
```

Expected: the full preprocessing suite passes; `--input-glob` is optional and both documented defaults appear in help.

- [ ] **Step 6: Commit the preprocessing consolidation**

```bash
git add data/preprocess.py tests/test_preprocess.py
git add -u data tests
git commit -m "refactor: expose one FineWeb-Edu preprocessor"
```

---

### Task 3: Remove Cluster and Benchmark Surfaces

**Files:**
- Create: `tests/test_review_package.py`
- Modify: `tests/test_attnres_kernel_mode.py`
- Delete: `scripts/launch_7b_ssh_common.sh`
- Delete: `scripts/launch_baseline_ssh.sh`
- Delete: `scripts/train_7b_attnres_4node.sh`
- Delete: `scripts/train_7b_baseline_4node.sh`
- Delete: `scripts/train_7b_dar_4node.sh`
- Delete: `scripts/train_attnres_multi_node.sh`
- Delete: `scripts/train_baseline_multi_node.sh`
- Delete: `scripts/train_dual_multi_node.sh`
- Delete: `tests/test_7b_multi_node_overrides.py`
- Delete: `tests/test_launch_baseline_ssh.py`
- Delete: `tests/test_three_cluster_7b_launchers.py`
- Delete: `assets/dar_overview.png`
- Delete untracked: `configs/1b_dense_attnres_benchmark.yaml`
- Delete untracked: `configs/1b_dense_baseline_benchmark.yaml`
- Delete untracked: `configs/1b_dense_dar_benchmark.yaml`

**Interfaces:**
- Produces: exactly three `scripts/train_*_single_node.sh` launchers and three `configs/7b_moe_*.yaml` configurations
- Retains: `scripts/attnres_kernel_mode.sh` as the Attention Residual launcher's internal helper

- [ ] **Step 1: Write a failing public-surface test**

Create `tests/test_review_package.py` with:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_training_surface_contains_only_three_review_launchers() -> None:
    scripts = {path.name for path in (ROOT / "scripts").glob("train_*.sh")}
    assert scripts == {
        "train_baseline_single_node.sh",
        "train_attnres_single_node.sh",
        "train_dual_single_node.sh",
    }


def test_configuration_surface_contains_only_three_7b_models() -> None:
    configs = {path.name for path in (ROOT / "configs").glob("*.yaml")}
    assert configs == {
        "7b_moe_baseline.yaml",
        "7b_moe_attnres.yaml",
        "7b_moe_dual.yaml",
    }


def test_scripts_contain_no_cluster_orchestration_artifacts() -> None:
    names = {path.name for path in (ROOT / "scripts").iterdir() if path.is_file()}
    assert not any(
        token in name
        for name in names
        for token in ("multi_node", "4node", "ssh", "hostfile")
    )
```

- [ ] **Step 2: Run the surface tests and verify they fail**

Run: `python -m pytest tests/test_review_package.py -q`

Expected: failures enumerate the current multi-node scripts and 1B configs.

- [ ] **Step 3: Delete approved files**

Use `apply_patch` to delete all tracked files listed in this task. Use explicit
paths to remove the three approved untracked 1B configs. Do not delete or stage
any other untracked file.

- [ ] **Step 4: Remove the deleted multi-node kernel-mode test**

In `tests/test_attnres_kernel_mode.py`, delete the `MULTI_NODE_LAUNCHER`
constant and `test_multi_node_launcher_rejects_invalid_mode_before_python`.
Keep the reference/fused environment tests and the single-node invalid-mode
test.

- [ ] **Step 5: Run public-surface and kernel-mode tests**

Run:

```bash
python -m pytest tests/test_review_package.py tests/test_attnres_kernel_mode.py -q
bash -n scripts/*.sh
```

Expected: all tests pass and every retained shell script parses successfully.

- [ ] **Step 6: Commit the surface reduction**

```bash
git add scripts tests assets
git add -u configs
git commit -m "refactor: keep reviewer training surface minimal"
```

---

### Task 4: Rewrite the Anonymous Reviewer README

**Files:**
- Modify: `README.md`
- Modify: `tests/test_review_package.py`

**Interfaces:**
- Consumes: commands exposed by `data/download.py`, `data/preprocess.py`, and the three training launchers
- Produces: the complete reviewer workflow without external links or cluster terminology

- [ ] **Step 1: Add failing README hygiene tests**

Append to `tests/test_review_package.py`:

```python
import re


def test_readme_is_anonymous_and_link_free() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"https?://|www\.", text, flags=re.IGNORECASE) is None
    assert re.search(r"/(?:home|root|gfs|gemini|workspace)/", text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text) is None


def test_readme_documents_only_public_commands() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for command in (
        "python data/download.py",
        "python data/preprocess.py",
        "bash scripts/train_baseline_single_node.sh",
        "bash scripts/train_attnres_single_node.sh",
        "bash scripts/train_dual_single_node.sh",
    ):
        assert command.lower() in text
    for forbidden in ("hostfile", "ssh", "multi-node", "multi node", "node_rank", "master_addr"):
        assert forbidden not in text
```

- [ ] **Step 2: Run README tests and verify they fail**

Run: `python -m pytest tests/test_review_package.py -q`

Expected: existing external URLs and cluster documentation cause failures.

- [ ] **Step 3: Replace README with the reviewer workflow**

Write a concise README containing exactly these sections:

1. `# Dual Attention Residuals` — one-paragraph description and names of the three model variants.
2. `## Environment` — Python 3.10; CUDA-capable PyTorch; tested package versions `torch 2.10.0`, `megatron-core 0.18.2`, `transformer-engine 2.10.0`; installation commands using package names only, with no index or repository URL.
3. `## Data` — `python data/download.py`, then `python data/preprocess.py`; describe the default Hugging Face identifier, 100BT pattern, output files, `--num-workers`, `--resume`, and a bounded `--max-files` smoke path.
4. `## Training` — a common environment block using relative paths and `GPUS_PER_NODE`, then one command for each retained launcher. Do not describe topology limitations.
5. `## Tests` — `python -m pytest tests -q`.
6. `## License` — refer only to the local `LICENSE` file without a Markdown link.

Do not embed an image, badge, citation URL, checkpoint location, personal path,
author name, host address, or unsupported workflow explanation.

- [ ] **Step 4: Run README and launcher checks**

Run:

```bash
python -m pytest tests/test_review_package.py -q
DRY_RUN=1 GPUS_PER_NODE=8 PYTHON_BIN=python bash scripts/train_baseline_single_node.sh
DRY_RUN=1 GPUS_PER_NODE=8 PYTHON_BIN=python bash scripts/train_attnres_single_node.sh
DRY_RUN=1 GPUS_PER_NODE=8 PYTHON_BIN=python bash scripts/train_dual_single_node.sh
```

Expected: tests pass; each dry run prints a `torch.distributed.run --standalone --nproc-per-node=8` command with the matching configuration.

- [ ] **Step 5: Commit the README**

```bash
git add README.md tests/test_review_package.py
git commit -m "docs: add anonymous reviewer workflow"
```

---

### Task 5: Remove Internal Plans and Run Final Verification

**Files:**
- Delete: `docs/superpowers/**`
- Verify only: all retained Python, shell, config, tokenizer, license, and test files

**Interfaces:**
- Produces: final anonymous reviewer package and evidence that no user-owned work was staged

- [ ] **Step 1: Record user-owned state before final cleanup**

Run:

```bash
git diff --binary -- dual_attentions_residuals/mcore_compat.py | sha256sum
sha256sum main.pdf
git status --short
```

Store the two hashes in the session output; do not write them into the repository.

- [ ] **Step 2: Delete internal design and plan documents**

Delete `docs/superpowers` with an explicit `apply_patch` covering its tracked
files. Confirm that no unrelated `docs` content is targeted.

- [ ] **Step 3: Run focused and full tests**

Run:

```bash
python -m pytest tests/test_download.py tests/test_preprocess.py tests/test_review_package.py -q
python -m pytest tests -q
```

Expected: every retained test passes; hardware-dependent cases may skip only when their declared prerequisites are unavailable.

- [ ] **Step 4: Run static and launcher verification**

Run:

```bash
python -m compileall -q data dual_attentions_residuals train.py tests
bash -n scripts/*.sh
for script in scripts/train_baseline_single_node.sh scripts/train_attnres_single_node.sh scripts/train_dual_single_node.sh; do
  DRY_RUN=1 GPUS_PER_NODE=8 PYTHON_BIN=python bash "$script"
done
rg -n 'https?://|www\.|/(home|root|gfs|gemini|workspace)/|hostfile|multi[- ]node|node_rank|master_addr' README.md
```

Expected: compilation and shell parsing succeed; three dry runs print valid
commands; `rg` exits 1 with no matches.

- [ ] **Step 5: Audit final files and user-owned state**

Run:

```bash
rg --files | sort
git diff --check
git diff --binary -- dual_attentions_residuals/mcore_compat.py | sha256sum
sha256sum main.pdf
git status --short
```

Expected: hashes match Step 1; `main.pdf` remains untracked; the
`mcore_compat.py` modification remains unstaged; only intended reviewer-package
changes are staged or committed.

- [ ] **Step 6: Commit internal-document removal**

```bash
git add -u docs/superpowers
git commit -m "chore: remove internal development notes"
```

- [ ] **Step 7: Inspect final history and status**

Run:

```bash
git log --oneline -8
git status --short
git diff --cached --stat
```

Expected: reviewer-package commits are present, the index is empty, and status
contains only the preserved `mcore_compat.py` modification plus untracked
`main.pdf`.
