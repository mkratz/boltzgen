# Boltz-2 Bridge: Template-Enforced Refolding for BoltzGen

## Problem

BoltzGen's built-in refolding step **intentionally discards template and conditioning information** during refolding ([issue #153](https://github.com/HannesStark/boltzgen/issues/153), [#116](https://github.com/HannesStark/boltzgen/issues/116)). For multi-chain or difficult target proteins, this causes the target structure to collapse during refolding, disrupting binding sites and producing meaningless RMSD scores.

## Solution

Replace BoltzGen's internal folding step with a call to **vanilla Boltz-2's CLI** (`boltz predict`), which supports hard template enforcement via `force: true` and configurable `threshold` potentials. A bridge module (`boltz2_bridge.py`) converts between formats so downstream steps (affinity, analysis, filtering) work unchanged.

## Usage

```bash
boltzgen run design_spec.yaml \
  --use_boltz2_refolding \
  --template_threshold 2.0 \
  --boltz2_diffusion_samples 5 \
  --boltz2_sampling_steps 200
```

| Flag | Default | Description |
|------|---------|-------------|
| `--use_boltz2_refolding` | off | Use Boltz-2 CLI instead of BoltzGen's internal refolding |
| `--template_threshold` | 2.0 | Angstrom deviation threshold for template enforcement |
| `--boltz2_diffusion_samples` | 5 | Number of diffusion samples per design |
| `--boltz2_sampling_steps` | 200 | Number of diffusion sampling steps |

## Architecture

```
BoltzGen pipeline (normal):
  design → inverse_folding → [folding] → design_folding → affinity → analysis → filtering

With bridge:
  design → inverse_folding → [boltz2_refolding] → design_folding → affinity → analysis → filtering
                                    │
                                    ├─ generate YAML (sequences + templates)
                                    ├─ run `boltz predict` (subprocess)
                                    └─ convert output → BoltzGen format
```

The bridge step writes to the same output directories (`fold_out_npz/`, `refold_cif/`) that downstream steps expect.

## Files

| File | Role |
|------|------|
| `src/boltzgen/task/predict/boltz2_bridge.py` | YAML generation, output conversion, orchestration |
| `src/boltzgen/cli/boltzgen.py` | CLI flags and pipeline integration |

## Key Implementation Details

### 1. Atom Ordering and Residue Numbering (Critical)

**The most important detail in this implementation.** Getting this wrong produces a pipeline that runs to completion with no errors but generates silently wrong metrics.

BoltzGen's RMSD computation (`rmsd_computation.py`) compares `coords` and `input_coords` **by array index**, not by atom name. This means the Boltz-2 predicted coordinates must be placed in **exactly the same atom ordering** as BoltzGen's internal tokenizer/featurizer produces.

The bridge matches atoms between Boltz-2 output and BoltzGen's internal representation using `(chain_id, residue_number, atom_name)` tuples as join keys. This requires two things to be correct:

#### a) Metadata source

The original plan assumed structural metadata (`input_coords`, `res_type`, `token_index`, etc.) could be carried from the inverse folding metadata `.npz`. This was wrong:

- The inverse folding metadata `.npz` only contains `design_mask`, `mol_type`, `ss_type`, `binding_type`, `token_resolved_mask`
- The structural fields (`input_coords`, `res_type`, `token_index`, `atom_resolved_mask`, `atom_to_token`, `mol_type`, `backbone_mask`) are generated **during** BoltzGen's folding step from the featurized batch
- Specifically in `boltz.py:1313`: `pred_dict["input_coords"] = batch["coords"]`

**Fix**: `_get_boltzgen_atom_keys_and_feats()` re-runs BoltzGen's `Tokenizer` + `Featurizer` pipeline on the design CIF to produce the canonical atom ordering and extract all required metadata arrays.

#### b) Residue numbering must be 1-based sequential per chain

**This is the subtlest and most dangerous part of the implementation.** BoltzGen and Boltz-2 use different residue numbering schemes, and a mismatch here produces a pipeline that appears to work perfectly — all 6 steps complete, metrics CSV is generated, designs are ranked — but the underlying RMSD computations compare wrong atom pairs, producing garbage.

The problem arises from a chain of numbering transformations:

1. **Native CIF** (e.g., `9bkq-assembly2.cif`): Chain A residues numbered 3–203 (leading unresolved residues SER, ALA were stripped, so numbering starts at 3)
2. **Template renumbering**: The bridge renumbers the template CIF to 1-based sequential for Boltz-2 (see Section 7), so Boltz-2's output has chain A residues 1–201
3. **BoltzGen featurizer**: Parses the design CIF and internally indexes residues. The `structure.residues` array uses a global `res_idx` field — chain A gets indices 0–200, chain B gets 201–331

If the bridge builds atom keys using `res_idx + 1` (global 1-based), chain B keys become `(B, 202, ...)` through `(B, 332, ...)`. But Boltz-2's output has chain B as `(B, 1, ...)` through `(B, 131, ...)`. The keys don't match, those atoms get NaN, and the NaN-fill with `input_coords` silently makes the RMSD computation "work" by comparing each atom against itself — producing artificially low (zero) RMSD for the entire design chain.

**What silent failure looks like**: With wrong numbering, a typical output shows:
- `bb_rmsd_design = 0.0` (design chain comparing against itself due to NaN fill)
- `bb_rmsd_target = <not computed>` (target chain atoms shifted by 2 residues)
- 38–40% of atoms identical to `input_coords` (all design chain atoms NaN-filled)
- Pipeline completes successfully, metrics CSV is valid, no errors anywhere

**Fix**: Atom keys must use **1-based sequential numbering per chain** (resetting to 1 at each chain boundary), not global residue indices:

```python
# WRONG: global indexing, chain B starts at 202
for res_idx in range(res_start, res_end):
    res_seq = res_idx + 1

# CORRECT: per-chain sequential, chain B starts at 1
for seq_idx, res_idx in enumerate(range(res_start, res_end)):
    res_seq = seq_idx + 1
```

**Important**: The function `_get_boltzgen_atom_keys_and_feats()` builds atom keys in TWO places — once before featurization and once after. Both must use per-chain sequential numbering. Similarly, the fallback function `_get_atom_keys_from_cif()` must also use per-chain sequential numbering (not `residue.seqid.num` from gemmi, which preserves original CIF numbering).

**What correct output looks like** (from nanobody-against-penguinpox test case):
- `bb_rmsd_target ≈ 1.0 Å` (target chain held in place by template enforcement)
- `bb_rmsd_design ≈ 0.4–1.6 Å` (design chain refolds to similar but not identical structure)
- Only 0–0.4% atoms NaN-filled (just featurizer padding atoms)
- `filter_rmsd ≈ 2.5–10 Å` (overall complex RMSD, dominated by relative chain orientation)

### 2. Confidence Metric Mapping

Boltz-2 uses different keys and structures for confidence metrics:

| BoltzGen field | Boltz-2 source | Notes |
|----------------|---------------|-------|
| `iptm` | `confidence.json → iptm` | Direct |
| `ptm` | `confidence.json → ptm` | Direct |
| `complex_plddt` | `confidence.json → complex_plddt` | Direct |
| `design_ptm` | `chains_ptm[design_chain_idx]` | Indexed by 0-based position, not chain ID |
| `design_to_target_iptm` | `pair_chains_iptm[design][target]` | Average across all design-target chain pairs |

`chains_ptm` is a dict keyed by 0-based chain position as strings (`"0"`, `"1"`, ...). The chain ordering matches the sequence order in the YAML input (target chains first, then design chains).

### 3. Best Sample Selection

Matches BoltzGen's logic in `get_best_folding_sample()`:
```python
score = 0.8 * design_to_target_iptm + 0.2 * design_ptm
best_idx = nanargmax(score)
```

Falls back to `0.8 * iptm + 0.2 * ptm` if per-chain metrics are unavailable.

### 4. Pipeline Integration

BoltzGen's pipeline uses Hydra configs — each step gets a YAML written by `configure_command()` then executed via subprocess. The bridge step doesn't fit this pattern, so it uses:

- A sentinel `config_path="__boltz2_bridge__"` to skip Hydra config validation
- Simple (non-Hydra) YAML with bridge parameters written by `configure_command()`
- Special-casing in `execute_command()` to call `_run_boltz2_refolding_step()` directly

### 5. Chain Role Detection

`_detect_chain_roles()` determines which chains are design vs target by loading `design_mask` from the inverse folding metadata `.npz`. Tokens where `design_mask > 0` belong to design chains; others are target chains.

### 6. Target CIF Discovery

The target structure is found by looking for `*_native.cif` files in the design directory — this is the convention BoltzGen uses when copying the original target into the output directory.

**Important**: The `--use_boltz2_refolding` flag automatically enables `writer.write_native=true` on the inverse folding step so that `*_native.cif` files are produced. Without this, the bridge has no target structure to use as a template.

### 7. Template CIF Renumbering

Boltz-2's mmCIF parser (`parse_polymer`) assumes 1-based sequential residue numbering. BoltzGen's native CIF files may have non-sequential numbering (e.g., when leading unresolved residues are stripped, chain A residues might start at 3 instead of 1). This causes an `IndexError` in Boltz-2's parser.

The bridge automatically renumbers the template CIF before passing it to Boltz-2:
- `_atom_site.auth_seq_id` and `_atom_site.label_seq_id` are renumbered to start at 1 per chain
- `_entity_poly_seq.num` is also renumbered to match (required for gemmi's `setup_entities()` to populate `entity.full_sequence`)

This renumbering is what makes per-chain sequential atom keys (Section 1b) necessary — Boltz-2's output inherits the renumbered residue IDs, so the bridge's atom key scheme must match.

### 8. Boltz-2 Output Directory Structure

Boltz-2 nests its results under a `boltz_results_<stem>/` subdirectory inside the `--out_dir`:

```
<out_dir>/
  boltz_results_<stem>/
    predictions/
      <stem>/
        <stem>_model_0.cif
        confidence_<stem>_model_0.json
        ...
```

The bridge accounts for this nesting when locating prediction files.

### 9. Featurizer Design Flag

The BoltzGen featurizer's `process()` method must be called with `design=True` when `atom14=True` is set. Calling with `design=False` raises an assertion error. This is required because the atom14 representation is specific to design mode.

### 10. Coordinate NaN Handling

After reordering Boltz-2 coordinates to match BoltzGen's atom ordering, a small number of atoms (typically < 1%, mostly featurizer padding atoms from the atom14 representation) may not have a Boltz-2 counterpart, leaving NaN values.

These NaN values would cause `linalg.svd` failures in the downstream RMSD computation. The bridge fills unmapped atom coordinates with the corresponding reference (`input_coords`) values. With correct residue numbering (Section 1b), this affects only a handful of padding atoms rather than entire chains.

## Output Format

The bridge produces exactly what downstream steps expect:

```
fold_out_npz/{id}.npz:
  coords              [N_samples, N_atoms, 3]    # Boltz-2 predictions, reordered
  input_coords        [N_atoms, 3]               # From BoltzGen featurizer
  res_type            [N_tokens]                  # From BoltzGen featurizer
  token_index         [N_tokens, ...]             # From BoltzGen featurizer
  atom_resolved_mask  [N_atoms]                   # From BoltzGen featurizer
  atom_to_token       [N_atoms]                   # From BoltzGen featurizer
  mol_type            [N_tokens]                  # From BoltzGen featurizer
  backbone_mask       [N_atoms]                   # From BoltzGen featurizer
  iptm                [N_samples]                 # From Boltz-2 confidence
  ptm                 [N_samples]                 # From Boltz-2 confidence
  design_ptm          [N_samples]                 # From Boltz-2 chains_ptm
  design_to_target_iptm [N_samples]               # From Boltz-2 pair_chains_iptm
  complex_plddt       [N_samples]                 # From Boltz-2 confidence
  ... (other eval_keys filled with NaN)

refold_cif/{id}.cif:
  Best sample mmCIF (copied from Boltz-2 output)
```

## Verification Checklist

Use these checks to confirm the bridge is working correctly. A pipeline that produces metrics without errors is **not sufficient** — the residue numbering bug (Section 1b) demonstrates that silently wrong results are the primary failure mode.

### Quick sanity checks

```python
import numpy as np

d = np.load("fold_out_npz/<id>.npz")
coords = d["coords"][0]
input_coords = d["input_coords"].reshape(-1, 3)

# 1. NaN-filled atom fraction should be < 1%
identical = np.all(np.isclose(coords, input_coords, atol=1e-6), axis=1)
pct = 100 * identical.sum() / len(coords)
print(f"Identical atoms: {pct:.1f}%")  # Should be < 1%
assert pct < 5, f"Too many NaN-filled atoms ({pct:.1f}%) — likely residue numbering mismatch"

# 2. bb_rmsd_design should be non-zero
bb_mask = d["backbone_mask"].flatten().astype(bool)
bb_diff = coords[bb_mask] - input_coords[bb_mask]
bb_rmsd = np.sqrt(np.mean(np.sum(bb_diff**2, axis=1)))
print(f"Backbone RMSD: {bb_rmsd:.2f} Å")  # Should be > 0 for design chain
```

### Full checklist

1. **NaN fraction**: < 1% of atoms identical to `input_coords` (if 30%+, residue numbering is wrong)
2. **Target stability**: `bb_rmsd_target` in metrics CSV should be < `template_threshold` (typically ~1 Å)
3. **Design chain RMSD**: `bb_rmsd_design` should be non-zero (if 0.0, design chain atoms are NaN-filled)
4. **Confidence sanity**: `iptm` and `ptm` should be in [0, 1] and non-NaN
5. **Format**: All `const.eval_keys` present in `.npz`
6. **Downstream**: Analysis step produces valid metrics CSV; filtering step produces ranked designs
7. **Comparison**: Run same designs with/without `--use_boltz2_refolding`, compare target RMSD distributions

### Red flags that indicate silent failure

| Symptom | Likely cause |
|---------|-------------|
| `bb_rmsd_design = 0.0` | Design chain atoms NaN-filled (numbering mismatch) |
| 30%+ atoms identical to `input_coords` | Per-chain residue numbering not resetting |
| `bb_rmsd_target` is NaN or absent | Target chain atoms matched to wrong residues |
| Backbone RMSD > 30 Å on "predicted" atoms | Atoms matched to wrong residues (shifted by N) |
| Pipeline succeeds but all designs fail filters | Metrics computed on garbage coordinates |
