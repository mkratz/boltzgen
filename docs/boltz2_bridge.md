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

### 1. Atom Ordering (Critical)

**The most important detail in this implementation.**

BoltzGen's RMSD computation (`rmsd_computation.py`) compares `coords` and `input_coords` **by array index**, not by atom name. This means the Boltz-2 predicted coordinates must be placed in **exactly the same atom ordering** as BoltzGen's internal tokenizer/featurizer produces.

The original plan assumed structural metadata (`input_coords`, `res_type`, `token_index`, etc.) could be carried from the inverse folding metadata `.npz`. This was wrong:

- The inverse folding metadata `.npz` only contains `design_mask`, `mol_type`, `ss_type`, `binding_type`, `token_resolved_mask`
- The structural fields (`input_coords`, `res_type`, `token_index`, `atom_resolved_mask`, `atom_to_token`, `mol_type`, `backbone_mask`) are generated **during** BoltzGen's folding step from the featurized batch
- Specifically in `boltz.py:1313`: `pred_dict["input_coords"] = batch["coords"]`

**Fix**: `_get_boltzgen_atom_keys_and_feats()` re-runs BoltzGen's `Tokenizer` + `Featurizer` pipeline on the design CIF to:
1. Produce the canonical atom ordering as `(chain_id, residue_number, atom_name)` keys
2. Extract all required metadata arrays with correct shapes

The Boltz-2 mmCIF coordinates are then reordered to match this ordering by joining on `(chain, residue, atom_name)` keys. Without this, every RMSD computation downstream would compare wrong atom pairs.

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

1. **Format**: Load `.npz` and verify all `const.eval_keys` are present
2. **Atom ordering**: Compute self-RMSD between `input_coords` and `coords[best]` for a known-good structure (should be near-zero for matching residues)
3. **Target stability**: Target chain RMSD should be < `template_threshold` in refolded structures
4. **Downstream**: Analysis step produces valid metrics CSV; filtering step produces ranked designs
5. **Comparison**: Run same designs with/without `--use_boltz2_refolding`, compare target RMSD distributions
