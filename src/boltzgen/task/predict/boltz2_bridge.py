"""Bridge between BoltzGen's pipeline and Boltz-2's CLI for template-enforced refolding.

Replaces BoltzGen's internal refolding step (which discards templates) with
Boltz-2's ``boltz predict`` CLI, which supports hard template enforcement via
``force: true`` and ``threshold``.

Three main entry points:
    generate_boltz2_yaml   – writes a Boltz-2 input YAML for one design
    convert_boltz2_to_boltzgen – maps Boltz-2 outputs → BoltzGen .npz / .cif
    run_boltz2_refolding   – orchestrates the full replacement for a design dir
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import yaml

from boltzgen.data import const

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _renumber_cif_residues(cif_path: Path, output_path: Path) -> Path:
    """Write a copy of an mmCIF with residues renumbered sequentially from 1.

    Boltz-2's template parser assumes 1-based sequential residue numbering.
    BoltzGen's native CIF files may have non-sequential numbering (e.g. when
    leading unresolved residues were stripped), which causes an IndexError
    in Boltz-2's ``parse_polymer``.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]

    # Build per-chain renumbering: old auth_seq_id -> new sequential id
    atom_site = block.find("_atom_site.", ["auth_asym_id", "auth_seq_id"])
    if not atom_site:
        doc.write_file(str(output_path))
        return output_path

    chain_seq_ids: Dict[str, list] = {}
    for row in atom_site:
        cid = row[0]
        sid = row[1]
        if cid not in chain_seq_ids:
            chain_seq_ids[cid] = []
        if not chain_seq_ids[cid] or chain_seq_ids[cid][-1] != sid:
            chain_seq_ids[cid].append(sid)

    # Map old -> new for each chain
    renumber_map: Dict[str, Dict[str, str]] = {}
    for cid, sids in chain_seq_ids.items():
        seen: Dict[str, str] = {}
        counter = 0
        for sid in sids:
            if sid not in seen:
                counter += 1
                seen[sid] = str(counter)
        renumber_map[cid] = seen

    # Apply renumbering to auth_seq_id and label_seq_id in _atom_site
    for tag in ["auth_seq_id", "label_seq_id"]:
        col_chain = block.find("_atom_site.", ["auth_asym_id", tag])
        if col_chain:
            for row in col_chain:
                cid = row[0]
                old_id = row[1]
                if cid in renumber_map and old_id in renumber_map[cid]:
                    row[1] = renumber_map[cid][old_id]

    # Also renumber _entity_poly_seq.num (gemmi needs 1-based sequential here)
    # Build entity_id -> chain mapping from _struct_asym
    entity_to_chain: Dict[str, str] = {}
    struct_asym = block.find("_struct_asym.", ["id", "entity_id"])
    if struct_asym:
        for row in struct_asym:
            entity_to_chain[row[1]] = row[0]

    eps = block.find("_entity_poly_seq.", ["entity_id", "num"])
    if eps:
        for row in eps:
            eid = row[0]
            old_num = row[1]
            cid = entity_to_chain.get(eid)
            if cid and cid in renumber_map and old_num in renumber_map[cid]:
                row[1] = renumber_map[cid][old_num]

    doc.write_file(str(output_path))
    return output_path


def _extract_sequences_from_cif(
    cif_path: Path,
) -> Dict[str, str]:
    """Return ``{chain_name: one_letter_sequence}`` from an mmCIF file.

    Uses Gemmi to parse the structure and read per-residue information.
    Only protein chains (standard amino acids) are included.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)
    st.setup_entities()

    sequences: Dict[str, str] = {}
    for model in st:
        for chain in model:
            seq_letters = []
            for residue in chain:
                info = gemmi.find_tabulated_residue(residue.name)
                if not info.is_amino_acid():
                    continue
                letter = info.one_letter_code if info.one_letter_code != "?" else "X"
                # D-amino acids map to lowercase L-counterparts (e.g. DPR -> 'p');
                # uppercase to get the standard residue for Boltz-2.
                letter = letter.upper()
                seq_letters.append(letter)
            if seq_letters:
                sequences[chain.name] = "".join(seq_letters)
    return sequences


def _extract_atom_coords_from_mmcif(
    mmcif_path: Path,
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """Parse an mmCIF and return atom coordinates keyed by ``(chain, res_seq, atom_name)``.

    This is used to reorder Boltz-2's predicted coordinates to match
    BoltzGen's internal atom ordering.
    """
    doc = gemmi.cif.read(str(mmcif_path))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)

    coord_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    key = (chain.name, residue.seqid.num, atom.name)
                    coord_map[key] = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                    )
    return coord_map


def _get_boltzgen_atom_keys_and_feats(
    design_cif_path: Path,
    metadata_npz_path: Path,
    moldir: Path,
) -> Tuple[List[Tuple[str, int, str]], dict, "Structure"]:
    """Re-featurize the design CIF through BoltzGen's pipeline to get the
    correct atom ordering and structural metadata fields.

    Returns
    -------
    atom_keys : list of (chain, res_seq, atom_name)
        The atom ordering used by BoltzGen's featurizer.
    feat_arrays : dict
        Contains ``input_coords``, ``res_type``, ``token_index``,
        ``atom_resolved_mask``, ``atom_to_token``, ``mol_type``,
        ``backbone_mask`` as numpy arrays.
    structure : Structure
        The parsed BoltzGen Structure object (for CIF generation).
    """
    from boltzgen.data.data import Structure
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.parse import mmcif as mmcif_parser
    from boltzgen.data.tokenize.tokenizer import Tokenizer

    metadata = np.load(metadata_npz_path, allow_pickle=True)
    design_mask = metadata["design_mask"]

    # Load molecules
    canonicals = load_canonicals(moldir)
    molecules = dict(canonicals)

    # Parse the design CIF
    parsed = mmcif_parser.parse_mmcif(
        design_cif_path, molecules, moldir=moldir, use_original_res_idx=False
    )
    structure = parsed.data

    # Tokenize
    tokenizer = Tokenizer()
    tokenized = tokenizer.tokenize(structure)

    # Build atom key list from the structure's atom ordering.
    # Use 1-based sequential residue numbering per chain to match
    # the renumbered Boltz-2 output (see _renumber_cif_residues).
    atom_keys: List[Tuple[str, int, str]] = []
    for chain in structure.chains:
        chain_name = chain["name"]
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        for seq_idx, res in enumerate(structure.residues[res_start:res_end]):
            atom_start = res["atom_idx"]
            atom_end = atom_start + res["atom_num"]
            res_num = seq_idx + 1  # 1-based sequential, matches Boltz-2 output
            for atom in structure.atoms[atom_start:atom_end]:
                atom_keys.append((chain_name, res_num, atom["name"]))

    # Extract coordinate and metadata arrays that the folding step normally produces
    import torch
    from boltzgen.data.data import Input
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.template.features import load_dummy_templates

    featurizer = Featurizer()

    input_data = Input(
        tokens=tokenized.tokens,
        bonds=tokenized.bonds,
        token_to_res=tokenized.token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )

    features = featurizer.process(
        input_data,
        molecules=molecules,
        random=np.random.default_rng(42),
        training=False,
        max_seqs=1,
        backbone_only=False,
        atom14=True,
        design=True,
        override_method="X-RAY DIFFRACTION",
    )

    # Add dummy templates
    templates_features = load_dummy_templates(
        tdim=1, num_tokens=len(features["res_type"])
    )
    features.update(templates_features)

    # Extract the arrays we need
    feat_arrays = {}
    feat_arrays["input_coords"] = features["coords"].numpy()
    feat_arrays["res_type"] = features["res_type"].numpy()
    feat_arrays["token_index"] = features["token_index"].numpy()
    feat_arrays["atom_resolved_mask"] = features["atom_resolved_mask"].numpy()
    feat_arrays["atom_to_token"] = features["atom_to_token"].numpy()
    feat_arrays["mol_type"] = features["mol_type"].numpy()
    feat_arrays["backbone_mask"] = features["backbone_mask"].numpy()

    # Build the atom key list from the featurized structure.
    # Use 1-based sequential numbering PER CHAIN to match the renumbered
    # Boltz-2 output (see _renumber_cif_residues).
    featurized_atom_keys: List[Tuple[str, int, str]] = []
    for chain in structure.chains:
        chain_name = chain["name"]
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        for seq_idx, res_idx in enumerate(range(res_start, res_end)):
            res = structure.residues[res_idx]
            atom_start = res["atom_idx"]
            atom_end = atom_start + res["atom_num"]
            res_seq = seq_idx + 1  # 1-based sequential per chain
            for atom in structure.atoms[atom_start:atom_end]:
                featurized_atom_keys.append((chain_name, res_seq, atom["name"]))

    return featurized_atom_keys, feat_arrays, structure


# ---------------------------------------------------------------------------
# YAML generator
# ---------------------------------------------------------------------------


def generate_boltz2_yaml(
    design_cif: Path,
    target_cif: Path,
    design_chain_ids: List[str],
    target_chain_ids: List[str],
    output_path: Path,
    template_threshold: float = 2.0,
) -> Path:
    """Create a Boltz-2 input YAML for one design.

    Parameters
    ----------
    design_cif : Path
        Inverse-folded design structure (contains designed binder + target).
    target_cif : Path
        Original target structure used as template.
    design_chain_ids : list[str]
        Chain IDs of the designed binder in ``design_cif``.
    target_chain_ids : list[str]
        Chain IDs of the target protein in ``design_cif``.
    output_path : Path
        Where to write the YAML file.
    template_threshold : float
        Angstrom threshold for Boltz-2 template enforcement.

    Returns
    -------
    Path
        The written YAML path.
    """
    design_seqs = _extract_sequences_from_cif(design_cif)
    target_seqs = _extract_sequences_from_cif(target_cif)

    sequences = []

    # Target chains – use sequences from the *target* CIF (authoritative)
    for cid in target_chain_ids:
        seq = target_seqs.get(cid)
        if seq is None:
            seq = design_seqs.get(cid)
        if seq is None:
            raise ValueError(
                f"Could not find sequence for target chain {cid} in "
                f"{target_cif} or {design_cif}"
            )
        sequences.append({"protein": {"id": cid, "sequence": seq}})

    # Designed binder chains – always from design CIF
    for cid in design_chain_ids:
        seq = design_seqs.get(cid)
        if seq is None:
            raise ValueError(
                f"Could not find sequence for design chain {cid} in {design_cif}"
            )
        sequences.append({"protein": {"id": cid, "sequence": seq}})

    # Renumber template CIF residues sequentially (Boltz-2 requires 1-based)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renumbered_cif = output_path.parent / f"{output_path.stem}_template.cif"
    _renumber_cif_residues(target_cif, renumbered_cif)

    # Template block – constrain target chains to their known structure
    templates = [
        {
            "cif": str(renumbered_cif.resolve()),
            "chain_id": target_chain_ids,
            "template_id": target_chain_ids,
            "force": True,
            "threshold": template_threshold,
        }
    ]

    boltz2_input = {"sequences": sequences, "templates": templates}

    with open(output_path, "w") as f:
        yaml.dump(boltz2_input, f, default_flow_style=False, sort_keys=False)

    return output_path


# ---------------------------------------------------------------------------
# Output converter
# ---------------------------------------------------------------------------


def convert_boltz2_to_boltzgen(
    boltz2_output_dir: Path,
    design_metadata_npz: Path,
    design_cif_path: Path,
    output_npz_path: Path,
    output_cif_path: Path,
    design_chain_ids: List[str],
    target_chain_ids: List[str],
    yaml_stem: str,
    diffusion_samples: int = 1,
    moldir: Optional[Path] = None,
) -> bool:
    """Convert Boltz-2 prediction outputs into BoltzGen-compatible format.

    Parameters
    ----------
    boltz2_output_dir : Path
        Root Boltz-2 output directory (contains ``predictions/``).
    design_metadata_npz : Path
        BoltzGen metadata ``.npz`` from the inverse folding step.
    design_cif_path : Path
        The inverse-folded design CIF (used for atom key ordering).
    output_npz_path : Path
        Where to write the BoltzGen-compatible ``.npz``.
    output_cif_path : Path
        Where to write the BoltzGen-compatible best-sample ``.cif``.
    design_chain_ids : list[str]
        Chain IDs of the designed binder.
    target_chain_ids : list[str]
        Chain IDs of the target protein.
    yaml_stem : str
        Stem name of the input YAML (used to locate Boltz-2 output files).
    diffusion_samples : int
        Number of diffusion samples that were generated.
    moldir : Path, optional
        Path to molecule directory for BoltzGen featurization.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    pred_dir = boltz2_output_dir / "predictions" / yaml_stem

    if not pred_dir.exists():
        logger.warning("Prediction directory not found: %s", pred_dir)
        return False

    # --- Get BoltzGen's atom ordering by re-featurizing the design CIF ---
    boltzgen_structure = None
    if moldir is not None:
        boltzgen_atom_keys, feat_arrays, boltzgen_structure = (
            _get_boltzgen_atom_keys_and_feats(
                design_cif_path, design_metadata_npz, moldir
            )
        )
    else:
        # Fallback: use Gemmi raw ordering (less reliable but avoids heavy deps)
        boltzgen_atom_keys = _get_atom_keys_from_cif(design_cif_path)
        feat_arrays = None

    # Use featurizer atom count if available (may include padding atoms for atom14)
    if feat_arrays is not None and "input_coords" in feat_arrays:
        n_atoms = feat_arrays["input_coords"].shape[-2]
    else:
        n_atoms = len(boltzgen_atom_keys)

    # --- Collect Boltz-2 outputs across all samples ---
    all_coords = []
    all_confidence = []

    for rank in range(diffusion_samples):
        mmcif_file = pred_dir / f"{yaml_stem}_model_{rank}.cif"
        if not mmcif_file.exists():
            logger.warning("Missing model file: %s", mmcif_file)
            continue

        conf_file = pred_dir / f"confidence_{yaml_stem}_model_{rank}.json"
        if not conf_file.exists():
            logger.warning("Missing confidence file: %s", conf_file)
            continue

        # Parse atom coordinates from Boltz-2 mmCIF
        boltz2_coords = _extract_atom_coords_from_mmcif(mmcif_file)

        # Reorder to match BoltzGen atom ordering
        reordered = np.full((n_atoms, 3), np.nan, dtype=np.float32)
        for i, key in enumerate(boltzgen_atom_keys):
            if key in boltz2_coords:
                reordered[i] = boltz2_coords[key]

        all_coords.append(reordered)

        with open(conf_file) as f:
            conf = json.load(f)
        all_confidence.append(conf)

    if not all_coords:
        logger.error("No valid model outputs found in %s", pred_dir)
        return False

    coords_stack = np.stack(all_coords, axis=0)  # [N_samples, N_atoms, 3]
    n_samples = len(all_coords)

    # --- Map confidence metrics ---
    def _get_metric(conf_list, key, default=np.nan):
        return np.array([c.get(key, default) for c in conf_list], dtype=np.float32)

    iptm = _get_metric(all_confidence, "iptm")
    ptm = _get_metric(all_confidence, "ptm")
    complex_plddt = _get_metric(all_confidence, "complex_plddt")

    # Extract per-chain and pair-chain metrics
    design_ptm_vals = []
    design_to_target_iptm_vals = []

    for conf in all_confidence:
        chains_ptm = conf.get("chains_ptm", {})
        pair_iptm = conf.get("pair_chains_iptm", {})

        # Boltz-2 indexes chains by 0-based position in the YAML sequences list
        all_chain_ids = target_chain_ids + design_chain_ids
        chain_idx_map = {cid: str(i) for i, cid in enumerate(all_chain_ids)}

        # design_ptm: average ptm across design chains
        d_ptms = []
        for cid in design_chain_ids:
            idx = chain_idx_map.get(cid)
            if idx and idx in chains_ptm:
                d_ptms.append(chains_ptm[idx])
        design_ptm_vals.append(np.mean(d_ptms) if d_ptms else np.nan)

        # design_to_target_iptm: average pair iptm between design and target
        dt_iptms = []
        for dcid in design_chain_ids:
            didx = chain_idx_map.get(dcid)
            for tcid in target_chain_ids:
                tidx = chain_idx_map.get(tcid)
                if didx and tidx and didx in pair_iptm and tidx in pair_iptm[didx]:
                    dt_iptms.append(pair_iptm[didx][tidx])
        design_to_target_iptm_vals.append(np.mean(dt_iptms) if dt_iptms else np.nan)

    design_ptm = np.array(design_ptm_vals, dtype=np.float32)
    design_to_target_iptm = np.array(design_to_target_iptm_vals, dtype=np.float32)

    # --- Build output NPZ ---
    out_dict = {}

    # Fill NaN coords with input_coords so analysis RMSD doesn't hit non-finite values.
    # Unmapped atoms (side chains with different naming) get their reference positions.
    if feat_arrays is not None and "input_coords" in feat_arrays:
        ref = feat_arrays["input_coords"]
        # ref may be (1, N, 3) or (N, 3); broadcast to match coords_stack
        if ref.ndim == 3:
            ref_broadcast = np.broadcast_to(ref, coords_stack.shape)
        else:
            ref_broadcast = np.broadcast_to(ref[np.newaxis], coords_stack.shape)
        nan_mask = np.isnan(coords_stack)
        coords_stack = np.where(nan_mask, ref_broadcast, coords_stack)

    out_dict["coords"] = coords_stack

    # Structural metadata from BoltzGen featurization
    if feat_arrays is not None:
        for key in [
            "input_coords",
            "res_type",
            "token_index",
            "atom_resolved_mask",
            "atom_to_token",
            "mol_type",
            "backbone_mask",
        ]:
            out_dict[key] = feat_arrays[key]
    else:
        # Fallback: try loading from metadata (won't have all fields)
        metadata = np.load(design_metadata_npz, allow_pickle=True)
        for key in [
            "input_coords",
            "res_type",
            "token_index",
            "atom_resolved_mask",
            "atom_to_token",
            "mol_type",
            "backbone_mask",
        ]:
            if key in metadata:
                out_dict[key] = metadata[key]
            else:
                logger.warning(
                    "Field '%s' not available (run with --moldir for full support)",
                    key,
                )

    # Confidence metrics (per-sample arrays)
    out_dict["iptm"] = iptm
    out_dict["ptm"] = ptm
    out_dict["complex_plddt"] = complex_plddt
    out_dict["design_ptm"] = design_ptm
    out_dict["design_to_target_iptm"] = design_to_target_iptm

    # Fill missing confidence keys with NaN
    nan_arr = np.full(n_samples, np.nan, dtype=np.float32)
    for key in const.eval_keys_confidence:
        if key not in out_dict:
            out_dict[key] = nan_arr.copy()
    for key in const.eval_keys_affinity:
        if key not in out_dict:
            out_dict[key] = nan_arr.copy()

    # Save NPZ
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, **out_dict)

    # --- Write best sample CIF ---
    score = np.where(
        np.isnan(design_to_target_iptm),
        0.8 * iptm + 0.2 * ptm,
        0.8 * design_to_target_iptm + 0.2 * design_ptm,
    )
    best_idx = int(np.nanargmax(score))

    # Generate the CIF from BoltzGen's structure with Boltz-2 coordinates
    # so that it only contains resolved atoms (matching analysis masks).
    # The raw Boltz-2 CIF predicts ALL atoms including those unresolved in
    # the original structure, which causes mask/atom dimension mismatches.
    output_cif_path.parent.mkdir(parents=True, exist_ok=True)
    if boltzgen_structure is not None:
        from boltzgen.data.write.mmcif import to_mmcif

        best_coords = all_coords[best_idx]
        n_struct_atoms = len(boltzgen_structure.atoms)
        boltzgen_structure.coords["coords"][:] = best_coords[:n_struct_atoms]
        cif_text = to_mmcif(boltzgen_structure)
        output_cif_path.write_text(cif_text)
    else:
        best_mmcif = pred_dir / f"{yaml_stem}_model_{best_idx}.cif"
        shutil.copy2(best_mmcif, output_cif_path)

    logger.info(
        "Converted %s: %d samples, best=%d (score=%.3f)",
        yaml_stem,
        n_samples,
        best_idx,
        score[best_idx],
    )
    return True


def _get_atom_keys_from_cif(
    cif_path: Path,
) -> List[Tuple[str, int, str]]:
    """Fallback: get atom keys from CIF using Gemmi (raw ordering).

    Uses 1-based sequential numbering per chain to match renumbered Boltz-2 output.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)

    keys: List[Tuple[str, int, str]] = []
    for model in st:
        for chain in model:
            for seq_idx, residue in enumerate(chain):
                res_num = seq_idx + 1  # 1-based sequential
                for atom in residue:
                    keys.append((chain.name, res_num, atom.name))
    return keys


# ---------------------------------------------------------------------------
# Bridge runner
# ---------------------------------------------------------------------------


def _detect_chain_roles(
    design_cif: Path,
    metadata_npz: Path,
) -> Tuple[List[str], List[str]]:
    """Detect which chains are design vs target from BoltzGen metadata.

    Returns (design_chain_ids, target_chain_ids).
    """
    metadata = np.load(metadata_npz, allow_pickle=True)
    design_mask = metadata["design_mask"]  # per-token boolean

    doc = gemmi.cif.read(str(design_cif))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)

    # Build chain_name -> token indices mapping
    chain_residues: Dict[str, List[int]] = {}
    token_idx = 0
    for model in st:
        for chain in model:
            chain_residues[chain.name] = []
            for residue in chain:
                info = gemmi.find_tabulated_residue(residue.name)
                if info.is_amino_acid():
                    chain_residues[chain.name].append(token_idx)
                    token_idx += 1

    design_chains = []
    target_chains = []

    for chain_name, token_indices in chain_residues.items():
        if not token_indices:
            continue
        # Check bounds to handle mismatches gracefully
        valid_indices = [i for i in token_indices if i < len(design_mask)]
        if not valid_indices:
            target_chains.append(chain_name)
            continue
        chain_mask_vals = design_mask[valid_indices]
        if np.any(chain_mask_vals > 0):
            design_chains.append(chain_name)
        else:
            target_chains.append(chain_name)

    return design_chains, target_chains


def run_boltz2_refolding(
    design_dir: Path,
    target_cif: Path,
    template_threshold: float = 2.0,
    diffusion_samples: int = 5,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = True,
    moldir: Optional[Path] = None,
    boltz_extra_args: Optional[List[str]] = None,
) -> None:
    """Replace BoltzGen's folding step with Boltz-2 CLI predictions.

    Scans the design directory for inverse-folded CIF files, generates
    Boltz-2 YAMLs, runs ``boltz predict``, and converts outputs to
    BoltzGen-compatible format in ``fold_out_npz/`` and ``refold_cif/``.

    Parameters
    ----------
    design_dir : Path
        Directory containing inverse-folded designs (``*.cif`` + ``*.npz``).
    target_cif : Path
        Path to the original target structure CIF.
    template_threshold : float
        Angstrom deviation threshold for template enforcement.
    diffusion_samples : int
        Number of diffusion samples per design.
    sampling_steps : int
        Number of diffusion sampling steps.
    recycling_steps : int
        Number of recycling steps.
    use_msa_server : bool
        Whether to use ``--use_msa_server`` for MSA generation.
    moldir : Path, optional
        Path to molecule directory for BoltzGen featurization.
    boltz_extra_args : list[str], optional
        Additional CLI arguments to pass to ``boltz predict``.
    """
    design_dir = Path(design_dir)
    target_cif = Path(target_cif)

    fold_out_dir = design_dir / const.folding_dirname
    refold_cif_dir = design_dir / const.refold_cif_dirname
    fold_out_dir.mkdir(parents=True, exist_ok=True)
    refold_cif_dir.mkdir(parents=True, exist_ok=True)

    # Find all design CIF files (exclude _native.cif)
    design_cifs = sorted(
        p
        for p in design_dir.iterdir()
        if p.suffix == ".cif" and "_native" not in p.stem
    )

    if not design_cifs:
        logger.warning("No design CIF files found in %s", design_dir)
        return

    logger.info("Found %d designs to refold with Boltz-2", len(design_cifs))
    print(f"Found {len(design_cifs)} designs to refold with Boltz-2")  # noqa: T201

    succeeded = 0
    failed = 0

    for design_cif in design_cifs:
        sample_id = design_cif.stem
        metadata_npz = design_cif.with_suffix(".npz")

        # Skip if output already exists
        output_npz = fold_out_dir / f"{sample_id}.npz"
        output_cif = refold_cif_dir / f"{sample_id}.cif"
        if output_npz.exists() and output_cif.exists():
            logger.info("Skipping %s (already processed)", sample_id)
            succeeded += 1
            continue

        if not metadata_npz.exists():
            logger.warning("Missing metadata for %s, skipping", sample_id)
            failed += 1
            continue

        try:
            # Detect chain roles
            design_chain_ids, target_chain_ids = _detect_chain_roles(
                design_cif, metadata_npz
            )

            if not target_chain_ids:
                logger.warning(
                    "No target chains detected for %s, skipping", sample_id
                )
                failed += 1
                continue

            # Create temp directory for this design's Boltz-2 run
            with tempfile.TemporaryDirectory(prefix=f"boltz2_{sample_id}_") as tmpdir:
                tmpdir = Path(tmpdir)

                # Generate YAML
                yaml_path = tmpdir / f"{sample_id}.yaml"
                generate_boltz2_yaml(
                    design_cif=design_cif,
                    target_cif=target_cif,
                    design_chain_ids=design_chain_ids,
                    target_chain_ids=target_chain_ids,
                    output_path=yaml_path,
                    template_threshold=template_threshold,
                )

                # Run Boltz-2
                cmd = [
                    "boltz",
                    "predict",
                    str(yaml_path),
                    "--out_dir",
                    str(tmpdir / "output"),
                    "--recycling_steps",
                    str(recycling_steps),
                    "--sampling_steps",
                    str(sampling_steps),
                    "--diffusion_samples",
                    str(diffusion_samples),
                    "--output_format",
                    "mmcif",
                ]
                if use_msa_server:
                    cmd.append("--use_msa_server")
                if boltz_extra_args:
                    cmd.extend(boltz_extra_args)

                logger.info("Running: %s", " ".join(cmd))
                print(f"  [{sample_id}] Running Boltz-2...")  # noqa: T201
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout per design
                )
                if result.returncode != 0:
                    logger.error(
                        "Boltz-2 failed for %s:\nstdout: %s\nstderr: %s",
                        sample_id,
                        result.stdout[-2000:] if result.stdout else "",
                        result.stderr[-2000:] if result.stderr else "",
                    )
                    print(f"  [{sample_id}] FAILED (exit code {result.returncode})")  # noqa: T201
                    if result.stderr:
                        print(f"  stderr: {result.stderr[-500:]}")  # noqa: T201
                    failed += 1
                    continue

                # Debug: show boltz2 output and any errors
                if result.stderr:
                    for line in result.stderr.splitlines():
                        if "Error" in line or "Failed" in line or "Traceback" in line or "File " in line:
                            print(f"  [{sample_id}] boltz2: {line.strip()}")  # noqa: T201

                # Convert outputs
                # Boltz-2 nests results under boltz_results_<stem>/
                boltz2_out = tmpdir / "output" / f"boltz_results_{sample_id}"
                if not boltz2_out.exists():
                    # List what's actually in the output dir for debugging
                    out_dir = tmpdir / "output"
                    contents = list(out_dir.iterdir()) if out_dir.exists() else []
                    print(f"  [{sample_id}] Expected {boltz2_out} but not found")  # noqa: T201
                    print(f"  [{sample_id}] Output dir contents: {[p.name for p in contents]}")  # noqa: T201
                success = convert_boltz2_to_boltzgen(
                    boltz2_output_dir=boltz2_out,
                    design_metadata_npz=metadata_npz,
                    design_cif_path=design_cif,
                    output_npz_path=output_npz,
                    output_cif_path=output_cif,
                    design_chain_ids=design_chain_ids,
                    target_chain_ids=target_chain_ids,
                    yaml_stem=sample_id,
                    diffusion_samples=diffusion_samples,
                    moldir=moldir,
                )

                if success:
                    print(f"  [{sample_id}] OK")  # noqa: T201
                    succeeded += 1
                else:
                    print(f"  [{sample_id}] conversion failed")  # noqa: T201
                    failed += 1

        except Exception:
            logger.exception("Failed to process %s", sample_id)
            failed += 1

    print(  # noqa: T201
        f"Boltz-2 refolding complete: {succeeded} succeeded, {failed} failed "
        f"out of {len(design_cifs)} designs"
    )
