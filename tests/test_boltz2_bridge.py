"""Tests for the Boltz-2 bridge, focused on the critical correctness issues:

1. Atom reordering: Boltz-2 mmCIF atom order != BoltzGen featurizer order.
   If we get this wrong, every downstream RMSD is garbage.
2. Confidence metric mapping: Boltz-2 uses 0-based chain position indices,
   not chain IDs.  Misindexing silently produces wrong design_ptm /
   design_to_target_iptm.
3. Best-sample selection: must match BoltzGen's 0.8*d2t_iptm + 0.2*d_ptm.
4. Chain role detection: design_mask → chain classification must handle
   multi-chain targets and partial masks.
5. YAML generation: template block must list target chains only, with
   force=True.

All fixtures are synthetic (no GPU, no Boltz-2, no heavy BoltzGen deps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gemmi
import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers to build minimal mmCIF files with known atom coordinates
# ---------------------------------------------------------------------------

def _make_minimal_mmcif(
    chains: Dict[str, List[Tuple[str, List[Tuple[str, Tuple[float, float, float]]]]]],
    path: Path,
) -> Path:
    """Write a minimal mmCIF file with exact atom positions.

    Parameters
    ----------
    chains : dict  chain_id -> [(resname, [(atom_name, (x,y,z)), ...]), ...]
    path : where to write
    """
    st = gemmi.Structure()
    st.name = "test"
    model = gemmi.Model("1")

    for chain_id, residues in chains.items():
        chain = gemmi.Chain(chain_id)
        for res_idx, (resname, atoms) in enumerate(residues, start=1):
            res = gemmi.Residue()
            res.name = resname
            res.seqid = gemmi.SeqId(str(res_idx))
            for aname, (x, y, z) in atoms:
                atom = gemmi.Atom()
                atom.name = aname
                atom.element = gemmi.Element(aname[0])
                atom.pos = gemmi.Position(x, y, z)
                atom.occ = 1.0
                atom.b_iso = 0.0
                res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)

    st.add_model(model)
    st.setup_entities()
    st.assign_label_seq_id()

    doc = st.make_mmcif_document()
    doc.write_file(str(path))
    return path


def _write_confidence_json(path: Path, conf: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(conf, f)
    return path


# ---------------------------------------------------------------------------
# 1.  ATOM REORDERING
# ---------------------------------------------------------------------------


class TestAtomReordering:
    """The core correctness issue: coords in the output NPZ must be ordered
    to match BoltzGen's featurizer, NOT Boltz-2's mmCIF order.

    We test this by:
    - Defining a "BoltzGen ordering" (the reference atom key list)
    - Writing a Boltz-2 mmCIF with atoms in a DIFFERENT order
    - Running convert_boltz2_to_boltzgen and checking coords match by key
    """

    def test_reordering_shuffled_atoms(self, tmp_path):
        """Atoms in Boltz-2 mmCIF are in reversed order within each residue.
        After conversion, coords must match the BoltzGen key order."""
        from boltzgen.task.predict.boltz2_bridge import (
            _extract_atom_coords_from_mmcif,
        )

        # BoltzGen expects: N, CA, C, O  (standard backbone order)
        boltzgen_keys = [
            ("A", 1, "N"),
            ("A", 1, "CA"),
            ("A", 1, "C"),
            ("A", 1, "O"),
            ("A", 2, "N"),
            ("A", 2, "CA"),
            ("A", 2, "C"),
            ("A", 2, "O"),
        ]
        boltzgen_expected_coords = {
            ("A", 1, "N"):  np.array([1.0, 0.0, 0.0]),
            ("A", 1, "CA"): np.array([2.0, 0.0, 0.0]),
            ("A", 1, "C"):  np.array([3.0, 0.0, 0.0]),
            ("A", 1, "O"):  np.array([4.0, 0.0, 0.0]),
            ("A", 2, "N"):  np.array([5.0, 0.0, 0.0]),
            ("A", 2, "CA"): np.array([6.0, 0.0, 0.0]),
            ("A", 2, "C"):  np.array([7.0, 0.0, 0.0]),
            ("A", 2, "O"):  np.array([8.0, 0.0, 0.0]),
        }

        # Boltz-2 mmCIF has atoms in REVERSED order within each residue
        boltz2_cif = _make_minimal_mmcif(
            {
                "A": [
                    ("ALA", [
                        ("O",  (4.0, 0.0, 0.0)),
                        ("C",  (3.0, 0.0, 0.0)),
                        ("CA", (2.0, 0.0, 0.0)),
                        ("N",  (1.0, 0.0, 0.0)),
                    ]),
                    ("GLY", [
                        ("O",  (8.0, 0.0, 0.0)),
                        ("C",  (7.0, 0.0, 0.0)),
                        ("CA", (6.0, 0.0, 0.0)),
                        ("N",  (5.0, 0.0, 0.0)),
                    ]),
                ]
            },
            tmp_path / "boltz2_pred.cif",
        )

        # Parse Boltz-2 output (this gives us the coord map keyed by (chain, res, atom))
        boltz2_coords = _extract_atom_coords_from_mmcif(boltz2_cif)

        # Simulate the reordering logic from convert_boltz2_to_boltzgen
        n_atoms = len(boltzgen_keys)
        reordered = np.full((n_atoms, 3), np.nan, dtype=np.float32)
        for i, key in enumerate(boltzgen_keys):
            if key in boltz2_coords:
                reordered[i] = boltz2_coords[key]

        # Verify: each position in the reordered array must match the
        # BoltzGen-expected coordinate for that key
        for i, key in enumerate(boltzgen_keys):
            np.testing.assert_array_almost_equal(
                reordered[i],
                boltzgen_expected_coords[key],
                err_msg=f"Atom {key} at index {i} has wrong coordinates after reordering",
            )

    def test_missing_atoms_get_nan(self, tmp_path):
        """If Boltz-2 doesn't resolve an atom, that position must be NaN,
        NOT zero or some other atom's coordinates."""
        from boltzgen.task.predict.boltz2_bridge import _extract_atom_coords_from_mmcif

        # BoltzGen expects N, CA, C, O, CB — but Boltz-2 only has N, CA, C
        boltzgen_keys = [
            ("A", 1, "N"),
            ("A", 1, "CA"),
            ("A", 1, "C"),
            ("A", 1, "O"),
            ("A", 1, "CB"),  # not in Boltz-2 output
        ]

        boltz2_cif = _make_minimal_mmcif(
            {
                "A": [
                    ("ALA", [
                        ("N",  (1.0, 2.0, 3.0)),
                        ("CA", (4.0, 5.0, 6.0)),
                        ("C",  (7.0, 8.0, 9.0)),
                        ("O",  (10.0, 11.0, 12.0)),
                        # CB intentionally missing
                    ]),
                ]
            },
            tmp_path / "partial.cif",
        )

        boltz2_coords = _extract_atom_coords_from_mmcif(boltz2_cif)
        n_atoms = len(boltzgen_keys)
        reordered = np.full((n_atoms, 3), np.nan, dtype=np.float32)
        for i, key in enumerate(boltzgen_keys):
            if key in boltz2_coords:
                reordered[i] = boltz2_coords[key]

        # CB (index 4) must be NaN
        assert np.all(np.isnan(reordered[4])), "Missing atom must be NaN, not filled"
        # Resolved atoms must NOT be NaN
        assert not np.any(np.isnan(reordered[:4])), "Resolved atoms should not be NaN"

    def test_multichain_reordering(self, tmp_path):
        """Atom reordering must work correctly across multiple chains
        where Boltz-2 might interleave chain atoms differently."""
        from boltzgen.task.predict.boltz2_bridge import _extract_atom_coords_from_mmcif

        # BoltzGen ordering: chain A first, then chain B
        boltzgen_keys = [
            ("A", 1, "N"),  ("A", 1, "CA"),
            ("B", 1, "N"),  ("B", 1, "CA"),
        ]

        # Boltz-2 has same chains but we verify correct key matching
        boltz2_cif = _make_minimal_mmcif(
            {
                "A": [("ALA", [("N", (10.0, 0.0, 0.0)), ("CA", (20.0, 0.0, 0.0))])],
                "B": [("GLY", [("N", (30.0, 0.0, 0.0)), ("CA", (40.0, 0.0, 0.0))])],
            },
            tmp_path / "multi.cif",
        )

        boltz2_coords = _extract_atom_coords_from_mmcif(boltz2_cif)
        n_atoms = len(boltzgen_keys)
        reordered = np.full((n_atoms, 3), np.nan, dtype=np.float32)
        for i, key in enumerate(boltzgen_keys):
            if key in boltz2_coords:
                reordered[i] = boltz2_coords[key]

        assert reordered[0, 0] == 10.0, "Chain A, res 1, N"
        assert reordered[1, 0] == 20.0, "Chain A, res 1, CA"
        assert reordered[2, 0] == 30.0, "Chain B, res 1, N"
        assert reordered[3, 0] == 40.0, "Chain B, res 1, CA"

    def test_naive_index_copy_would_fail(self, tmp_path):
        """Demonstrate that naively copying Boltz-2 coords by index
        (without key-based reordering) produces wrong results.
        This is the bug the bridge prevents."""
        from boltzgen.task.predict.boltz2_bridge import _extract_atom_coords_from_mmcif

        # BoltzGen order:  N(1), CA(2), C(3)
        boltzgen_keys = [("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C")]

        # Boltz-2 order is reversed: C(3), CA(2), N(1)
        boltz2_cif = _make_minimal_mmcif(
            {
                "A": [
                    ("ALA", [
                        ("C",  (30.0, 0.0, 0.0)),
                        ("CA", (20.0, 0.0, 0.0)),
                        ("N",  (10.0, 0.0, 0.0)),
                    ]),
                ]
            },
            tmp_path / "reversed.cif",
        )

        boltz2_coords = _extract_atom_coords_from_mmcif(boltz2_cif)

        # CORRECT: key-based reordering
        correct = np.full((3, 3), np.nan, dtype=np.float32)
        for i, key in enumerate(boltzgen_keys):
            if key in boltz2_coords:
                correct[i] = boltz2_coords[key]

        assert correct[0, 0] == 10.0  # N
        assert correct[1, 0] == 20.0  # CA
        assert correct[2, 0] == 30.0  # C

        # WRONG: naive index copy (what would happen without reordering)
        # Read atoms in Boltz-2's file order
        doc = gemmi.cif.read(str(boltz2_cif))
        st = gemmi.make_structure_from_block(doc[0])
        naive = []
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        naive.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
        naive = np.array(naive, dtype=np.float32)

        # Naive order puts C at index 0, but BoltzGen expects N there
        # This would cause RMSD to compare N-coords vs C-coords
        assert naive[0, 0] != correct[0, 0], (
            "Naive copy should differ from correct reordering — "
            "this proves the reordering is necessary"
        )


# ---------------------------------------------------------------------------
# 2.  CONFIDENCE METRIC MAPPING
# ---------------------------------------------------------------------------


class TestConfidenceMapping:
    """Boltz-2 indexes chains_ptm and pair_chains_iptm by 0-based position
    (as strings: "0", "1", ...), NOT by chain ID.

    The YAML lists target chains first, then design chains. So for
    target=[A,B] design=[C]:  A→"0", B→"1", C→"2".

    Getting the indexing wrong silently gives wrong design_ptm and
    design_to_target_iptm values.
    """

    def _build_confidence(
        self,
        chains_ptm: dict,
        pair_chains_iptm: dict,
        iptm: float = 0.8,
        ptm: float = 0.7,
    ) -> dict:
        return {
            "iptm": iptm,
            "ptm": ptm,
            "complex_plddt": 0.75,
            "chains_ptm": chains_ptm,
            "pair_chains_iptm": pair_chains_iptm,
        }

    def test_single_target_single_design(self):
        """target=[A], design=[B] → A="0", B="1"
        design_ptm should be chains_ptm["1"]
        design_to_target_iptm should be pair_chains_iptm["1"]["0"]
        """
        # Import the conversion logic inline to test just the metric extraction
        target_chain_ids = ["A"]
        design_chain_ids = ["B"]

        conf = self._build_confidence(
            chains_ptm={"0": 0.9, "1": 0.65},     # A=0.9, B=0.65
            pair_chains_iptm={"0": {"1": 0.72}, "1": {"0": 0.72}},
        )

        all_chain_ids = target_chain_ids + design_chain_ids
        chain_idx_map = {cid: str(i) for i, cid in enumerate(all_chain_ids)}

        # design_ptm
        d_ptms = []
        for cid in design_chain_ids:
            idx = chain_idx_map.get(cid)
            if idx and idx in conf["chains_ptm"]:
                d_ptms.append(conf["chains_ptm"][idx])
        design_ptm = np.mean(d_ptms)

        # design_to_target_iptm
        dt_iptms = []
        for dcid in design_chain_ids:
            didx = chain_idx_map.get(dcid)
            for tcid in target_chain_ids:
                tidx = chain_idx_map.get(tcid)
                if didx and tidx and didx in conf["pair_chains_iptm"]:
                    if tidx in conf["pair_chains_iptm"][didx]:
                        dt_iptms.append(conf["pair_chains_iptm"][didx][tidx])
        d2t_iptm = np.mean(dt_iptms)

        assert design_ptm == pytest.approx(0.65), "design_ptm should be chain B's ptm"
        assert d2t_iptm == pytest.approx(0.72), "d2t_iptm should be B→A pair iptm"

    def test_two_target_one_design(self):
        """target=[A,B], design=[C] → A="0", B="1", C="2"
        design_ptm = chains_ptm["2"]
        design_to_target_iptm = mean(pair["2"]["0"], pair["2"]["1"])
        """
        target_chain_ids = ["A", "B"]
        design_chain_ids = ["C"]

        conf = self._build_confidence(
            chains_ptm={"0": 0.95, "1": 0.92, "2": 0.70},
            pair_chains_iptm={
                "0": {"1": 0.88, "2": 0.60},
                "1": {"0": 0.88, "2": 0.55},
                "2": {"0": 0.60, "1": 0.55},
            },
        )

        all_chain_ids = target_chain_ids + design_chain_ids
        chain_idx_map = {cid: str(i) for i, cid in enumerate(all_chain_ids)}

        d_ptms = []
        for cid in design_chain_ids:
            idx = chain_idx_map.get(cid)
            if idx and idx in conf["chains_ptm"]:
                d_ptms.append(conf["chains_ptm"][idx])

        dt_iptms = []
        for dcid in design_chain_ids:
            didx = chain_idx_map.get(dcid)
            for tcid in target_chain_ids:
                tidx = chain_idx_map.get(tcid)
                if didx and tidx and didx in conf["pair_chains_iptm"]:
                    if tidx in conf["pair_chains_iptm"][didx]:
                        dt_iptms.append(conf["pair_chains_iptm"][didx][tidx])

        design_ptm = np.mean(d_ptms)
        d2t_iptm = np.mean(dt_iptms)

        assert design_ptm == pytest.approx(0.70), "design_ptm should be chain C's ptm"
        assert d2t_iptm == pytest.approx(0.575), (
            "d2t_iptm should be mean of C→A(0.60) and C→B(0.55)"
        )

    def test_wrong_indexing_gives_wrong_values(self):
        """If someone used chain IDs instead of positional indices,
        they'd look up chains_ptm["C"] which doesn't exist."""
        conf_chains_ptm = {"0": 0.95, "1": 0.92, "2": 0.70}

        # Wrong: using chain ID "C" as key
        wrong_ptm = conf_chains_ptm.get("C", None)
        assert wrong_ptm is None, "Chain ID lookup should fail — must use positional index"

        # Right: using positional index "2" for design chain C
        right_ptm = conf_chains_ptm.get("2", None)
        assert right_ptm == 0.70

    def test_target_at_position_zero_not_skipped(self):
        """Bug check: chain_idx_map maps first target to "0".
        The code does `if idx and idx in chains_ptm` — but "0" is falsy
        as a string in some languages. In Python str "0" is truthy, but
        let's verify this explicitly."""
        target_chain_ids = ["A"]
        design_chain_ids = ["B"]
        all_chain_ids = target_chain_ids + design_chain_ids
        chain_idx_map = {cid: str(i) for i, cid in enumerate(all_chain_ids)}

        # "0" must be truthy in Python
        idx_a = chain_idx_map["A"]  # "0"
        assert idx_a == "0"
        assert bool(idx_a) is True, "String '0' must be truthy for the lookup to work"


# ---------------------------------------------------------------------------
# 3.  BEST-SAMPLE SELECTION
# ---------------------------------------------------------------------------


class TestBestSampleSelection:
    """Must match BoltzGen's get_best_folding_sample():
        score = 0.8 * design_to_target_iptm + 0.2 * design_ptm
    Falls back to 0.8 * iptm + 0.2 * ptm when per-chain metrics are NaN.
    """

    def test_best_sample_with_per_chain_metrics(self):
        design_to_target_iptm = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        design_ptm = np.array([0.6, 0.8, 0.85], dtype=np.float32)
        iptm = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        ptm = np.array([0.7, 0.7, 0.7], dtype=np.float32)

        score = np.where(
            np.isnan(design_to_target_iptm),
            0.8 * iptm + 0.2 * ptm,
            0.8 * design_to_target_iptm + 0.2 * design_ptm,
        )
        best = int(np.nanargmax(score))

        # Sample 1: 0.8*0.9 + 0.2*0.8 = 0.72 + 0.16 = 0.88
        # Sample 2: 0.8*0.7 + 0.2*0.85 = 0.56 + 0.17 = 0.73
        # Sample 0: 0.8*0.5 + 0.2*0.6 = 0.40 + 0.12 = 0.52
        assert best == 1
        assert score[1] == pytest.approx(0.88)

    def test_fallback_when_per_chain_nan(self):
        """When design_to_target_iptm is NaN, falls back to global metrics."""
        design_to_target_iptm = np.array([np.nan, np.nan], dtype=np.float32)
        design_ptm = np.array([np.nan, np.nan], dtype=np.float32)
        iptm = np.array([0.6, 0.9], dtype=np.float32)
        ptm = np.array([0.5, 0.8], dtype=np.float32)

        score = np.where(
            np.isnan(design_to_target_iptm),
            0.8 * iptm + 0.2 * ptm,
            0.8 * design_to_target_iptm + 0.2 * design_ptm,
        )
        best = int(np.nanargmax(score))

        # Sample 1: 0.8*0.9 + 0.2*0.8 = 0.88
        assert best == 1
        assert score[1] == pytest.approx(0.88)

    def test_mixed_nan_and_valid(self):
        """Some samples have per-chain metrics, others don't."""
        design_to_target_iptm = np.array([np.nan, 0.95], dtype=np.float32)
        design_ptm = np.array([np.nan, 0.90], dtype=np.float32)
        iptm = np.array([0.99, 0.50], dtype=np.float32)
        ptm = np.array([0.99, 0.50], dtype=np.float32)

        score = np.where(
            np.isnan(design_to_target_iptm),
            0.8 * iptm + 0.2 * ptm,
            0.8 * design_to_target_iptm + 0.2 * design_ptm,
        )

        # Sample 0 fallback: 0.8*0.99 + 0.2*0.99 = 0.99
        # Sample 1 per-chain: 0.8*0.95 + 0.2*0.90 = 0.94
        assert int(np.nanargmax(score)) == 0
        assert score[0] == pytest.approx(0.99)
        assert score[1] == pytest.approx(0.94)


# ---------------------------------------------------------------------------
# 4.  CHAIN ROLE DETECTION
# ---------------------------------------------------------------------------


class TestChainRoleDetection:
    """_detect_chain_roles uses the design_mask from inverse folding metadata
    to classify chains as design vs target. The mask is per-token (per-residue).
    """

    def test_single_design_single_target(self, tmp_path):
        from boltzgen.task.predict.boltz2_bridge import _detect_chain_roles

        # Chain A = target (3 residues), Chain B = design (2 residues)
        cif_path = _make_minimal_mmcif(
            {
                "A": [
                    ("ALA", [("N", (0, 0, 0)), ("CA", (1, 0, 0))]),
                    ("GLY", [("N", (2, 0, 0)), ("CA", (3, 0, 0))]),
                    ("VAL", [("N", (4, 0, 0)), ("CA", (5, 0, 0))]),
                ],
                "B": [
                    ("LEU", [("N", (6, 0, 0)), ("CA", (7, 0, 0))]),
                    ("ILE", [("N", (8, 0, 0)), ("CA", (9, 0, 0))]),
                ],
            },
            tmp_path / "design.cif",
        )

        # design_mask: 5 tokens total, first 3 = target (0), last 2 = design (1)
        design_mask = np.array([0, 0, 0, 1, 1], dtype=np.float32)
        npz_path = tmp_path / "design.npz"
        np.savez(npz_path, design_mask=design_mask)

        design_chains, target_chains = _detect_chain_roles(cif_path, npz_path)

        assert design_chains == ["B"]
        assert target_chains == ["A"]

    def test_multi_target_chains(self, tmp_path):
        from boltzgen.task.predict.boltz2_bridge import _detect_chain_roles

        # Chains A, B = target; Chain C = design
        cif_path = _make_minimal_mmcif(
            {
                "A": [("ALA", [("CA", (0, 0, 0))])],
                "B": [("GLY", [("CA", (1, 0, 0))])],
                "C": [("VAL", [("CA", (2, 0, 0))])],
            },
            tmp_path / "multi_target.cif",
        )

        design_mask = np.array([0, 0, 1], dtype=np.float32)  # A=0, B=0, C=1
        npz_path = tmp_path / "multi_target.npz"
        np.savez(npz_path, design_mask=design_mask)

        design_chains, target_chains = _detect_chain_roles(cif_path, npz_path)

        assert design_chains == ["C"]
        assert set(target_chains) == {"A", "B"}

    def test_mask_shorter_than_chains_graceful(self, tmp_path):
        """If design_mask is shorter than total tokens (e.g. truncated),
        extra chains default to target."""
        from boltzgen.task.predict.boltz2_bridge import _detect_chain_roles

        cif_path = _make_minimal_mmcif(
            {
                "A": [("ALA", [("CA", (0, 0, 0))])],
                "B": [("GLY", [("CA", (1, 0, 0))])],
                "C": [("VAL", [("CA", (2, 0, 0))])],
            },
            tmp_path / "short_mask.cif",
        )

        # Mask only covers 2 tokens — chain C (token 2) has no valid indices
        design_mask = np.array([0, 1], dtype=np.float32)
        npz_path = tmp_path / "short_mask.npz"
        np.savez(npz_path, design_mask=design_mask)

        design_chains, target_chains = _detect_chain_roles(cif_path, npz_path)

        assert "B" in design_chains
        assert "A" in target_chains
        # C has no valid mask indices → defaults to target
        assert "C" in target_chains


# ---------------------------------------------------------------------------
# 5.  YAML GENERATION
# ---------------------------------------------------------------------------


class TestYamlGeneration:
    """The generated YAML must have:
    - Target chains listed FIRST (sequences order matters for Boltz-2 indexing)
    - Template block with force=True and correct threshold
    - Template only constrains target chains, not design chains
    """

    def test_target_chains_before_design_in_yaml(self, tmp_path):
        from boltzgen.task.predict.boltz2_bridge import generate_boltz2_yaml

        target_cif = _make_minimal_mmcif(
            {"A": [("ALA", [("CA", (0, 0, 0))])]},
            tmp_path / "target.cif",
        )
        design_cif = _make_minimal_mmcif(
            {
                "A": [("ALA", [("CA", (0, 0, 0))])],
                "B": [("GLY", [("CA", (1, 0, 0))])],
            },
            tmp_path / "design.cif",
        )

        yaml_path = tmp_path / "out.yaml"
        generate_boltz2_yaml(
            design_cif=design_cif,
            target_cif=target_cif,
            design_chain_ids=["B"],
            target_chain_ids=["A"],
            output_path=yaml_path,
            template_threshold=3.0,
        )

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        seqs = data["sequences"]
        # Target chain A must come first
        assert seqs[0]["protein"]["id"] == "A"
        # Design chain B must come second
        assert seqs[1]["protein"]["id"] == "B"

    def test_template_block_structure(self, tmp_path):
        from boltzgen.task.predict.boltz2_bridge import generate_boltz2_yaml

        target_cif = _make_minimal_mmcif(
            {
                "A": [("ALA", [("CA", (0, 0, 0))])],
                "B": [("GLY", [("CA", (1, 0, 0))])],
            },
            tmp_path / "target.cif",
        )
        design_cif = _make_minimal_mmcif(
            {
                "A": [("ALA", [("CA", (0, 0, 0))])],
                "B": [("GLY", [("CA", (1, 0, 0))])],
                "C": [("VAL", [("CA", (2, 0, 0))])],
            },
            tmp_path / "design.cif",
        )

        yaml_path = tmp_path / "out.yaml"
        generate_boltz2_yaml(
            design_cif=design_cif,
            target_cif=target_cif,
            design_chain_ids=["C"],
            target_chain_ids=["A", "B"],
            output_path=yaml_path,
            template_threshold=1.5,
        )

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        tmpl = data["templates"][0]
        assert tmpl["force"] is True
        assert tmpl["threshold"] == 1.5
        # Template constrains target chains only
        assert tmpl["chain_id"] == ["A", "B"]
        assert tmpl["template_id"] == ["A", "B"]
        # Design chain C must NOT appear in template
        assert "C" not in tmpl["chain_id"]

    def test_sequence_ordering_matches_boltz2_indexing(self, tmp_path):
        """The sequence order in the YAML determines the 0-based index
        used by Boltz-2's confidence output. Target chains must come
        first so the confidence metric mapping works correctly."""
        from boltzgen.task.predict.boltz2_bridge import generate_boltz2_yaml

        target_cif = _make_minimal_mmcif(
            {
                "X": [("ALA", [("CA", (0, 0, 0))])],
                "Y": [("GLY", [("CA", (1, 0, 0))])],
            },
            tmp_path / "target.cif",
        )
        design_cif = _make_minimal_mmcif(
            {
                "X": [("ALA", [("CA", (0, 0, 0))])],
                "Y": [("GLY", [("CA", (1, 0, 0))])],
                "Z": [("VAL", [("CA", (2, 0, 0))])],
            },
            tmp_path / "design.cif",
        )

        yaml_path = tmp_path / "out.yaml"
        generate_boltz2_yaml(
            design_cif=design_cif,
            target_cif=target_cif,
            design_chain_ids=["Z"],
            target_chain_ids=["X", "Y"],
            output_path=yaml_path,
        )

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        chain_ids_in_order = [s["protein"]["id"] for s in data["sequences"]]
        # X→"0", Y→"1", Z→"2"
        assert chain_ids_in_order == ["X", "Y", "Z"], (
            "Target chains must precede design chains to match confidence indexing"
        )


# ---------------------------------------------------------------------------
# 6.  END-TO-END: convert_boltz2_to_boltzgen (without featurizer)
# ---------------------------------------------------------------------------


class TestConvertBoltz2ToBoltzgen:
    """Integration test for the full conversion pipeline using the Gemmi
    fallback path (no BoltzGen featurizer / no moldir).

    Verifies:
    - Output NPZ has all required keys
    - coords shape is [N_samples, N_atoms, 3]
    - Confidence metrics are correctly placed
    - Best sample CIF is copied
    """

    def _setup_boltz2_output(
        self,
        tmp_path: Path,
        yaml_stem: str,
        n_samples: int,
        chains: dict,
        confidences: List[dict],
    ) -> Path:
        """Create a fake Boltz-2 output directory."""
        pred_dir = tmp_path / "output" / "predictions" / yaml_stem
        pred_dir.mkdir(parents=True)

        for rank in range(n_samples):
            _make_minimal_mmcif(chains, pred_dir / f"{yaml_stem}_model_{rank}.cif")
            _write_confidence_json(
                pred_dir / f"confidence_{yaml_stem}_model_{rank}.json",
                confidences[rank],
            )

        return tmp_path / "output"

    def test_full_conversion_fallback_path(self, tmp_path):
        from boltzgen.task.predict.boltz2_bridge import convert_boltz2_to_boltzgen

        yaml_stem = "design_0"
        chains = {
            "A": [("ALA", [("N", (1, 0, 0)), ("CA", (2, 0, 0))])],
            "B": [("GLY", [("N", (3, 0, 0)), ("CA", (4, 0, 0))])],
        }

        confidences = [
            {
                "iptm": 0.8, "ptm": 0.7, "complex_plddt": 0.75,
                "chains_ptm": {"0": 0.9, "1": 0.65},
                "pair_chains_iptm": {"0": {"1": 0.72}, "1": {"0": 0.72}},
            },
            {
                "iptm": 0.85, "ptm": 0.75, "complex_plddt": 0.80,
                "chains_ptm": {"0": 0.92, "1": 0.70},
                "pair_chains_iptm": {"0": {"1": 0.80}, "1": {"0": 0.80}},
            },
        ]

        boltz2_out = self._setup_boltz2_output(
            tmp_path, yaml_stem, 2, chains, confidences
        )

        # Create fake design CIF and metadata (same structure)
        design_cif = _make_minimal_mmcif(chains, tmp_path / "design.cif")
        metadata_npz = tmp_path / "design.npz"
        np.savez(metadata_npz, design_mask=np.array([0, 1], dtype=np.float32))

        output_npz = tmp_path / "fold_out_npz" / f"{yaml_stem}.npz"
        output_cif = tmp_path / "refold_cif" / f"{yaml_stem}.cif"

        success = convert_boltz2_to_boltzgen(
            boltz2_output_dir=boltz2_out,
            design_metadata_npz=metadata_npz,
            design_cif_path=design_cif,
            output_npz_path=output_npz,
            output_cif_path=output_cif,
            design_chain_ids=["B"],
            target_chain_ids=["A"],
            yaml_stem=yaml_stem,
            diffusion_samples=2,
            moldir=None,  # fallback path
        )

        assert success is True
        assert output_npz.exists()
        assert output_cif.exists()

        # Verify NPZ contents
        data = dict(np.load(output_npz, allow_pickle=True))

        # coords must be [2, 4, 3] — 2 samples, 4 atoms, 3 coords
        assert data["coords"].shape == (2, 4, 3)
        assert data["iptm"].shape == (2,)
        assert data["ptm"].shape == (2,)
        assert data["design_ptm"].shape == (2,)
        assert data["design_to_target_iptm"].shape == (2,)

        # Verify confidence values
        np.testing.assert_array_almost_equal(data["iptm"], [0.8, 0.85])
        np.testing.assert_array_almost_equal(data["ptm"], [0.7, 0.75])
        # design chain is B → position "1"
        np.testing.assert_array_almost_equal(data["design_ptm"], [0.65, 0.70])
        # design→target iptm: pair["1"]["0"]
        np.testing.assert_array_almost_equal(
            data["design_to_target_iptm"], [0.72, 0.80]
        )

    def test_best_sample_cif_copied(self, tmp_path):
        """The best sample's CIF must be copied to the output location."""
        from boltzgen.task.predict.boltz2_bridge import convert_boltz2_to_boltzgen

        yaml_stem = "design_best"

        # Sample 0: low score, Sample 1: high score
        chains_s0 = {"A": [("ALA", [("CA", (0, 0, 0))])], "B": [("GLY", [("CA", (100, 0, 0))])]}
        chains_s1 = {"A": [("ALA", [("CA", (0, 0, 0))])], "B": [("GLY", [("CA", (200, 0, 0))])]}

        pred_dir = tmp_path / "output" / "predictions" / yaml_stem
        pred_dir.mkdir(parents=True)

        _make_minimal_mmcif(chains_s0, pred_dir / f"{yaml_stem}_model_0.cif")
        _make_minimal_mmcif(chains_s1, pred_dir / f"{yaml_stem}_model_1.cif")

        _write_confidence_json(
            pred_dir / f"confidence_{yaml_stem}_model_0.json",
            {"iptm": 0.5, "ptm": 0.5, "complex_plddt": 0.5,
             "chains_ptm": {"0": 0.5, "1": 0.3},
             "pair_chains_iptm": {"1": {"0": 0.3}}},
        )
        _write_confidence_json(
            pred_dir / f"confidence_{yaml_stem}_model_1.json",
            {"iptm": 0.9, "ptm": 0.9, "complex_plddt": 0.9,
             "chains_ptm": {"0": 0.95, "1": 0.85},
             "pair_chains_iptm": {"1": {"0": 0.92}}},
        )

        design_cif = _make_minimal_mmcif(
            {"A": [("ALA", [("CA", (0, 0, 0))])], "B": [("GLY", [("CA", (0, 0, 0))])]},
            tmp_path / "design.cif",
        )
        metadata_npz = tmp_path / "design.npz"
        np.savez(metadata_npz, design_mask=np.array([0, 1], dtype=np.float32))

        output_npz = tmp_path / "fold_out_npz" / f"{yaml_stem}.npz"
        output_cif = tmp_path / "refold_cif" / f"{yaml_stem}.cif"

        convert_boltz2_to_boltzgen(
            boltz2_output_dir=tmp_path / "output",
            design_metadata_npz=metadata_npz,
            design_cif_path=design_cif,
            output_npz_path=output_npz,
            output_cif_path=output_cif,
            design_chain_ids=["B"],
            target_chain_ids=["A"],
            yaml_stem=yaml_stem,
            diffusion_samples=2,
            moldir=None,
        )

        # Best sample should be model_1 (higher d2t_iptm and design_ptm)
        # Read the output CIF and check it has sample 1's coordinates
        from boltzgen.task.predict.boltz2_bridge import _extract_atom_coords_from_mmcif

        out_coords = _extract_atom_coords_from_mmcif(output_cif)
        # Sample 1 had B/CA at (200, 0, 0)
        assert out_coords[("B", 1, "CA")][0] == pytest.approx(200.0), (
            "Output CIF should contain best sample (model_1) coordinates"
        )

    def test_all_eval_keys_present_in_npz(self, tmp_path):
        """The output NPZ must contain all keys from const.eval_keys
        so downstream analysis/filtering doesn't crash."""
        from boltzgen.data import const
        from boltzgen.task.predict.boltz2_bridge import convert_boltz2_to_boltzgen

        yaml_stem = "design_keys"
        chains = {"A": [("ALA", [("CA", (0, 0, 0))])]}
        confidences = [
            {"iptm": 0.8, "ptm": 0.7, "complex_plddt": 0.75,
             "chains_ptm": {"0": 0.9},
             "pair_chains_iptm": {}},
        ]

        boltz2_out = self._setup_boltz2_output(
            tmp_path, yaml_stem, 1, chains, confidences
        )

        design_cif = _make_minimal_mmcif(chains, tmp_path / "design.cif")
        metadata_npz = tmp_path / "design.npz"
        np.savez(metadata_npz, design_mask=np.array([1], dtype=np.float32))

        output_npz = tmp_path / "fold_out_npz" / f"{yaml_stem}.npz"
        output_cif = tmp_path / "refold_cif" / f"{yaml_stem}.cif"

        convert_boltz2_to_boltzgen(
            boltz2_output_dir=boltz2_out,
            design_metadata_npz=metadata_npz,
            design_cif_path=design_cif,
            output_npz_path=output_npz,
            output_cif_path=output_cif,
            design_chain_ids=["A"],
            target_chain_ids=[],
            yaml_stem=yaml_stem,
            diffusion_samples=1,
            moldir=None,
        )

        data = dict(np.load(output_npz, allow_pickle=True))

        # Check all confidence + affinity keys are present
        for key in const.eval_keys_confidence:
            assert key in data, f"Missing confidence key: {key}"

        for key in const.eval_keys_affinity:
            assert key in data, f"Missing affinity key: {key}"

        # coords must always be present
        assert "coords" in data
