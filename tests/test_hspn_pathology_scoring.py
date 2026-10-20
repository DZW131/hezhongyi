import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_hspn_pathology import (
    GlomerulusEvidence,
    group_and_score,
    row_to_glomerulus,
    score_iskdc,
    score_oxford,
    Thresholds,
)


def make_glom(index, **kwargs):
    defaults = {
        "slide_id": "case_a",
        "glomerulus_id": "g{}".format(index),
    }
    defaults.update(kwargs)
    return GlomerulusEvidence(**defaults)


def test_oxford_mestc_thresholds():
    glomeruli = []
    for index in range(10):
        glomeruli.append(
            make_glom(
                index,
                mesangial=index < 5,
                endocapillary=index == 0,
                segmental_sclerosis=index == 1,
                cellular_crescent=index in {0, 1},
                fibrocellular_crescent=index == 2,
                ifta_percent=0.30,
            )
        )

    oxford, counts, percentages, warnings = score_oxford(glomeruli)

    assert oxford["M"] == "M1"
    assert oxford["E"] == "E1"
    assert oxford["S"] == "S1"
    assert oxford["T"] == "T1"
    assert oxford["C"] == "C2"
    assert oxford["MEST_C"] == "M1E1S1T1C2"
    assert counts["active_crescent"] == 3
    assert percentages["active_crescent"] == 0.3
    assert warnings == []


def test_oxford_missing_t_warns_but_scores_other_components():
    glomeruli = [make_glom(index) for index in range(8)]

    oxford, _, _, warnings = score_oxford(glomeruli)

    assert oxford["MEST_C"] == "M0E0S0T_NAC0"
    assert warnings


def test_iskdc_rule_boundaries():
    minimal = [make_glom(index) for index in range(8)]
    mesangial_only = [make_glom(index, mesangial=index < 3) for index in range(8)]
    focal_iii = [make_glom(index, mesangial=index < 3, fibrinoid_necrosis=index == 0) for index in range(8)]
    diffuse_iii = [make_glom(index, mesangial=index < 5, endocapillary=index == 0) for index in range(8)]
    class_iv = [make_glom(index, cellular_crescent=index < 4) for index in range(8)]
    class_v = [make_glom(index, cellular_crescent=index < 6) for index in range(8)]
    class_vi = [make_glom(index, membranoproliferative_pattern=index == 0) for index in range(8)]

    assert score_iskdc(minimal)["grade"] == "I"
    assert score_iskdc(mesangial_only)["grade"] == "II"
    assert score_iskdc(focal_iii)["grade"] == "IIIa"
    assert score_iskdc(diffuse_iii)["grade"] == "IIIb"
    assert score_iskdc(class_iv)["grade"] == "IV"
    assert score_iskdc(class_v)["grade"] == "V"
    assert score_iskdc(class_vi)["grade"] == "VI"


def test_fibrous_crescents_count_for_iskdc_not_oxford_c():
    glomeruli = [make_glom(index, fibrous_crescent=index < 4) for index in range(8)]

    oxford, counts, percentages, warnings = score_oxford(glomeruli)

    assert score_iskdc(glomeruli)["grade"] == "IV"
    assert score_iskdc(glomeruli)["crescent_percent"] == "0.5000"
    assert oxford["C"] == "C0"
    assert counts["any_crescent"] == 4
    assert counts["active_crescent"] == 0
    assert percentages["fibrous_crescent"] == 0.5
    assert warnings


def test_row_parser_accepts_probabilities_subtypes_and_percent_strings():
    row = {
        "slide_id": "case_b",
        "glomerulus_id": "g1",
        "mesangial_prob": "0.71",
        "crescent_subtype": "fibrocellular",
        "ifta_percent": "30%",
        "bbox": "1,2,11,12",
    }

    glom = row_to_glomerulus(row, 0, Thresholds(), generic_crescent_active=True)

    assert glom.slide_id == "case_b"
    assert glom.mesangial
    assert glom.fibrocellular_crescent
    assert glom.active_crescent
    assert glom.ifta_percent == 0.30
    assert glom.bbox == (1.0, 2.0, 11.0, 12.0)


def test_row_parser_uses_subtype_specific_thresholds():
    row = {
        "slide_id": "case_b",
        "glomerulus_id": "g2",
        "crescent_prob": "0.31",
        "cellular_crescent_prob": "0.60",
    }

    glom = row_to_glomerulus(
        row,
        0,
        Thresholds(crescent=0.30, cellular_crescent=0.70),
        generic_crescent_active=False,
    )

    assert glom.generic_crescent
    assert glom.any_crescent
    assert not glom.cellular_crescent
    assert not glom.active_crescent


def test_group_and_score_merges_multiple_rows_per_glomerulus():
    glomeruli = [
        make_glom(0, glomerulus_id="g0", mesangial=True),
        make_glom(0, glomerulus_id="g0", cellular_crescent=True),
        make_glom(1, glomerulus_id="g1"),
    ]

    scores = group_and_score(glomeruli, min_glomeruli=1)

    assert len(scores) == 1
    assert scores[0].total_glomeruli == 2
    assert scores[0].counts["mesangial"] == 1
    assert scores[0].counts["active_crescent"] == 1
