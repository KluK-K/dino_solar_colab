"""Read-only byte/link/coverage checks. Standard library only; no model imports."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]
BUNDLE = ROOT / "reference/completed_review_bundle"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def payloads():
    return {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(ROOT).parts
            and "__pycache__" not in p.relative_to(ROOT).parts and p.name != ".DS_Store"}


def main():
    files = payloads()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    expected = {x["path"]: x for x in manifest["files"]}
    assert set(files) - {"MANIFEST.json"} == set(expected), "Unexpected or missing payload"
    parsed = 0
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
                          ".joblib", ".sqlite", ".db", ".key", ".pem"}
    secret_patterns = [
        re.compile(r"gh[pousr]_[A-Za-z0-9]{25,}"),
        re.compile(r"AKIA[A-Z0-9]{16}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}"),
    ]
    for name, record in expected.items():
        path = files[name]
        assert not path.is_symlink(), name
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], name
        assert path.suffix.lower() not in forbidden_suffixes, name
        assert not any(part.startswith(".env") or part in {".venv", "site-packages"} for part in path.parts), name
        assert "OWNER_ONLY" not in name and not name.endswith("frozen_registry.json"), name
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
            parsed += 1
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".svg"}:
            content = path.read_text()
            assert not any(pattern.search(content) for pattern in secret_patterns), "Credential-like bytes: " + name

    context = json.loads((ROOT / "PACKAGE_CONTEXT.json").read_text())
    assert context["course_final_project_included"] is False
    assert context["journal_submission_ready"] is False
    assert context["manuscript_revision"] == 63
    assert context["original_reference_bundle_revision"] == 61
    assert context["current_main_pages"] == 14 and context["current_supplement_pages"] == 21
    assert context["selected_fixed_case_illustration_shared_with_owner_authorization"]
    assert not context["raw_dataset_files_or_separate_prediction_arrays_included"]
    assert digest(BUNDLE / "MANIFEST.json") == context["source_bundle_manifest_sha256"]
    for record in context["copies"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], record["path"]
    visual = json.loads((ROOT / "paper/VISUAL_UPDATE_CONTEXT.json").read_text())
    assert visual["owner_permission"] == "允许同步这张展示图和新版 PDF"
    assert visual["selected_figure"]["cases"] == 6 and visual["selected_figure"]["systems"] == 4
    assert visual["selected_figure"]["illustrative_seed"] == 11
    assert visual["selected_figure"]["all_cases_fixed_before_predictive_scoring"]
    assert visual["selected_figure"]["shared_pixel_softmax_scale"] == [0, 1]
    assert not visual["selected_figure"]["per_image_normalization"]
    assert not visual["selected_figure"]["separate_prediction_arrays_or_masks_included"]
    assert not visual["new_training_inference_or_statistical_replay"]
    assert visual["all92_scientific_bindings_unchanged"]

    check = subprocess.run([sys.executable, str(BUNDLE / "verify_package.py")],
                           cwd=BUNDLE, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    for line in check.stdout.splitlines():
        json.loads(line)

    core = {x["contrast_id"]: x for x in rows(BUNDLE / "tables/article_tables/all456_existing_core_matrix_comparisons.csv")}
    held = rows(ROOT / "results/held_source_lora_vs_cnn.csv")
    assert len(held) == 12
    assert {(x["source"], x["budget"]) for x in held} == {
        (s, b) for s in ["ARTS", "CFVS", "CSIR"] for b in ["20%", "100%"]}
    for record in held:
        original = core[record["contrast_id"]]
        assert all(record[k] == v for k, v in original.items())
    auroc = [x for x in held if x["endpoint"] == "AUROC"]
    dice = [x for x in held if x["endpoint"] == "positive_image_Dice"]
    assert len(auroc) == len(dice) == 6
    assert all(x["pair_signs"] == "+,+,+" and float(x["difference"]) > 0 for x in auroc)
    assert sum(float(x["difference"]) < 0 for x in dice) == 4

    absolute = {(x["family_code"], x["cohort"], x["stratum"], x["endpoint"]): x for x in
                rows(BUNDLE / "tables/article_tables/all3686_absolute_family_endpoint_cells.csv")}
    for file, count in [("matched_ucf20_absolute.csv", 40), ("pvel_original_dense_regimes.csv", 18)]:
        selected = rows(ROOT / "results" / file)
        assert len(selected) == count
        for record in selected:
            original = absolute[tuple(record[k] for k in ["family_code", "cohort", "stratum", "endpoint"])]
            assert all(record[k] == v for k, v in original.items()), file
    assert len(rows(ROOT / "training/all1061_source_recipe_records_sanitized.csv")) == 1061
    assert len(rows(ROOT / "training/all427_source_recipe_families_sanitized.csv")) == 427

    links = 0
    front = [ROOT / "README.md", ROOT / "RIGHTS_AND_SCOPE.md"] + list((ROOT / "docs").glob("*.md"))
    front += [ROOT / d / "README.md" for d in ["results", "training", "figures", "reference", "paper"]]
    for path in front:
        content = path.read_text()
        assert not re.search(r"<!-- (?:HELD|MATCHED|PVEL)_TABLE -->", content), "Unfilled table"
        for raw in re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", content):
            target = raw.strip().strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            linked = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            assert linked.is_relative_to(ROOT) and linked.exists(), (path.name, target)
            links += 1
    print(json.dumps({
        "state": "read_only_package_checks_passed",
        "manifest_payload_files": len(expected),
        "python_AST_files": parsed,
        "front_door_local_links": links,
        "original_bundle_manifest_files": 311,
        "all6212_original_contrasts_retained": True,
        "all1061_training_view_records_retained": True,
        "reader_subset_rows": {"held_source": 12, "matched_UCF20": 40, "PVEL": 18},
        "original_numeric_fields_unchanged": True,
        "model_import_training_inference_download_or_scientific_replay": False,
        "authorized_six_case_flattened_illustration_included": True,
        "raw_dataset_weights_optimizer_arrays_or_owner_inventories_included": False,
        "current_article_revision": 63,
        "current_PDF_pages": {"main": 14, "supplement": 21},
        "journal_submission_ready": False
    }, indent=2))


if __name__ == "__main__":
    main()
