"""Reader presentation of original saved aggregate controls; no model/data imports."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).absolute().parents[1]
BUNDLE = ROOT / "reference/completed_review_bundle"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, records):
    with (ROOT / "results" / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fill_table(key, text):
    path = ROOT / "docs/02_RESULTS.md"
    original = path.read_text()
    block = "<!-- BEGIN " + key + " -->\n" + text + "\n<!-- END " + key + " -->"
    pattern = r"<!-- BEGIN " + key + r" -->.*?<!-- END " + key + r" -->"
    if "<!-- " + key + " -->" in original:
        updated = original.replace("<!-- " + key + " -->", block)
    else:
        updated, count = re.subn(pattern, lambda match: block, original, flags=re.S)
        assert count == 1, key
    path.write_text(updated)


def main():
    legend = {x["code"]: x for x in rows(BUNDLE / "tables/all427_unambiguous_recipe_codes.csv")}
    core = rows(BUNDLE / "tables/article_tables/all456_existing_core_matrix_comparisons.csv")
    absolute = rows(BUNDLE / "tables/article_tables/all3686_absolute_family_endpoint_cells.csv")

    held = []
    for x in core:
        if x["endpoint"] not in {"AUROC", "positive_image_Dice"}:
            continue
        family = legend[x["second_code"]]
        if not (family["recipe"] == "dinov3_dense_lora"
                and family["source_unit"].startswith("holdout_")
                and legend[x["first_code"]]["recipe"] == "deeplabv3plus_full"):
            continue
        held.append({"source": family["source_unit"].replace("holdout_", ""),
                     "budget": {"b0p2": "20%", "b1": "100%"}[family["budget"]], **x})
    assert len(held) == 12
    write_csv("held_source_lora_vs_cnn.csv", held)

    names = {"F381": "Frozen DINO", "F382": "LoRA", "F383": "Normal reference",
             "F384": "Full supervised port"}
    endpoint_set = {"AUROC", "positive_image_Dice", "pooled_native_pixel_AP",
                    "recall_at_source_threshold", "review_rate_at_source_threshold"}
    matched = [{"system": names[x["family_code"]], **x} for x in absolute
               if x["family_code"] in names and x["endpoint"] in endpoint_set
               and x["cohort"] in {"ucf_reported_real_confirmation_v1", "scdd_publication_v1"}]
    assert len(matched) == 40
    write_csv("matched_ucf20_absolute.csv", matched)

    pvel_codes = {"F295": ("SCDD 20%", "CNN"), "F304": ("SCDD 20%", "LoRA"),
                  "F336": ("SCDD 100%", "CNN"), "F345": ("SCDD 100%", "LoRA"),
                  "F389": ("UCF 100%", "CNN"), "F398": ("UCF 100%", "LoRA")}
    pvel = [{"source_regime": pvel_codes[x["family_code"]][0],
             "system": pvel_codes[x["family_code"]][1], **x} for x in absolute
            if x["cohort"] == "pvelad_restricted_confirmation_v2" and x["family_code"] in pvel_codes]
    assert len(pvel) == 18
    write_csv("pvel_original_dense_regimes.csv", pvel)

    table = ["| Held source | Budget | CNN AUROC | LoRA AUROC | CNN Dice | LoRA Dice |",
             "|---|---|---:|---:|---:|---:|"]
    conditions = [(s, b) for s in ["ARTS", "CFVS", "CSIR"] for b in ["20%", "100%"]]
    lookup = {(x["source"], x["budget"], x["endpoint"]): x for x in held}
    for source, budget in conditions:
        a = lookup[source, budget, "AUROC"]
        d = lookup[source, budget, "positive_image_Dice"]
        table.append(f"| {source} | {budget} | {float(a['first']):.3f} | {float(a['second']):.3f} | "
                     f"{float(d['first']):.3f} | {float(d['second']):.3f} |")
    fill_table("HELD_TABLE", "\n".join(table))

    table = ["| System | AUROC | Native AP | Positive Dice | Recall | Review fraction |",
             "|---|---:|---:|---:|---:|---:|"]
    ml = {(x["family_code"], x["cohort"], x["endpoint"]): x for x in matched}
    ucf = "ucf_reported_real_confirmation_v1"
    for code, name in names.items():
        values = [float(ml[code, ucf, ep]["descriptive_fitted_mean"]) for ep in
                  ["AUROC", "pooled_native_pixel_AP", "positive_image_Dice",
                   "recall_at_source_threshold", "review_rate_at_source_threshold"]]
        table.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    fill_table("MATCHED_TABLE", "\n".join(table))

    table = ["| Source regime | Known-crack recall | Author-normal FPR | Box-point hit |",
             "|---|---:|---:|---:|"]
    pl = {(x["source_regime"], x["system"], x["endpoint"]): x for x in pvel}
    for source in ["SCDD 20%", "SCDD 100%", "UCF 100%"]:
        cells = []
        for ep in ["known_crack_recall", "author_normal_FPR", "box_point_hit_rate"]:
            cells.append(" / ".join(f"{float(pl[source, system, ep]['descriptive_fitted_mean']):.3f}"
                                    for system in ["CNN", "LoRA"]))
        table.append("| " + source + " | " + " | ".join(cells) + " |")
    fill_table("PVEL_TABLE", "\n".join(table))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    y = np.arange(6)
    for ax, ep, title, color in zip(axes, ["AUROC", "positive_image_Dice"],
                                   ["Image recognition", "Native crack overlap"],
                                   ["#087F8C", "#8856A7"]):
        entries = [lookup[s, b, ep] for s, b in conditions]
        values = np.array([float(x["difference"]) for x in entries])
        low = np.array([float(x["ci95_low"]) for x in entries])
        high = np.array([float(x["ci95_high"]) for x in entries])
        ax.errorbar(values, y, xerr=[values-low, high-values], fmt="o", color=color,
                    capsize=3, elinewidth=1.6, markersize=6)
        ax.axvline(0, color="#687580", lw=1)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("LoRA minus adapted CNN")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.set_yticks(y, [s + "  " + b for s, b in conditions])
    axes[0].invert_yaxis()
    fig.suptitle("Recognition improves; localization remains mixed", fontsize=15, fontweight="bold", y=.98)
    fig.text(.5, .025, "All six original conditions. Fitted-system conditional 95% intervals; unadjusted across 6,212 contrasts.\n"
             "Three paired seeds per condition; provided acquisition families are not verified physical modules.",
             ha="center", fontsize=9, color="#435463")
    fig.tight_layout(rect=[0, .12, 1, .92])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(ROOT / f"figures/held_source_tradeoff.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    colors = ["#7A8A99", "#087F8C", "#9C85B5", "#C68952"]
    short = ["Frozen\nDINO", "LoRA", "Normal\nreference", "Supervised\nlocal port"]
    for ax, cohort, title in zip(axes, [ucf, "scdd_publication_v1"],
                                 ["UCF 507 confirmation", "Transfer to SCDD 117"]):
        entries = [ml[code, cohort, "positive_image_Dice"] for code in names]
        mean = [float(x["descriptive_fitted_mean"]) for x in entries]
        ax.bar(range(4), mean, color=colors, width=.65, alpha=.85)
        for i, x in enumerate(entries):
            points = [float(x["seed" + s]) for s in ["11", "22", "33"]]
            ax.scatter([i-.12, i, i+.12], points, s=24, color="#182B3A", zorder=4)
        ax.set_xticks(range(4), short)
        ax.set_ylim(0, .65)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Positive-image Dice")
    fig.suptitle("Same UCF training masks, different transfer behavior", fontsize=15, fontweight="bold", y=.98)
    fig.text(.5, .025, "All four original central systems: 86 training + 144 validation masks, within 574 development images.\n"
             "Bars: per-fit means. Points: seeds 11/22/33, not confidence intervals. HPO, pretraining and work differ.",
             ha="center", fontsize=9, color="#435463")
    fig.tight_layout(rect=[0, .12, 1, .92])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(ROOT / f"figures/matched_ucf20_transfer.{ext}", dpi=180)
    plt.close(fig)
    print("Reader summaries: 12 held-source rows, 40 matched-system cells, 18 PVEL cells. No scientific fitting/scoring.")


if __name__ == "__main__":
    main()
