from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path("analysis/scc_bcc")
OUT = ROOT / "consolidated_evidence"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def main():
    print("=" * 80)
    print("DERMASENSE SCC CONSOLIDATED EVIDENCE SUMMARY")
    print("=" * 80)

    rows = []

    # ------------------------------------------------------------------
    # 1. C1 vs F1 paired lesion displacement
    # ------------------------------------------------------------------
    displacement_path = (
        ROOT / "f1" / "c1_vs_f1_lesion_displacement.csv"
    )

    if displacement_path.exists():
        df = load_csv(displacement_path)

        for group_name in ("problematic", "clean"):
            g = df[df["group"] == group_name]

            rows.append({
                "category": "Paired geometry",
                "analysis": "C1 → F1 margin change",
                "group": group_name,
                "n": len(g),
                "C1_value": g["bcc_minus_scc_distance_c1"].mean(),
                "F1_value": g["bcc_minus_scc_distance_f1"].mean(),
                "change": g["delta_margin"].mean(),
                "p_value": None,
                "interpretation": (
                    "Negative change = movement toward SCC"
                ),
            })

        # Overall paired change
        rows.append({
            "category": "Paired geometry",
            "analysis": "Overall C1 → F1 margin change",
            "group": "all",
            "n": len(df),
            "C1_value": df["bcc_minus_scc_distance_c1"].mean(),
            "F1_value": df["bcc_minus_scc_distance_f1"].mean(),
            "change": df["delta_margin"].mean(),
            "p_value": 0.222107,
            "interpretation": (
                "Wilcoxon paired test; not statistically significant"
            ),
        })

    # ------------------------------------------------------------------
    # 2. C1/F1 raw geometry
    # ------------------------------------------------------------------
    size_path = (
        ROOT / "size_adjusted_geometry"
        / "size_adjusted_geometry_results.csv"
    )

    if size_path.exists():
        df = load_csv(size_path)

        # Be deliberately permissive because the script evolved.
        for _, r in df.iterrows():
            analysis = str(r.get("analysis", r.get("test", "")))
            variable = str(r.get("variable", ""))

            if (
                "RAW GROUP" in analysis.upper()
                or "raw" in analysis.lower()
            ):
                rows.append({
                    "category": "Raw geometry",
                    "analysis": f"{analysis} {variable}".strip(),
                    "group": str(r.get("group", "")),
                    "n": r.get("n", None),
                    "C1_value": r.get("mean", r.get("value", None)),
                    "F1_value": None,
                    "change": None,
                    "p_value": r.get("p_value", None),
                    "interpretation": "Raw problematic vs clean comparison",
                })

    # ------------------------------------------------------------------
    # 3. Size ↔ margin
    # ------------------------------------------------------------------
    corr_path = (
        ROOT / "size_margin_relationship"
        / "size_margin_correlations.csv"
    )

    if corr_path.exists():
        df = load_csv(corr_path)

        for _, r in df.iterrows():
            rows.append({
                "category": "Clinical confound audit",
                "analysis": f"{r.get('model', '')} size ↔ margin "
                            f"{r.get('variable', '')}".strip(),
                "group": "all SCC lesions",
                "n": r.get("n", 22),
                "C1_value": r.get("spearman_rho", None)
                    if str(r.get("model", "")).upper() == "C1"
                    else None,
                "F1_value": r.get("spearman_rho", None)
                    if str(r.get("model", "")).upper() == "F1"
                    else None,
                "change": None,
                "p_value": r.get("spearman_p", None),
                "interpretation": "Weak/non-significant monotonic association",
            })

    # ------------------------------------------------------------------
    # 4. Clinical metadata
    # ------------------------------------------------------------------
    numeric_path = (
        ROOT / "clinical_metadata"
        / "numeric_comparisons.csv"
    )

    if numeric_path.exists():
        df = load_csv(numeric_path)

        for _, r in df.iterrows():
            rows.append({
                "category": "Clinical metadata",
                "analysis": str(r.get("variable", "")),
                "group": "problematic vs clean",
                "n": None,
                "C1_value": r.get(
                    "problematic_mean",
                    r.get("problematic", None),
                ),
                "F1_value": None,
                "change": None,
                "p_value": r.get("p_value", None),
                "interpretation": "Exploratory clinical confound comparison",
            })

    categorical_path = (
        ROOT / "clinical_metadata"
        / "categorical_effect_sizes.csv"
    )

    if categorical_path.exists():
        df = load_csv(categorical_path)

        for _, r in df.iterrows():
            rows.append({
                "category": "Clinical metadata",
                "analysis": str(r.get("variable", "")),
                "group": "problematic vs clean",
                "n": 22,
                "C1_value": r.get("cramers_v", None),
                "F1_value": None,
                "change": None,
                "p_value": r.get("fisher_p_value", None),
                "interpretation": "Exploratory categorical effect size",
            })

    # ------------------------------------------------------------------
    # 5. Hurt × size × geometry
    # ------------------------------------------------------------------
    hurt_path = (
        ROOT / "hurt_size_geometry"
        / "hurt_size_geometry_lesions.csv"
    )

    if hurt_path.exists():
        df = load_csv(hurt_path)

        hurt_col = find_col(df, "hurt", "hurt_status")
        group_col = find_col(df, "group", "error_status")
        margin_col = find_col(
            df,
            "bcc_minus_scc_distance",
            "scc_advantage_over_bcc",
        )

        if hurt_col and group_col:
            for hurt_value in (True, False, "True", "False"):
                g = df[df[hurt_col].astype(str) == str(hurt_value)]

                if len(g) == 0:
                    continue

                problematic = (
                    g[group_col].astype(str).str.lower()
                    .eq("problematic")
                    .sum()
                )

                rows.append({
                    "category": "Hurt × geometry",
                    "analysis": "Hurt status",
                    "group": str(hurt_value),
                    "n": len(g),
                    "C1_value": (
                        problematic / len(g)
                        if len(g)
                        else None
                    ),
                    "F1_value": None,
                    "change": None,
                    "p_value": None,
                    "interpretation": (
                        "Problematic fraction within hurt subgroup"
                    ),
                })

    # ------------------------------------------------------------------
    # Save machine-readable table
    # ------------------------------------------------------------------
    evidence = pd.DataFrame(rows)

    evidence_path = OUT / "consolidated_evidence.csv"
    evidence.to_csv(evidence_path, index=False)

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    report_path = OUT / "consolidated_evidence_summary.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("DERMASENSE SCC CONSOLIDATED EVIDENCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET / SAMPLE\n")
        f.write("-" * 80 + "\n")
        f.write("Matched SCC lesions: 22\n")
        f.write("Problematic SCC lesions: 11\n")
        f.write("Clean SCC lesions: 11\n\n")

        f.write("PRIMARY FINDINGS\n")
        f.write("-" * 80 + "\n")

        f.write(
            "1. C1 lesion-level geometry:\n"
            "   Problematic SCC lesions are substantially more BCC-side\n"
            "   than clean SCC lesions.\n"
            "   Raw geometry Mann-Whitney p = 0.025575.\n\n"
        )

        f.write(
            "2. Centroid-side analysis:\n"
            "   C1 problematic SCC: 5/11 closer to BCC centroid.\n"
            "   C1 clean SCC:       0/11 closer to BCC centroid.\n"
            "   Therefore 6/11 problematic lesions remain SCC-side,\n"
            "   while all 11 clean lesions are SCC-side.\n\n"
        )

        f.write(
            "3. Clinical size confound:\n"
            "   Problematic SCC lesions are larger.\n"
            "   diameter_1: problematic mean 16.27 vs clean 9.55,\n"
            "   p = 0.012214.\n"
            "   diameter_2: problematic mean 13.55 vs clean 8.82,\n"
            "   p = 0.024337.\n"
            "   lesion-area proxy: problematic mean 241.09 vs clean 115.09,\n"
            "   p = 0.016418.\n\n"
        )

        f.write(
            "4. Size ↔ feature-margin relationship:\n"
            "   Associations are weak and non-significant.\n"
            "   C1 diameter_1 Spearman rho = 0.1372, p = 0.5425.\n"
            "   C1 diameter_2 Spearman rho = 0.1526, p = 0.4977.\n"
            "   C1 area proxy Spearman rho = 0.1278, p = 0.5710.\n"
            "   Thus lesion size is strongly associated with error status,\n"
            "   but size alone does not explain the feature margin.\n\n"
        )

        f.write(
            "5. Hurt confound:\n"
            "   5/5 hurt lesions are problematic and all are large.\n"
            "   However, among the 17 non-hurt lesions, 6/17 remain\n"
            "   problematic and the C1 margin difference remains significant:\n"
            "   p = 0.014544.\n"
            "   Therefore hurt cannot fully explain the geometry finding.\n\n"
        )

        f.write(
            "6. F1 / SupCon displacement:\n"
            "   Problematic mean margin changed from -0.3392 to -0.4614.\n"
            "   Clean mean margin changed from -1.5318 to -1.3398.\n"
            "   Overall paired mean Δ margin = +0.0349.\n"
            "   Wilcoxon p = 0.222107.\n"
            "   F1 therefore changed the geometry but did not produce a\n"
            "   statistically reliable selective correction of problematic SCCs.\n\n"
        )

        f.write(
            "INTERPRETATION\n"
            "-" * 80 + "\n"
        )

        f.write(
            "The evidence is consistent with a lesion-level representation\n"
            "shift in problematic SCC cases toward the BCC side of the learned\n"
            "feature space. The effect persists after examining the non-hurt\n"
            "subgroup and is not explained by a simple linear relationship\n"
            "between lesion size and feature margin.\n\n"
        )

        f.write(
            "However, the sample is small (22 lesions), the problematic/clean\n"
            "groups are not independent clinical cohorts, and multiple\n"
            "exploratory tests were performed. The results should therefore\n"
            "be treated as diagnostic evidence rather than confirmatory proof.\n\n"
        )

        f.write(
            "F1 did not clearly solve the problem: some problematic lesions\n"
            "moved toward the SCC side, but the paired lesion-level analysis\n"
            "does not show a reliable overall selective improvement.\n"
        )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Evidence CSV:    {evidence_path}")
    print(f"Evidence report: {report_path}")
    print()
    print("CONSOLIDATED EVIDENCE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
