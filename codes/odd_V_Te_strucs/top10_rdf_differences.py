#!/usr/bin/env python3
"""Report RDF bins with the largest deviation from the background mean.

The background mean is calculated from every structure except 16 and 28.
For each of structures 16 and 28, the script reports the bins with the
largest absolute difference from that background mean.

Usage:
    python top10_rdf_differences.py V_Bi_struc_comp.csv

If no CSV path is provided, the script expects V_Bi_struc_comp.csv in the
current directory.
"""

import csv
import sys
from pathlib import Path


TARGET_STRUCTURES = (22,33,87)
TOP_N = 10


def read_rdf_data(csv_path: Path):
    """Read RDF columns and RDF values indexed by structure number."""
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rdf_columns = [name for name in reader.fieldnames if name.startswith("v_")]

        if not rdf_columns:
            raise ValueError("No RDF columns beginning with 'v_' were found.")
        if "structures" not in reader.fieldnames:
            raise ValueError("The CSV must contain a 'structures' column.")

        rdf_by_structure = {}
        for row in reader:
            structure_id = int(row["structures"])
            if structure_id in rdf_by_structure:
                raise ValueError(f"Structure {structure_id} appears more than once.")
            rdf_by_structure[structure_id] = [float(row[column]) for column in rdf_columns]

    return rdf_columns, rdf_by_structure


def mean_rdf(rdf_vectors):
    """Calculate a pointwise mean RDF without requiring external packages."""
    n_vectors = len(rdf_vectors)
    if n_vectors == 0:
        raise ValueError("No background RDFs are available.")

    return [sum(bin_values) / n_vectors for bin_values in zip(*rdf_vectors)]


def top_difference_bins(target_rdf, background_rdf, rdf_columns, top_n):
    """Return radial bins ordered by decreasing absolute RDF difference."""
    results = []
    for column, target_value, background_value in zip(
        rdf_columns, target_rdf, background_rdf
    ):
        radial_distance = float(column.removeprefix("v_"))
        signed_difference = target_value - background_value
        results.append(
            {
                "r_angstrom": radial_distance,
                "target_rdf": target_value,
                "background_mean_rdf": background_value,
                "signed_difference": signed_difference,
                "absolute_difference": abs(signed_difference),
            }
        )

    return sorted(results, key=lambda item: item["absolute_difference"], reverse=True)[:top_n]


def print_results(structure_id, results):
    print(f"\nTop {len(results)} RDF-bin deviations: Structure {structure_id} vs background mean")
    print(
        f"{'r (Å)':>8}  {'RDF structure':>15}  {'RDF background mean':>20}  "
        f"{'signed difference':>18}  {'absolute difference':>21}"
    )
    for result in results:
        print(
            f"{result['r_angstrom']:8.2f}  "
            f"{result['target_rdf']:15.8f}  "
            f"{result['background_mean_rdf']:20.8f}  "
            f"{result['signed_difference']:18.8f}  "
            f"{result['absolute_difference']:21.8f}"
        )


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("V_Te1_struc_comp.csv")
    rdf_columns, rdf_by_structure = read_rdf_data(csv_path)

    missing_targets = set(TARGET_STRUCTURES) - rdf_by_structure.keys()
    if missing_targets:
        raise ValueError(f"Missing target structure(s): {sorted(missing_targets)}")

    background_vectors = [
        rdf
        for structure_id, rdf in rdf_by_structure.items()
        if structure_id not in TARGET_STRUCTURES
    ]
    background_mean = mean_rdf(background_vectors)

    print(f"Background mean: {len(background_vectors)} structures (excluding 16 and 28)")
    for structure_id in TARGET_STRUCTURES:
        results = top_difference_bins(
            rdf_by_structure[structure_id], background_mean, rdf_columns, TOP_N
        )
        print_results(structure_id, results)


if __name__ == "__main__":
    main()
