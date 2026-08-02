import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_needle_output(output: str) -> dict:
    """
    Extract alignment statistics from EMBOSS needle output.

    Expected lines include:
        # Identity:      120/200 (60.0%)
        # Similarity:    145/200 (72.5%)
        # Gaps:           10/200 ( 5.0%)
        # Score: 456.5
    """
    patterns = {
        "identity": r"#\s*Identity:\s+(\d+)/(\d+)\s+\(\s*([\d.]+)%\)",
        "similarity": r"#\s*Similarity:\s+(\d+)/(\d+)\s+\(\s*([\d.]+)%\)",
        "gaps": r"#\s*Gaps:\s+(\d+)/(\d+)\s+\(\s*([\d.]+)%\)",
        "score": r"#\s*Score:\s+([-\d.]+)",
    }

    results = {}

    for name, pattern in patterns.items():
        match = re.search(pattern, output)

        if match is None:
            if name == "score":
                results[name] = None
                continue
            raise ValueError(
                f"Could not find {name!r} statistics in the needle output."
            )

        if name == "score":
            results[name] = float(match.group(1))
        else:
            results[name] = {
                "count": int(match.group(1)),
                "alignment_length": int(match.group(2)),
                "percent": float(match.group(3)),
            }

    return results


def run_needle(
    sequence_a: str,
    sequence_b: str,
    gap_open: float = 10.0,
    gap_extend: float = 0.5,
    matrix: str = "EBLOSUM62",
) -> tuple[dict, str]:
    """
    Run EMBOSS needle on two protein sequences.

    Parameters
    ----------
    sequence_a : str
        First protein sequence.
    sequence_b : str
        Second protein sequence.

    Returns
    -------
    statistics : dict
        Parsed identity, similarity, gaps and score.
    alignment_output : str
        Full EMBOSS needle output.
    """

    if shutil.which("needle") is None:
        raise RuntimeError(
            "The EMBOSS 'needle' executable was not found in PATH.\n"
            "Install with: conda install -c bioconda emboss"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        fasta_a = tmpdir / "seqA.fasta"
        fasta_b = tmpdir / "seqB.fasta"
        output_file = tmpdir / "alignment.needle"

        fasta_a.write_text(f">seqA\n{sequence_a.strip()}\n")
        fasta_b.write_text(f">seqB\n{sequence_b.strip()}\n")

        command = [
            "needle",
            "-asequence",
            str(fasta_a),
            "-bsequence",
            str(fasta_b),
            "-gapopen",
            str(gap_open),
            "-gapextend",
            str(gap_extend),
            "-datafile",
            matrix,
            "-outfile",
            str(output_file),
            "-auto",
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"needle failed:\n{e.stderr.strip()}") from e

        alignment_output = output_file.read_text()
        statistics = parse_needle_output(alignment_output)

    return statistics, alignment_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an EMBOSS needle global protein alignment and report "
            "identity and similarity percentages."
        )
    )

    parser.add_argument(
        "sequence_a",
        type=Path,
        help="First protein sequence in FASTA format.",
    )
    parser.add_argument(
        "sequence_b",
        type=Path,
        help="Second protein sequence in FASTA format.",
    )
    parser.add_argument(
        "--gap-open",
        type=float,
        default=10.0,
        help="Gap-opening penalty. Default: 10.0",
    )
    parser.add_argument(
        "--gap-extend",
        type=float,
        default=0.5,
        help="Gap-extension penalty. Default: 0.5",
    )
    parser.add_argument(
        "--matrix",
        default="EBLOSUM62",
        help="EMBOSS substitution matrix. Default: EBLOSUM62",
    )
    parser.add_argument(
        "--alignment-output",
        type=Path,
        help="Optional file in which to save the complete needle alignment.",
    )

    args = parser.parse_args()

    try:
        statistics, alignment_output = run_needle(
            sequence_a=args.sequence_a,
            sequence_b=args.sequence_b,
            gap_open=args.gap_open,
            gap_extend=args.gap_extend,
            matrix=args.matrix,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    identity = statistics["identity"]
    similarity = statistics["similarity"]
    gaps = statistics["gaps"]

    print(
        f"Identity:   {identity['percent']:.2f}% "
        f"({identity['count']}/{identity['alignment_length']})"
    )
    print(
        f"Similarity: {similarity['percent']:.2f}% "
        f"({similarity['count']}/{similarity['alignment_length']})"
    )
    print(
        f"Gaps:       {gaps['percent']:.2f}% "
        f"({gaps['count']}/{gaps['alignment_length']})"
    )

    if statistics["score"] is not None:
        print(f"Score:      {statistics['score']:.2f}")

    if args.alignment_output is not None:
        args.alignment_output.write_text(alignment_output)
        print(f"Alignment:  {args.alignment_output}")


if __name__ == "__main__":
    main()
