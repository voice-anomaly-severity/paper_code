"""Write tables as CSV, Markdown and LaTeX. Only aggregate tables are ever written."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if value != 0 and (abs(value) < 10 ** (-digits) or abs(value) >= 1e6):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _formatted(df: pd.DataFrame, digits: dict[str, int] | None, default_digits: int) -> pd.DataFrame:
    digits = digits or {}
    out = df.copy().astype(object)
    for column in df.columns:
        d = digits.get(column, default_digits)
        out[column] = [_fmt(v, d) for v in df[column]]
    return out


def _latex_escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("–", "--"), ("Δ", r"$\Delta$"), ("ρ", r"$\rho$"), ("²", r"$^2$")]:
        text = text.replace(a, b)
    return text


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str,
                digits: dict[str, int] | None = None, default_digits: int = 3) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / f"{name}.csv", index=False)
    shown = _formatted(df, digits, default_digits)

    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    (table_dir / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join([header, rule, *body]) + "\n", encoding="utf-8")

    align = "".join("r" if shown[c].map(_is_number).all() else "l" for c in shown.columns)
    lines = [
        r"\begin{table}[ht]", r"\centering", rf"\caption{{{_latex_escape(caption)}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(_latex_escape(str(c)) for c in shown.columns) + r" \\", r"\midrule",
        *[" & ".join(_latex_escape(str(v)) for v in row) + r" \\" for row in shown.itertuples(index=False)],
        r"\bottomrule", r"\end{tabular}", rf"\label{{tab:{name}}}", r"\end{table}",
    ]
    (table_dir / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_number(text: str) -> bool:
    try:
        float(str(text).replace("–", "nan"))
        return True
    except ValueError:
        return False
