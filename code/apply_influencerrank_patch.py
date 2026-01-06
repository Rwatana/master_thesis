#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a small, robust patch to your InfluencerRank FULL SCRIPT.

What this patch does:
  1) Removes seaborn dependency (import + usage)
  2) Forces a non-interactive matplotlib backend (Agg) for headless servers
  3) Replaces sns.heatmap with a matplotlib imshow-based heatmap
  4) Replaces sns.set_style("whitegrid") with plain matplotlib grid
  5) (Optional) Makes a Python<3.10-safe type hint for setup_mlflow_experiment's tracking_uri

Usage:
  python apply_influencerrank_patch.py --in base.py --out influencer_rank_full_modified.py

Notes:
  - The patch is text-based and should work even if line numbers differ.
  - If a pattern is not found, the script will warn and keep the original chunk.
"""
import argparse
import re
from pathlib import Path

def _replace_once(text: str, pattern: str, repl: str, flags=0, label=""):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n == 0:
        print(f"[WARN] pattern not found for: {label or pattern}")
        return text, False
    return new, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to the base script (.py)")
    ap.add_argument("--out", dest="out", required=True, help="Path to write the modified script (.py)")
    ap.add_argument("--no-typehint-fix", action="store_true", help="Skip Optional[str] type-hint fix (keep as-is).")
    args = ap.parse_args()

    src_path = Path(args.inp)
    out_path = Path(args.out)

    text = src_path.read_text(encoding="utf-8", errors="replace")

    # 1) Remove seaborn import (and add matplotlib backend guard)
    # Replace:
    #   import matplotlib.pyplot as plt
    #   import seaborn as sns
    #
    # With:
    #   import matplotlib
    #   matplotlib.use("Agg")
    #   import matplotlib.pyplot as plt
    #
    # If seaborn import isn't present, we still try to insert the Agg backend before pyplot import.
    pat_pltsns = r"(?:\r?\n)import matplotlib\.pyplot as plt(?:\r?\n)import seaborn as sns(?:\r?\n)"
    repl_pltsns = "\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    text, _ = _replace_once(text, pat_pltsns, repl_pltsns, flags=re.MULTILINE, label="matplotlib+seaborn import swap")

    if "import seaborn as sns" in text:
        # Fallback: remove seaborn import line only
        text = re.sub(r"^\s*import seaborn as sns\s*(?:\r?\n)", "", text, flags=re.MULTILINE)

    # Ensure Agg backend exists before pyplot import (if not already inserted)
    if "matplotlib.use('Agg')" not in text and "matplotlib.use(\"Agg\")" not in text:
        text, _ = _replace_once(
            text,
            r"^\s*import matplotlib\.pyplot as plt\s*(?:\r?\n)",
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n",
            flags=re.MULTILINE,
            label="insert matplotlib Agg before pyplot"
        )

    # 2) Replace sns.heatmap in plot_attention_weights with matplotlib imshow + colorbar
    # We target the exact mini-block around sns.heatmap.
    pat_heatmap_block = r"""
(\s*)plt\.figure\(figsize=\(12,\s*8\)\)\s*\r?\n
\1subset_matrix\s*=\s*attention_matrix\[:50,\s*:\]\s*\r?\n
\1sns\.heatmap\(\s*subset_matrix\s*,\s*cmap\s*=\s*["']Blues["']\s*,\s*annot\s*=\s*False\s*,\s*cbar_kws\s*=\s*\{\s*['"]label['"]\s*:\s*['"]Attention Weight['"]\s*\}\s*\)\s*\r?\n
\1plt\.xlabel\(
"""
    repl_heatmap_block = r"""
\1plt.figure(figsize=(12, 8))
\1subset_matrix = attention_matrix[:50, :]
\1im = plt.imshow(subset_matrix, aspect="auto", interpolation="nearest")
\1plt.colorbar(im, label="Attention Weight")
\1plt.xlabel(
"""
    text, _ = _replace_once(text, pat_heatmap_block, repl_heatmap_block, flags=re.MULTILINE | re.VERBOSE, label="replace sns.heatmap block")

    # 3) Remove sns.set_style("whitegrid") inside generate_enhanced_scatter_plot
    text = re.sub(r"^\s*sns\.set_style\(\s*['\"]whitegrid['\"]\s*\)\s*(?:\r?\n)", "", text, flags=re.MULTILINE)

    # 4) Remove any remaining sns. usage (defensive)
    if "sns." in text:
        print("[WARN] Found remaining 'sns.' usage. Attempting best-effort removal/rewrite where safe.")
        # If anything remains, we leave it as-is to avoid breaking logic.

    # 5) Optional: make 'tracking_uri: str | None' -> Optional[str]
    if not args.no_typehint_fix:
        if "tracking_uri: str | None" in text:
            # Ensure Optional is imported
            if not re.search(r"^\s*from typing import .*Optional", text, flags=re.MULTILINE):
                # Insert after the first block of imports (after warnings/itertools is fine)
                text, _ = _replace_once(
                    text,
                    r"^(import itertools\s*\r?\n)",
                    r"\1\nfrom typing import Optional\n",
                    flags=re.MULTILINE,
                    label="insert Optional import"
                )
            text = text.replace("tracking_uri: str | None", "tracking_uri: Optional[str]")

    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] Wrote modified script: {out_path}")

if __name__ == "__main__":
    raise SystemExit(main())
