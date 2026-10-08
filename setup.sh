#!/usr/bin/env bash
set -euo pipefail
RGI_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$RGI_ROOT"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing prerequisite: python3 (3.9+)" >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ required"'
python3 -m json.tool evidence/common-claims.json >/dev/null
python3 -m json.tool evidence/common-local/stats.json >/dev/null
echo "Prerequisites and compact-evidence JSON checks passed."
echo "Verify: python3 verify_evidence.py"
if command -v xelatex >/dev/null 2>&1 && command -v latexmk >/dev/null 2>&1; then
  echo "Build: latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error paper/main.tex"
else
  echo "Optional paper rebuild requires XeLaTeX and latexmk. The compiled PDF is included."
fi
