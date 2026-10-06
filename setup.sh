#!/usr/bin/env bash
set -euo pipefail
RGI_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$RGI_ROOT"
for command_name in python3 xelatex latexmk; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command_name" >&2
    exit 1
  fi
done
python3 -m json.tool evidence/claims.json >/dev/null
python3 -m json.tool evidence/numerical-theory-summary.json >/dev/null
echo "Prerequisites and JSON checks passed."
echo "Verify: python3 verify_evidence.py"
echo "Build: latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error paper/main.tex"
