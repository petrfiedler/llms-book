#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_FILE="${1:-llms-book.pdf}"
BUILD_DIR=".build"
MERGED_MD="$BUILD_DIR/llms-book-merged.md"
TITLE_TEX="$BUILD_DIR/title-page.tex"
VERBATIM_TEX="$BUILD_DIR/verbatim-wrap.tex"
H1_FILTER="$BUILD_DIR/h1-pagebreak.lua"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON" ]]; then
  echo "Error: python3 is not available." >&2
  exit 1
fi

for i in $(seq 1 13); do
  if [[ ! -f "$i.md" ]]; then
    echo "Error: missing chapter file $i.md" >&2
    exit 1
  fi
done

mkdir -p "$BUILD_DIR"

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

if ! "$PYTHON" -c "import pypandoc" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --upgrade pypandoc-binary
fi

cat > "$TITLE_TEX" <<'EOF'
\begin{titlepage}
\centering
\vspace*{\fill}
{\Huge\bfseries LLM pod kapotou: Co\\[0.35em]vás ve škole nenaučili\par}
\vspace*{\fill}
\end{titlepage}
\clearpage
EOF

cat > "$VERBATIM_TEX" <<'EOF'
\usepackage{xurl}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,breaknonspaceingroup,commandchars=\\\{\}}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines,breakanywhere,breaknonspaceingroup}
EOF

cat > "$H1_FILTER" <<'EOF'
function Header(el)
  if el.level == 1 then
    return {pandoc.RawBlock('latex', '\\clearpage'), el}
  end
end
EOF

export OUTPUT_FILE MERGED_MD TITLE_TEX
"$PYTHON" - <<'PY'
from pathlib import Path
import os
import pypandoc

chapters = [Path(f"{i}.md") for i in range(1, 14)]
output = Path(os.environ["OUTPUT_FILE"])
merged = Path(os.environ["MERGED_MD"])

header = "---\nlang: cs\ntoc-title: Obsah\n---\n\n"
parts = [chapter.read_text(encoding="utf-8") for chapter in chapters]
merged.write_text(header + "\n\n".join(parts), encoding="utf-8")

pypandoc.convert_file(
    str(merged),
    "pdf",
    format="markdown+lists_without_preceding_blankline+raw_tex",
    outputfile=str(output),
    extra_args=[
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=2",
        f"--include-before-body={os.environ['TITLE_TEX']}",
        f"--include-in-header={Path('.build/verbatim-wrap.tex').resolve()}",
        f"--lua-filter={Path('.build/h1-pagebreak.lua').resolve()}",
    ],
)

print(f"OK: generated {output}")
PY
