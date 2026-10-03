#!/usr/bin/env bash
# Runs the dataset survey as a loop of fresh, non-interactive devin sessions.
# Each iteration does exactly one unchecked task in TASKS.md, then the script commits.
# Stops when TASKS.md has no unchecked items, or after MAX_ITERS.
#
# Usage: scripts/research_loop.sh [MAX_ITERS]
# Env:   DEVIN_BIN (default: Devin.app bundled CLI), DEVIN_PERMISSION_MODE (default: dangerous)
set -euo pipefail

cd "$(dirname "$0")/.."

DIR=research/dataset-survey
MAX_ITERS="${1:-12}"
DEVIN_BIN="${DEVIN_BIN:-/Applications/Devin.app/Contents/Resources/app/extensions/windsurf/devin/bin/devin}"
MODE="${DEVIN_PERMISSION_MODE:-dangerous}"

read -r -d '' PROMPT <<EOF || true
You are one iteration of an automated research loop. Previous iterations have no memory; the files are the memory.

1. Read $DIR/BRIEF.md (the research brief), $DIR/TASKS.md (checklist), and $DIR/FINDINGS.md (what has already been found).
2. Take ONLY the first unchecked task in TASKS.md. Do not start any other task.
3. Do thorough web research for that task using web search and page fetches. Open the actual dataset/repo pages; do not report from memory. Be skeptical: verify license, size, and label format from the page itself, and mark anything you could not verify as "unverified".
4. Append your results to $DIR/FINDINGS.md using the template at the top of that file. Do not modify earlier entries. Do not add entries that duplicate an existing one (same URL).
5. For task 9 only: write $DIR/REPORT.md as specified in the task.
6. Change the task's "- [ ]" to "- [x]" in TASKS.md.
7. Do NOT run git commands; the loop script commits for you. Do not edit files outside $DIR.
8. Finish with a 2-3 line summary of what you added.
EOF

for i in $(seq 1 "$MAX_ITERS"); do
  if ! grep -q '^- \[ \]' "$DIR/TASKS.md"; then
    echo "All tasks complete."
    break
  fi
  task=$(grep -m1 '^- \[ \]' "$DIR/TASKS.md" | sed 's/^- \[ \] //' | cut -c1-80)
  echo "=== Iteration $i: $task"

  # Clean env: when launched from inside Devin Desktop's terminal, inherited IDE vars
  # make the CLI ignore ~/.local/share/devin/credentials.toml and fail with "Login canceled".
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" TERM="${TERM:-xterm}" LANG="${LANG:-en_US.UTF-8}" \
    "$DEVIN_BIN" --permission-mode "$MODE" --respect-workspace-trust false -p "$PROMPT" || {
    echo "devin exited non-zero on iteration $i; stopping." >&2
    exit 1
  }

  if git diff --quiet -- "$DIR"; then
    echo "Iteration $i made no changes; stopping to avoid a stuck loop." >&2
    exit 1
  fi
  git add "$DIR"
  git commit -q -m "dataset survey: $task"
  echo "=== Committed iteration $i"
done

grep -q '^- \[ \]' "$DIR/TASKS.md" && echo "Stopped with tasks remaining (MAX_ITERS=$MAX_ITERS)." || echo "Done. See $DIR/REPORT.md"
