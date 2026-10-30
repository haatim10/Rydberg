#!/usr/bin/env bash
# Delete the pre-rename branch names from GitHub (renamed on 30 October 2026).
#
# Each new branch is the old branch plus one README/link commit. Before deleting,
# this script checks on GitHub that the new branch's parent is exactly the old
# branch's current tip, so no commit can be lost. Anything that fails the check
# is skipped and reported.
#
# Usage:  tools/delete_old_branches.sh            # dry run: show what would happen
#         tools/delete_old_branches.sh --delete   # actually delete
set -euo pipefail

REMOTE=${REMOTE:-origin}
DELETE=${1:-}

# old-name new-name
PAIRS=(
  "codex archive/complete"
  "research/self-supervised-20261030 research/01-foundations"
  "research/tcom-receiver-20261030 research/02-receiver"
  "research/urformer-comparison-20261030 research/03-urformer-comparison"
  "research/urformer-audits-20261030 research/04-urformer-audits"
  "paper/merged-tcom-20261030 paper/tcom-merged"
  "track-a-cui-reproduction track-a/cui-reproduction"
  "cursor/ula-config-channel-8c77 track-a/early-modules"
  "paper1-restructure paper/paper1-restructure"
  "paper2-style paper/paper2-style-rewrite"
  "claude/track-b-exact history/track-b-exact"
  "track-b-ula-channel-estimation history/track-b-ula"
  "report-corrections history/report-corrections"
  "claude/adversarial-audit-gs6iid history/adversarial-audit"
  "paper1-review history/paper1-review"
  "claude/new-paper-implementation-fs4ako history/paper1-review"
  "cursor/fig5-norm-diag-8c77 history/fig5-norm-diag"
)
# Test branch left by the reorganisation session; it equals the old main.
STRAY=(tmp/delete-probe)

git fetch --quiet "$REMOTE" '+refs/heads/*:refs/remotes/'"$REMOTE"'/*'

to_delete=()
for pair in "${PAIRS[@]}"; do
  read -r old new <<<"$pair"
  old_sha=$(git rev-parse --verify --quiet "$REMOTE/$old" || true)
  new_sha=$(git rev-parse --verify --quiet "$REMOTE/$new" || true)
  if [[ -z $old_sha ]]; then echo "gone    $old"; continue; fi
  if [[ -z $new_sha ]]; then echo "SKIP    $old: $new is missing"; continue; fi
  parent=$(git rev-parse "$new_sha^")
  if [[ $parent != "$old_sha" ]]; then
    echo "SKIP    $old: moved since the rename ($old_sha is not the parent of $new)"
    continue
  fi
  echo "ok      $old -> $new"
  to_delete+=("$old")
done
for b in "${STRAY[@]}"; do
  if git rev-parse --verify --quiet "$REMOTE/$b" >/dev/null; then
    echo "ok      $b (test branch)"
    to_delete+=("$b")
  fi
done

if [[ ${#to_delete[@]} -eq 0 ]]; then echo "Nothing to delete."; exit 0; fi
if [[ $DELETE != --delete ]]; then
  echo; echo "Dry run. Re-run with --delete to remove ${#to_delete[@]} branches."
  exit 0
fi
git push "$REMOTE" --delete "${to_delete[@]}"
