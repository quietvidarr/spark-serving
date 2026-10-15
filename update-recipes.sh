#!/bin/bash
# Update the upstream recipes without losing our local patches.
#
# Each recipe repo keeps our edits as commit(s) on the branch `local-patches`.
# This fetches upstream's latest main and replays our commits on top (git rebase).
# If upstream changed the same lines, the rebase is aborted and that repo is left
# exactly as it was — resolve by hand.
#
# A recipe with its own local/check-update.sh + local/update.sh (both TensorFold recipes) is
# checked for compatibility with our patches first and updated only through its own script
# (see its LOCAL_CHANGES.md).
#
# usage: update-recipes.sh           # update all
#        update-recipes.sh --check   # only show what's new upstream
set -u
R=~/recipes
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

for repo in qwen38-flash-next-dgx-spark qwen36-35b-a3b-exl3-tensorfold-dgx-spark; do
  d=$R/$repo; [ -d "$d/.git" ] || continue
  echo "=================== $repo"
  if [ -x "$d/local/update.sh" ]; then       # checks compatibility itself before touching anything
    if [ $CHECK = 1 ]; then "$d/local/check-update.sh" | sed 's/^/  /'; else "$d/local/update.sh" | sed 's/^/  /'; fi
    continue
  fi
  git -C "$d" fetch -q origin || { echo "  fetch failed (network?) — skipped"; continue; }
  up=$(git -C "$d" rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main)
  base=$(git -C "$d" merge-base HEAD "$up")
  new=$(git -C "$d" rev-list --count "$base..$up")
  if [ "$new" = 0 ]; then echo "  up to date"; continue; fi
  echo "  $new new upstream commit(s):"
  git -C "$d" log --oneline "$base..$up" | sed 's/^/    /'
  git -C "$d" diff --stat "$base" "$up" | tail -1 | sed 's/^/   /'
  [ $CHECK = 1 ] && continue

  if git -C "$d" rebase --autostash "$up" >/dev/null 2>&1; then
    echo "  updated; local patches re-applied:"
    git -C "$d" log --oneline "$up..HEAD" | sed 's/^/    /'
  else
    git -C "$d" rebase --abort 2>/dev/null
    echo "  !! CONFLICT: upstream changed the same lines as our patches."
    echo "     Left unchanged. Files involved:"
    git -C "$d" diff --name-only "$base" "$up" | grep -Fx -f <(git -C "$d" diff --name-only "$up" HEAD 2>/dev/null; git -C "$d" diff --name-only "$base" HEAD) | sort -u | sed 's/^/       /'
    continue
  fi

  # .env.sample changes may introduce new knobs our .env doesn't have
  if git -C "$d" diff --quiet "$base" HEAD -- .env.sample 2>/dev/null; then :; else
    echo "  note: .env.sample changed upstream — compare with your .env:"
    git -C "$d" diff "$base" HEAD -- .env.sample | grep '^[+-][^+-]' | head -20 | sed 's/^/    /'
  fi
  if [ "$repo" = dashboard ]; then
    echo "  rebuilding dashboard..."
    (cd "$d" && docker compose up --build -d 2>&1 | tail -1)
  fi
done
echo
echo "Models pick up recipe changes the next time the router starts them."
echo "If a recipe now uses a new Docker image, that start will first download it (slow on Wi-Fi)."
echo "(TensorFold: new patches = new image; fetch it ahead with the Model updates panel or POST /router/update.)"
