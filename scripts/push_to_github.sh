#!/usr/bin/env bash
# Create the GitHub repository and push this project to it.
#
#   ./scripts/push_to_github.sh [repo-name] [public|private]
#
# Requires the GitHub CLI (`gh`) and an authenticated session:
#   gh auth login
#
# If you would rather not use `gh`, create the empty repo in the browser and
# then run the three commands printed at the bottom of this script.

set -euo pipefail

REPO_NAME="${1:-biu-bigdata-support-tickets}"
VISIBILITY="${2:-public}"
ACCOUNT="yaeldorin-glitch"

cd "$(dirname "$0")/.."

echo "==> repository : ${ACCOUNT}/${REPO_NAME} (${VISIBILITY})"

# --- sanity checks ---------------------------------------------------------
if [ -f .env ]; then
  echo "!!  .env exists locally. It is gitignored, but double-check it is not staged."
fi

if git status --porcelain | grep -qE '^\?\? data/raw/'; then
  echo "!!  data/raw/ is untracked and gitignored - the 26MB CSV will NOT be pushed. Good."
fi

# --- git init --------------------------------------------------------------
if [ ! -d .git ]; then
  git init -b main
  echo "==> initialised a new git repository"
fi

git add -A
if ! git diff --cached --quiet; then
  git commit -m "Big Data & AI course project: streaming ticket pipeline with semantic search"
  echo "==> committed"
else
  echo "==> nothing new to commit"
fi

# --- create + push ---------------------------------------------------------
if command -v gh >/dev/null 2>&1; then
  if gh repo view "${ACCOUNT}/${REPO_NAME}" >/dev/null 2>&1; then
    echo "==> repo already exists; adding it as a remote"
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/${ACCOUNT}/${REPO_NAME}.git"
    git push -u origin main
  else
    gh repo create "${ACCOUNT}/${REPO_NAME}" \
      --"${VISIBILITY}" \
      --source=. \
      --remote=origin \
      --description "End-to-end big data pipeline over customer IT support tickets: Kafka + Spark Structured Streaming + MinIO + Elasticsearch, with embedding-based semantic search and RAG." \
      --push
  fi
  echo
  echo "==> done: https://github.com/${ACCOUNT}/${REPO_NAME}"
else
  cat <<EOF

The GitHub CLI is not installed, so the repo was not created automatically.

  1. Create an empty repository at:
       https://github.com/new
     Name it: ${REPO_NAME}
     Do NOT add a README, .gitignore or licence - this project already has them.

  2. Then run:

       git remote add origin https://github.com/${ACCOUNT}/${REPO_NAME}.git
       git branch -M main
       git push -u origin main

EOF
fi
