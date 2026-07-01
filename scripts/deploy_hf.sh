#!/usr/bin/env bash
# Deploy a clean single-commit snapshot to the Hugging Face Space.
#
# Why a snapshot (not `git push space main`): HF rejects binary files in history
# (the SHL assignment PDF), and we should not republish SHL's assignment on a public
# Space. This pushes only the current tree, minus the assignment docs, as one commit.
#
# Prereqs (one-time):
#   hf auth login                 # token stored at ~/.cache/huggingface/token
#   git remote add space https://huggingface.co/spaces/<user>/<space>
#   git config credential.helper '!f() { echo "username=<user>"; echo "password=$(cat $HOME/.cache/huggingface/token)"; }; f'
set -euo pipefail

BRANCH="$(git branch --show-current)"
cleanup() { git checkout -f "$BRANCH" >/dev/null 2>&1 || true; git branch -D hf-deploy >/dev/null 2>&1 || true; }
trap cleanup EXIT

git checkout -q --orphan hf-deploy
git add -A
# Exclude SHL's assignment docs from the public Space.
git rm --cached -q docs/SHL_AI_Intern_Assignment.pdf docs/SHL_AI_Intern_Assignment.md 2>/dev/null || true
git commit -q -m "SHL Assessment Recommender — deploy snapshot"
git push space hf-deploy:main --force
echo "Pushed. The Space will rebuild; check /health once it is RUNNING."
