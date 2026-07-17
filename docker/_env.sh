#!/bin/bash
# Refresh docker/.env, the machine-local settings compose reads automatically
# (it sits next to docker-compose.yml, so plain `docker compose -f ... run` picks
# it up too, not just the helper scripts here). Called by build.sh and start.sh;
# safe to run by hand.
#
# It only rewrites the two lines it owns -- DL_UID/DL_GID, which pin the
# container user to you so pipeline output stays deletable without sudo.
# Everything else you keep in .env is preserved, because secrets live there:
#
#   HF_TOKEN=hf_...        gated facebook/sam3 weights (Stage 2)
#   DL_BAGS_DIR=/path/to   read bags from outside the repo (see .env.example)
#   CVAT_USER / CVAT_PASSWORD
#
# .env is gitignored and chmod 600 precisely because of that. Never commit it.
set -e
cd "$(dirname "$0")"

touch .env
chmod 600 .env

# Keep every line we don't own, then re-append ours.
grep -vE '^(DL_UID|DL_GID)=' .env > .env.tmp 2>/dev/null || true
printf 'DL_UID=%s\nDL_GID=%s\n' "$(id -u)" "$(id -g)" >> .env.tmp
mv .env.tmp .env
chmod 600 .env

# Bind sources must exist or docker creates them as root, reintroducing exactly
# the root-owned dirs the DL_UID pinning is here to avoid.
mkdir -p ../data/bags ../work checkpoints
