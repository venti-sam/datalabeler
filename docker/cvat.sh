#!/bin/bash
# Manage a local CVAT server for Stage 3 (human correction of SAM 3 masks).
#
# CVAT is a big, separate multi-container app (server, UI, db, redis, workers).
# We do NOT vendor it -- this script clones the official cvat-ai/cvat repo and
# drives its own docker compose. Our pipeline talks to it over HTTP; the only
# coupling is the URL in config (`cvat.host`) and the login in docker/.env
# (`CVAT_USER`/`CVAT_PASSWORD`).
#
#   ./cvat.sh up          # clone if needed, start CVAT, wait for the UI
#   ./cvat.sh superuser   # create the admin login (uses CVAT_USER/PASSWORD if set)
#   ./cvat.sh status      # show CVAT containers
#   ./cvat.sh logs        # tail the CVAT server log (Ctrl-C to stop)
#   ./cvat.sh down        # stop CVAT (its data is kept in docker volumes)
#
# Where CVAT is checked out: $CVAT_DIR (set in docker/.env), default a sibling
# `cvat/` next to this repo. Pin a release with CVAT_REF (default: latest tag).
set -e
cd "$(dirname "$0")"

# Pull CVAT_DIR / CVAT_REF / CVAT_USER / CVAT_PASSWORD from .env if present.
[ -f .env ] && { set -a; . ./.env; set +a; }

REPO_ROOT="$(cd .. && pwd)"
CVAT_REPO="https://github.com/cvat-ai/cvat.git"
CVAT_DIR="${CVAT_DIR:-$(dirname "$REPO_ROOT")/cvat}"
CVAT_URL="http://localhost:8080"

cvat_compose() { ( cd "$CVAT_DIR" && docker compose "$@" ); }

server_container() {
    docker ps --format '{{.Names}}' | grep -m1 -E 'cvat.*server' || true
}

clone_if_needed() {
    if [ -d "$CVAT_DIR/.git" ]; then
        return
    fi
    echo "Cloning CVAT into $CVAT_DIR ..."
    git clone "$CVAT_REPO" "$CVAT_DIR"
    # Pin to a release tag by default -- the default branch (develop) can be
    # mid-change. Override with CVAT_REF (a tag, branch, or commit).
    local ref="${CVAT_REF:-$(cd "$CVAT_DIR" && git tag | sort -V | tail -1)}"
    if [ -n "$ref" ]; then
        echo "Checking out CVAT $ref"
        ( cd "$CVAT_DIR" && git checkout -q "$ref" )
    fi
}

case "${1:-}" in
  up)
    clone_if_needed
    echo "Starting CVAT (first run pulls several GB of images)..."
    cvat_compose up -d
    echo -n "Waiting for the UI at $CVAT_URL "
    for _ in $(seq 1 60); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "$CVAT_URL" || true)
        if [ "$code" = "200" ] || [ "$code" = "302" ]; then
            echo " ready."
            echo "Open $CVAT_URL in a browser. Create a login with: ./cvat.sh superuser"
            exit 0
        fi
        echo -n "."
        sleep 3
    done
    echo
    echo "UI did not answer yet; check './cvat.sh status' and './cvat.sh logs'."
    ;;

  superuser)
    SERVER="$(server_container)"
    [ -z "$SERVER" ] && { echo "CVAT server not running. Run ./cvat.sh up first."; exit 1; }
    if [ -n "$CVAT_USER" ] && [ -n "$CVAT_PASSWORD" ]; then
        echo "Creating superuser '$CVAT_USER' (from docker/.env) ..."
        # ~ is kept literal by the host (inside double quotes) and expands to the
        # django user's home *inside* the container, where manage.py lives.
        docker exec -e DJANGO_SUPERUSER_PASSWORD="$CVAT_PASSWORD" "$SERVER" \
            bash -c "python3 ~/manage.py createsuperuser --noinput \
                --username '$CVAT_USER' --email '${CVAT_EMAIL:-admin@example.com}'" \
            && echo "Done. These are the credentials the pipeline uses." \
            || echo "createsuperuser failed (does '$CVAT_USER' already exist?)."
    else
        echo "CVAT_USER/CVAT_PASSWORD not set in docker/.env; creating interactively."
        docker exec -it "$SERVER" bash -ic 'python3 ~/manage.py createsuperuser'
    fi
    ;;

  status)
    [ -d "$CVAT_DIR" ] && cvat_compose ps || echo "CVAT not cloned yet ($CVAT_DIR)."
    ;;

  logs)
    SERVER="$(server_container)"
    [ -z "$SERVER" ] && { echo "CVAT server not running."; exit 1; }
    docker logs -f "$SERVER"
    ;;

  down)
    [ -d "$CVAT_DIR" ] && cvat_compose down && echo "CVAT stopped (data kept)." \
        || echo "CVAT not cloned yet ($CVAT_DIR)."
    ;;

  *)
    echo "usage: ./cvat.sh {up|superuser|status|logs|down}"
    exit 1
    ;;
esac
