#!/bin/bash
# Start a long-running dev container for a pipeline service so you can drop into
# a shell with join.sh and run `datalabeler <cmd>` by hand. The compose services
# are batch jobs (their entrypoint runs one command and exits), so here we
# override the entrypoint to keep the container alive.
#
#   ./start.sh            # pick a service from a menu
#   ./start.sh extract    # start the extract dev container (CPU: stages 1,3,4)
#   ./start.sh sam3       # start the sam3 dev container   (GPU: stage 2)
#
# Container is named datalabeler-<service>. Stop it with ./stop.sh.
set -e

cd "$(dirname "$0")"
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: 'docker-compose.yml' not found next to start.sh."
    exit 1
fi

echo "=================================="
echo "     Start Dev Container"
echo "=================================="

# Build the service list from the compose file.
mapfile -t SERVICES < <(docker compose config --services)

SERVICE="$1"
if [ -z "$SERVICE" ]; then
    for i in "${!SERVICES[@]}"; do echo "$i) ${SERVICES[$i]}"; done
    echo ""
    read -p "Service name or number to start: " SEL
    if [[ "$SEL" =~ ^[0-9]+$ ]] && [ -n "${SERVICES[$SEL]}" ]; then
        SERVICE="${SERVICES[$SEL]}"
    else
        SERVICE="$SEL"
    fi
fi

# Validate.
if ! printf '%s\n' "${SERVICES[@]}" | grep -qx "$SERVICE"; then
    echo "Error: service '$SERVICE' not in docker-compose.yml."
    echo "Available: ${SERVICES[*]}"
    exit 1
fi

CONTAINER="datalabeler-${SERVICE}"

# Already running? Nothing to do.
if [ "$(docker ps -q -f name=^/${CONTAINER}$)" ]; then
    echo "Container '$CONTAINER' is already running. Join it with: ./join.sh $SERVICE"
    exit 0
fi

# Exists but stopped? Restart it.
if [ "$(docker ps -aq -f name=^/${CONTAINER}$)" ]; then
    echo "Restarting existing container '$CONTAINER'..."
    docker start "$CONTAINER" >/dev/null
    echo "Started. Join it with: ./join.sh $SERVICE"
    exit 0
fi

echo "----------------------------------"
echo "Starting '$CONTAINER' (kept alive; entrypoint overridden to sleep)..."
# `run -d` honours the service's volumes, env, and GPU reservation from compose.
docker compose run -d --name "$CONTAINER" --entrypoint sleep "$SERVICE" infinity >/dev/null
echo "----------------------------------"
echo "Success. Join it with: ./join.sh $SERVICE"
