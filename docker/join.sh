#!/bin/bash
# Open an interactive shell inside a pipeline dev container. If the container
# isn't up yet, it is started first (via start.sh). Inside, run the pipeline by
# hand, e.g.  datalabeler extract   /   datalabeler autolabel   /   datalabeler status
#
#   ./join.sh            # pick a service from a menu
#   ./join.sh extract    # shell into the extract container (CPU: stages 1,3,4)
#   ./join.sh sam3       # shell into the sam3 container    (GPU: stage 2)
set -e

cd "$(dirname "$0")"
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: 'docker-compose.yml' not found next to join.sh."
    exit 1
fi

echo "=================================="
echo "     Join Dev Container"
echo "=================================="

mapfile -t SERVICES < <(docker compose config --services)

SERVICE="$1"
if [ -z "$SERVICE" ]; then
    for i in "${!SERVICES[@]}"; do echo "$i) ${SERVICES[$i]}"; done
    echo ""
    read -p "Service name or number to join: " SEL
    if [[ "$SEL" =~ ^[0-9]+$ ]] && [ -n "${SERVICES[$SEL]}" ]; then
        SERVICE="${SERVICES[$SEL]}"
    else
        SERVICE="$SEL"
    fi
fi

if ! printf '%s\n' "${SERVICES[@]}" | grep -qx "$SERVICE"; then
    echo "Error: service '$SERVICE' not in docker-compose.yml."
    echo "Available: ${SERVICES[*]}"
    exit 1
fi

CONTAINER="datalabeler-${SERVICE}"

# Not running yet? Start (or restart) it via start.sh first.
if [ -z "$(docker ps -q -f name=^/${CONTAINER}$)" ]; then
    echo "Container '$CONTAINER' is not running; starting it..."
    ./start.sh "$SERVICE"
fi

echo "Connecting to '$CONTAINER' ..."
docker exec -e DISPLAY="${DISPLAY}" -ti "$CONTAINER" bash
