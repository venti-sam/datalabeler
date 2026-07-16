#!/bin/bash
# Stop and remove a pipeline dev container started by start.sh.
#
#   ./stop.sh            # pick a service from a menu
#   ./stop.sh extract    # stop+remove datalabeler-extract
#   ./stop.sh sam3       # stop+remove datalabeler-sam3
set -e

cd "$(dirname "$0")"
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: 'docker-compose.yml' not found next to stop.sh."
    exit 1
fi

echo "=================================="
echo "      Stop Dev Container"
echo "=================================="

mapfile -t SERVICES < <(docker compose config --services)

SERVICE="$1"
if [ -z "$SERVICE" ]; then
    for i in "${!SERVICES[@]}"; do echo "$i) ${SERVICES[$i]}"; done
    echo ""
    read -p "Service name or number to stop: " SEL
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

if [ -z "$(docker ps -aq -f name=^/${CONTAINER}$)" ]; then
    echo "Container '$CONTAINER' does not exist. Nothing to stop."
    exit 0
fi

echo "Stopping and removing '$CONTAINER'..."
docker rm -f "$CONTAINER" >/dev/null
echo "Done."
