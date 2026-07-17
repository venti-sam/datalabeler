#!/bin/bash
# Build a pipeline image (extract | sam3). With no argument, builds all services.
#   ./build.sh            # build every service
#   ./build.sh extract    # build just the extract image
#   ./build.sh sam3       # build just the sam3 image
set -e

# docker-compose.yml lives next to this script.
cd "$(dirname "$0")"
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: 'docker-compose.yml' not found next to build.sh."
    exit 1
fi

./_env.sh

echo "=================================="
echo "        Build Image(s)"
echo "=================================="

SERVICES=$(docker compose config --services)

if [ -z "$1" ]; then
    echo "Building all services: $(echo $SERVICES | tr '\n' ' ')"
    docker compose build
    echo "----------------------------------"
    echo "Done. Joinable with: ./join.sh <service>"
    exit 0
fi

# Validate the requested service against the compose file.
if ! echo "$SERVICES" | grep -qx "$1"; then
    echo "Error: service '$1' not in docker-compose.yml."
    echo "Available: $(echo $SERVICES | tr '\n' ' ')"
    exit 1
fi

echo "Building '$1'..."
docker compose build "$1"
echo "----------------------------------"
echo "Done. Join it with: ./join.sh $1"
