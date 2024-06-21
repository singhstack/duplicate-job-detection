#!/bin/bash

# Stop and remove all Docker containers and networks defined in the Compose file
docker-compose down

# Remove Docker volumes
docker volume prune -f

# Remove Docker networks
docker network prune -f

# Remove any stopped containers
docker container prune -f

# Remove dangling images
docker image prune -f

# Remove specific volumes and configuration files
rm -rf ./volumes
# sudo rm -rf ./embedEtcd.yaml

echo "Cleanup completed successfully."
