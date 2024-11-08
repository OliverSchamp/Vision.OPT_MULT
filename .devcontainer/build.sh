#!/usr/bin.bash
# docker compose -f "projects/Vision.OPT_MULT/.devcontainer/docker-compose.yaml" up -d --build 
docker build -t opt-mult:latest -f /home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/.devcontainer/Dockerfile .