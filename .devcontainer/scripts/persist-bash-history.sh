#!/bin/bash
set -e

sudo mkdir -p /command-history
sudo chown -R $USER:$USER /command-history
touch /command-history/.bash_history
echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/command-history/.bash_history" >> ~/.bashrc

# this script works with mounts property in devcontainer.json
# refs: https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
