#!/bin/bash
# Runs stop.sh on every camera listed in .env (any branch: raspbian, jetson, linux). See all-cameras.sh.
exec "$(dirname "$0")/all-cameras.sh" stop
