#!/bin/bash

# SOS serve script with 3 screen sessions
echo "Starting 3 SOS serve sessions..."

# Set up SOS path (from the export_sos_serve.sh)
export PATH="$PWD/target/release:$PATH"

# Start SOS serve 1 on port 3000
screen -dmS sos1 bash -c "
export PATH='$PWD/target/release:\$PATH'
sos serve --port 3000
"

# Start SOS serve 2 on port 3001
screen -dmS sos2 bash -c "
export PATH='$PWD/target/release:\$PATH'
sos serve --port 3001
"

# Start SOS serve 3 on port 3002
screen -dmS sos3 bash -c "
export PATH='$PWD/target/release:\$PATH'
sos serve --port 3002
"

echo "Started 3 SOS serve sessions:"
echo "- sos1 (port 3000): screen -r sos1"
echo "- sos2 (port 3001): screen -r sos2"
echo "- sos3 (port 3002): screen -r sos3"
echo ""
echo "To stop all SOS servers, run: ./stop_sos_serve.sh"

screen -ls