#!/bin/bash

# Stop all SOS serve screen sessions
echo "Stopping SOS serve sessions..."

screen -S sos1 -X quit 2>/dev/null && echo "- Stopped sos1 (port 3000)" || echo "- sos1 not running"
screen -S sos2 -X quit 2>/dev/null && echo "- Stopped sos2 (port 3001)" || echo "- sos2 not running"
screen -S sos3 -X quit 2>/dev/null && echo "- Stopped sos3 (port 3002)" || echo "- sos3 not running"

echo ""
echo "Done. Remaining screen sessions:"
screen -ls