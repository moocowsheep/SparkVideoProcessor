#!/bin/bash
# Point BD2's monitor receiver at our raw 2160p29.97 SR output (IS-05 PATCH).
# Usage: bd2_route.sh [sdp-file]   (default: tx_2160p2997_raw.sdp next to this script)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SDP="${1:-$DIR/tx_2160p2997_raw.sdp}"
RX="http://192.168.18.195:8090/x-nmos/connection/v1.0/single/receivers/3fd6ea3d-295a-588f-b046-61b3fdc4a77a/staged"
python3 - "$SDP" <<'EOF' > /tmp/bd2_patch.json
import json, sys
sdp = open(sys.argv[1]).read()
print(json.dumps({
  "master_enable": True,
  "activation": {"mode": "activate_immediate"},
  "transport_file": {"data": sdp, "type": "application/sdp"},
}))
EOF
curl -s -X PATCH -H 'Content-Type: application/json' --data @/tmp/bd2_patch.json "$RX" | head -c 400
echo
