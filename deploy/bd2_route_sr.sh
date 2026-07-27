#!/bin/bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Point BD2's monitor receiver at our raw 2160p29.97 SR output (IS-05 PATCH).
# Usage: bd2_route.sh [sdp-file]   (default: tx_2160p2997_raw.sdp next to this script)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
# Site-local addressing (gitignored). Everything below defaults to the IETF documentation
# ranges, so a fresh checkout is inert until this supplies real values. See lab.env.example.
# shellcheck disable=SC1091
[ -f "$DIR/lab.env" ] && . "$DIR/lab.env"
SDP="${1:-$DIR/tx_2160p2997_raw.sdp}"
BD2_HOST="${BD2_HOST:-192.0.2.195:8090}"
BD2_RX="${BD2_RX:-00000000-0000-5000-8000-000000000001}"
TX_ADDR="${TX_ADDR:-192.0.2.110}"
GMID="${GMID:-00-00-5e-ff-fe-00-53-88}"
RX="http://$BD2_HOST/x-nmos/connection/v1.0/single/receivers/$BD2_RX/staged"

# The checked-in SDPs carry documentation addresses (RFC 5737 / RFC 7042), so substitute this site's
# real egress source and grandmaster before sending. The source-filter especially: SSM receivers join
# on (group, source) and silently drop everything when it disagrees with what the engine transmits.
python3 - "$SDP" "$TX_ADDR" "$GMID" <<'EOF' > /tmp/bd2_patch.json
import json, re, sys
sdp, tx, gmid = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
sdp = re.sub(r'192\.0\.2\.110', tx, sdp)
sdp = re.sub(r'(?<=IEEE1588-2008:)[0-9A-Fa-f-]{23}', gmid.upper(), sdp)
print(json.dumps({
  "master_enable": True,
  "activation": {"mode": "activate_immediate"},
  "transport_file": {"data": sdp, "type": "application/sdp"},
}))
EOF
# These receivers answer a zero-body HTTP 500 for a while after boot, then serve the identical
# request normally — retry rather than believe the first refusal (see docs/M7-ip10.md).
for try in 1 2 3 4; do
  code=$(curl -s -o /tmp/bd2_patch_resp.json -w '%{http_code}' -X PATCH \
              -H 'Content-Type: application/json' --data @/tmp/bd2_patch.json "$RX")
  [ "$code" = "200" ] && break
  sleep 1
done
echo "PATCH $RX -> HTTP $code"
head -c 400 /tmp/bd2_patch_resp.json; echo
