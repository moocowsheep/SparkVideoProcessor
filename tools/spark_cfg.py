#!/usr/bin/env python3
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

"""
spark_cfg.py — read/modify single fields of the running control daemon's PipelineConfig.

The daemon's POST /api/config REPLACES the whole config and is refused while the engine is
RUNNING ("cannot change config while running") — see control/daemon.cpp. So there is no
partial update: changing one field means stop -> read -> merge -> post -> start. Doing that
by hand is easy to get wrong in two ways this script guards against:

  * posting a partial object silently resets every omitted field to its proto default;
  * the daemon strict-parses first and falls back to a lenient parse, so a MISSPELLED key is
    not rejected — it comes back as "config updated (unknown fields ignored...)".

The engine's prior state is preserved: if it was STOPPED, it is left stopped.

Fields owned by the NMOS layer are re-derived on every IS-05 activation and will overwrite
what you set here (reconcile() in control/nmos/spark_node.cpp) — setting one is warned about
but not blocked, since it is still useful for a no-NMOS bring-up.

  ./tools/spark_cfg.py                             # show the whole config
  ./tools/spark_cfg.py rxPci                       # show one field
  ./tools/spark_cfg.py rxPci=0000:01:00.1          # set one field
  ./tools/spark_cfg.py dstMac=00:00:5e:00:53:89 txFill=0.9
  SPARK_CONTROL_URL=http://host:8080 ./tools/spark_cfg.py ...
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("SPARK_CONTROL_URL", "http://127.0.0.1:8080").rstrip("/")

# Re-derived from the activated sender's transport params + SDP on every IS-05 activation.
NMOS_OWNED = {
    "rxMcastGroup", "rxSrcIp", "rxDstPort", "rxIfaceIp",
    "rxAudioMcastGroup", "rxAudioSrcIp", "rxAudioDstPort",
    "txMcastGroup", "txDstPort", "txSrc", "txAudioMcastGroup", "txAudioDstPort",
    "inWidth", "inHeight", "inExactframerate", "inDepth", "inSampling",
    "inIp10", "inJxs",
}


def get_status():
    with urllib.request.urlopen(BASE + "/api/status", timeout=10) as r:
        return json.load(r)


def post(path, body=""):
    req = urllib.request.Request(
        BASE + path, data=body.encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode()
    try:
        ack = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "message": raw.strip()}
    return ack


def die(msg):
    sys.stderr.write("spark_cfg: %s\n" % msg)
    sys.exit(1)


def main(argv):
    try:
        st = get_status()
    except urllib.error.URLError as e:
        die("cannot reach daemon at %s (%s)" % (BASE, e))

    cfg = dict(st["config"])
    sets = [a for a in argv if "=" in a]
    gets = [a for a in argv if "=" not in a]

    if not sets:
        if gets:
            for k in gets:
                if k not in cfg:
                    die("no such field %r" % k)
                print(json.dumps(cfg[k]))
        else:
            print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    # Parse values as JSON where possible so booleans/numbers keep their type; a bare string
    # like a PCI address or MAC is not valid JSON and falls through unchanged.
    changes = {}
    for arg in sets:
        k, _, v = arg.partition("=")
        if k not in cfg:
            die("no such field %r — the daemon would silently ignore it "
                "(check spelling against /api/status)" % k)
        try:
            changes[k] = json.loads(v)
        except json.JSONDecodeError:
            changes[k] = v

    was_running = st["state"] == "RUNNING"
    print("daemon %s, engine %s" % (BASE, st["state"]))
    for k, v in changes.items():
        print("  %-20s %r -> %r" % (k, cfg.get(k), v))
        if k in NMOS_OWNED:
            print("  %-20s ^ WARNING: re-derived on the next IS-05 activation, which will "
                  "overwrite this" % "")
        cfg[k] = v

    if was_running:
        ack = post("/api/stop")
        if not ack.get("ok"):
            die("stop failed: %s" % ack.get("message"))

    ack = post("/api/config", json.dumps(cfg))
    if not ack.get("ok"):
        die("config rejected: %s" % ack.get("message"))
    msg = ack.get("message", "")
    if "unknown" in msg.lower():
        die("daemon reported: %s — aborting before restart" % msg)

    if was_running:
        ack = post("/api/start")
        if not ack.get("ok"):
            die("start failed: %s" % ack.get("message"))

    after = get_status()
    print("engine %s | %s" % (after["state"], after.get("message", "")))
    bad = [k for k, v in changes.items() if after["config"].get(k) != v]
    for k in changes:
        print("  verify %-20s = %r%s" % (k, after["config"].get(k),
                                         "   MISMATCH" if k in bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
