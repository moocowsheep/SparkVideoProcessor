#!/usr/bin/env python3
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

"""
nmos_mdns_proxy.py — a registry-free "virtual registry" for the Spark dashboard.

Browses _nmos-node._tcp over mDNS (via avahi-browse), aggregates each discovered NMOS
node's IS-04 resources by querying its Node API server-side, and re-serves them in the
IS-04 Query API shape (/x-nmos/query/v1.3/{nodes,devices,sources,flows,senders,receivers})
with permissive CORS. Sender manifest_href is rewritten through this proxy so the browser
can fetch SDPs even from non-CORS nodes (e.g. Blackmagic on :8090).

Point the dashboard's Registry URL at this proxy and discovery works with NO registry —
the proxy never talks to a registry, only to mDNS + each node directly. Runs as a normal
user (no sudo / DPDK). Default port 3290.
  SPARK_MDNS_PROXY_PORT (3290), SPARK_MDNS_PROXY_TTL (4s cache).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("SPARK_MDNS_PROXY_PORT", "3290"))
SERVICE = "_nmos-node._tcp"
CACHE_TTL = float(os.environ.get("SPARK_MDNS_PROXY_TTL", "4"))
HTTP_TIMEOUT = 2.5
# Query API collection -> Node API endpoint (self is a single object -> the nodes collection).
NODE_EP = {"nodes": "self", "devices": "devices", "sources": "sources",
           "flows": "flows", "senders": "senders", "receivers": "receivers"}
RESOURCES = list(NODE_EP)

_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()


def _unescape(s):
    return re.sub(r"\\(\d{3})", lambda m: chr(int(m.group(1))), s)


def browse_nodes():
    """One entry per node {instance, host(IPv4), port, proto, vers[]} from the avahi cache."""
    try:
        out = subprocess.run(["avahi-browse", "-rpt", SERVICE],
                             capture_output=True, text=True, timeout=8).stdout
    except Exception as e:  # avahi missing / daemon down
        sys.stderr.write("avahi-browse failed: %s\n" % e)
        return []
    by_instance = {}
    for line in out.splitlines():
        if not line.startswith("="):
            continue
        f = line.split(";")
        if len(f) < 9 or f[2] != "IPv4":
            continue
        instance = _unescape(f[3])
        addr, port = f[7], f[8]
        kv = dict(re.findall(r'"([^"=]+)=([^"]*)"', f[9] if len(f) > 9 else ""))
        ent = by_instance.setdefault(instance, {"addrs": [], "port": port,
                                                 "proto": kv.get("api_proto", "http"),
                                                 "vers": [v.strip() for v in kv.get("api_ver", "v1.3").split(",") if v.strip()]})
        ent["addrs"].append(addr)
    nodes = []
    for instance, ent in by_instance.items():
        routable = [a for a in ent["addrs"] if not a.startswith(("127.", "169.254."))]
        host = (routable or ent["addrs"])[0]
        nodes.append({"instance": instance, "host": host, "port": ent["port"],
                      "proto": ent["proto"], "vers": ent["vers"]})
    return nodes


def _pick_ver(vers):
    for v in ("v1.3", "v1.2", "v1.1", "v1.0"):
        if v in vers:
            return v
    return vers[0] if vers else "v1.3"


def _get_json(url):
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode())


def aggregate():
    agg = {k: [] for k in RESOURCES}
    for n in browse_nodes():
        base = "%s://%s:%s" % (n["proto"], n["host"], n["port"])
        ver = _pick_ver(n["vers"])
        for coll in RESOURCES:
            try:
                data = _get_json("%s/x-nmos/node/%s/%s" % (base, ver, NODE_EP[coll]))
            except Exception:
                continue  # node/endpoint unreachable -> just skip it
            for it in (data if isinstance(data, list) else [data]):
                if coll == "senders" and it.get("manifest_href"):
                    it["_orig_manifest"] = it["manifest_href"]
                agg[coll].append(it)
    for k in RESOURCES:  # dedupe by id (a node reachable on >1 IP returns the same ids)
        uniq = {}
        for it in agg[k]:
            if "id" in it:
                uniq[it["id"]] = it
        agg[k] = list(uniq.values())
    return agg


def get_agg():
    with _lock:
        if _cache["data"] is None or time.time() - _cache["t"] > CACHE_TTL:
            _cache["data"] = aggregate()
            _cache["t"] = time.time()
        return _cache["data"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self, ctype="application/json"):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Type", ctype)

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self._cors(ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        m = re.match(r"^/x-nmos/query/v1\.\d+/(\w+)/?$", u.path)
        if m and m.group(1) in RESOURCES:
            coll = m.group(1)
            items = get_agg()[coll]
            if coll == "senders":  # rewrite manifest_href through this proxy (CORS + non-CORS nodes)
                proxy = "http://" + (self.headers.get("Host") or ("127.0.0.1:%d" % PORT))
                rewritten = []
                for it in items:
                    it = dict(it)
                    orig = it.pop("_orig_manifest", None) or it.get("manifest_href")
                    if orig:
                        it["manifest_href"] = "%s/manifest?u=%s" % (proxy, urllib.parse.quote(orig, safe=""))
                    rewritten.append(it)
                items = rewritten
            return self._send(200, json.dumps(items))
        if u.path == "/manifest":
            url = (urllib.parse.parse_qs(u.query).get("u") or [""])[0]
            if not url:
                return self._send(400, '{"error":"missing u"}')
            try:
                with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:
                    return self._send(200, r.read(), "application/sdp")
            except Exception as e:
                return self._send(502, json.dumps({"error": str(e)}))
        if u.path in ("/", "/discover", "/healthz"):
            return self._send(200, json.dumps({"service": SERVICE, "nodes": browse_nodes()}))
        return self._send(404, '{"error":"not found"}')


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    sys.stderr.write("nmos-mdns-proxy listening on :%d, browsing %s\n" % (PORT, SERVICE))
    srv.serve_forever()
