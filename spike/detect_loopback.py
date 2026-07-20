#!/usr/bin/env python3
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# M1 task 1 — detect which ConnectX-7 QSFP ports are physically cabled together.
#
# The DGX Spark's CX-7 exposes 4× 100G ports; for the M1 loopback self-test some pair (or
# pairs) are cabled port<->port. All 4 show carrier-up, so link state alone can't tell us the
# wiring. This probe sends a uniquely-tagged raw L2 frame out each port and listens on every
# other port to see where it lands — the receiving port is the cabled partner.
#
# Method (no extra deps — pure AF_PACKET, both TX and RX):
#   * auto-discover CX-7 ports (PCI vendor 0x15b3) that are up,
#   * open one raw RX socket per port (ethertype 0x88B5, IEEE local-experimental),
#   * for each src port: blast K tagged frames, then select() on the other ports' sockets,
#   * an edge src->dst means a frame sent on src arrived on dst (i.e. they share a wire).
#
# A confirmed cable is a bidirectional edge (src->dst AND dst->src). Output ends with
# shell-sourceable LOOP_* vars so the gate-4 loopback test can consume the result directly.
#
# Run as root (raw sockets):   sudo python3 spike/detect_loopback.py
#
# Caveat: this finds a *direct* L2 path. If ports were attached to a switch instead of a DAC
# loopback, broadcast frames would flood and the pairing would be ambiguous — but the M1
# topology is direct port<->port cables, so a clean 1:1 pairing is expected.
import fcntl
import os
import select
import socket
import struct
import sys
import time

ETH_P_PROBE = 0x88B5          # IEEE Std 802 local experimental ethertype 1
MAGIC = b"SPRKLOOP"           # payload tag so we ignore unrelated 0x88B5 traffic
BCAST = b"\xff\xff\xff\xff\xff\xff"
FRAMES_PER_SRC = 16           # send a burst so a single drop doesn't hide a real cable
RX_WINDOW_S = 0.40            # listen window after each burst
SIOCGIFINDEX = 0x8933


def die(msg, code=1):
    print(f"  [FAIL] {msg}", file=sys.stderr)
    sys.exit(code)


def is_cx7(ifname):
    """True if ifname is a Mellanox/CX-7 (PCI vendor 0x15b3) port."""
    try:
        with open(f"/sys/class/net/{ifname}/device/vendor") as f:
            return f.read().strip().lower() == "0x15b3"
    except OSError:
        return False


def iface_up(ifname):
    try:
        with open(f"/sys/class/net/{ifname}/carrier") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def iface_mac(ifname):
    with open(f"/sys/class/net/{ifname}/address") as f:
        return bytes.fromhex(f.read().strip().replace(":", ""))


def iface_pci(ifname):
    try:
        return os.path.basename(os.path.realpath(f"/sys/class/net/{ifname}/device"))
    except OSError:
        return "?"


def discover_ports():
    ports = []
    for ifname in sorted(os.listdir("/sys/class/net")):
        if is_cx7(ifname) and iface_up(ifname):
            ports.append(ifname)
    return ports


def build_frame(src_mac, src_idx):
    # dst=broadcast so the partner NIC always accepts it; payload = MAGIC + src index.
    payload = MAGIC + bytes([src_idx])
    frame = BCAST + src_mac + struct.pack("!H", ETH_P_PROBE) + payload
    return frame + b"\x00" * max(0, 60 - len(frame))   # pad to min Ethernet frame


def open_socket(ifname):
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_PROBE))
    s.bind((ifname, ETH_P_PROBE))
    s.setblocking(False)
    return s


def drain(sock):
    while True:
        try:
            sock.recv(2048)
        except (BlockingIOError, OSError):
            break


def main():
    if os.geteuid() != 0:
        die("must run as root for raw sockets — re-run: sudo python3 spike/detect_loopback.py")

    ports = discover_ports()
    print("===== ConnectX-7 loopback detection =====")
    if len(ports) < 2:
        die(f"need >=2 up CX-7 ports, found {len(ports)}: {ports} "
            f"(check cabling / 'ip link set <if> up')")
    for i, p in enumerate(ports):
        m = iface_mac(p)
        print(f"  port[{i}] {p:<16} pci={iface_pci(p):<12} "
              f"mac={':'.join('%02x' % b for b in m)}")

    socks = {p: open_socket(p) for p in ports}
    macs = {p: iface_mac(p) for p in ports}
    # edges[(src, dst)] = count of src-tagged frames seen arriving on dst
    edges = {}

    for si, src in enumerate(ports):
        for s in socks.values():
            drain(s)
        tx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        tx.bind((src, 0))
        frame = build_frame(macs[src], si)
        for _ in range(FRAMES_PER_SRC):
            tx.send(frame)
        tx.close()

        deadline = time.monotonic() + RX_WINDOW_S
        watch = [socks[p] for p in ports if p != src]
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            ready, _, _ = select.select(watch, [], [], timeout)
            if not ready:
                break
            for s in ready:
                try:
                    data = s.recv(2048)
                except (BlockingIOError, OSError):
                    continue
                # ethertype 0x88B5 frames only; confirm our magic + sender index.
                if data[12:14] != struct.pack("!H", ETH_P_PROBE):
                    continue
                if data[14:22] != MAGIC or data[22] != si:
                    continue
                dst = next(p for p in ports if socks[p] is s)
                edges[(src, dst)] = edges.get((src, dst), 0) + 1

    print("\n----- observed L2 edges (src -> dst : frames) -----")
    if not edges:
        print("  (none — no frames crossed any port pair)")
    for (src, dst), n in sorted(edges.items()):
        print(f"  {src} -> {dst} : {n}/{FRAMES_PER_SRC}")

    # Confirmed cable = bidirectional edge. Dedupe to unordered pairs.
    cables = []
    seen = set()
    for (src, dst) in edges:
        key = frozenset((src, dst))
        if key in seen:
            continue
        if (dst, src) in edges:
            cables.append((src, dst))
            seen.add(key)

    print("\n===== RESULT =====")
    if not cables:
        die("no L2 path between any CX-7 ports (no bidirectional edge). "
            "Check that the ports are cabled (DAC or to a switch) and up.")

    n = len(ports)
    full_mesh = len(edges) == n * (n - 1)
    if full_mesh:
        # Every port reaches every other => one L2 broadcast domain. With direct DAC loopback
        # only the two cabled ports would see each other's broadcasts; a full mesh means the
        # ports are attached to a switch. Either way, any TX/RX pair traverses the fabric, and
        # tx_pp pacing precision is a NIC TX-side property independent of the fabric.
        print(f"  [PASS] switched fabric: all {n} CX-7 ports share one L2 domain (switch).")
        print("         Any TX/RX port pair works for the gate-4 loopback test.")
    else:
        for src, dst in cables:
            print(f"  [PASS] direct pair: {src} ({iface_pci(src)}) <-> {dst} ({iface_pci(dst)})")
        if len(cables) > 1:
            print(f"  note: {len(cables)} cabled pairs found — using the first for gate 4.")

    a, b = cables[0]
    pa, pb = iface_pci(a), iface_pci(b)
    print(f"\n  chosen gate-4 pair:  TX {a} ({pa})  ->  RX {b} ({pb})")
    print("\n----- copy-paste the gate-4 run (PCI addresses, root) -----")
    print(f"  sudo -E LOOP_TX_PCI={pa} LOOP_RX_PCI={pb} PROFILE=1080p bash spike/st2110_loopback_gate4.sh")
    print(f"  sudo -E LOOP_TX_PCI={pa} LOOP_RX_PCI={pb} PROFILE=2160p bash spike/st2110_loopback_gate4.sh")


if __name__ == "__main__":
    main()
