# deploy/ — host provisioning

Reproducible setup for the ST 2110 IO plane the engine needs (M1 open item #5). Dev/build setup
(Holoscan SDK, protobuf, web UI) is separate — see [`docs/SETUP.md`](../docs/SETUP.md).

## `provision.sh`
Idempotent; safe to re-run. Provisions three things found necessary during M0/M1 bring-up:

1. **packages** — `mft` (for `mlxconfig`), `dpdk` + `dpdk-dev` (mlx5 PMD), `linuxptp`.
2. **NIC firmware** — `REAL_TIME_CLOCK_ENABLE=1` on every ConnectX-7. This is the **`tx_pp`
   prerequisite**: without it the mlx5 PMD reports *"Packet pacing is not supported"* and ST 2110-21
   pacing is impossible (see `docs/M0-feasibility-findings.md`). **Requires a cold reboot to apply.**
3. **hugepages** — `2048 × 2MB` (4 GiB) for DPDK/EAL: allocated now, persisted via
   `/etc/sysctl.d/80-spark-hugepages.conf`, and `/dev/hugepages` mounted.

```bash
bash deploy/provision.sh --check        # read-only: report current state (run anytime)
sudo bash deploy/provision.sh           # apply (idempotent)
HUGEPAGES=4096 sudo bash deploy/provision.sh   # override page count

# if it reports REAL_TIME_CLOCK_ENABLE was changed:
sudo reboot
bash deploy/provision.sh --check        # should now be all [ok]
```

Detects the CX-7 cards automatically (PCI vendor `15b3`, function-0 BDF per card). On this box that's
`0000:01:00.0` and `0002:01:00.0`. If no CX-7 is on the bus, the firmware step warns and skips (cable
it first — the card is hot-plug).

After `--check` shows all `[ok]`, the engine pipeline runs:
`./engine/build/st2110_pipeline` (+ a generator; see `docs/M2-processing.md`).

## `ptp.sh` + `ptp4l.conf` — PTP discipline (production timing)
Disciplines the shared CX-7 PHC (`ptp0`) with `ptp4l`/`phc2sys` so the engine's RTP timestamps and
`tx_pp` pacing track a reference clock (ST 2110-10 / ST 2059). The engine reads the same PHC, so this
needs no engine change. **Loopback on one box doesn't need it** (all ports share `ptp0`); it's for
production interop with a facility grandmaster.

```bash
bash deploy/ptp.sh --check               # read-only: HW-timestamp caps, /dev/ptp*, tools (no root)
sudo bash deploy/ptp.sh --test           # 15s master-mode self-test (validates the stack, no GM needed)
sudo bash deploy/ptp.sh                   # SLAVE: discipline PHC to a network grandmaster (production)
sudo bash deploy/ptp.sh --master          # this box AS the time source (small setup / no external GM)
```
`ptp4l.conf` is an ST 2059-2 media-profile starting point — **align its domain/intervals with your
facility grandmaster**. `--check` here shows the CX-7 PTP-capable with `ptp0` present (M0 gate 1).

## `systemd/` — run it as two daemons

Two units, installed from and running out of **this checkout** (a rebuild takes effect on the next
`systemctl restart`). The `.deb` ships its own units under `debian/` for installed paths.

| unit | what it runs |
|---|---|
| `spark-ptp.service` | `deploy/ptp.sh slave` — `ptp4l` disciplines the CX-7 PHC to the facility grandmaster, `phc2sys` syncs the system clock. |
| `spark.service` | `deploy/spark_run.sh` — `spark_controld` (which spawns `st2110_pipeline` and serves the dashboard on :8080) **and** `spark_nmos_node`. |

```bash
sudo bash deploy/systemd/install.sh --start   # install + enable at boot + start now
systemctl status spark-ptp spark
journalctl -fu spark                          # daemon + engine + node, interleaved
sudo bash deploy/systemd/install.sh --uninstall
```

`spark.service` is `Wants=`/`After=spark-ptp.service`: timing comes up first, but a PTP outage
degrades genlock rather than tearing the media plane down. Inside `spark.service` the daemon and the
node are one failure domain — the node's IS-05 endpoints route the engine *through* the daemon's
control API, so `spark_run.sh` waits for `/api/status` before launching the node and exits as soon as
either process does, letting `Restart=on-failure` bring the pair back together.

Site addressing comes from `deploy/lab.env` (gitignored); per-unit overrides go in
`/etc/default/spark-ptp` (e.g. `IFACE=`) and `/etc/default/spark-video-processor` (engine knobs).
`spark-ptp.service` declares `Conflicts=` against the NTP clients — `phc2sys` owns `CLOCK_REALTIME`,
and a second discipline sawtooths the wall clock; `install.sh` warns if one is still enabled at boot.

## `make_deb.sh` + `debian/` — the installable package

Builds `spark-video-processor_<version>_arm64.deb` from the built trees: the three binaries plus the
non-distro shared-library closure (Holoscan/GXF/UCX/RMM, CUDA runtime — never the NVIDIA driver
libs), `web/`, `deploy/`, the docs, and the **same two daemons** as above in their installed-path
form (`debian/spark-ptp.service`, `debian/spark-video-processor.service`).

```bash
bash deploy/make_deb.sh                    # version 0.8.0~beta into the repo root
bash deploy/make_deb.sh 0.8.1~beta /tmp
sudo dpkg -i spark-video-processor_0.8.1~beta_arm64.deb
sudo systemctl start spark-ptp spark-video-processor
```

Both units are **enabled at boot but not started** by `postinst` — installing must never yank the
NICs/GPU out from under a manually launched instance. `postinst` also flags any enabled NTP client,
since `phc2sys` owns `CLOCK_REALTIME` while `spark-ptp` runs.

Configuration lives in three conffiles (`dpkg` preserves your edits across upgrades):
`/etc/default/spark-ptp` (`IFACE=`), `/etc/default/spark-video-processor` (engine knobs,
`SPARK_NO_NMOS=1` to run without the node) and `/etc/default/spark-nmos-node` (registry + advertised
addresses). `Depends` is computed from the actual link closure via `dpkg -S`, plus `linuxptp` and
`curl`.

> **Upgrading from ≤ 0.8.0:** the NMOS node had its own `spark-nmos-node.service`. It now runs inside
> `spark-video-processor.service`, and `postinst` stops and disables the old unit so an upgrade
> doesn't leave two nodes registering.
