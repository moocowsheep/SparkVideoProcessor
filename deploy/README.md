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
