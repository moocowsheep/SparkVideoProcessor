#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# deploy/make_deb.sh — build the spark-video-processor Debian package for
# DGX Spark (arm64). Run from anywhere inside the repo; needs built trees
# (engine/build, control/build incl. spark_nmos_node) and dpkg-deb.
#
#   bash deploy/make_deb.sh [VERSION] [OUTDIR]
#   bash deploy/make_deb.sh 0.8.0~beta /tmp
#
# What it does:
#  - stages binaries to /usr/lib/spark-video-processor
#  - bundles the non-distro shared-library closure (Holoscan/GXF/UCX/RMM,
#    CUDA npp/cudart — anything resolving outside /lib and /usr/lib) into
#    /usr/lib/spark-video-processor/lib; the systemd units point
#    LD_LIBRARY_PATH there, which overrides the build-tree RUNPATH
#  - NVIDIA driver libs (libcuda, libnvidia-*) are never bundled
#  - computes Depends from the distro-lib closure via dpkg -S
#  - installs web/, deploy/, git-tracked docs/, systemd units and
#    /etc/default files, then dpkg-deb --build
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-0.8.0~beta}"
OUTDIR="${2:-$ROOT}"
PKG=spark-video-processor
BINS=(
  "$ROOT/engine/build/st2110_pipeline"
  "$ROOT/control/build/spark_controld"
  "$ROOT/control/build/spark_nmos_node"
)
for b in "${BINS[@]}"; do
  [ -x "$b" ] || { echo "missing binary: $b (build engine/ and control/ first)" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
LIBDIR="$STAGE/usr/lib/$PKG"
mkdir -p "$LIBDIR/lib" "$STAGE/usr/share/$PKG" "$STAGE/usr/share/doc/$PKG" \
         "$STAGE/lib/systemd/system" "$STAGE/etc/default" "$STAGE/DEBIAN"

install -m755 "${BINS[@]}" "$LIBDIR/"

# Bundle every dependency that does not come from a distro package
# (i.e. resolves outside /lib and /usr/lib), except NVIDIA driver libs.
ldd "${BINS[@]}" 2>/dev/null | awk '$3 ~ /^\// && $3 !~ /^\/(lib|usr\/lib)\// {print $3}' \
  | grep -v 'libnvidia\|libcuda\.so' | sort -u | while read -r so; do
    cp -L "$so" "$LIBDIR/lib/"
done

# Depends: map the distro-lib closure back to package names. NVIDIA driver
# packages are excluded (driver version varies per system) and noted in the
# description instead.
# realpath first: /lib is a /usr/lib symlink and dpkg's database only knows
# the real paths; tolerate dpkg -S misses (e.g. runfile-installed driver libs).
DEPENDS=$(ldd "${BINS[@]}" 2>/dev/null \
  | awk '$3 ~ /^\/(lib|usr\/lib)\// {print $3}' | sort -u | xargs -r realpath | sort -u \
  | { xargs -r dpkg -S 2>/dev/null || true; } | cut -d: -f1 | sort -u \
  | { grep -vi 'nvidia\|^cuda' || true; } | paste -sd, - | sed 's/,/, /g')
[ -n "$DEPENDS" ] || { echo "Depends computation came up empty — dpkg -S failed?" >&2; exit 1; }
# linuxptp: spark-ptp.service (ptp4l/phc2sys). curl: spark_run.sh waits on the
# daemon's /api/status before launching the node.
DEPENDS="$DEPENDS, linuxptp, curl"

cp -r "$ROOT/web" "$STAGE/usr/share/$PKG/web"
mkdir -p "$STAGE/usr/share/$PKG/deploy"
# install -D (not a flat cp): deploy/ has subdirectories (systemd/) whose files
# would otherwise collide into deploy/. Modes are preserved so the scripts the
# units exec stay executable.
# deploy/systemd/ is excluded too: those units are the source-tree variants, and
# their install.sh would rewrite /etc/systemd/system to point at the package's
# share dir (no build tree there) — the package ships debian/*.service instead.
(cd "$ROOT" && git ls-files deploy | grep -vE '^deploy/(debian|systemd)/' | while read -r f; do
  install -D -m "$(stat -c %a "$f")" "$f" "$STAGE/usr/share/$PKG/$f"
done)
(cd "$ROOT" && git ls-files docs | xargs -I{} cp {} "$STAGE/usr/share/doc/$PKG/")
cp "$ROOT/README.md" "$STAGE/usr/share/doc/$PKG/"

# Two units: PTP discipline, and the media plane (control daemon + engine + NMOS
# node, supervised together by deploy/spark_run.sh).
cp "$ROOT/deploy/debian/spark-ptp.service" \
   "$ROOT/deploy/debian/spark-video-processor.service" "$STAGE/lib/systemd/system/"
cp "$ROOT/deploy/debian/default-spark-video-processor" "$STAGE/etc/default/spark-video-processor"
cp "$ROOT/deploy/debian/default-spark-nmos-node" "$STAGE/etc/default/spark-nmos-node"
cp "$ROOT/deploy/debian/default-spark-ptp" "$STAGE/etc/default/spark-ptp"

SIZE=$(du -sk --exclude=DEBIAN "$STAGE" | cut -f1)
sed -e "s/@VERSION@/$VERSION/" -e "s/@SIZE@/$SIZE/" -e "s/@DEPENDS@/$DEPENDS/" \
    "$ROOT/deploy/debian/control.in" > "$STAGE/DEBIAN/control"
install -m755 "$ROOT/deploy/debian/postinst" "$ROOT/deploy/debian/prerm" \
              "$ROOT/deploy/debian/postrm" "$STAGE/DEBIAN/"
printf '/etc/default/spark-video-processor\n/etc/default/spark-nmos-node\n/etc/default/spark-ptp\n' \
  > "$STAGE/DEBIAN/conffiles"

# Sanity: with the bundled dir on LD_LIBRARY_PATH nothing may resolve to the
# build tree or be missing.
BAD=$(LD_LIBRARY_PATH="$LIBDIR/lib" ldd "$LIBDIR/st2110_pipeline" \
      | grep -c 'not found\|/home/' || true)
[ "$BAD" -eq 0 ] || { echo "bundled closure incomplete:" >&2
  LD_LIBRARY_PATH="$LIBDIR/lib" ldd "$LIBDIR/st2110_pipeline" | grep 'not found\|/home/' >&2; exit 1; }

DEB="$OUTDIR/${PKG}_${VERSION}_arm64.deb"
dpkg-deb --build --root-owner-group "$STAGE" "$DEB"
echo "built: $DEB"
