#!/usr/bin/env bash
#
# On-device functional test for the Idein "userqpu" kernel patch.
#
# The patched kernel makes the vc4 driver intercept the mailbox property
# tags RPI_FIRMWARE_EXECUTE_QPU (0x00030011) and RPI_FIRMWARE_SET_ENABLE_QPU
# (0x00030012) issued through /dev/vcio, so that user QPU programs are
# executed by vc4 instead of the VideoCore firmware.  qmkl drives exactly
# this path (launch_qpu_code.c -> libmailbox -> /dev/vcio), so running the
# qmkl test binaries on the target and verifying their numerical results
# exercises the whole patch: SET_ENABLE_QPU filtering, EXECUTE_QPU
# submission, job completion interrupt (DBQITC) and seqno wakeup.
#
# Target : Raspberry Pi Zero 2 / 2 / 3 (VideoCore IV) booted with the
#          patched arm64 kernel (e.g. 6.18.34-idein-rpi-v8, tag 1.20260609.0)
#          and dtoverlay=vc4-kms-v3d enabled.
#          The v3d hang_limit_ms change (Pi 4/5) is NOT covered by this test.
# Prereq : qmkl built in this tree (test/sgemm, test/scopy, test/vsAbs) with
#          libmailbox / librpimemmgr / vcsm installed.  Run as root or as a
#          user in the "video" group.
#
# Usage  : test/userqpu-kernel-test.sh [iterations]      (default: 3)
#
# Environment overrides:
#   TIMEOUT_SECS      per-test timeout, default 180.  A broken patch makes
#                     the EXECUTE_QPU ioctl wait forever (the kernel waits
#                     with an infinite but killable timeout), so every test
#                     runs under timeout(1) and a timeout is reported as HANG.
#   SGEMM_MAX_ABS_ERR pass threshold for sgemm "Maximum absolute error"
#                     (GPU vs CPU), default 0.1.  A non-executed or garbage
#                     QPU run yields errors orders of magnitude above this.
#   SKIP_PM_CHECK=1   skip the runtime-PM check (use when another GL client
#                     such as a compositor may keep V3D powered).

set -u

ITERATIONS=${1:-3}
TIMEOUT_SECS=${TIMEOUT_SECS:-180}
SGEMM_MAX_ABS_ERR=${SGEMM_MAX_ABS_ERR:-0.1}

TEST_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=$(mktemp -d /tmp/userqpu-kernel-test.XXXXXX)
PASS=0
FAIL=0

note() { printf '%s\n' "$*"; }
ok()   { printf 'PASS: %s\n' "$*"; PASS=$((PASS + 1)); }
ng()   { printf 'FAIL: %s\n' "$*"; FAIL=$((FAIL + 1)); }

# ---------------------------------------------------------------- preflight

note "== Preflight =="
note "kernel: $(uname -rm)"
case "$(uname -r)" in
    *idein*) : ;;
    *) note "WARN: kernel does not look like an Idein build; is the patched kernel installed?" ;;
esac

if [ ! -e /dev/vcio ]; then
    note "ERROR: /dev/vcio not found (rpi firmware mailbox driver missing)"
    exit 2
fi
if [ ! -r /dev/vcio ] || [ ! -w /dev/vcio ]; then
    note "ERROR: no permission on /dev/vcio (run as root or join the 'video' group)"
    exit 2
fi

# The interception only happens while the vc4 v3d driver is bound.  Without
# it the tags fall through to the firmware and this test would not exercise
# the kernel patch at all, so treat that as a hard error.
if ! ls /sys/bus/platform/drivers/vc4_v3d/ 2>/dev/null | grep -q v3d; then
    note "ERROR: vc4_v3d driver is not bound (need dtoverlay=vc4-kms-v3d on a VideoCore IV Pi)"
    exit 2
fi
note "vc4_v3d: bound"

V3D_PM_STATUS=$(ls -d /sys/bus/platform/drivers/vc4_v3d/*v3d*/power/runtime_status 2>/dev/null | head -1)

for bin in scopy vsAbs sgemm; do
    if [ ! -x "$TEST_DIR/$bin" ]; then
        note "ERROR: $TEST_DIR/$bin not found. Build qmkl first (cmake . && make)."
        exit 2
    fi
done

DMESG_BEFORE=$(dmesg 2>/dev/null | wc -l)

# ------------------------------------------------------------------- tests

# run_case NAME CMD...  -- runs CMD under timeout, stores log, returns 0/1/2
# (2 = hang).  Output is kept in $LOG_DIR/NAME.ITER.log.
run_case() {
    local name=$1; shift
    local log="$LOG_DIR/$name.$iter.log"
    timeout -k 10 "$TIMEOUT_SECS" "$@" >"$log" 2>&1
    local rc=$?
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        ng "$name (iter $iter): HANG (killed after ${TIMEOUT_SECS}s) -- EXECUTE_QPU never completed; see $log"
        return 2
    elif [ $rc -ne 0 ]; then
        ng "$name (iter $iter): exited with $rc; see $log"
        return 1
    fi
    return 0
}

for iter in $(seq 1 "$ITERATIONS"); do
    note ""
    note "== Iteration $iter/$ITERATIONS =="

    # scopy / vsAbs: exit code is always 0; a mismatch is reported on stdout.
    for name in scopy vsAbs; do
        if run_case "$name" "$TEST_DIR/$name"; then
            log="$LOG_DIR/$name.$iter.log"
            if grep -q 'GPU and CPU differ' "$log"; then
                ng "$name (iter $iter): GPU result differs from CPU; see $log"
            elif ! grep -q '^GPU:' "$log"; then
                ng "$name (iter $iter): unexpected output (no GPU run?); see $log"
            else
                ok "$name (iter $iter)"
            fi
        fi
    done

    # sgemm: judge by the first (GPU vs CPU) "Maximum absolute error" line.
    if run_case sgemm "$TEST_DIR/sgemm"; then
        log="$LOG_DIR/sgemm.$iter.log"
        err=$(awk -F': ' '/^Maximum absolute error:/ { print $2; exit }' "$log")
        if [ -z "$err" ]; then
            ng "sgemm (iter $iter): no error report in output; see $log"
        elif ! awk -v e="$err" -v t="$SGEMM_MAX_ABS_ERR" \
                 'BEGIN { exit !(e == e+0 && e < t) }'; then
            ng "sgemm (iter $iter): max abs error $err >= $SGEMM_MAX_ABS_ERR; see $log"
        else
            ok "sgemm (iter $iter): max abs error $err"
        fi
    fi
done

# -------------------------------------------------------- runtime PM check

# With the "Fix v3d runtime PM refcount leak" commit, V3D must go back to
# runtime suspend once no job is running (vc4 uses a short autosuspend
# delay).  On the unfixed kernel the refcount leaks once per job and the
# status stays "active" forever after the first job.
note ""
note "== Runtime PM check =="
if [ "${SKIP_PM_CHECK:-0}" = "1" ]; then
    note "skipped (SKIP_PM_CHECK=1)"
elif [ -z "$V3D_PM_STATUS" ] || [ ! -r "$V3D_PM_STATUS" ]; then
    note "WARN: v3d runtime_status not readable; skipping PM check"
else
    status=
    for _ in $(seq 1 15); do
        status=$(cat "$V3D_PM_STATUS")
        [ "$status" = "suspended" ] && break
        sleep 1
    done
    if [ "$status" = "suspended" ]; then
        ok "v3d runtime PM: suspended after jobs (no refcount leak)"
    else
        ng "v3d runtime PM: still '$status' 15s after last job (refcount leak, or another GL client is active -- rerun headless or with SKIP_PM_CHECK=1)"
    fi
fi

# ------------------------------------------------------------ dmesg check

note ""
note "== Kernel log check =="
DMESG_AFTER=$(dmesg 2>/dev/null | wc -l)
if [ "$DMESG_AFTER" -eq 0 ]; then
    note "WARN: dmesg not readable; skipping (run as root for this check)"
else
    new_errors=$(dmesg 2>/dev/null | tail -n "$((DMESG_AFTER - DMESG_BEFORE))" \
                 | grep -iE 'vc4|v3d' | grep -iE 'error|fail|timeout|hang|oops|bug' || true)
    if [ -n "$new_errors" ]; then
        ng "new vc4/v3d kernel errors during the run:"
        printf '%s\n' "$new_errors"
    else
        ok "no new vc4/v3d errors in dmesg"
    fi
fi

# ----------------------------------------------------------------- summary

note ""
note "== Summary: $PASS passed, $FAIL failed (logs in $LOG_DIR) =="
[ $FAIL -eq 0 ]
