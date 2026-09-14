#!/usr/bin/env bash
# Sample the RDMA adapters' operation counters into a CSV, one row per counter per sample.
#
# The expert all-to-all reaches no RCCL log and a trace names kernels without saying what went on
# the wire, so the adapter is the only source. Reading sysfs perturbs nothing.
#
# Counts are per adapter and per node, never per rank or kernel, and include every other user of
# the NIC (here the mooncake KV transfer), so an absolute count is a ceiling; in an A/B that
# traffic is common-mode and largely cancels.
#
# Usage:
#   ./rdma_counters.sh --out <file.csv> [--interval 30]  # sample until killed
#   ./rdma_counters.sh --out <file.csv> --once           # one sample and exit
#   ./rdma_counters.sh --out <file.csv> --devices mlx5_0,mlx5_2   # only these adapters
#
# --devices: a node here has ten adapters and the run is given eight (IB_DEVICES); the other two
# carry other jobs' traffic. Default is every adapter, so no device is dropped silently.
#
# RDMA_SYSFS_ROOT overrides the sysfs root, so this can be exercised on a machine with no adapter.
#
# Size, measured here: 10 mlx5 adapters x 53 counters is ~530 rows and 28 kB a sample, so the
# default 30 s interval costs about 30 MB per node over an eight-hour job.
#
# Writes `<out>.started` once its first complete sample has landed, so a caller can wait for
# a baseline instead of guessing from the line count.
#
# Exit codes: 0 sampled (or a host with no RDMA at all), 2 bad arguments, 3 a requested
# adapter exposes no counters, 4 the output could not be written.
#
# Output columns: epoch_ns,host,device,port,counter,value
# The host is provenance: a logical label like decode_NODE2 comes from the file name and
# says nothing about which machine it was, so a reader cannot otherwise tell two
# allocations apart. It is recorded, not used as an identity check -- two arms of an A/B
# always land on different physical nodes.
set -uo pipefail

OUT=""
INTERVAL=30
ONCE=0
DEVICES=""
#: `device:port` pairs, `*` meaning every port of that device.
WANTED_PORTS=""

# `set -u` turns a missing option value into an unbound-variable error and status 1 rather than
# the documented 2, so the argument count is checked before `$2` is read.
_needs_value() {
    if (( $2 < 2 )); then
        echo "rdma_counters.sh: $1 needs a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) _needs_value "$1" $#; OUT="$2"; shift 2 ;;
        --interval) _needs_value "$1" $#; INTERVAL="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        --devices) _needs_value "$1" $#; DEVICES="${DEVICES:+${DEVICES},}$2"; shift 2 ;;
        *) echo "rdma_counters.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$OUT" ]]; then
    echo "rdma_counters.sh: --out is required" >&2
    exit 2
fi

# A zero interval makes the inner countdown run zero times, sampling as fast as the shared
# filesystem allows; only a positive integer is accepted.
if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 1 )); then
    echo "rdma_counters.sh: --interval must be a positive whole number of seconds, got '${INTERVAL}'" >&2
    exit 2
fi

# A node with no RDMA device is not an error; the same launcher runs on such hosts. A header-only
# file records that the sampler ran and found nothing, which is not the same fact as not running.
#
# Every write to the output is checked. `errexit` is not an option here -- it would fire inside
# the TERM/sleep shutdown path -- and an unchecked redirection fails silently: a truncate that did
# not happen leaves the previous attempt's file looking like this run's, and a failed append exits
# 0 having collected nothing. Both report success with no data.
_cannot_write() {
    echo "rdma_counters.sh: cannot write ${OUT}: $1" >&2
    exit 4
}

mkdir -p "$(dirname "$OUT")" || _cannot_write "cannot create its directory"
if [[ "$ONCE" == "1" ]]; then
    # A single snapshot appends: two `--once` calls are how a caller makes a window.
    [[ -s "$OUT" ]] || echo "epoch_ns,host,device,port,counter,value" > "$OUT" \
        || _cannot_write "header"
else
    # The sampling loop owns its file and truncates it: on a SLURM requeue the same path comes
    # back, and appending would splice two attempts into one window.
    echo "epoch_ns,host,device,port,counter,value" > "$OUT" || _cannot_write "header"
    # A previous attempt's failure marker goes with the window it described: the contents have
    # just been replaced, so leaving it would have a successful requeue read as incomplete for
    # ever. `--once` appends into an existing window, so it does not clear the marker.
    rm -f "${OUT}.failed" "${OUT}.started"
fi

SYSFS_ROOT="${RDMA_SYSFS_ROOT:-/sys/class/infiniband}"
if [[ ! -d "$SYSFS_ROOT" ]] || [[ -z "$(ls -A "$SYSFS_ROOT" 2>/dev/null)" ]]; then
    echo "rdma_counters.sh: no RDMA devices under ${SYSFS_ROOT}; nothing to sample" >&2
    # Same rule as the per-adapter check below, which this used to jump over: a host with no RDMA
    # is supported and exits 0, but adapters that were *asked for* and are not there is a failed
    # measurement. Exiting 0 left a header-only file indistinguishable from a node that sent
    # nothing, and no marker for the report to withhold on.
    [[ -n "$DEVICES" ]] && exit 3
    exit 0
fi

# Devices present but no counters is the container case: `/sys/class/infiniband/<dev>` is a
# symlink into `/sys/devices/...`, and docker gives the class directory with the targets absent.
# Measured here: 10 devices visible, 0 counter files, until `/sys/devices` is bind-mounted
# read-only.
#
# Checked per *requested* device: an unused adapter exposing counters would mask a requested one
# that does not, and the totals would come out short while looking complete.
_missing=""
_present=""
# Repeatable and merged, because the transports name their adapters separately -- IB_DEVICES for
# the KV transfer, MORI_RDMA_DEVICES for the expert exchange, NCCL_IB_HCA for RCCL -- and sampling
# one list would leave the others' traffic out of a total that looks whole. Duplicates collapse
# and an `^` exclusion list is refused rather than guessed.
#
# `mlx5_0:1` names a port, not just a device. The suffix used to be dropped, so a request for one
# port sampled every port of that HCA and a multi-port card contributed traffic from a port this
# run was never given. Kept as `device:port`; a bare name means every port of that device, which
# is what asking for the device says, and a device named both ways keeps the wider intent.
if [[ -n "$DEVICES" ]]; then
    _seen=""
    _merged=""
    _wanted_ports=""
    IFS=',' read -ra _raw <<< "$DEVICES"
    for _name in "${_raw[@]}"; do
        # NCCL's exact-match prefix: `NCCL_IB_HCA="=mlx5_0:1"` means that device and no other.
        # Left on, the name matches no sysfs entry, and the same adapter arriving unprefixed from
        # another list makes the set look partial -- so the sampler refused all of it.
        _name="${_name#=}"
        [[ -n "$_name" ]] || continue
        if [[ "$_name" == ^* ]]; then
            echo "rdma_counters.sh: cannot resolve an exclusion list ('${_name}'); name the adapters instead" >&2
            exit 2
        fi
        _port="*"
        [[ "$_name" == *:* ]] && _port="${_name##*:}"
        _name="${_name%%:*}"
        [[ -n "$_name" ]] || continue
        [[ ",${_wanted_ports}," == *",${_name}:${_port},"* ]] || \
            _wanted_ports="${_wanted_ports}${_name}:${_port},"
        [[ ",${_seen}," == *",${_name},"* ]] && continue
        _seen="${_seen}${_name},"
        _merged="${_merged}${_name},"
    done
    DEVICES="${_merged%,}"
    WANTED_PORTS="${_wanted_ports%,}"
    IFS=',' read -ra _wanted <<< "$DEVICES"
else
    _wanted=()
    for _dev in "$SYSFS_ROOT"/*; do
        [[ -d "$_dev" ]] && _wanted+=("$(basename "$_dev")")
    done
fi

# Per requested `device:port`, not per device. Validating the device alone passed a request for
# `mlx5_0:2` whenever port 1 had counters, and `_sample` then filtered port 1 out and exited 0
# with a header only -- a failed measurement wearing the shape of a node that sent nothing.
_check_entries=()
if [[ -n "$WANTED_PORTS" ]]; then
    IFS=',' read -ra _check_entries <<< "$WANTED_PORTS"
else
    for _name in "${_wanted[@]}"; do
        [[ -n "$_name" ]] && _check_entries+=("${_name}:*")
    done
fi

for _entry in "${_check_entries[@]}"; do
    [[ -n "$_entry" ]] || continue
    _name="${_entry%%:*}"
    _port="${_entry##*:}"
    _dev="${SYSFS_ROOT}/${_name}"
    if [[ -d "$_dev" ]] \
       && { compgen -G "${_dev}/ports/${_port}/counters/*" >/dev/null \
            || compgen -G "${_dev}/ports/${_port}/hw_counters/*" >/dev/null; }; then
        _present="${_present}${_entry} "
    else
        _missing="${_missing}${_entry} "
    fi
done

if [[ -n "$_missing" ]]; then
    echo "rdma_counters.sh: no counters under: ${_missing% }" >&2
    echo "rdma_counters.sh: in a container the class entries are symlinks into /sys/devices;" >&2
    echo "rdma_counters.sh: add '-v /sys/devices:/sys/devices:ro' to the docker run options." >&2
fi

if [[ -z "$_present" ]]; then
    # Two outcomes, two exit codes: a host with no RDMA at all is supported (the empty DEVICES
    # default) and exits 0 with a header-only file; adapters that were *asked for* and cannot be
    # sampled are a failed measurement, since exit 0 there reads as a node that sent nothing.
    echo "rdma_counters.sh: none of the requested adapters exposes counters; nothing to sample" >&2
    [[ -n "$DEVICES" ]] && exit 3
    exit 0
fi

# A partial set is refused rather than sampled: totals over some of a run's adapters are a wrong
# number that cannot be told from a right one. In the default mode every discovered adapter *is*
# the requested set, so a mix of usable and counterless ones is the same partial total -- it used
# to warn and exit 0. Total absence stays supported: that is a host with no RDMA, handled above.
if [[ -n "$_missing" ]]; then
    echo "rdma_counters.sh: refusing to sample a partial set of the requested adapters" >&2
    [[ -n "$DEVICES" ]] || echo "rdma_counters.sh: name the ones this run uses with --devices" >&2
    exit 3
fi

_HOST="$(hostname 2>/dev/null || echo unknown)"

#: Counters that have read as a number at least once, and how many of them failed in the sample
#: just taken. A failure here is only meaningful against a previous success.
declare -A _read_ok=()
_lost=0

_sample() {
    local now dev port group file name value
    now="$(date +%s%N)"
    for dev in "$SYSFS_ROOT"/*; do
        [[ -d "$dev" ]] || continue
        if [[ -n "$DEVICES" ]] && [[ ",${DEVICES}," != *",$(basename "$dev"),"* ]]; then
            continue
        fi
        for port in "$dev"/ports/*; do
            [[ -d "$port" ]] || continue
            # The port, not only the device: a request for `mlx5_0:1` must not sample port 2.
            if [[ -n "$WANTED_PORTS" ]] \
               && [[ ",${WANTED_PORTS}," != *",${dev##*/}:${port##*/},"* ]] \
               && [[ ",${WANTED_PORTS}," != *",${dev##*/}:*,"* ]]; then
                continue
            fi
            # `counters` are the portable port totals, `hw_counters` the vendor's own, where the
            # per-operation-type ones live under names mlx5 and bnxt_re spell differently -- so
            # nothing is hardcoded and the reader decides what is interesting.
            for group in counters hw_counters; do
                [[ -d "$port/$group" ]] || continue
                for file in "$port/$group"/*; do
                    [[ -f "$file" ]] || continue
                    # One unreadable counter (permissions, EINVAL for an unsupported one) must
                    # not end the whole sample: it leaves `value` empty and the digit test drops
                    # it, which also rejects a non-numeric counter.
                    # Builtins, not `cat` and three `basename`s: at 10 adapters x 53 counters
                    # that was ~2100 process launches per interval on a node being timed.
                    value=""
                    read -r value < "$file" 2>/dev/null
                    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
                        # A counter that read as a number before and does not now shortens this
                        # node's window without shortening the file: the parser still sees many
                        # samples and reports the delta as the whole run. Counted, so the caller
                        # can mark the series. An entry that never read as a number is a
                        # non-numeric counter, which is normal and stays silent.
                        [[ -n "${_read_ok[$file]:-}" ]] && _lost=$(( _lost + 1 ))
                        continue
                    fi
                    _read_ok["$file"]=1
                    # `|| return 1`: otherwise the loop carries on and the function's status is
                    # the last echo's, so a sample that lost rows mid-write reports success.
                    echo "${now},${_HOST},${dev##*/},${port##*/},${file##*/},${value}" || return 1
                done
            done
        done
    done
}

if [[ "$ONCE" == "1" ]]; then
    _sample >> "$OUT" || _cannot_write "sample"
    exit 0
fi

# SIGTERM is how the launcher stops this; the last sample lands before it does, so the window
# ends when the servers stop, not one interval earlier.
_stop=0
trap '_stop=1' TERM INT
# One complete sample, then a marker. A reader cannot tell a finished sample from a half-written
# one by counting lines -- rows arrive one echo at a time -- so the launcher waits for this file
# rather than for the output to be non-empty.
_first=1
while [[ "$_stop" == "0" ]]; do
    # A failed write ends the sampler rather than being retried: once the file is unwritable
    # every later sample is lost too, and a short window that looks whole is worse than a
    # sampler the launcher can see died.
    _lost=0
    _sample >> "$OUT" || _cannot_write "sample"
    if (( _lost > 0 )); then
        # Marked rather than fatal: what landed is real, and the report reads a marked series as
        # a floor. Silence was the problem -- a shortened window wearing the shape of a whole one.
        echo "rdma_counters.sh: ${_lost} counter(s) stopped reading; the window is short" >&2
        echo "lost ${_lost} counter(s) mid-run" > "${OUT}.failed" 2>/dev/null || true
    fi
    if (( _first )); then
        # Only when something was actually read: an adapter whose counter files exist but never
        # yield a number would otherwise have its empty baseline declared complete.
        if (( ${#_read_ok[@]} == 0 )); then
            echo "rdma_counters.sh: no counter read a value on the first sample" >&2
            echo "no counter readable at the baseline" > "${OUT}.failed" 2>/dev/null || true
        fi
        : > "${OUT}.started" 2>/dev/null || true
        _first=0
    fi
    for ((_i = 0; _i < INTERVAL && _stop == 0; _i++)); do sleep 1; done
done
# The same loss check as inside the loop. A counter lost between the last interval and SIGTERM
# shortens the window at the end, where it matters most, and this sample used to go unchecked.
_lost=0
_sample >> "$OUT" || _cannot_write "final sample"
if (( _lost > 0 )); then
    echo "rdma_counters.sh: ${_lost} counter(s) stopped reading before the final sample" >&2
    echo "lost ${_lost} counter(s) mid-run" > "${OUT}.failed" 2>/dev/null || true
fi
