#!/bin/bash
# Login-node watchdog — every 5 min, kill all $USER processes on the login
# node except self, ancestors, and systemd user-instance.
#
# Designed for WCSS ui.wcss.pl: zombie bashes / find / tar / git gc from
# timeout'd ssh sessions accumulate, exhaust the per-user shell quota, and
# lock the user out for 30min–2h. This watchdog cleans them up automatically.
#
# Launch (once, detached):
#   nohup bash hpc/watchdog.sh > logs/watchdog.log 2>&1 < /dev/null &
#   disown
#
# Stop:
#   pkill -9 -f 'hpc/watchdog.sh'
#
# WARNING: this WILL kill any interactive ssh session you have open from
# another terminal at the next sweep. SLURM jobs on compute nodes are NOT
# affected (they run under their own cgroup on different machines).

INTERVAL="${WATCHDOG_INTERVAL:-300}"

SELF=$$

# Walk up parent chain so the watchdog doesn't kill its own shell.
get_ancestors() {
    local pid=$SELF
    local out="$pid"
    while :; do
        local ppid
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -z "$ppid" ] && break
        [ "$ppid" -le 1 ] && break
        out="$out $ppid"
        pid=$ppid
    done
    echo "$out"
}

ANCESTORS=$(get_ancestors)
echo "$(date -Iseconds) WATCHDOG up: pid=$SELF ancestors='$ANCESTORS' interval=${INTERVAL}s"

while :; do
    sleep "$INTERVAL"
    KILLED=0
    while read -r pid comm; do
        [ -z "$pid" ] && continue
        # skip ancestors
        skip=0
        for anc in $ANCESTORS; do
            [ "$pid" = "$anc" ] && skip=1 && break
        done
        [ $skip -eq 1 ] && continue
        # skip systemd user instance and (sd-pam)
        case "$comm" in
            systemd|"(sd-pam)") continue ;;
        esac
        if kill -9 "$pid" 2>/dev/null; then
            KILLED=$((KILLED + 1))
        fi
    done < <(ps -u "$USER" -o pid=,comm= --no-headers)
    echo "$(date -Iseconds) sweep killed=$KILLED"
done
