#!/bin/bash
#
# Put the Nano in its fastest state, then hand off to the stream process.
# The container is privileged, so these sysfs writes reach the host.

# holly-stream only runs between run.sh and stop.sh. run.sh records the boot it was started in; if Docker restarts
# this container after a reboot or power cut, exit cleanly (0, so on-failure does not retry) until run.sh again.
if [ -n "${HOLLY_BOOT_ID}" ] && [ "${HOLLY_BOOT_ID}" != "$(cat /proc/sys/kernel/random/boot_id)" ]; then
    echo "[INFO] Not started by run.sh since this boot; staying stopped until run.sh."
    exit 0
fi

if [ "${MAX_PERFORMANCE:-True}" == "True" ]; then
    # Equivalent of jetson_clocks: pin GPU and CPU frequencies at maximum so
    # inference latency does not jitter while the governors ramp up and down.
    GPU=/sys/devices/57000000.gpu/devfreq/57000000.gpu
    if [ -w "$GPU/min_freq" ]; then
        cat "$GPU/max_freq" > "$GPU/min_freq"
    fi
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
        [ -w "$cpu/scaling_governor" ] && echo performance > "$cpu/scaling_governor"
    done
fi

exec /opt/holly-stream/holly-stream
