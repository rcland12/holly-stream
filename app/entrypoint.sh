#!/bin/bash
#
# Put the Nano in its fastest state, then hand off to the stream process.
# The container is privileged, so these sysfs writes reach the host.

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
    [ -w /sys/devices/pwm-fan/target_pwm ] && echo "${FAN_PWM:-255}" > /sys/devices/pwm-fan/target_pwm
fi

exec /opt/holly-stream/holly-stream
