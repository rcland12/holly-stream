#!/bin/bash
#
# Shows the cameras connected to the Holly server: where they come from, whether detection is running on them,
# and what the relay is sending on. Works with either compose stack, with or without the detector running.
#
# Usage (on the server): ./status.sh

set -euo pipefail

container_for() {
    docker ps --filter "label=com.docker.compose.service=$1" --format '{{.Names}}' | head -n1
}

ingest="$(container_for holly-ingest)"
if [[ -z "${ingest}" ]]; then
    echo "holly-ingest is not running." >&2
    exit 1
fi
detector="$(container_for holly-detector)"

api() {
    docker exec "${ingest}" wget -q -O - "http://127.0.0.1:9997/v3/$1?itemsPerPage=1000"
}

export PATHS_JSON="$(api paths/list)"
export SRT_JSON="$(api srtconns/list)"
export RTSP_JSON="$(api rtspsessions/list)"
export DETECTOR_JSON="$([[ -n "${detector}" ]] && docker exec "${detector}" cat /data/status.json 2>/dev/null || true)"
export DETECTOR_RUNNING="$([[ -n "${detector}" ]] && echo yes || echo no)"

python3 - <<'EOF'
import json, os
from urllib.parse import parse_qs

paths = json.loads(os.environ["PATHS_JSON"])["items"]
srt = {c["id"]: c for c in json.loads(os.environ["SRT_JSON"])["items"]}
rtsp = json.loads(os.environ["RTSP_JSON"])["items"]
detector = json.loads(os.environ["DETECTOR_JSON"]) if os.environ["DETECTOR_JSON"] else {}
detector_running = os.environ["DETECTOR_RUNNING"] == "yes"
workers = detector.get("cameras", {})

# The relay runs inside the ingest container, so its RTSP sessions come from localhost
relays = {s["path"]: s for s in rtsp if s.get("remoteAddr", "").startswith("127.0.0.1:") and s.get("state") == "read"}

rows = [("STREAM", "FROM", "ROTATE", "DETECTION", "RELAYING")]
for path in sorted(paths, key=lambda p: p["name"]):
    name, source = path["name"], path.get("source") or {}
    if name.startswith("annotated/") or not path.get("ready") or source.get("type") != "srtConn":
        continue
    conn = srt.get(source.get("id"), {})
    options = parse_qs(conn.get("query", ""))
    if options.get("detect", ["0"])[0] != "1":
        detection = "off"
    elif not detector_running:
        detection = "on, detector not running"
    elif name in workers and workers[name].get("state") == "running":
        w = workers[name]
        detection = f"on, {w.get('fps', '?')} fps, {w.get('inference_ms', '?')} ms"
    else:
        detection = f"on, {workers.get(name, {}).get('state', 'starting')}"
    if f"annotated/{name}" in relays:
        relaying = "annotated"
    elif name in relays:
        relaying = "plain"
    else:
        relaying = "no"
    rows.append((name, conn.get("remoteAddr", "?").rsplit(":", 1)[0], options.get("rotate", ["0"])[0], detection, relaying))

if len(rows) == 1:
    print("No cameras connected.")
else:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
    print()
    print("RELAYING is what goes to RELAY_URL: no means none is set, or the relay is reconnecting.")
    print("Watch on the LAN: http://<server>:8889/<stream> (WebRTC), http://<server>:8888/<stream> (HLS),")
    print("rtsp://<server>:8554/<stream>. Use annotated/<stream> for the copy with detections.")
print(f"Detector: {'running, model ' + detector.get('model', '?') if detector_running else 'not running'}")
EOF
