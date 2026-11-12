#!/bin/bash
# keeps at most 3 final_method.py trainings on the GPU (each ~3.5 GB)
cd "$(dirname "$0")"
n_running() { powershell.exe -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'final_method.py --arms' }).Count" | tr -d '\r'; }
for a in "$@"; do
  while [ "$(n_running)" -ge 3 ]; do sleep 60; done
  python -u final_method.py --arms $a > fm_$a.log 2>&1 &
  sleep 90
done
wait
