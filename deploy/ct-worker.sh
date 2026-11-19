#!/usr/bin/env bash
set -euo pipefail
APP=/opt/ai-checker
CONF=/etc/ai-checker
ENV_FILE="$CONF/worker.env"
SECRET_FILE="$CONF/worker.secret"
RULES_FILE="$CONF/firewall.nft"
WORKER_UNIT=/etc/systemd/system/ai-checker-worker.service
FIREWALL_UNIT=/etc/systemd/system/ai-checker-firewall.service
STATE=/root
BACKUP_POINTER="$STATE/ai-checker-worker-last-backup"
BEFORE_POINTER="$STATE/ai-checker-worker-code-before"
FRESH_INSTALL_MARK="$STATE/ai-checker-worker-fresh-install"
WORKER_DATA_ROOT="${WORKER_DATA_ROOT:?WORKER_DATA_ROOT is missing: deploy.sh passes it from deploy/site.env}"
VENV="$WORKER_DATA_ROOT/venv"
CONTROLLER_URL="${CONTROLLER_URL:-}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-$WORKER_DATA_ROOT/gpu.lock}"
WORKER_HOST="${WORKER_HOST:?WORKER_HOST is missing: deploy.sh passes it from deploy/site.env}"
WORKER_PORT="${WORKER_PORT:?WORKER_PORT is missing: deploy.sh passes it from deploy/site.env}"
APP_HOST="${APP_HOST:?APP_HOST is missing: deploy.sh passes it from deploy/site.env}"
KEPT_CODE_COPIES=3
SECRET_BYTES=48

fill_in_site_values() {
  sed -e "s/__WORKER_HOST__/$WORKER_HOST/g" -e "s/__WORKER_PORT__/$WORKER_PORT/g" -e "s/__APP_HOST__/$APP_HOST/g" \
      -e "s|__WORKER_DATA_ROOT__|$WORKER_DATA_ROOT|g" -e "s|__CONTROLLER_URL__|$CONTROLLER_URL|g" -e "s|__GPU_LOCK_FILE__|$GPU_LOCK_FILE|g" "$@"
}

do_backup() {
  local stamp backup parts path
  stamp="$(date +%Y%m%d-%H%M%S)"
  backup="$STATE/ai-checker-worker-before-$stamp.tgz"
  parts=()
  for path in opt/ai-checker etc/ai-checker etc/systemd/system/ai-checker-worker.service etc/systemd/system/ai-checker-firewall.service; do
    if [ -e "/$path" ]; then parts+=("$path"); fi
  done
  if [ "${#parts[@]}" = 0 ]; then
    ( umask 077 && tar -czf "$backup" --files-from /dev/null )
  else
    ( umask 077 && tar -czf "$backup" -C / "${parts[@]}" )
  fi
  chmod 600 "$backup"
  echo "$backup" > "$BACKUP_POINTER"
  chmod 600 "$BACKUP_POINTER"
  echo "worker backup: $backup ($(du -h "$backup" | cut -f1)), nothing in it is printed"
}

ensure_secret() {
  install -d -m 750 -o root -g root "$CONF"
  if [ -s "$SECRET_FILE" ]; then
    echo "secret: kept the one already on this box"
    return
  fi
  ( umask 077 && head -c "$SECRET_BYTES" /dev/urandom | base64 -w0 > "$SECRET_FILE" )
  chmod 600 "$SECRET_FILE"
  echo "secret: made a new one on this box (not shown)"
}

install_env_file() {
  if [ -f "$ENV_FILE" ]; then
    echo "env: $ENV_FILE kept as it is"
    return
  fi
  install -m 600 -o root -g root /dev/null "$ENV_FILE"
  fill_in_site_values "$APP/deploy/worker.env" > "$ENV_FILE"
  echo "env: wrote $ENV_FILE"
}

install_code() {
  local bundle="$1" stamp="$2" fresh="$APP-new-$2"
  mkdir "$fresh"
  if ! tar -xzf "$bundle" -C "$fresh" --no-same-owner; then
    rm -rf "$fresh" "$bundle"
    echo "refusing: the code bundle could not be unpacked; nothing was changed"
    exit 1
  fi
  rm -f "$bundle"
  chown -R root:root "$fresh"
  chmod -R u=rwX,go=rX "$fresh"
  if [ -d "$APP" ]; then
    mv "$APP" "$APP-before-$stamp"
    echo "$APP-before-$stamp" > "$BEFORE_POINTER"
  else
    echo none > "$BEFORE_POINTER"
    : > "$FRESH_INSTALL_MARK"
  fi
  mv "$fresh" "$APP"
}

keep_newest_copies() {
  local pattern="$1" copies surplus index
  shopt -s nullglob
  copies=($pattern)
  shopt -u nullglob
  surplus=$(( ${#copies[@]} - KEPT_CODE_COPIES ))
  for (( index = 0; index < surplus; index++ )); do
    rm -rf -- "${copies[index]}"
    echo "old copy removed: ${copies[index]}"
  done
}

install_units() {
  fill_in_site_values "$APP/deploy/ai-checker.nft" > "$RULES_FILE"
  chmod 600 "$RULES_FILE"
  install -m 644 "$APP/deploy/ai-checker-firewall.service" "$FIREWALL_UNIT"
  fill_in_site_values "$APP/deploy/ai-checker-worker.service" > "$WORKER_UNIT"
  chmod 644 "$WORKER_UNIT"
  systemctl daemon-reload
}

check_python_stack() {
  "$VENV/bin/python" -I -c "import torch, transformers, safetensors; print('stack: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

wait_for_listener() {
  for _ in $(seq 1 60); do
    if ss -ltn "sport = :$WORKER_PORT" | grep -q "$WORKER_HOST:$WORKER_PORT"; then
      return 0
    fi
    sleep 1
  done
  echo "the worker did not listen on $WORKER_HOST:$WORKER_PORT within 60 s"
  return 1
}

do_apply() {
  local bundle="$1" stamp since
  [ -s "$BACKUP_POINTER" ] && [ -s "$(cat "$BACKUP_POINTER")" ] || { echo "refusing: run backup first"; exit 1; }
  rm -f "$BEFORE_POINTER" "$FRESH_INSTALL_MARK"
  check_python_stack
  stamp="$(date +%Y%m%d-%H%M%S)"
  install_code "$bundle" "$stamp"
  ensure_secret
  install_env_file
  install_units
  systemctl enable ai-checker-firewall > /dev/null 2>&1
  systemctl restart ai-checker-firewall
  systemctl enable ai-checker-worker > /dev/null 2>&1
  since="$(date +%s)"
  systemctl restart ai-checker-worker
  wait_for_listener
  keep_newest_copies "$APP-before-*"
  keep_newest_copies "$APP-undone-*"
  sleep 1
  echo "earlier code: $(cat "$BEFORE_POINTER")"
  do_check "$since"
}

do_check() {
  local since="${1:-}" failures=0 listening refused firewall tracebacks mode
  echo "worker service: $(systemctl is-active ai-checker-worker || true)"
  echo "firewall service: $(systemctl is-active ai-checker-firewall || true)"
  listening="$(ss -ltn "sport = :$WORKER_PORT" | awk 'NR>1 {print $4}' | tr '\n' ' ')"
  echo "listening on: ${listening:-nothing}"
  refused="$(curl -s -o /dev/null -w '%{http_code}' "http://$WORKER_HOST:$WORKER_PORT/v1/health" || true)"
  echo "this container asking without the secret: $refused (403 expected: only $APP_HOST may ask)"
  firewall="$(nft list table inet ai_checker 2>/dev/null | grep -c "tcp dport $WORKER_PORT" || true)"
  echo "firewall rules for the port: $firewall (2 expected)"
  mode="$(stat -c '%a %U' "$SECRET_FILE")"
  echo "secret file mode: $mode (600 root expected)"
  echo "env file mode: $(stat -c '%a %U' "$ENV_FILE")"
  [ "$(systemctl is-active ai-checker-worker || true)" = active ] || failures=$((failures + 1))
  [ "$(systemctl is-active ai-checker-firewall || true)" = active ] || failures=$((failures + 1))
  [ "$listening" = "$WORKER_HOST:$WORKER_PORT " ] || failures=$((failures + 1))
  [ "$refused" = 403 ] || failures=$((failures + 1))
  [ "$firewall" = 2 ] || failures=$((failures + 1))
  [ "$mode" = "600 root" ] || failures=$((failures + 1))
  if [ -n "$since" ]; then
    tracebacks="$(journalctl -u ai-checker-worker --since "@$since" --no-pager | grep -c Traceback || true)"
    echo "tracebacks in the journal since restart: $tracebacks"
    [ "$tracebacks" = 0 ] || failures=$((failures + 1))
  fi
  echo "worker check failures: $failures"
  [ "$failures" = 0 ]
}

do_undo() {
  local before stamp
  before="$(cat "$BEFORE_POINTER" 2>/dev/null || true)"
  if [ -z "$before" ]; then
    echo "nothing to undo: the last apply changed nothing, or undo already ran"
    return 0
  fi
  if [ "$before" != none ] && [ ! -d "$before" ]; then
    echo "refusing: the earlier code $before is missing, so nothing was changed"
    exit 1
  fi
  stamp="$(date +%Y%m%d-%H%M%S)"
  if [ "$before" = none ]; then
    systemctl disable --now ai-checker-worker > /dev/null 2>&1 || true
    systemctl disable --now ai-checker-firewall > /dev/null 2>&1 || true
    rm -f "$WORKER_UNIT" "$FIREWALL_UNIT"
    systemctl daemon-reload
    mv "$APP" "$APP-undone-$stamp"
    rm -f "$BEFORE_POINTER" "$FRESH_INSTALL_MARK"
    echo "worker and firewall stopped and removed, code kept in $APP-undone-$stamp, $CONF and the venv and models left in place"
  else
    mv "$APP" "$APP-undone-$stamp"
    mv "$before" "$APP"
    rm -f "$BEFORE_POINTER"
    install_units
    systemctl restart ai-checker-firewall
    systemctl restart ai-checker-worker
    wait_for_listener
    echo "earlier code back in place, undone code kept in $APP-undone-$stamp"
  fi
}

case "${1:-}" in
  backup) do_backup ;;
  apply) do_apply "${2:?apply needs the code bundle}" ;;
  check) do_check ;;
  undo) do_undo ;;
  sums) cd "$APP" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum ;;
  *) echo "unknown ct action"; exit 2 ;;
esac
