#!/usr/bin/env bash
set -euo pipefail
APP=/opt/ai-checker
DATA=/var/lib/ai-checker
CONF=/etc/ai-checker
ENV_FILE="$CONF/app.env"
SECRET_FILE="$CONF/worker.secret"
UNIT=/etc/systemd/system/ai-checker-app.service
SIGNIN=/srv/signin/ai-checker
CADDYFILE=/etc/caddy/Caddyfile
STATE=/root
BACKUP_POINTER="$STATE/ai-checker-app-last-backup"
BEFORE_POINTER="$STATE/ai-checker-app-code-before"
CADDY_CHANGE="$STATE/ai-checker-app-caddy-change"
CADDY_BLOCK_BEFORE="$STATE/ai-checker-app-caddy-block-before"
SIGNIN_BEFORE="$STATE/ai-checker-app-signin-before"
CADDY_BLOCK_BEGIN="# >>> ai-checker block, added and removed by the ai-checker deploy.sh"
CADDY_BLOCK_END="# <<< ai-checker block"
CADDY_BLOCK_SOURCE="$APP/deploy/Caddyfile.ai-checker"
WORKER_HOST="${WORKER_HOST:?WORKER_HOST is missing: deploy.sh passes it from deploy/site.env}"
WORKER_PORT="${WORKER_PORT:?WORKER_PORT is missing: deploy.sh passes it from deploy/site.env}"
APP_HOST="${APP_HOST:?APP_HOST is missing: deploy.sh passes it from deploy/site.env}"
APP_PORT="${APP_PORT:?APP_PORT is missing: deploy.sh passes it from deploy/site.env}"
LAN_PORT="${LAN_PORT:?LAN_PORT is missing: deploy.sh passes it from deploy/site.env}"
SIGNED_IN_USER="${SIGNED_IN_USER:?SIGNED_IN_USER is missing: deploy.sh passes it from deploy/site.env}"
SERVICE_USER=ai-checker
KEPT_CODE_COPIES=3
LOCAL="http://127.0.0.1:$APP_PORT"
LAN_URL="https://$APP_HOST:$LAN_PORT"

fill_in_site_values() {
  sed -e "s/__WORKER_HOST__/$WORKER_HOST/g" -e "s/__WORKER_PORT__/$WORKER_PORT/g" -e "s/__APP_HOST__/$APP_HOST/g" \
      -e "s/__APP_PORT__/$APP_PORT/g" -e "s/__LAN_PORT__/$LAN_PORT/g" -e "s/__SIGNED_IN_USER__/$SIGNED_IN_USER/g" "$@"
}

code_of() {
  curl -s -o /dev/null -w '%{http_code}' "$@"
}

wait_for_health() {
  for _ in $(seq 1 30); do
    if [ "$(code_of "$LOCAL/api/health")" = 200 ]; then
      return 0
    fi
    sleep 1
  done
  echo "AI Checker did not answer /api/health within 30 s"
  return 1
}

do_backup() {
  local stamp backup parts path
  stamp="$(date +%Y%m%d-%H%M%S)"
  backup="$STATE/ai-checker-app-before-$stamp.tgz"
  parts=(etc/caddy/Caddyfile)
  for path in opt/ai-checker etc/ai-checker etc/systemd/system/ai-checker-app.service srv/signin/ai-checker var/lib/ai-checker; do
    if [ -e "/$path" ]; then parts+=("$path"); fi
  done
  ( umask 077 && tar -czf "$backup" -C / --exclude=opt/ai-checker/.venv "${parts[@]}" )
  chmod 600 "$backup"
  echo "$backup" > "$BACKUP_POINTER"
  chmod 600 "$BACKUP_POINTER"
  echo "app backup: $backup ($(du -h "$backup" | cut -f1)), nothing in it is printed"
}

port_is_taken_by_another_program() {
  local listeners main_pid
  listeners="$(ss -ltnp "sport = :$APP_PORT" | grep LISTEN || true)"
  [ -n "$listeners" ] || return 1
  main_pid="$(systemctl show -p MainPID --value ai-checker-app 2>/dev/null || true)"
  case "$main_pid" in ''|0) return 0 ;; esac
  ! grep -q "pid=$main_pid," <<< "$listeners"
}

ensure_user_and_folders() {
  if ! id -u "$SERVICE_USER" > /dev/null 2>&1; then
    useradd --system --home-dir "$DATA" --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    echo "user: made $SERVICE_USER"
  fi
  install -d -m 700 -o "$SERVICE_USER" -g "$SERVICE_USER" "$DATA"
  install -d -m 750 -o root -g "$SERVICE_USER" "$CONF"
}

require_secret() {
  if [ ! -s "$SECRET_FILE" ]; then
    echo "refusing: $SECRET_FILE is missing; deploy.sh copies it from the worker container before apply"
    exit 1
  fi
  chown root:"$SERVICE_USER" "$SECRET_FILE"
  chmod 640 "$SECRET_FILE"
}

pin_proxy_mode_in_older_env() {
  if grep -q '^AUTH_MODE=' "$ENV_FILE"; then
    return
  fi
  printf 'AUTH_MODE=proxy\n' >> "$ENV_FILE"
  echo "env: $ENV_FILE had no AUTH_MODE, so AUTH_MODE=proxy was added (this install sits behind the proxy sign-in)"
}

install_env_file() {
  if [ -f "$ENV_FILE" ]; then
    echo "env: $ENV_FILE kept as it is"
    pin_proxy_mode_in_older_env
    return
  fi
  install -m 640 -o root -g "$SERVICE_USER" /dev/null "$ENV_FILE"
  fill_in_site_values "$APP/deploy/app.env" > "$ENV_FILE"
  echo "env: wrote $ENV_FILE"
}

install_code() {
  local bundle="$1" stamp="$2" fresh="$APP-new-$2"
  mkdir "$fresh"
  if ! { tar -xzf "$bundle" -C "$fresh" --no-same-owner && python3 -m venv "$fresh/.venv" &&
         "$fresh/.venv/bin/pip" install -q --disable-pip-version-check -r "$fresh/requirements-app.txt"; }; then
    rm -rf "$fresh" "$bundle"
    echo "refusing: the new code could not be unpacked or its packages installed; nothing was changed"
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

prune_old_copies() {
  keep_newest_copies "$APP-before-*"
  keep_newest_copies "$APP-undone-*"
  keep_newest_copies "$SIGNIN-before-*"
  keep_newest_copies "$SIGNIN-undone-*"
  keep_newest_copies "$CADDYFILE.bak-ai-checker-[0-9]*"
  keep_newest_copies "$CADDYFILE.bak-ai-checker-undo-*"
}

install_signin_page() {
  local bundle="$1" stamp="$2"
  if [ -d "$SIGNIN" ]; then
    mv "$SIGNIN" "$SIGNIN-before-$stamp"
    echo "$SIGNIN-before-$stamp" > "$SIGNIN_BEFORE"
  else
    echo none > "$SIGNIN_BEFORE"
  fi
  mkdir -p "$SIGNIN"
  tar -xzf "$bundle" -C "$SIGNIN" --no-same-owner
  rm -f "$bundle"
  chmod -R u=rwX,go=rX "$SIGNIN"
}

marker_count() {
  grep -cxF -- "$1" "$CADDYFILE" || true
}

own_block_state() {
  local begins ends
  begins="$(marker_count "$CADDY_BLOCK_BEGIN")"
  ends="$(marker_count "$CADDY_BLOCK_END")"
  if [ "$begins" = 1 ] && [ "$ends" = 1 ]; then
    echo one
  elif [ "$begins" = 0 ] && [ "$ends" = 0 ]; then
    echo none
  else
    echo damaged
  fi
}

print_own_block() {
  awk -v begin="$CADDY_BLOCK_BEGIN" -v end="$CADDY_BLOCK_END" '
    $0 == begin { inside = 1 }
    inside { print }
    $0 == end { inside = 0 }
  ' "$CADDYFILE"
}

render_caddy_block() {
  printf '%s\n' "$CADDY_BLOCK_BEGIN"
  sed -e '/./,$!d' "$CADDY_BLOCK_SOURCE" | fill_in_site_values
  printf '%s\n' "$CADDY_BLOCK_END"
}

caddyfile_without_own_block() {
  awk -v begin="$CADDY_BLOCK_BEGIN" -v end="$CADDY_BLOCK_END" '
    $0 == begin { inside = 1; held = 0; next }
    inside { if ($0 == end) inside = 0; next }
    held { print ""; held = 0 }
    $0 == "" { held = 1; next }
    { print }
    END { if (held) print "" }
  ' "$CADDYFILE"
}

caddyfile_with_own_block_from() {
  awk -v begin="$CADDY_BLOCK_BEGIN" -v end="$CADDY_BLOCK_END" -v replacement="$1" '
    $0 == begin { inside = 1; while ((getline line < replacement) > 0) print line; next }
    inside { if ($0 == end) inside = 0; next }
    { print }
  ' "$CADDYFILE"
}

caddyfile_is_valid() {
  local report
  if ! command -v caddy > /dev/null 2>&1; then
    echo "caddy: no caddy command here, the Caddyfile was not validated"
    return 1
  fi
  if ! report="$(caddy validate --config "$CADDYFILE" --adapter caddyfile 2>&1)"; then
    printf '%s\n' "$report" | tail -n 5
    return 1
  fi
}

switch_caddyfile() {
  local next="$1" backup="$2"
  cp -a "$CADDYFILE" "$backup" || return 1
  cat "$next" > "$CADDYFILE"
  if ! caddyfile_is_valid; then
    cat "$backup" > "$CADDYFILE"
    echo "caddy: the changed Caddyfile did not validate, so it was put back from $backup"
    return 1
  fi
  if ! systemctl reload caddy; then
    cat "$backup" > "$CADDYFILE"
    echo "caddy: the reload failed, so the Caddyfile was put back from $backup; Caddy keeps the config it was running"
    return 1
  fi
}

add_caddy_site() {
  local stamp="$1" work change
  rm -f "$CADDY_CHANGE" "$CADDY_BLOCK_BEFORE"
  work="$(mktemp -d)"
  case "$(own_block_state)" in
    one)
      render_caddy_block > "$work/block"
      if [ "$(print_own_block)" = "$(cat "$work/block")" ]; then
        rm -rf "$work"
        echo "caddy: the marked ai-checker block is up to date, left as is"
        return 0
      fi
      print_own_block > "$work/before"
      caddyfile_with_own_block_from "$work/block" > "$work/next"
      change=replaced
      ;;
    none)
      { cat "$CADDYFILE"; printf '\n'; render_caddy_block; } > "$work/next"
      change=added
      ;;
    *)
      rm -rf "$work"
      echo "caddy: the ai-checker markers in $CADDYFILE are damaged (not one begin and one end line); fix them by hand"
      return 1
      ;;
  esac
  if ! switch_caddyfile "$work/next" "$CADDYFILE.bak-ai-checker-$stamp"; then
    rm -rf "$work"
    return 1
  fi
  if [ "$change" = replaced ]; then
    mv "$work/before" "$CADDY_BLOCK_BEFORE"
  fi
  echo "$change" > "$CADDY_CHANGE"
  rm -rf "$work"
  echo "caddy: $change the marked ai-checker block for $LAN_URL and reloaded (earlier file: $CADDYFILE.bak-ai-checker-$stamp)"
}

undo_caddy_site() {
  local change state stamp work
  change="$(cat "$CADDY_CHANGE" 2>/dev/null || true)"
  [ -n "$change" ] || return 0
  state="$(own_block_state)"
  if [ "$state" = none ]; then
    rm -f "$CADDY_CHANGE" "$CADDY_BLOCK_BEFORE"
    echo "caddy: the marked ai-checker block is already gone, Caddyfile left as is"
    return 0
  fi
  if [ "$state" != one ]; then
    echo "caddy: the ai-checker markers in $CADDYFILE are damaged; nothing changed, fix them by hand"
    return 1
  fi
  stamp="$(date +%Y%m%d-%H%M%S)"
  work="$(mktemp -d)"
  case "$change" in
    added) caddyfile_without_own_block > "$work/next" ;;
    replaced) caddyfile_with_own_block_from "$CADDY_BLOCK_BEFORE" > "$work/next" ;;
    *)
      rm -rf "$work"
      echo "caddy: $CADDY_CHANGE holds '$change', which this script does not know; nothing changed"
      return 1
      ;;
  esac
  if ! switch_caddyfile "$work/next" "$CADDYFILE.bak-ai-checker-undo-$stamp"; then
    rm -rf "$work"
    return 1
  fi
  rm -rf "$work"
  rm -f "$CADDY_CHANGE" "$CADDY_BLOCK_BEFORE"
  echo "caddy: undid the $change ai-checker block only, other sites untouched (earlier file: $CADDYFILE.bak-ai-checker-undo-$stamp)"
}

lan_answer() {
  curl -sk -o /dev/null -w '%{http_code} %{redirect_url}' --resolve "$APP_HOST:$LAN_PORT:127.0.0.1" "$@" "$LAN_URL/"
}

worker_answer() {
  curl -s -o /dev/null -w '%{http_code}' -m 10 -H "X-Checker-Secret: $(cat "$SECRET_FILE")" "http://$WORKER_HOST:$WORKER_PORT/v1/health" || true
}

worker_answer_without_secret() {
  curl -s -o /dev/null -w '%{http_code}' -m 10 "http://$WORKER_HOST:$WORKER_PORT/v1/health" || true
}

do_check() {
  local since="${1:-}" failures=0 health direct plain dummy signin tracebacks worker_with worker_without public_names auth_mode
  echo "service: $(systemctl is-active ai-checker-app || true)"
  health="$(curl -s "$LOCAL/api/health" || true)"
  echo "health: $health"
  direct="$(code_of "$LOCAL/api/session")"
  echo "app without the user header: $direct (401 expected)"
  plain="$(lan_answer)"
  dummy="$(lan_answer -H 'Cookie: starfire_session=not-a-session')"
  signin="$(code_of -k --resolve "$APP_HOST:$LAN_PORT:127.0.0.1" "$LAN_URL/signin/")"
  echo "through Caddy on $LAN_URL: no cookie -> $plain, made-up cookie -> $dummy, /signin/ -> $signin"
  worker_with="$(worker_answer)"
  worker_without="$(worker_answer_without_secret)"
  echo "worker from this container: with the secret -> $worker_with (200 expected), without -> $worker_without (401 expected)"
  public_names="$(grep -c "ai-checker" "$CADDYFILE" || true)"
  echo "ai-checker lines in the Caddyfile: $public_names (the marked block only, no public host name: $(awk -v b="$CADDY_BLOCK_BEGIN" -v e="$CADDY_BLOCK_END" '$0==b{i=1} i && /^http:\/\//{n++} $0==e{i=0} END{print n+0}' "$CADDYFILE") http:// sites inside it, 0 expected)"
  auth_mode="$(sed -n 's/^AUTH_MODE=//p' "$ENV_FILE" | tail -n 1)"
  echo "sign-in mode in the env file: ${auth_mode:-none} (proxy expected behind the proxy sign-in)"
  echo "modes: env $(stat -c '%a %U:%G' "$ENV_FILE"), secret $(stat -c '%a %U:%G' "$SECRET_FILE"), data $(stat -c '%a %U:%G' "$DATA")"
  [ "$(systemctl is-active ai-checker-app || true)" = active ] || failures=$((failures + 1))
  case "$health" in *'"ok":true'*|*'"ok": true'*) ;; *) failures=$((failures + 1)) ;; esac
  [ "$direct" = 401 ] || failures=$((failures + 1))
  [ "$auth_mode" = proxy ] || failures=$((failures + 1))
  for answer in "$plain" "$dummy"; do
    case "$answer" in "303 "*"/signin/") ;; *) failures=$((failures + 1)) ;; esac
  done
  [ "$signin" = 200 ] || failures=$((failures + 1))
  [ "$worker_with" = 200 ] || failures=$((failures + 1))
  [ "$worker_without" = 401 ] || failures=$((failures + 1))
  [ "$(stat -c '%a' "$ENV_FILE")" = 640 ] || failures=$((failures + 1))
  [ "$(stat -c '%a' "$SECRET_FILE")" = 640 ] || failures=$((failures + 1))
  [ "$(stat -c '%a' "$DATA")" = 700 ] || failures=$((failures + 1))
  if [ -n "$since" ]; then
    tracebacks="$(journalctl -u ai-checker-app --since "@$since" --no-pager | grep -c Traceback || true)"
    echo "tracebacks in the journal since restart: $tracebacks"
    [ "$tracebacks" = 0 ] || failures=$((failures + 1))
  fi
  echo "app check failures: $failures"
  [ "$failures" = 0 ]
}

do_apply() {
  local code_bundle="$1" signin_bundle="$2" stamp since
  [ -s "$BACKUP_POINTER" ] && [ -s "$(cat "$BACKUP_POINTER")" ] || { echo "refusing: run backup first"; exit 1; }
  rm -f "$BEFORE_POINTER" "$CADDY_CHANGE" "$CADDY_BLOCK_BEFORE" "$SIGNIN_BEFORE"
  if port_is_taken_by_another_program; then
    echo "refusing: something other than ai-checker listens on port $APP_PORT"
    exit 1
  fi
  stamp="$(date +%Y%m%d-%H%M%S)"
  ensure_user_and_folders
  require_secret
  install_code "$code_bundle" "$stamp"
  install_env_file
  install -m 644 "$APP/deploy/ai-checker-app.service" "$UNIT"
  systemctl daemon-reload
  systemctl enable ai-checker-app > /dev/null
  since="$(date +%s)"
  systemctl restart ai-checker-app
  wait_for_health
  install_signin_page "$signin_bundle" "$stamp"
  add_caddy_site "$stamp"
  prune_old_copies
  sleep 1
  echo "earlier code: $(cat "$BEFORE_POINTER")"
  do_check "$since"
}

undo_signin_page() {
  local stamp="$1" before
  before="$(cat "$SIGNIN_BEFORE" 2>/dev/null || true)"
  [ -n "$before" ] || return 0
  if [ "$before" != none ] && [ ! -d "$before" ]; then
    rm -f "$SIGNIN_BEFORE"
    echo "sign-in page: the earlier copy $before is gone, the current page stays"
    return 0
  fi
  if [ -d "$SIGNIN" ]; then
    mv "$SIGNIN" "$SIGNIN-undone-$stamp"
  fi
  if [ "$before" != none ]; then
    mv "$before" "$SIGNIN"
  fi
  rm -f "$SIGNIN_BEFORE"
  echo "sign-in page: earlier version back in place (undone one kept in $SIGNIN-undone-$stamp)"
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
  if ! undo_caddy_site; then
    echo "undo stopped before touching the code; the Caddyfile is as it was before undo"
    exit 1
  fi
  undo_signin_page "$stamp"
  if [ "$before" = none ]; then
    systemctl disable --now ai-checker-app > /dev/null 2>&1 || true
    rm -f "$UNIT"
    systemctl daemon-reload
    mv "$APP" "$APP-undone-$stamp"
    rm -f "$BEFORE_POINTER"
    echo "service stopped and removed, code kept in $APP-undone-$stamp, voice library left in $DATA"
  else
    mv "$APP" "$APP-undone-$stamp"
    mv "$before" "$APP"
    rm -f "$BEFORE_POINTER"
    install -m 644 "$APP/deploy/ai-checker-app.service" "$UNIT"
    systemctl daemon-reload
    systemctl restart ai-checker-app
    wait_for_health
    echo "earlier code back in place, undone code kept in $APP-undone-$stamp"
  fi
  prune_old_copies
}

case "${1:-}" in
  backup) do_backup ;;
  apply) do_apply "${2:?apply needs the code bundle}" "${3:?apply needs the sign-in bundle}" ;;
  check) do_check ;;
  undo) do_undo ;;
  sums) cd "$APP" && find . -type f ! -path './.venv/*' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum ;;
  *) echo "unknown ct action"; exit 2 ;;
esac
