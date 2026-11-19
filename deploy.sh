#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_ENV="${SITE_ENV:-$REPO/deploy/site.env}"
SITE_NAMES=(PROXMOX WORKER_CT APP_CT WORKER_HOST APP_HOST WORKER_PORT APP_PORT LAN_PORT SIGNED_IN_USER)
SITE_VALUE_PATTERN='^[A-Za-z0-9][A-Za-z0-9._:-]*$'
SITE_PATH_PATTERN='^/[A-Za-z0-9._/-]+$'
SITE_URL_PATTERN='^https?://[A-Za-z0-9][A-Za-z0-9._:-]*$'
SITE_SIGNIN="${SITE_SIGNIN:-$HOME/Projects/site-signin}"
WORKER_CT_SCRIPT=/root/ai-checker-deploy-worker.sh
APP_CT_SCRIPT=/root/ai-checker-deploy-app.sh
SECRET_PATH=/etc/ai-checker/worker.secret

usage() {
  cat <<'USAGE'
usage: deploy.sh backup | apply [commit] | check | undo | worker-apply [commit] | app-apply [commit]

Runs from the laptop and reaches the containers through ssh to the Proxmox host
and `pct`. Hosts, containers, ports and the signed-in user come from
deploy/site.env (copy deploy/site.env.example).

  backup          timestamped backups in both containers (mode 600): code, config,
                  units, and on the app side the Caddyfile and sign-in page
  apply [commit]  worker first (code, secret generated on that box, nft rules that let
                  only the app container reach the port, service), then the secret is
                  copied container to container without being shown, then the app
                  (service user, venv, unit, sign-in page, a marked Caddy site on the
                  LAN address only, validated before reload), then both checks
  check           run both checks again
  undo            app first (Caddy block, sign-in page, code, service), then the worker
                  (service, firewall, code); the voice library, models, venv and the
                  secret are never deleted

The worker's Python environment (torch and friends) and the models are prepared
once by hand under WORKER_DATA_ROOT (venv/ and models/ inside it) and are only checked here, never rebuilt.
USAGE
}

site_value() {
  sed -n "s/^$1=//p" "$SITE_ENV" | tail -n 1
}

load_site() {
  local name value
  if [ ! -f "$SITE_ENV" ]; then
    echo "refusing: $SITE_ENV is missing; copy deploy/site.env.example to deploy/site.env and fill it in"
    exit 1
  fi
  for name in "${SITE_NAMES[@]}"; do
    value="${!name:-$(site_value "$name")}"
    if [[ ! "$value" =~ $SITE_VALUE_PATTERN ]]; then
      echo "refusing: $name in $SITE_ENV must be a plain host name, address, number or user name, got '$value'"
      exit 1
    fi
    printf -v "$name" '%s' "$value"
  done
  load_site_value WORKER_DATA_ROOT "$SITE_PATH_PATTERN" required "an absolute path of plain characters"
  load_site_value GPU_LOCK_FILE "$SITE_PATH_PATTERN" optional "an absolute path of plain characters"
  load_site_value CONTROLLER_URL "$SITE_URL_PATTERN" optional "like http://host:port"
}

load_site_value() {
  local name="$1" pattern="$2" need="$3" description="$4" value
  value="${!name:-$(site_value "$name")}"
  if [ -z "$value" ] && [ "$need" = optional ]; then
    printf -v "$name" '%s' ""
    return
  fi
  if [[ ! "$value" =~ $pattern ]]; then
    echo "refusing: $name in $SITE_ENV must be $description, got '$value'"
    exit 1
  fi
  printf -v "$name" '%s' "$value"
}

push_to_ct() {
  local ct="$1" source="$2" target="$3" mode="$4" name
  name="$(basename "$target")"
  scp -q "$source" "$PROXMOX:/root/$name"
  ssh -o BatchMode=yes "$PROXMOX" "pct push $ct /root/$name $target --perms $mode && rm /root/$name"
}

site_arguments() {
  echo "WORKER_HOST=$WORKER_HOST WORKER_PORT=$WORKER_PORT APP_HOST=$APP_HOST APP_PORT=$APP_PORT LAN_PORT=$LAN_PORT SIGNED_IN_USER=$SIGNED_IN_USER WORKER_DATA_ROOT=$WORKER_DATA_ROOT CONTROLLER_URL=$CONTROLLER_URL GPU_LOCK_FILE=$GPU_LOCK_FILE"
}

in_worker() {
  ssh -o BatchMode=yes "$PROXMOX" "pct exec $WORKER_CT -- env $(site_arguments) bash $WORKER_CT_SCRIPT $*"
}

in_app() {
  ssh -o BatchMode=yes "$PROXMOX" "pct exec $APP_CT -- env $(site_arguments) bash $APP_CT_SCRIPT $*"
}

push_ct_scripts() {
  push_to_ct "$WORKER_CT" "$REPO/deploy/ct-worker.sh" "$WORKER_CT_SCRIPT" 700
  push_to_ct "$APP_CT" "$REPO/deploy/ct-app.sh" "$APP_CT_SCRIPT" 700
}

copy_secret_between_containers() {
  ssh -o BatchMode=yes "$PROXMOX" \
    "pct exec $WORKER_CT -- cat $SECRET_PATH | pct exec $APP_CT -- bash -c 'umask 077 && install -d -m 750 /etc/ai-checker && cat > $SECRET_PATH.new && mv $SECRET_PATH.new $SECRET_PATH'"
  echo "secret: copied from container $WORKER_CT to container $APP_CT, not shown"
}

build_signin_bundle() {
  local work="$1"
  [ -f "$SITE_SIGNIN/build.py" ] || { echo "refusing: no site-signin builder at $SITE_SIGNIN (set SITE_SIGNIN)"; exit 1; }
  python3 "$SITE_SIGNIN/build.py" --first AI --second Checker --tagline "Sign in to check and rewrite text." \
    --icon "$REPO/aichecker/web/static/icon.svg" --out "$work/signin"
  tar -czf "$work/signin.tgz" -C "$work/signin" .
}

export_tree() {
  local commit="$1" work="$2"
  mkdir "$work/tree"
  git -C "$REPO" archive --format=tar "$commit" | tar -x -C "$work/tree"
  ( cd "$work/tree" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$work/expected.sums"
  tar -czf "$work/code.tgz" -C "$work/tree" .
}

verify_sums() {
  local run_in="$1" work="$2" commit="$3" where="$4"
  if diff <(sort -k2 "$work/expected.sums") <("$run_in" sums | sort -k2) > /dev/null; then
    echo "sha256: all $(wc -l < "$work/expected.sums") files in $where match $(git -C "$REPO" rev-parse --short "$commit")"
  else
    echo "sha256 MISMATCH between $where and $commit"
    return 1
  fi
}

require_calibration() {
  local commit="$1"
  if ! git -C "$REPO" cat-file -e "$commit:aichecker/data/calibration.json" 2> /dev/null; then
    echo "refusing: $commit has no aichecker/data/calibration.json; calibrate and commit it first"
    exit 1
  fi
}

apply_worker() {
  local work="$1" commit="$2"
  push_to_ct "$WORKER_CT" "$work/code.tgz" /root/ai-checker-code.tgz 600
  if ! in_worker apply /root/ai-checker-code.tgz; then
    echo "worker apply reported failures: read them, then run undo if needed"
    return 1
  fi
  copy_secret_between_containers
  verify_sums in_worker "$work" "$commit" "the worker's /opt/ai-checker"
}

apply_app() {
  local work="$1" commit="$2"
  build_signin_bundle "$work"
  push_to_ct "$APP_CT" "$work/code.tgz" /root/ai-checker-code.tgz 600
  push_to_ct "$APP_CT" "$work/signin.tgz" /root/ai-checker-signin.tgz 600
  if ! in_app apply /root/ai-checker-code.tgz /root/ai-checker-signin.tgz; then
    echo "app apply reported failures: read them, then run undo if needed"
    return 1
  fi
  verify_sums in_app "$work" "$commit" "the app's /opt/ai-checker"
}

run_apply() {
  local part="$1" commit="${2:-HEAD}" work
  load_site
  require_calibration "$commit"
  push_ct_scripts
  work="$(mktemp -d)"
  export_tree "$commit" "$work"
  case "$part" in
    both) in_worker backup && in_app backup && apply_worker "$work" "$commit" && apply_app "$work" "$commit" ;;
    worker) in_worker backup && apply_worker "$work" "$commit" ;;
    app) in_app backup && apply_app "$work" "$commit" ;;
  esac || { rm -rf "$work"; exit 1; }
  rm -rf "$work"
}

main() {
  local action="${1:-}"
  case "$action" in
    backup)
      load_site
      push_ct_scripts
      in_worker backup
      in_app backup
      ;;
    check)
      load_site
      push_ct_scripts
      in_worker check
      in_app check
      ;;
    undo)
      load_site
      push_ct_scripts
      in_app undo
      in_worker undo
      ;;
    apply) run_apply both "${2:-HEAD}" ;;
    worker-apply) run_apply worker "${2:-HEAD}" ;;
    app-apply) run_apply app "${2:-HEAD}" ;;
    -h|--help|help) usage ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
