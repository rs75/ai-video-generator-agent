#!/bin/sh
set -e

# If GOOGLE_CLOUD_PROJECT wasn't passed in, derive it from the gcloud config you
# mounted at /root/.config/gcloud — the same value as `gcloud config get-value
# project` on your host. This lets `docker run ... video-agent` work without an
# explicit -e flag, while keeping no project id baked into the image.
if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
  gdir="${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}"
  active="default"
  [ -f "$gdir/active_config" ] && active="$(cat "$gdir/active_config")"
  cfg="$gdir/configurations/config_$active"
  if [ -f "$cfg" ]; then
    GOOGLE_CLOUD_PROJECT="$(sed -n 's/^[[:space:]]*project[[:space:]]*=[[:space:]]*//p' "$cfg" | head -n1)"
  fi
  # Fall back to the ADC file's quota project.
  if [ -z "$GOOGLE_CLOUD_PROJECT" ] && [ -f "$gdir/application_default_credentials.json" ]; then
    GOOGLE_CLOUD_PROJECT="$(python3 -c "import json;print(json.load(open('$gdir/application_default_credentials.json')).get('quota_project_id') or '')" 2>/dev/null || true)"
  fi
  export GOOGLE_CLOUD_PROJECT
fi

if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
  echo "ERROR: GOOGLE_CLOUD_PROJECT is not set and could not be derived from the" >&2
  echo "mounted gcloud config. On your host run:" >&2
  echo "    gcloud config set project YOUR_PROJECT_ID" >&2
  echo "or pass it explicitly:" >&2
  echo "    docker run ... -e GOOGLE_CLOUD_PROJECT=your-project-id ..." >&2
  exit 1
fi

echo "-> Using Google Cloud project: $GOOGLE_CLOUD_PROJECT"
exec "$@"
