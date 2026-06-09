#!/usr/bin/env bash
#
# deploy.sh — Build and deploy the Video Script Agent to Google Cloud Run.
#
# The whole stack (the Python ADK agent + the Remotion renderer with headless
# Chrome) ships as ONE container, built in the cloud by Cloud Build from the
# repo's Dockerfile and deployed to Cloud Run. On Cloud Run the app authenticates
# as its runtime service account through the metadata server (Application Default
# Credentials) — the same ADC path it uses locally — so no keys are needed.
#
# The ONLY required input is your Google Cloud project id.
#
#   ./deploy.sh PROJECT_ID                 # private (default), 7-scene-ready
#   ./deploy.sh PROJECT_ID --public        # open to the internet (no auth)
#   REGION=europe-west1 ./deploy.sh PROJECT_ID
#
# Run `./deploy.sh --help` for all options.

set -euo pipefail

# --- Defaults (override via env or the matching --flag) ----------------------
REGION="${REGION:-us-central1}"                       # Cloud Run region
VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"     # Vertex AI (Gemini/Imagen) location
SERVICE="${SERVICE:-video-agent}"                     # Cloud Run service name
MEMORY="${MEMORY:-4Gi}"                               # per-instance memory (Chrome render is heavy)
CPU="${CPU:-2}"                                       # per-instance vCPUs
TIMEOUT="${TIMEOUT:-3600}"                            # request timeout, secs (max 3600; a render can take minutes)
MAX_INSTANCES="${MAX_INSTANCES:-1}"                   # scale cap (cost guard)
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-false}"
SKIP_IAM="${SKIP_IAM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

usage() {
  cat <<'EOF'
Deploy the Video Script Agent (ADK + Remotion) to Google Cloud Run.

Usage:
  ./deploy.sh PROJECT_ID [--public]
  ./deploy.sh --project=PROJECT_ID [--region=us-central1] [--service=video-agent]

Arguments:
  PROJECT_ID               Google Cloud project to deploy to (required).

Options:
  --public                 Expose the service publicly (no auth). Default: PRIVATE.
  --region=REGION          Cloud Run region (default: us-central1).
  --service=NAME           Cloud Run service name (default: video-agent).
  --vertex-location=LOC    Vertex AI location for Gemini/Imagen (default: us-central1).
  -h, --help               Show this help.

Environment overrides:
  REGION, VERTEX_LOCATION, SERVICE, MEMORY (4Gi), CPU (2), TIMEOUT (3600),
  MAX_INSTANCES (2), ALLOW_UNAUTHENTICATED (false), SKIP_IAM (false).

NOTE: the deployed adk web UI has NO authentication of its own, and every video
makes real, BILLED Imagen + Cloud TTS calls — so the service is deployed PRIVATE
by default. Reach a private service with an authenticated tunnel:
  gcloud run services proxy SERVICE --region REGION --project PROJECT_ID
EOF
}

# --- Parse arguments ---------------------------------------------------------
PROJECT=""
for arg in "$@"; do
  case "$arg" in
    --public|--allow-unauthenticated) ALLOW_UNAUTHENTICATED="true" ;;
    --project=*)         PROJECT="${arg#*=}" ;;
    --region=*)          REGION="${arg#*=}" ;;
    --service=*)         SERVICE="${arg#*=}" ;;
    --vertex-location=*) VERTEX_LOCATION="${arg#*=}" ;;
    -h|--help)           usage; exit 0 ;;
    --*)                 echo "ERROR: unknown option: $arg" >&2; echo >&2; usage >&2; exit 2 ;;
    *)
      if [ -z "$PROJECT" ]; then
        PROJECT="$arg"
      else
        echo "ERROR: unexpected extra argument: $arg" >&2; exit 2
      fi
      ;;
  esac
done

if [ -z "$PROJECT" ]; then
  echo "ERROR: a Google Cloud project id is required." >&2
  echo >&2
  usage >&2
  exit 2
fi

# --- Preflight ---------------------------------------------------------------
command -v gcloud >/dev/null 2>&1 || {
  echo "ERROR: gcloud CLI not found. Install it: https://cloud.google.com/sdk/docs/install" >&2
  exit 1
}

if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | grep -q .; then
  echo "ERROR: no active gcloud account. Run: gcloud auth login" >&2
  exit 1
fi

if ! gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  echo "ERROR: project '$PROJECT' not found or not accessible by the active account." >&2
  echo "       Check the id and your access (gcloud projects list)." >&2
  exit 1
fi

echo "=================================================================="
echo " Deploying '$SERVICE' to Cloud Run"
echo "   project:         $PROJECT"
echo "   run region:      $REGION"
echo "   vertex location: $VERTEX_LOCATION"
echo "   resources:       cpu=$CPU memory=$MEMORY timeout=${TIMEOUT}s min-instances=0 max-instances=$MAX_INSTANCES"
echo "   public access:   $ALLOW_UNAUTHENTICATED"
echo "=================================================================="

# All gcloud calls target this project and run non-interactively.
GCLOUD=(gcloud "--project=$PROJECT" --quiet)

# --- 1. Enable the APIs the app and the build need (idempotent) --------------
echo "-> Enabling required APIs…"
"${GCLOUD[@]}" services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  texttospeech.googleapis.com

# --- 2. Build (Cloud Build, from the Dockerfile) and deploy ------------------
if [ "$ALLOW_UNAUTHENTICATED" = "true" ]; then
  AUTH_FLAG="--allow-unauthenticated"
else
  AUTH_FLAG="--no-allow-unauthenticated"
fi

echo "-> Building with Cloud Build and deploying to Cloud Run (this can take a few minutes)…"
"${GCLOUD[@]}" run deploy "$SERVICE" \
  --source "$SCRIPT_DIR" \
  --region "$REGION" \
  --platform managed \
  --execution-environment gen2 \
  --port 8080 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --timeout "$TIMEOUT" \
  --min-instances 0 \
  --max-instances "$MAX_INSTANCES" \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$VERTEX_LOCATION" \
  "$AUTH_FLAG"

# --- 3. Give the runtime service account access to Vertex AI -----------------
if [ "$SKIP_IAM" != "true" ]; then
  RUNTIME_SA="$("${GCLOUD[@]}" run services describe "$SERVICE" --region "$REGION" \
      --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || true)"
  if [ -z "$RUNTIME_SA" ]; then
    PROJECT_NUMBER="$("${GCLOUD[@]}" projects describe "$PROJECT" --format='value(projectNumber)')"
    RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
  fi
  echo "-> Granting roles/aiplatform.user to runtime SA: $RUNTIME_SA"
  if ! "${GCLOUD[@]}" projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:${RUNTIME_SA}" \
        --role="roles/aiplatform.user" \
        --condition=None >/dev/null 2>&1; then
    echo "   WARNING: could not grant roles/aiplatform.user (insufficient permission?)." >&2
    echo "   Vertex calls will fail until that SA has it. Grant it manually:" >&2
    echo "     gcloud projects add-iam-policy-binding $PROJECT \\" >&2
    echo "       --member=serviceAccount:${RUNTIME_SA} --role=roles/aiplatform.user" >&2
  fi
fi

# --- 4. Report ---------------------------------------------------------------
URL="$("${GCLOUD[@]}" run services describe "$SERVICE" --region "$REGION" \
        --format="value(status.url)" 2>/dev/null || true)"

echo
echo "=================================================================="
echo " Deployed: $SERVICE"
[ -n "$URL" ] && echo "   URL: $URL"
if [ "$ALLOW_UNAUTHENTICATED" = "true" ]; then
  echo "   This service is PUBLIC (no auth) — anyone with the URL can run it,"
  echo "   and every video is a real, billed call. Open the URL and give the"
  echo "   agent a video title."
else
  echo "   This service is PRIVATE. Reach it with an authenticated tunnel:"
  echo "     gcloud run services proxy $SERVICE --region $REGION --project $PROJECT"
  echo "   then open the http://localhost:8080 it prints."
  echo "   (Re-run with --public to expose it on the internet, or grant another"
  echo "    user roles/run.invoker on the service to let them reach it.)"
fi
echo
echo " Dry-run with ZERO API cost (placeholder media instead of Imagen/TTS):"
echo "   gcloud run services update $SERVICE --region $REGION --project $PROJECT \\"
echo "     --update-env-vars VIDEO_AGENT_FAKE_MEDIA=1"
echo "=================================================================="