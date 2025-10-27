#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./deploy_cloud_run.sh <PROJECT_ID> <REGION> [SERVICE_NAME] [WEIGHTS_URL]
# Example:
#   ./deploy_cloud_run.sh my-project us-central1 u2net-remover https://storage.googleapis.com/bucket/u2net_clothing.pth

PROJECT_ID=${1:?project id}
REGION=${2:?region}
SERVICE_NAME=${3:-u2net-remover}
WEIGHTS_URL=${4:-}

IMAGE=gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest

echo "Building: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" .

echo "Deploying to Cloud Run: ${SERVICE_NAME} in ${REGION}"
CMD=(gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi --max-instances 3)

if [[ -n "${WEIGHTS_URL}" ]]; then
  CMD+=(--set-env-vars WEIGHTS_URL="${WEIGHTS_URL}")
fi

"${CMD[@]}"

echo "Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)'
