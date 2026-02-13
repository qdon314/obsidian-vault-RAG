#!/usr/bin/env bash
# Launch a distributed ingestion run on ECS.
#
# Usage:
#   scripts/ecs_run_ingest.sh --workers 5 --corpus-id regulations_v1 --index-name regulatory
#
# This script:
#   1. Scales the ingest-worker service to desired count
#   2. Launches the ingest-orchestrator task
#   3. Streams CloudWatch logs
#   4. Scales workers back to 0 when the orchestrator exits

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────
CLUSTER="${ECS_CLUSTER:-obsidian-rag}"
WORKERS="${WORKERS:-3}"
CORPUS_ID=""
INDEX_NAME=""
CORPUS_PATH=""
MAX_DOCS=""
EXTRA_ARGS=""

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)      WORKERS="$2"; shift 2 ;;
        --corpus-id)    CORPUS_ID="$2"; shift 2 ;;
        --index-name)   INDEX_NAME="$2"; shift 2 ;;
        --corpus)       CORPUS_PATH="$2"; shift 2 ;;
        --max-docs)     MAX_DOCS="$2"; shift 2 ;;
        --cluster)      CLUSTER="$2"; shift 2 ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$CORPUS_ID" || -z "$INDEX_NAME" ]]; then
    echo "Usage: $0 --corpus-id <id> --index-name <name> [--workers N] [--corpus /path]"
    exit 1
fi

WORKER_SERVICE="${CLUSTER}-ingest-worker"
ORCH_TASK_DEF="${CLUSTER}-ingest-orchestrator"

echo "=== Distributed Ingestion ==="
echo "Cluster:      $CLUSTER"
echo "Workers:      $WORKERS"
echo "Corpus ID:    $CORPUS_ID"
echo "Index Name:   $INDEX_NAME"
echo ""

# ── Step 1: Scale up workers ──────────────────────────────────
echo ">>> Scaling ingest-worker service to $WORKERS..."
aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "$WORKER_SERVICE" \
    --desired-count "$WORKERS" \
    --no-cli-pager > /dev/null

echo ">>> Waiting for workers to stabilize..."
aws ecs wait services-stable \
    --cluster "$CLUSTER" \
    --services "$WORKER_SERVICE"
echo ">>> $WORKERS workers running."

# ── Step 2: Build command override ────────────────────────────
CMD_ARGS="--corpus-id $CORPUS_ID --index-name $INDEX_NAME"
if [[ -n "$CORPUS_PATH" ]]; then
    CMD_ARGS="$CMD_ARGS --corpus $CORPUS_PATH"
else
    CMD_ARGS="$CMD_ARGS --corpus /data/vault"
fi
if [[ -n "$MAX_DOCS" ]]; then
    CMD_ARGS="$CMD_ARGS --max-docs $MAX_DOCS"
fi

# ── Step 3: Retrieve network config from existing worker ──────
NETWORK_CONFIG=$(aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "$WORKER_SERVICE" \
    --query 'services[0].networkConfiguration' \
    --output json --no-cli-pager)

SUBNETS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); print(','.join(nc['awsvpcConfiguration']['subnets']))")
SECURITY_GROUPS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); sgs=nc['awsvpcConfiguration'].get('securityGroups',[]); print(','.join(sgs))" 2>/dev/null || echo "")

NETWORK_OVERRIDE="awsvpcConfiguration={subnets=[$SUBNETS],assignPublicIp=ENABLED"
if [[ -n "$SECURITY_GROUPS" ]]; then
    NETWORK_OVERRIDE="$NETWORK_OVERRIDE,securityGroups=[$SECURITY_GROUPS]"
fi
NETWORK_OVERRIDE="$NETWORK_OVERRIDE}"

# ── Step 4: Launch orchestrator task ──────────────────────────
echo ">>> Launching orchestrator task..."
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$ORCH_TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_OVERRIDE" \
    --overrides "{\"containerOverrides\":[{\"name\":\"ingest-orchestrator\",\"command\":[\"python\",\"scripts/run_orchestrator.py\",$( echo $CMD_ARGS | python3 -c "import sys; args=sys.stdin.read().split(); print(','.join(['\"'+a+'\"' for a in args]))")]}]}" \
    --query 'tasks[0].taskArn' \
    --output text --no-cli-pager)

TASK_ID=$(basename "$TASK_ARN")
echo ">>> Orchestrator task: $TASK_ID"

# ── Step 5: Wait for completion ───────────────────────────────
echo ">>> Waiting for orchestrator to complete (tailing logs)..."
LOG_GROUP="/ecs/${CLUSTER}/ingest-orchestrator"

# Wait briefly for the task to start before tailing
sleep 10

# Tail logs in background
aws logs tail "$LOG_GROUP" --follow --format short &
TAIL_PID=$!

# Wait for the task to stop
aws ecs wait tasks-stopped \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" 2>/dev/null || true

# Stop log tailing
kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

# Check exit code
EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text --no-cli-pager)

echo ""
echo ">>> Orchestrator exited with code: $EXIT_CODE"

# ── Step 6: Scale down workers ────────────────────────────────
echo ">>> Scaling ingest-worker service to 0..."
aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "$WORKER_SERVICE" \
    --desired-count 0 \
    --no-cli-pager > /dev/null

echo "=== Done ==="
exit "${EXIT_CODE:-1}"
