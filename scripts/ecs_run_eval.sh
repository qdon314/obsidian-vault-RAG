#!/usr/bin/env bash
# Launch an eval run on ECS.
#
# Usage:
#   scripts/ecs_run_eval.sh [--query-set default] [--run-name my-run]

set -euo pipefail

CLUSTER="${ECS_CLUSTER:-obsidian-rag}"
QUERY_SET="${QUERY_SET:-default}"
RUN_NAME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --query-set)  QUERY_SET="$2"; shift 2 ;;
        --run-name)   RUN_NAME="$2"; shift 2 ;;
        --cluster)    CLUSTER="$2"; shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

EVAL_TASK_DEF="${CLUSTER}-query-eval"
WORKER_SERVICE="${CLUSTER}-ingest-worker"

echo "=== Remote Eval ==="
echo "Cluster:    $CLUSTER"
echo "Query Set:  $QUERY_SET"
echo "Run Name:   ${RUN_NAME:-<auto>}"
echo ""

# Build command
CMD_ARGS="--query-set $QUERY_SET"
if [[ -n "$RUN_NAME" ]]; then
    CMD_ARGS="$CMD_ARGS --run-name $RUN_NAME"
fi

CMD_ARGS="$CMD_ARGS $EXTRA_ARGS"

# Retrieve network config from existing service
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

# Launch task
echo ">>> Launching eval task..."
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$EVAL_TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_OVERRIDE" \
    --overrides "{\"containerOverrides\":[{\"name\":\"query-eval\",\"command\":[\"python\",\"scripts/run_remote_eval.py\",$( echo $CMD_ARGS | python3 -c "import sys; args=sys.stdin.read().split(); print(','.join(['\"'+a+'\"' for a in args]))")]}]}" \
    --query 'tasks[0].taskArn' \
    --output text --no-cli-pager)

TASK_ID=$(basename "$TASK_ARN")
echo ">>> Eval task: $TASK_ID"

# Poll until task stops (configurable via EVAL_TIMEOUT, default 30 min)
TIMEOUT="${EVAL_TIMEOUT:-1800}"
INTERVAL=30
ELAPSED=0

while true; do
    STATUS=$(aws ecs describe-tasks \
        --cluster "$CLUSTER" \
        --tasks "$TASK_ARN" \
        --query 'tasks[0].lastStatus' \
        --output text --no-cli-pager)

    if [[ "$STATUS" == "STOPPED" ]]; then
        echo ">>> Task stopped after ${ELAPSED}s"
        break
    fi

    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo ">>> ERROR: Task did not stop within ${TIMEOUT}s (last status: $STATUS)"
        exit 1
    fi

    echo ">>> Task status: $STATUS (${ELAPSED}s / ${TIMEOUT}s)"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

# Retrieve exit code and stop reason
EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text --no-cli-pager)

STOP_REASON=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].stoppedReason' \
    --output text --no-cli-pager)

echo ">>> Eval exited with code: ${EXIT_CODE:-None}"
if [[ -n "$STOP_REASON" && "$STOP_REASON" != "None" ]]; then
    echo ">>> Stop reason: $STOP_REASON"
fi

# Print recent logs (last 100 lines) for visibility
LOG_GROUP="/ecs/${CLUSTER}/query-eval"
echo ""
echo ">>> Recent logs:"
aws logs tail "$LOG_GROUP" --since 30m --format short 2>/dev/null | tail -100 || true
echo ""

if [[ "$EXIT_CODE" == "None" || -z "$EXIT_CODE" ]]; then
    echo ">>> ERROR: Container exit code unavailable (task may have been killed by ECS)"
    exit 1
fi

exit "$EXIT_CODE"
