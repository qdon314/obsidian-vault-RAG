#!/usr/bin/env bash
# Launch an ad-hoc query on ECS.
#
# Usage:
#   scripts/ecs_run_query.sh "What is 10 CFR 50.46?"

set -euo pipefail

CLUSTER="${ECS_CLUSTER:-obsidian-rag}"
QUERY_TEXT="${1:-}"

if [[ -z "$QUERY_TEXT" ]]; then
    echo "Usage: $0 \"your query here\""
    exit 1
fi

EVAL_TASK_DEF="${CLUSTER}-query-eval"
WORKER_SERVICE="${CLUSTER}-ingest-worker"

echo "=== Remote Query ==="
echo "Cluster: $CLUSTER"
echo "Query:   $QUERY_TEXT"
echo ""

# Retrieve network config
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

# Launch task with command override to run_remote_query.py
echo ">>> Launching query task..."
OVERRIDES_JSON=$(QUERY_TEXT="$QUERY_TEXT" python3 -c 'import json, os; print(json.dumps({"containerOverrides":[{"name":"query-eval","command":["python","scripts/run_remote_query.py","--query",os.environ["QUERY_TEXT"]]}]}))')
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$EVAL_TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_OVERRIDE" \
    --overrides "$OVERRIDES_JSON" \
    --query 'tasks[0].taskArn' \
    --output text --no-cli-pager)

TASK_ID=$(basename "$TASK_ARN")
echo ">>> Query task: $TASK_ID"

# Tail logs
LOG_GROUP="/ecs/${CLUSTER}/query-eval"
sleep 10
aws logs tail "$LOG_GROUP" --follow --format short &
TAIL_PID=$!

aws ecs wait tasks-stopped \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" 2>/dev/null || true

kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text --no-cli-pager)

echo ">>> Query exited with code: $EXIT_CODE"
exit "${EXIT_CODE:-1}"
