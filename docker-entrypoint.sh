#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-help}"
shift || true

case "$CMD" in
  build-index)
    exec python scripts/build_index.py "$@"
    ;;
  query)
    exec python scripts/ask.py "$@"
    ;;
  help)
    echo "Usage: docker run <image> <command> [args]"
    echo ""
    echo "Commands:"
    echo "  build-index  Build a RAG index from a corpus"
    echo "  query        Query a built index"
    echo ""
    echo "Examples:"
    echo "  docker run <image> build-index --corpus /data/vault --index-name my_index"
    echo "  docker run <image> query --index my_index --q 'What is X?'"
    exit 0
    ;;
  *)
    exec "$CMD" "$@"
    ;;
esac