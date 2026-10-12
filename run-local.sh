#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ".venv/bin/python" ]]; then
  echo "请先创建虚拟环境并安装依赖：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  
  source .env
  set +a
fi

export REDIS_URL="redis://:${REDIS_PASSWORD:-echomind123}@localhost:${ECHOMIND_REDIS_PORT:-6381}/0"
export CHROMA_HOST="localhost"
export CHROMA_PORT="8001"
export ECHOMIND_ALLOW_LOCAL_CHROMA_FALLBACK="true"

export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="$NO_PROXY"

export CHROMA_PERSIST_DIRECTORY="./data/chroma"
export ECHOMIND_SKILLS_DIR="./skills"

exec .venv/bin/python -m uvicorn api.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8002}" "$@"
