#!/usr/bin/env bash
# Claude Code PreToolUse 훅 — git commit 백프레셔
#
# 동작:
#   1. stdin에서 Claude Code가 전달하는 JSON tool input을 읽는다
#   2. command 필드에 "git commit"이 포함된 경우에만 개입
#   3. pytest를 실행하고, 실패 시 non-zero를 반환해 커밋을 차단 (백프레셔)
#
# 훅 등록 위치: .claude/settings.json > hooks > PreToolUse

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# stdin JSON에서 command 추출
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # 훅 입력 형식: { tool_input: { command: '...' } }
    print(d.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" 2>/dev/null || true)

# git commit 이 아니면 조용히 통과
if ! echo "$COMMAND" | grep -qE "git commit"; then
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [Pre-commit] 테스트 실행 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_DIR"

if uv run pytest tests/ -q --tb=short 2>&1; then
    echo ""
    echo "  ✓ 모든 테스트 통과 — 커밋을 진행합니다."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
else
    echo ""
    echo "  ✗ 테스트 실패 — 커밋이 차단됩니다."
    echo "    실패한 테스트를 수정한 뒤 다시 커밋하세요."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
