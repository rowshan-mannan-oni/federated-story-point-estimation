import sys
from pathlib import Path

# Ensure the project root is on sys.path so that both `import export_issues`
# and `from fl.data import ...` resolve correctly regardless of invocation path.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
