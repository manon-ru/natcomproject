## [2026-05-12] Session start
### Project conventions
- Python 3, uv-managed. Run with `uv run` or `python` after `sys.path.insert(0,'src')`.
- All src modules imported via `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))`.
- No pandas, no scipy, no logging framework, no tqdm. numpy + matplotlib only.
- No abstract base classes, no dataclasses, no type stubs. Match existing sparse annotation style.
- Existing algorithm result dicts MUST keep schema: `{"success", "iterations", "path", "snapshot", "snapshot_history", "disruption_iteration", "history"}`.
- Commits are atomic per task, conventional-commit style.
- Working dir: /Users/robert/Code/natcomproject
- Target branch: proposal-alignment (create from main)

## [2026-05-12] Config constants added
- Added src/config.py with proposal-aligned constants and exact dict/list values.
- Verified importability with the required Python assertion command; result: OK.
- LSP diagnostics on src/config.py returned clean.
