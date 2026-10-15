#!/bin/bash
# The TensorFold prompt cache on disk: points, pins, sizes per model key (read-only).
DB="${1:-$HOME/recipes/prompt-cache/store}/index.sqlite"
[ -f "$DB" ] || { echo "no store at $DB"; exit 1; }
python3 - "$DB" <<'EOF'
import sqlite3, sys, time
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
now = time.time()
print(f"{'model':42} {'key':16} {'kind':7} {'points':>6} {'GiB':>7} {'hits':>6} {'oldest h':>8} {'last use h':>10}")
for label, key, pinned, n, b, hits, created, used in db.execute(
        "SELECT label, model, pinned, COUNT(*), SUM(bytes), SUM(hits), MIN(created), MAX(used) "
        "FROM points GROUP BY label, model, pinned ORDER BY label, pinned"):
    print(f"{label[:42]:42} {key:16} {'pinned' if pinned else 'normal':7} {n:6} {b / 2**30:7.2f} {hits:6} "
          f"{(now - created) / 3600:8.1f} {(now - used) / 3600:10.1f}")
blk = db.execute("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM blocks").fetchone()
pts = db.execute("SELECT COALESCE(SUM(bytes),0) FROM points").fetchone()[0]
print(f"blocks: {blk[0]} ({blk[1] / 2**30:.2f} GiB); total on disk {(blk[1] + pts) / 2**30:.2f} GiB")
EOF
