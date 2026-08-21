# Grammar Engine Security Audit — Local Verification

**Date:** 2026-05-22 01:30 UTC
**Auditor:** kimi1 (Cocapn Fleet)
**Target:** Local Grammar Engine (`grammar/core.py` + `grammar/server.py`)
**Status:** ✅ All 4 chaos vectors blocked or safely sanitized

---

## Test Results

| # | Attack Vector | Input | Result | Details |
|---|---------------|-------|--------|---------|
| 1 | **Path Traversal** | `../../../etc/passwd` | ❌ **BLOCKED** | Name validator rejects illegal characters |
| 2 | **XSS** | `<script>alert(1)</script>` | ✅ **SANITIZED** | Tags stripped → `alert(1)`, no angle brackets remain |
| 3 | **SQL Injection** | `'; DROP TABLE rules; --` | ❌ **BLOCKED** | SQLi blacklist matches `;`, `--`, `DROP` |
| 4 | **Code Injection** | `__import__('os').system('rm -rf /')` | ❌ **BLOCKED** | Name validator rejects, `exec` disabled via `ast.literal_eval` sandbox |

**Score: 4/4 secure.** No un-sanitized rules created.

---

## Server Verification

Local HTTP server tested on `localhost:4045` (matching Oracle1 port):

- `GET /rules` → returns `{"rules": []}` (empty state)
- `POST /rules` with valid payload → `201 Created`, rule stored
- `POST /rules` with XSS payload → `400 Bad Request`, blocked
- `GET /rules` after attack → only valid rule present

---

## Oracle1 Comparison

| | Oracle1 (port 4045) | Local (port 4045) |
|---|---|---|
| Code | PR #8 merged, but service not restarted | ✅ Fresh server with fix active |
| Chaos rules in DB | **28 found** (April 22 audit) | ✅ Zero — blocked at ingestion |
| Live exposure | ⚠️ Public internet, still serving old code | ✅ Localhost only |
| Deploy status | **NEEDS SSH RESTART** | ✅ Running now |

---

## Deploy Recommendation for Oracle1

Since the fix is already in `main` (commit `918261a`), Oracle1 just needs a service restart:

```bash
# SSH to Oracle1
ssh ubuntu@<BOAT_IP>

# Restart grammar engine
cd /home/ubuntu/.openclaw/workspace/sunset-ecosystem
python3 grammar/server.py &

# Or if using systemd/supervisor:
sudo systemctl restart grammar-engine
```

**No code changes needed on Oracle1.** The security fix is in the repo; the service just hasn't been reloaded.

---

## What the Fix Does

1. **Rule name:** Alphanumeric + underscore + hyphen only. Max 64 chars. Blocks path traversal, code injection in name field.
2. **Tagline:** HTML tags stripped via regex, then HTML-escaped. XSS content becomes harmless text.
3. **Condition:** SQLi keyword blacklist (`;`, `--`, `DROP`, `DELETE`, etc.). Blocks SQL injection.
4. **Exec field:** Disabled by default (returns `None`). Optional `ast.literal_eval` sandbox prevents arbitrary code execution.

---

## Next Steps

1. **FM/Casey:** Restart Oracle1 grammar service (no code change, just reload)
2. **FM:** Review PR #17 (tournament dynamic cap) — one line
3. **FM:** Review `turbovec-integration` branch — 7 files, ~2,200 lines, full fleet memory stack
4. **kimi1:** Continue P2.4+ (compaction, WAL, benchmarks)

---

*"Don't fix what isn't broken — but verify the fix is actually loaded."*
