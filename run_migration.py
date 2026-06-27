"""
Run a migration SQL file against Supabase using the service role key.
Usage: python run_migration.py migrations/007_create_faqs_table.sql
"""
import sys
import os
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Load from .env if not in environment
if not SUPABASE_URL or not SUPABASE_KEY:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k == "SUPABASE_URL":
                    SUPABASE_URL = v.strip()
                elif k == "SUPABASE_KEY":
                    SUPABASE_KEY = v.strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
    sys.exit(1)

sql_file = sys.argv[1] if len(sys.argv) > 1 else "migrations/007_create_faqs_table.sql"
sql = open(sql_file).read()

# Split on semicolons to execute each statement individually
statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]

project_ref = SUPABASE_URL.replace("https://", "").split(".")[0]
mgmt_url = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"

# Try Management API first (requires PAT as bearer)
# Fall back to rpc exec approach
headers = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "apikey": SUPABASE_KEY,
}

# Try running full SQL as a single block via Management API
print(f"Running migration: {sql_file}")
resp = requests.post(mgmt_url, json={"query": sql}, headers=headers)
if resp.status_code == 200:
    print("Migration applied successfully via Management API.")
    sys.exit(0)

# Fallback: run each statement via the RPC exec_sql endpoint
print(f"Management API returned {resp.status_code}. Trying RPC exec_sql fallback...")
rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
errors = []
for stmt in statements:
    r = requests.post(rpc_url, json={"sql": stmt}, headers=headers)
    if r.status_code not in (200, 204):
        errors.append(f"  [{r.status_code}] {stmt[:80]}...")

if not errors:
    print("Migration applied successfully via RPC exec_sql.")
    sys.exit(0)

print("\nCould not apply migration automatically.")
print("Please run the following SQL in the Supabase SQL Editor")
print(f"(https://supabase.com/dashboard/project/{project_ref}/sql/new):\n")
print(open(sql_file).read())
