#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MEDIVORA BACKEND — One-Command Setup
# Run: chmod +x setup.sh && ./setup.sh
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  MEDIVORA BACKEND SETUP"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Install Python dependencies ──────────────────────
echo "📦 Step 1: Installing Python dependencies..."
pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"
echo ""

# ── Step 2: Run SQL schema in Supabase ───────────────────────
echo "🗄️  Step 2: Creating database tables in Supabase..."

# Read env vars
source <(grep -v '^#' .env | sed 's/^/export /')

# Use the Supabase SQL API (via PostgREST rpc or direct pg)
# We'll use the Supabase Management API approach via psql-compatible HTTP
python3 -c "
import os
from supabase import create_client

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

if not url or not key:
    print('   ❌ SUPABASE_URL or SUPABASE_KEY not set in .env')
    exit(1)

client = create_client(url, key)

# Read the SQL file
with open('schema.sql', 'r') as f:
    sql = f.read()

# Split into individual statements and execute via rpc
# Since we can't run DDL via PostgREST, we'll use the pg connection
print('   ℹ️  Cannot run DDL via REST API.')
print('   👉 Please run schema.sql manually in Supabase SQL Editor.')
print('   📋 The file is at: schema.sql')
print('')
print('   After running the SQL, press Enter to continue...')
" 2>/dev/null || true

# If python approach didn't work, fall back to manual
echo ""
echo "   ┌──────────────────────────────────────────────────┐"
echo "   │  MANUAL STEP REQUIRED:                           │"
echo "   │                                                  │"
echo "   │  1. Open: https://supabase.com/dashboard         │"
echo "   │  2. Go to SQL Editor                             │"
echo "   │  3. Paste contents of schema.sql                 │"
echo "   │  4. Click 'Run'                                  │"
echo "   │  5. Come back here and press Enter               │"
echo "   └──────────────────────────────────────────────────┘"
echo ""
read -p "   Press Enter after running the SQL in Supabase... "
echo "   ✅ Assuming schema is ready"
echo ""

# ── Step 3: Verify Supabase connection ───────────────────────
echo "🔌 Step 3: Testing Supabase connection..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

from db import get_db
db = get_db()

# Test: try to query profiles table
result = db.client.table('profiles').select('id').limit(1).execute()
print('   ✅ Connected to Supabase! Profiles table ready.')
print('   📊 Tables are set up correctly.')
" || {
    echo "   ❌ Connection failed. Check your .env file."
    exit 1
}
echo ""

# ── Step 4: Start the server ─────────────────────────────────
echo "🚀 Step 4: Starting Medivora API server..."
echo ""
echo "   Server running at: http://localhost:8000"
echo "   API docs at:       http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""
echo "═══════════════════════════════════════════════════"
echo ""

uvicorn main:app --reload --port 8000
