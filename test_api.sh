#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MEDIVORA API — Endpoint Tests
# Run this AFTER the server is running (in a separate terminal)
# Usage: chmod +x test_api.sh && ./test_api.sh
# ═══════════════════════════════════════════════════════════════

BASE="http://localhost:8000"
PASS=0
FAIL=0

# Generate random 10-digit phone numbers so tests are idempotent
RAND=$(( RANDOM % 9000 + 1000 ))
PATIENT_PHONE="7${RAND}43210"
DOCTOR_PHONE="8${RAND}43210"
OTP_PHONE="9${RAND}43210"
PASSWORD="test123456"

test_endpoint() {
    local method=$1
    local path=$2
    local data=$3
    local expected_status=$4
    local description=$5
    local token=$6

    local headers=(-H "Content-Type: application/json")
    if [ -n "$token" ]; then
        headers+=(-H "Authorization: Bearer $token")
    fi

    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "${headers[@]}" "$BASE$path" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "${headers[@]}" -d "$data" "$BASE$path" 2>/dev/null)
    fi

    status=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    if [ "$status" == "$expected_status" ]; then
        echo "  ✅ $description (HTTP $status)"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $description — Expected $expected_status, got $status"
        echo "     Response: $(echo "$body" | head -1)"
        FAIL=$((FAIL + 1))
    fi

    echo "$body" > /tmp/medivora_last_response.json
}

echo ""
echo "═══════════════════════════════════════════════════"
echo "  MEDIVORA API TEST SUITE"
echo "  Patient phone: $PATIENT_PHONE"
echo "  Doctor phone:  $DOCTOR_PHONE"
echo "  OTP phone:     $OTP_PHONE"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Health Check ─────────────────────────────────────────────
echo "1️⃣  Health Check"
test_endpoint "GET" "/health" "" "200" "GET /health"
test_endpoint "GET" "/" "" "200" "GET /"
echo ""

# ── Auth: Signup ─────────────────────────────────────────────
echo "2️⃣  Auth: Signup"
test_endpoint "POST" "/auth/signup" \
    "{\"name\":\"Test Patient\",\"phone\":\"$PATIENT_PHONE\",\"password\":\"$PASSWORD\"}" \
    "200" "POST /auth/signup (new patient)"

TOKEN=$(cat /tmp/medivora_last_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
USER_ID=$(cat /tmp/medivora_last_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('user_id',''))" 2>/dev/null)
echo "     Token: ${TOKEN:0:30}..."
echo "     User ID: $USER_ID"

# Duplicate signup should fail
test_endpoint "POST" "/auth/signup" \
    "{\"name\":\"Test Patient\",\"phone\":\"$PATIENT_PHONE\",\"password\":\"$PASSWORD\"}" \
    "409" "POST /auth/signup (duplicate — expect 409)"
echo ""

# ── Auth: Login ──────────────────────────────────────────────
echo "3️⃣  Auth: Login"
test_endpoint "POST" "/auth/login" \
    "{\"phone\":\"$PATIENT_PHONE\",\"password\":\"$PASSWORD\"}" \
    "200" "POST /auth/login (valid credentials)"

test_endpoint "POST" "/auth/login" \
    "{\"phone\":\"$PATIENT_PHONE\",\"password\":\"wrongpassword\"}" \
    "401" "POST /auth/login (wrong password — expect 401)"

test_endpoint "POST" "/auth/login" \
    '{"phone":"1111111111","password":"test123456"}' \
    "401" "POST /auth/login (non-existent — expect 401)"
echo ""

# ── Auth: OTP Flow ───────────────────────────────────────────
echo "4️⃣  Auth: OTP Flow"
test_endpoint "POST" "/auth/send-otp" \
    "{\"phone\":\"$OTP_PHONE\"}" \
    "200" "POST /auth/send-otp"

OTP=$(cat /tmp/medivora_last_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('otp_for_testing',''))" 2>/dev/null)
echo "     OTP (mock): $OTP"

test_endpoint "POST" "/auth/verify-otp" \
    "{\"phone\":\"$OTP_PHONE\",\"otp\":\"$OTP\",\"name\":\"OTP User\"}" \
    "200" "POST /auth/verify-otp (new user auto-create)"

IS_NEW=$(cat /tmp/medivora_last_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_new_user',''))" 2>/dev/null)
echo "     is_new_user: $IS_NEW"

# Verify OTP again (existing user — should login)
test_endpoint "POST" "/auth/send-otp" \
    "{\"phone\":\"$OTP_PHONE\"}" \
    "200" "POST /auth/send-otp (same user again)"
test_endpoint "POST" "/auth/verify-otp" \
    "{\"phone\":\"$OTP_PHONE\",\"otp\":\"123456\"}" \
    "200" "POST /auth/verify-otp (existing user login)"
echo ""

# ── Patient Profile ──────────────────────────────────────────
echo "5️⃣  Patient Profile"
test_endpoint "GET" "/patients/profile" "" "200" "GET /patients/profile" "$TOKEN"

test_endpoint "PUT" "/patients/profile" \
    '{"age":28,"gender":"male","blood_group":"O+","city":"Mumbai","allergies":["Penicillin","Dust"]}' \
    "200" "PUT /patients/profile (update)" "$TOKEN"

test_endpoint "GET" "/patients/profile" "" "200" "GET /patients/profile (after update)" "$TOKEN"
echo ""

# ── Medical Records ──────────────────────────────────────────
echo "6️⃣  Medical Records"
test_endpoint "POST" "/patients/medical-records" \
    '{"record_type":"condition","title":"Seasonal Allergies","description":"Allergic rhinitis during spring","diagnosis":"Allergic Rhinitis","status":"chronic"}' \
    "201" "POST /patients/medical-records (create)" "$TOKEN"

RECORD_ID=$(cat /tmp/medivora_last_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('record',{}).get('id',''))" 2>/dev/null)
echo "     Record ID: $RECORD_ID"

test_endpoint "POST" "/patients/medical-records" \
    '{"record_type":"surgery","title":"Appendectomy","description":"Emergency appendix removal","onset_date":"2023-06-15","status":"resolved"}' \
    "201" "POST /patients/medical-records (second record)" "$TOKEN"

test_endpoint "GET" "/patients/medical-records" "" "200" "GET /patients/medical-records (list)" "$TOKEN"

if [ -n "$RECORD_ID" ]; then
    test_endpoint "GET" "/patients/medical-records/$RECORD_ID" "" "200" "GET /patients/medical-records/:id" "$TOKEN"
fi
echo ""

# ── Auth: No Token ───────────────────────────────────────────
echo "7️⃣  Auth Guards"
test_endpoint "GET" "/patients/profile" "" "401" "GET /patients/profile (no token — expect 401)"
test_endpoint "GET" "/patients/medical-records" "" "401" "GET /patients/medical-records (no token — expect 401)"
echo ""

# ── Validation ───────────────────────────────────────────────
echo "8️⃣  Validation"
test_endpoint "POST" "/auth/signup" \
    '{"name":"X","phone":"123","password":"abc"}' \
    "422" "POST /auth/signup (invalid phone — expect 422)"

test_endpoint "POST" "/auth/signup" \
    '{"name":"Test","phone":"9876543211","password":"abc"}' \
    "422" "POST /auth/signup (weak password — expect 422)"
echo ""

# ── Doctor Signup ────────────────────────────────────────────
echo "9️⃣  Doctor Signup"
test_endpoint "POST" "/auth/signup" \
    "{\"name\":\"Dr. Sharma\",\"phone\":\"$DOCTOR_PHONE\",\"password\":\"doctor12345\",\"user_type\":\"doctor\",\"nmc_number\":\"NMC${RAND}\",\"specialties\":[\"Cardiology\",\"Internal Medicine\"],\"experience_years\":10}" \
    "200" "POST /auth/signup (doctor)"

DOC_TOKEN=$(cat /tmp/medivora_last_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
test_endpoint "GET" "/patients/profile" "" "200" "GET /patients/profile (doctor)" "$DOC_TOKEN"
echo ""

# ── Results ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════════════"
echo ""

rm -f /tmp/medivora_last_response.json

if [ $FAIL -eq 0 ]; then
    echo "  🎉 All tests passed! Backend is working correctly."
    exit 0
else
    echo "  ⚠️  Some tests failed. Check the output above."
    exit 1
fi
