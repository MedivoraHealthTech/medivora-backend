"""
Portals Router — Pharmacy & Lab portal login and order management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.jwt_handler import create_token
from auth.password_handler import verify_password
from db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["portals"])


# ── Schemas ────────────────────────────────────────────────────────────

class PortalLoginRequest(BaseModel):
    username: str
    password: str

class OrderStatusUpdate(BaseModel):
    status: str
    portal_notes: Optional[str] = None

class InvoiceRequest(BaseModel):
    total_amount: float
    invoice_data: Optional[dict] = None

class ReportRequest(BaseModel):
    report_url: Optional[str] = None
    report_notes: Optional[str] = None
    test_names: Optional[str] = None


# ── Helper: verify portal token has correct portal_type ───────────────

def require_portal(portal_type: str):
    """Returns a dependency that checks role=portal and matching portal_type."""
    async def _dep(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") != "portal":
            raise HTTPException(status_code=403, detail="Portal access required")
        if current_user.get("portal_type") != portal_type:
            raise HTTPException(status_code=403, detail=f"This endpoint is for {portal_type} portal only")
        return current_user
    return _dep


# ── POST /portal/login ────────────────────────────────────────────────

@router.post("/login")
async def portal_login(body: PortalLoginRequest):
    """Login for pharmacy or lab portals. Returns JWT with role=portal."""
    db = get_db()
    try:
        result = db.client.table("portal_users").select("*").eq("username", body.username.strip()).limit(1).execute()
        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        portal_user = result.data[0]
        if not verify_password(body.password, portal_user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_token(
            portal_user["id"],
            role="portal",
            extra={"portal_type": portal_user["portal_type"], "name": portal_user["name"]},
        )
        logger.info(f"Portal login: {body.username} ({portal_user['portal_type']})")
        return {
            "token": token,
            "portal_type": portal_user["portal_type"],
            "name": portal_user["name"],
            "username": portal_user["username"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portal login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


# ── GET /portal/orders ────────────────────────────────────────────────

@router.get("/orders")
async def list_portal_orders(current_user: dict = Depends(get_current_user)):
    """List all orders for this portal type (pharmacy or lab)."""
    if current_user.get("role") != "portal":
        raise HTTPException(status_code=403, detail="Portal access required")

    portal_type = current_user.get("portal_type")
    db = get_db()
    try:
        result = db.client.table("orders").select("*").eq("type", portal_type).order("created_at", desc=True).execute()
        return {"orders": result.data or []}
    except Exception as e:
        logger.error(f"List portal orders error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch orders")


# ── PATCH /portal/orders/{order_id}/status ────────────────────────────

@router.patch("/orders/{order_id}/status")
async def update_order_status(
    order_id: str,
    body: OrderStatusUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Portal updates order status."""
    if current_user.get("role") != "portal":
        raise HTTPException(status_code=403, detail="Portal access required")

    valid_statuses = {
        "pharmacy": ["confirmed", "processing", "out_for_delivery", "delivered", "cancelled"],
        "lab":      ["confirmed", "processing", "sample_collected", "report_ready", "completed", "cancelled"],
    }
    portal_type = current_user.get("portal_type")
    if body.status not in valid_statuses.get(portal_type, []):
        raise HTTPException(status_code=400, detail=f"Invalid status '{body.status}' for {portal_type}")

    db = get_db()
    try:
        update = {"status": body.status}
        if body.portal_notes:
            update["portal_notes"] = body.portal_notes
        result = db.client.table("orders").update(update).eq("id", order_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Order not found")
        return {"message": "Status updated", "order": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update order status error: {e}")
        raise HTTPException(status_code=500, detail="Could not update order")


# ── POST /portal/orders/{order_id}/invoice  (pharmacy) ────────────────

@router.post("/orders/{order_id}/invoice")
async def generate_invoice(
    order_id: str,
    body: InvoiceRequest,
    current_user: dict = Depends(require_portal("pharmacy")),
):
    """Pharmacy generates invoice for an order."""
    db = get_db()
    try:
        update = {
            "total_amount": body.total_amount,
            "invoice_data": body.invoice_data or {},
            "status": "processing",
        }
        result = db.client.table("orders").update(update).eq("id", order_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Order not found")
        return {"message": "Invoice generated", "order": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate invoice error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate invoice")


# ── POST /portal/orders/{order_id}/report  (lab) ─────────────────────

@router.post("/orders/{order_id}/report")
async def upload_report(
    order_id: str,
    body: ReportRequest,
    current_user: dict = Depends(get_current_user),
):
    """Lab uploads report for a test order."""
    if current_user.get("role") != "portal" or current_user.get("portal_type") != "lab":
        raise HTTPException(status_code=403, detail="Lab portal access required")

    db = get_db()
    try:
        # Update order
        order_update = {
            "status": "report_ready",
            "report_url": body.report_url,
            "report_notes": body.report_notes,
        }
        order_result = db.client.table("orders").update(order_update).eq("id", order_id).execute()
        if not order_result.data:
            raise HTTPException(status_code=404, detail="Order not found")

        order = order_result.data[0]

        # Insert lab_report record
        db.client.table("lab_reports").insert({
            "order_id": order_id,
            "patient_id": order["patient_id"],
            "test_names": body.test_names,
            "report_url": body.report_url,
            "report_notes": body.report_notes,
        }).execute()

        return {"message": "Report uploaded and shared with patient", "order": order}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload report error: {e}")
        raise HTTPException(status_code=500, detail="Could not upload report")
