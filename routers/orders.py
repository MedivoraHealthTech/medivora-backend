"""
Orders Router — Patient creates pharmacy / lab orders.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.dependencies import get_current_user
from db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orders", tags=["orders"])


# ── Schemas ────────────────────────────────────────────────────────────

class OrderItem(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    quantity: Optional[int] = 1

class CreateOrderRequest(BaseModel):
    type: str                          # 'pharmacy' | 'lab'
    prescription_id: Optional[str] = None
    items: List[OrderItem]
    delivery_type: Optional[str] = "home"    # 'home' | 'visit'
    delivery_address: Optional[str] = None
    notes: Optional[str] = None


# ── POST /orders ───────────────────────────────────────────────────────

@router.post("")
async def create_order(
    body: CreateOrderRequest,
    current_user: dict = Depends(get_current_user),
):
    """Patient creates a pharmacy or lab order from a prescription."""
    if body.type not in ("pharmacy", "lab"):
        raise HTTPException(status_code=400, detail="type must be 'pharmacy' or 'lab'")

    db = get_db()
    patient_id = current_user.get("sub")

    try:
        # Resolve patient row
        pat = db.client.table("patients").select("id").eq("profile_id", patient_id).limit(1).execute()
        if not pat.data:
            raise HTTPException(status_code=404, detail="Patient profile not found")
        resolved_patient_id = pat.data[0]["id"]

        # Get patient name/phone from profiles
        prof = db.client.table("profiles").select("first_name, last_name, phone").eq("id", patient_id).limit(1).execute()
        profile = prof.data[0] if prof.data else {}
        patient_name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        patient_phone = profile.get("phone", "")

        order_data = {
            "type": body.type,
            "patient_id": resolved_patient_id,
            "prescription_id": body.prescription_id,
            "items": [item.dict() for item in body.items],
            "delivery_type": body.delivery_type,
            "delivery_address": body.delivery_address,
            "notes": body.notes,
            "status": "pending",
            "patient_name": patient_name,
            "patient_phone": patient_phone,
        }

        result = db.client.table("orders").insert(order_data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Could not create order")

        logger.info(f"Order created: type={body.type} patient={patient_id}")
        return {"message": f"{body.type.capitalize()} order placed successfully", "order": result.data[0]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create order error: {e}")
        raise HTTPException(status_code=500, detail="Could not create order")


# ── GET /orders/my ─────────────────────────────────────────────────────

@router.get("/my")
async def my_orders(current_user: dict = Depends(get_current_user)):
    """Patient lists their own orders."""
    db = get_db()
    patient_id = current_user.get("sub")

    try:
        pat = db.client.table("patients").select("id").eq("profile_id", patient_id).limit(1).execute()
        if not pat.data:
            return {"orders": []}
        resolved_patient_id = pat.data[0]["id"]

        result = db.client.table("orders").select("*").eq("patient_id", resolved_patient_id).order("created_at", desc=True).execute()

        # Fetch lab reports for lab orders
        orders = result.data or []
        lab_order_ids = [o["id"] for o in orders if o["type"] == "lab"]
        report_map = {}
        if lab_order_ids:
            reports = db.client.table("lab_reports").select("*").in_("order_id", lab_order_ids).execute()
            for r in (reports.data or []):
                report_map[r["order_id"]] = r

        for o in orders:
            if o["type"] == "lab":
                o["lab_report"] = report_map.get(o["id"])

        return {"orders": orders}
    except Exception as e:
        logger.error(f"My orders error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch orders")
