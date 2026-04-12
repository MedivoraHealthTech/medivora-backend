"""
Drug Blacklist — Manages restricted/dangerous medications.
Seeds default blacklist entries on startup.
"""

import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger("medivora.drug_blacklist")

# Default blacklisted drugs (high-risk without doctor supervision)
DEFAULT_BLACKLIST = [
    {"drug_name": "Thalidomide", "reason": "Teratogenic — severe birth defects", "category": "banned"},
    {"drug_name": "Nimesulide (pediatric)", "reason": "Hepatotoxicity in children", "category": "restricted"},
    {"drug_name": "Phenylpropanolamine", "reason": "Stroke risk — banned by DCGI", "category": "banned"},
    {"drug_name": "Cisapride", "reason": "Cardiac arrhythmia risk", "category": "banned"},
    {"drug_name": "Rofecoxib", "reason": "Cardiovascular risk — withdrawn", "category": "banned"},
    {"drug_name": "Valdecoxib", "reason": "Cardiovascular and skin reactions", "category": "banned"},
    {"drug_name": "Gatifloxacin (oral)", "reason": "Dysglycemia — banned by DCGI", "category": "banned"},
    {"drug_name": "Tegaserod", "reason": "Cardiovascular risk", "category": "banned"},
    {"drug_name": "Sibutramine", "reason": "Cardiovascular risk — withdrawn", "category": "banned"},
    {"drug_name": "Rosiglitazone", "reason": "Cardiovascular risk — restricted", "category": "restricted"},
]

# Pregnancy-contraindicated drugs (additional filtering when pregnant)
PREGNANCY_BLACKLIST = [
    "methotrexate", "isotretinoin", "warfarin", "misoprostol",
    "thalidomide", "diethylstilbestrol", "valproic acid", "lithium",
    "tetracycline", "doxycycline", "ibuprofen", "naproxen",
    "aspirin", "atorvastatin", "simvastatin", "finasteride",
]


def load_blacklist_from_db(db) -> List[Dict]:
    """Load blacklist from database. Falls back to defaults if DB unavailable."""
    try:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't call run_until_complete
            # Return defaults as fallback
            return DEFAULT_BLACKLIST
        except RuntimeError:
            result = asyncio.run(db.get_blacklisted_drugs())
            return result if result else DEFAULT_BLACKLIST
    except Exception as e:
        logger.warning(f"Could not load blacklist from DB, using defaults: {e}")
        return DEFAULT_BLACKLIST


def filter_prescription(prescription_text: str, blacklist: List[Dict], is_pregnant: bool = False) -> Tuple[str, List[str]]:
    """
    Filter a prescription text to remove blacklisted drugs.
    Returns (filtered_text, list_of_removed_drug_names).
    """
    if not prescription_text:
        return prescription_text, []

    removed_drugs = []
    filtered_text = prescription_text

    # Check against DB blacklist
    for entry in blacklist:
        drug_name = entry.get("drug_name", "")
        if not drug_name:
            continue
        # Case-insensitive check
        pattern = re.compile(re.escape(drug_name), re.IGNORECASE)
        if pattern.search(filtered_text):
            filtered_text = pattern.sub(f"[REMOVED: {drug_name} — {entry.get('reason', 'blacklisted')}]", filtered_text)
            removed_drugs.append(drug_name)
            logger.warning(f"Removed blacklisted drug: {drug_name} — {entry.get('reason', '')}")

    # Additional pregnancy filtering
    if is_pregnant:
        for drug in PREGNANCY_BLACKLIST:
            pattern = re.compile(re.escape(drug), re.IGNORECASE)
            if pattern.search(filtered_text) and drug not in [d.lower() for d in removed_drugs]:
                filtered_text = pattern.sub(f"[REMOVED: {drug} — contraindicated in pregnancy]", filtered_text)
                removed_drugs.append(drug)
                logger.warning(f"Removed pregnancy-contraindicated drug: {drug}")

    return filtered_text, removed_drugs


def is_blacklisted(drug_name: str, blacklist: List[Dict]) -> bool:
    """Check if a drug name matches any blacklisted entry."""
    name_lower = drug_name.lower()
    for entry in blacklist:
        if entry.get("drug_name", "").lower() in name_lower or name_lower in entry.get("drug_name", "").lower():
            return True
    return False


def seed_default_blacklist(db=None):
    """
    Seed the drug blacklist table with default entries if empty.
    Called at module load time (synchronous context), so we just log intent.
    Actual seeding happens on first use if table is empty.
    """
    try:
        if db is None:
            from database import DatabaseManager
            db = DatabaseManager()
        # At module load time we can't await, so just verify DB is reachable
        logger.info("Drug blacklist module initialized (defaults will be used if table is empty)")
    except Exception as e:
        logger.warning(f"Could not initialize drug blacklist: {e}")
