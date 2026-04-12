"""
PDF Generator — Generates prescription PDFs with verification.
Stub implementation until reportlab/PDF generation is fully configured.
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("medivora.pdf")

PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", "data/prescriptions")


def compute_signature_hash(doctor_id: str, nmc_number: str, timestamp: str, approval_id: str) -> str:
    """Compute a SHA-256 hash for prescription verification."""
    payload = f"{doctor_id}:{nmc_number}:{timestamp}:{approval_id}"
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_prescription_pdf(
    patient_name: str = "",
    patient_age: int = 0,
    patient_gender: str = "",
    doctor_name: str = "",
    doctor_specialty: str = "",
    nmc_number: str = "",
    diagnosis: str = "",
    medications: list = None,
    instructions: str = "",
    approval_id: str = "",
    signature_hash: str = "",
    verification_url: str = "",
    **kwargs,
) -> str:
    """
    Generate a prescription PDF and return the file path.
    Falls back to a text file if reportlab is not available.
    """
    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
    filename = f"prescription_{approval_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(PDF_OUTPUT_DIR, filename)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4
        y = height - 50

        # Company logo — top left
        logo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "frontend", "public", "company-logo.jpeg",
        )
        logo_path = os.path.normpath(logo_path)
        logo_height = 48
        if os.path.exists(logo_path):
            try:
                img = ImageReader(logo_path)
                iw, ih = img.getSize()
                logo_width = logo_height * iw / ih
                c.drawImage(logo_path, 50, y - logo_height, width=logo_width, height=logo_height, mask="auto")
                y -= logo_height + 10
            except Exception:
                pass

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "MEDIVORA — Medical Prescription")
        y -= 30

        c.setFont("Helvetica", 11)
        lines = [
            f"Date: {datetime.now().strftime('%d %b %Y')}",
            f"Approval ID: {approval_id}",
            "",
            f"Patient: {patient_name}   Age: {patient_age}   Gender: {patient_gender}",
            f"Doctor: {doctor_name}   Specialty: {doctor_specialty}",
            f"NMC Registration: {nmc_number}",
            "",
            f"Diagnosis: {diagnosis}",
            "",
            "Medications:",
        ]
        for med in (medications or []):
            if isinstance(med, dict):
                lines.append(f"  - {med.get('name', '')}  {med.get('dosage', '')}  {med.get('frequency', '')}  x{med.get('duration', '')}")
            else:
                lines.append(f"  - {med}")
        lines += [
            "",
            f"Instructions: {instructions}",
            "",
            f"Signature Hash: {signature_hash[:16]}...",
            f"Verify: {verification_url}" if verification_url else "",
        ]

        for line in lines:
            c.drawString(50, y, line)
            y -= 16
            if y < 60:
                c.showPage()
                y = height - 60

        c.save()
        logger.info(f"Prescription PDF generated: {filepath}")

    except ImportError:
        # Fallback: write as text
        filepath = filepath.replace(".pdf", ".txt")
        with open(filepath, "w") as f:
            f.write("MEDIVORA — Medical Prescription\n")
            f.write(f"Approval ID: {approval_id}\n")
            f.write(f"Patient: {patient_name}, Age: {patient_age}\n")
            f.write(f"Doctor: {doctor_name} ({doctor_specialty})\n")
            f.write(f"Diagnosis: {diagnosis}\n")
            f.write(f"Medications: {medications}\n")
            f.write(f"Instructions: {instructions}\n")
        logger.warning(f"reportlab not installed; wrote text file: {filepath}")

    return filepath
