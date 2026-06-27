"""
PDF Generator — Medivora prescription with branded letterhead.
"""

import hashlib
import logging
import os
from datetime import datetime

logger = logging.getLogger("medivora.pdf")

PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", "data/prescriptions")

# ── Brand colours (r, g, b  0-1) ─────────────────────────────────────────────
NAVY       = (0.099, 0.188, 0.667)   # #1930AA
STEEL_BLUE = (0.290, 0.537, 0.780)   # #4A89C7
WHITE      = (1.0,   1.0,   1.0)
DARK       = (0.067, 0.067, 0.067)   # #111111
MUTED      = (0.267, 0.267, 0.267)   # #444444
LIGHT_RULE = (0.800, 0.800, 0.800)   # #CCCCCC
ROW_ALT    = (0.941, 0.949, 1.000)   # #F0F2FF
FOOTER_BG  = (0.176, 0.188, 0.259)   # #2D3042
WARN_RED   = (0.70,  0.10,  0.00)

# ── Contact details ───────────────────────────────────────────────────────────
PHONE   = "+91 99716 15161"
EMAIL   = "nikhil.syal@themedivora.com"
WEBSITE = "www.themedivora.com"
ADDRESS = ("Unit No 43, First Floor, M2K Corporate Park "
           "Sector 51, Gurugram, Haryana 122 003")


def compute_signature_hash(doctor_id: str, nmc_number: str,
                           timestamp: str, approval_id: str) -> str:
    payload = f"{doctor_id}:{nmc_number}:{timestamp}:{approval_id}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _wrap(text: str, max_chars: int) -> list[str]:
    """Word-wrap text into lines of at most max_chars each."""
    words = str(text).split()
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word) if cur else word
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            # If single word exceeds max_chars, hard-break it
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _extract_from_approval(approval_data: dict, doctor_data: dict) -> dict:
    doc  = doctor_data  or {}
    appr = approval_data or {}
    rx   = (appr.get("modified_prescription")
            or appr.get("proposed_prescription") or {})
    ai   = appr.get("ai_assessment") or {}
    pat  = appr.get("patient_profile") or {}

    meds = []
    for m in (rx.get("medicines") or rx.get("medications") or []):
        meds.append({
            "name":         m.get("medicine_name") or m.get("name", ""),
            "generic_name": m.get("generic_name", ""),
            "dosage":       m.get("dosage") or m.get("strength", ""),
            "frequency":    m.get("frequency", ""),
            "duration":     m.get("duration", ""),
            "instructions": m.get("instructions", ""),
            "before_food":  m.get("before_food"),
        })

    return dict(
        patient_name           = pat.get("name")    or appr.get("patient_name", ""),
        patient_age            = pat.get("age")     or appr.get("patient_age", 0),
        patient_gender         = pat.get("gender")  or appr.get("patient_gender", ""),
        doctor_name            = doc.get("name")    or doc.get("full_name", ""),
        doctor_specialty       = doc.get("specialty") or doc.get("specialization", ""),
        nmc_number             = appr.get("nmc_number") or doc.get("nmc_number", ""),
        diagnosis              = rx.get("diagnosis") or ai.get("diagnosis", ""),
        medications            = meds,
        general_instructions   = rx.get("general_instructions", []),
        dietary_advice         = rx.get("dietary_advice", []),
        warning_signs          = rx.get("warning_signs", []),
        follow_up_instructions = rx.get("follow_up_instructions", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────

def generate_prescription_pdf(
    patient_name: str           = "",
    patient_age: int            = 0,
    patient_gender: str         = "",
    doctor_name: str            = "",
    doctor_specialty: str       = "",
    nmc_number: str             = "",
    diagnosis: str              = "",
    medications: list           = None,
    instructions: str           = "",
    general_instructions: list  = None,
    dietary_advice: list        = None,
    warning_signs: list         = None,
    follow_up_instructions: str = "",
    approval_id: str            = "",
    signature_hash: str         = "",
    verification_url: str       = "",
    approval_data: dict         = None,
    doctor_data: dict           = None,
    is_provisional: bool        = False,
    **kwargs,
) -> str:

    if approval_data or doctor_data:
        flat = _extract_from_approval(approval_data or {}, doctor_data or {})
        patient_name           = flat["patient_name"]           or patient_name
        patient_age            = flat["patient_age"]            or patient_age
        patient_gender         = flat["patient_gender"]         or patient_gender
        doctor_name            = flat["doctor_name"]            or doctor_name
        doctor_specialty       = flat["doctor_specialty"]       or doctor_specialty
        nmc_number             = flat["nmc_number"]             or nmc_number
        diagnosis              = flat["diagnosis"]              or diagnosis
        medications            = flat["medications"]            or medications
        general_instructions   = flat["general_instructions"]   or general_instructions
        dietary_advice         = flat["dietary_advice"]         or dietary_advice
        warning_signs          = flat["warning_signs"]          or warning_signs
        follow_up_instructions = flat["follow_up_instructions"] or follow_up_instructions

    medications          = list(medications          or [])
    general_instructions = list(general_instructions or [])
    dietary_advice       = list(dietary_advice       or [])
    warning_signs        = list(warning_signs        or [])
    if instructions and instructions not in general_instructions:
        general_instructions.insert(0, instructions)

    # Filter out medicines with no name
    medications = [m for m in medications
                   if (isinstance(m, dict) and (m.get("medicine_name") or m.get("name", "")).strip())
                   or (not isinstance(m, dict) and str(m).strip())]

    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prescription_{approval_id or 'rx'}_{ts}.pdf"
    filepath = os.path.join(PDF_OUTPUT_DIR, filename)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.utils import ImageReader

        W, H = A4   # 595.27 × 841.89 pt
        c = rl_canvas.Canvas(filepath, pagesize=A4)

        def f(*rgb):  c.setFillColorRGB(*rgb)
        def s(*rgb):  c.setStrokeColorRGB(*rgb)

        logo_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "frontend", "public", "company-logo.jpeg",
        ))

        ML = 35
        MR = W - 35

        # ── HEADER BAND ──────────────────────────────────────────────────────
        HDR_H   = 78
        SPLIT_X = 95   # narrower navy block — just wide enough for the logo

        f(*NAVY)
        c.rect(0, H - HDR_H, SPLIT_X, HDR_H, fill=1, stroke=0)

        f(*STEEL_BLUE)
        c.rect(SPLIT_X, H - HDR_H, W - SPLIT_X, HDR_H, fill=1, stroke=0)

        if os.path.exists(logo_path):
            try:
                img    = ImageReader(logo_path)
                iw, ih = img.getSize()
                lh     = 48
                lw     = lh * iw / ih
                lw     = min(lw, SPLIT_X - 14)   # never overflow the navy block
                c.drawImage(logo_path,
                            (SPLIT_X - lw) / 2, H - HDR_H + (HDR_H - lh) / 2,
                            width=lw, height=lh, mask="auto")
            except Exception:
                pass

        f(*DARK)
        dr_label = doctor_name if doctor_name.startswith("Dr") else f"Dr. {doctor_name}"
        c.setFont("Helvetica-Bold", 13)
        c.drawRightString(W - 18, H - HDR_H + 52, dr_label)

        f(*MUTED)
        c.setFont("Helvetica", 10)
        c.drawRightString(W - 18, H - HDR_H + 36, doctor_specialty or "General Physician")

        c.setFont("Helvetica", 9)
        if nmc_number:
            c.drawRightString(W - 18, H - HDR_H + 20, f"Reg No: {nmc_number}")

        # ── DOUBLE RULE ───────────────────────────────────────────────────────
        s(*LIGHT_RULE)
        c.setLineWidth(0.6)
        c.line(0, H - HDR_H - 6,  W, H - HDR_H - 6)
        c.line(0, H - HDR_H - 12, W, H - HDR_H - 12)

        # ── WATERMARK  ("Rx" in light grey) ──────────────────────────────────
        c.saveState()
        f(0.91, 0.91, 0.93)
        c.setFont("Helvetica-Bold", 180)
        c.drawCentredString(W / 2, H / 2 - 80, "Rx")
        c.restoreState()

        # ── BODY ─────────────────────────────────────────────────────────────
        y = H - HDR_H - 32

        # Date + Rx No — top right
        f(*MUTED)
        c.setFont("Helvetica", 9)
        c.drawRightString(MR, y, f"Date: {datetime.now().strftime('%d %B %Y')}")
        if approval_id:
            c.drawRightString(MR, y - 13, f"Rx No: {approval_id[:8].upper()}")

        # Patient block
        f(*DARK)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(ML, y, "Patient Details")

        y -= 15
        c.setFont("Helvetica", 10)
        line = f"Name: {patient_name or 'N/A'}"
        if patient_age:
            line += f"    Age: {patient_age} yrs"
        if patient_gender:
            line += f"    Sex: {patient_gender.capitalize()}"
        c.drawString(ML, y, line)
        y -= 14

        # Diagnosis — wrapped across multiple lines if long
        if diagnosis:
            diag_lines = _wrap(str(diagnosis), 90)
            f(*DARK)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(ML, y, "Diagnosis:")
            c.setFont("Helvetica", 10)
            c.drawString(ML + 72, y, diag_lines[0])
            for dl in diag_lines[1:]:
                y -= 13
                c.drawString(ML + 72, y, dl)
            y -= 8

        # Thin rule
        y -= 8
        s(*LIGHT_RULE)
        c.setLineWidth(0.4)
        c.line(ML, y, MR, y)
        y -= 14

        # "Rx" label above table (replaces ℞ which won't render in Helvetica)
        f(*NAVY)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(ML, y, "Rx")
        y -= 20

        # ── MEDICATION TABLE ──────────────────────────────────────────────────
        if medications:
            # Col widths: #, Medicine, Dosage, Frequency, Duration, Instructions
            # Total = MR - ML ≈ 525pt
            COL_W = [20, 145, 58, 88, 55, 159]
            HDRS  = ["#", "Medicine", "Dosage", "Frequency", "Duration", "Instructions"]
            INST_COL_CHARS = 26   # ~159pt / ~6pt per char at 8pt font
            BASE_ROW_H     = 22
            LINE_H         = 10   # extra height per additional instruction line

            # Header row
            f(*NAVY)
            c.rect(ML, y - BASE_ROW_H + 5, sum(COL_W), BASE_ROW_H, fill=1, stroke=0)
            f(*WHITE)
            c.setFont("Helvetica-Bold", 8)
            x = ML
            for hdr, cw in zip(HDRS, COL_W):
                c.drawString(x + 3, y - BASE_ROW_H + 9, hdr)
                x += cw
            y -= BASE_ROW_H

            for idx, med in enumerate(medications):
                if isinstance(med, dict):
                    name    = (med.get("medicine_name") or med.get("name", "")).strip()
                    generic = med.get("generic_name", "").strip()
                    dosage  = (med.get("dosage") or med.get("strength", "")).strip()
                    freq    = med.get("frequency", "").strip()
                    dur     = med.get("duration", "").strip()
                    inst    = med.get("instructions", "").strip()
                    if not inst:
                        if med.get("before_food") is True:
                            inst = "Before food"
                        elif med.get("before_food") is False:
                            inst = "After food"
                else:
                    name = str(med).strip()
                    generic = dosage = freq = dur = inst = ""

                # Word-wrap the Instructions cell
                inst_lines = _wrap(inst, INST_COL_CHARS) if inst else []
                # Calculate dynamic row height
                extra_lines = max(0, len(inst_lines) - 1)
                row_h = BASE_ROW_H + extra_lines * LINE_H

                # Check if we need a new page before drawing this row
                if y - row_h < 160:
                    _draw_footer(c, W, signature_hash, doctor_name, doctor_specialty, nmc_number)
                    c.showPage()
                    y = H - 40

                # Alternate row shading
                if idx % 2 == 0:
                    f(*ROW_ALT)
                    c.rect(ML, y - row_h + 5, sum(COL_W), row_h, fill=1, stroke=0)

                # Row divider
                s(*LIGHT_RULE)
                c.setLineWidth(0.3)
                c.line(ML, y - row_h + 5, MR, y - row_h + 5)

                # Draw each cell
                x = ML
                for i, (val, cw) in enumerate(zip(
                    [str(idx + 1), name, dosage, freq, dur, inst], COL_W
                )):
                    cell_y = y - BASE_ROW_H + 12   # baseline aligned to top of row
                    if i == 1:
                        # "Medicine Name (Generic)" on the same line, wrapped if long
                        display = f"{name} ({generic})" if generic else name
                        f(*DARK)
                        c.setFont("Helvetica-Bold", 8)
                        disp_lines = _wrap(display, 26)
                        for li, dl in enumerate(disp_lines[:2]):
                            c.drawString(x + 3, cell_y - li * 9, dl)
                    elif i == 5:
                        # Instructions — multi-line
                        f(*DARK)
                        c.setFont("Helvetica", 8)
                        for li, iline in enumerate(inst_lines):
                            c.drawString(x + 3, cell_y - li * LINE_H, iline)
                    else:
                        f(*DARK)
                        c.setFont("Helvetica", 8)
                        c.drawString(x + 3, cell_y, str(val)[:22])
                    x += cw

                y -= row_h

            # Bottom border of table
            s(*NAVY)
            c.setLineWidth(0.5)
            c.line(ML, y + 5, MR, y + 5)

        y -= 18

        # ── INSTRUCTIONS / ADVICE ─────────────────────────────────────────────
        def _section(title: str, items: list, color=DARK):
            nonlocal y
            if not items:
                return
            if y < 170:
                return
            f(*color)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(ML, y, f"{title}:")
            y -= 13
            f(*DARK)
            c.setFont("Helvetica", 9)
            for item in items:
                # Wrap each bullet item
                item_lines = _wrap(str(item), 95)
                for li, iline in enumerate(item_lines):
                    prefix = "•  " if li == 0 else "    "
                    c.drawString(ML + 8, y, prefix + iline)
                    y -= 12
                    if y < 170:
                        break
                if y < 170:
                    break
            y -= 4

        _section("General Instructions", general_instructions)
        _section("Dietary Advice",       dietary_advice)
        if warning_signs:
            _section("Warning Signs — seek help if you notice",
                     warning_signs, color=WARN_RED)

        # Follow-up — word-wrapped
        if follow_up_instructions and y >= 170:
            f(*DARK)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(ML, y, "Follow-up:")
            c.setFont("Helvetica", 9)
            fu_lines = _wrap(str(follow_up_instructions), 88)
            c.drawString(ML + 65, y, fu_lines[0])
            for fl in fu_lines[1:]:
                y -= 13
                c.drawString(ML + 65, y, fl)
            y -= 16

        # ── DIGITAL SIGNATURE BLOCK ──────────────────────────────────────────
        sig_y = max(y - 20, 175)
        s(*DARK)
        c.setLineWidth(0.5)
        c.line(W - 220, sig_y, MR, sig_y)

        f(*DARK)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(W - 220, sig_y - 12, "Digitally Signed By:")
        dr_label = doctor_name if doctor_name.startswith("Dr") else f"Dr. {doctor_name}"
        c.setFont("Helvetica-Bold", 9)
        c.drawString(W - 220, sig_y - 23, dr_label)
        f(*MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(W - 220, sig_y - 34, doctor_specialty or "")
        if nmc_number:
            c.drawString(W - 220, sig_y - 44, f"Reg. No: {nmc_number}")
        if signature_hash:
            f(0.45, 0.45, 0.45)
            c.setFont("Helvetica", 6.5)
            c.drawString(W - 220, sig_y - 56, signature_hash[:40])
            c.drawString(W - 220, sig_y - 65, signature_hash[40:])

        # ── FOOTER ───────────────────────────────────────────────────────────
        _draw_footer(c, W, signature_hash, doctor_name, doctor_specialty, nmc_number)

        c.save()
        logger.info("Prescription PDF generated: %s", filepath)

    except ImportError:
        filepath = filepath.replace(".pdf", ".txt")
        with open(filepath, "w") as fp:
            fp.write("MEDIVORA — Medical Prescription\n")
            fp.write(f"Approval : {approval_id}\n")
            fp.write(f"Patient  : {patient_name}, {patient_age} yrs, {patient_gender}\n")
            fp.write(f"Doctor   : {doctor_name} ({doctor_specialty})\n")
            fp.write(f"Diagnosis: {diagnosis}\n\nMedications:\n")
            for m in medications:
                fp.write(f"  {m}\n")
            fp.write(f"\nInstructions: {'; '.join(general_instructions)}\n")
        logger.warning("reportlab not installed — wrote text fallback: %s", filepath)

    return filepath


# ── Footer ────────────────────────────────────────────────────────────────────

def _draw_footer(c, W, signature_hash, doctor_name, doctor_specialty, nmc_number):

    # Contact row
    CY     = 96
    BOX_SZ = 10
    c.setLineWidth(0.5)
    for text, x in [(PHONE, 35), (EMAIL, 215), (WEBSITE, 415)]:
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setFillColorRGB(1.0, 1.0, 1.0)
        c.rect(x, CY - 1, BOX_SZ, BOX_SZ, fill=1, stroke=1)
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 8.5)
        c.drawString(x + BOX_SZ + 4, CY, text)

    c.setStrokeColorRGB(*LIGHT_RULE)
    c.setLineWidth(0.4)
    c.line(0, 82, W, 82)

    # Address bar
    ADDR_H = 30
    c.setFillColorRGB(*FOOTER_BG)
    c.rect(0, 0, W, ADDR_H, fill=1, stroke=0)

    c.setFillColorRGB(*NAVY)
    c.circle(22, ADDR_H / 2, 12, fill=1, stroke=0)
    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(22, ADDR_H / 2 - 4, "m")

    c.setFillColorRGB(1.0, 1.0, 1.0)
    c.setFont("Helvetica", 8.5)
    c.drawString(42, ADDR_H / 2 - 4, ADDRESS)
