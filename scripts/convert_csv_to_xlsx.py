"""convert_csv_to_xlsx.py

Usage:
  python -m pip install openpyxl
  python scripts/convert_csv_to_xlsx.py integrations.csv integrations.xlsx

This reads a CSV and writes a simple .xlsx workbook (single sheet).
"""
import sys
import csv
from openpyxl import Workbook

def csv_to_xlsx(csv_path, xlsx_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Integrations"
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            ws.append(row)
    wb.save(xlsx_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_csv_to_xlsx.py <input.csv> <output.xlsx>")
        sys.exit(1)
    csv_to_xlsx(sys.argv[1], sys.argv[2])
    print(f"Written {sys.argv[2]}")
