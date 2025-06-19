   # file_conversion/office_2_pdf.py
import subprocess
import shlex
from pathlib import Path
import shutil
from fpdf import FPDF   # already in requirements

def office_to_pdf(file_path: str, destination_dir: str = "./source") -> Path:
    """
    Convert an Office, PowerPoint, Excel or plain-text file to PDF on Linux/macOS.
    Uses LibreOffice (--headless --convert-to pdf) for everything except .txt,
    which is rendered with FPDF.  Returns the final PDF Path.

    Dependencies (Ubuntu):
        sudo apt install libreoffice libreoffice-core libreoffice-writer libreoffice-calc libreoffice-impress
        sudo apt install fonts-liberation   # avoids blank glyphs in PDFs
    """
    in_path = Path(file_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    ext = in_path.suffix.lower()
    out_dir = in_path.parent        # LibreOffice writes here by default
    pdf_path = in_path.with_suffix(".pdf")

    try:
        # ── 1. Plain-text → PDF with FPDF ────────────────────────────────
        if ext == ".txt":
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)

            with open(in_path, "r", encoding="utf-8") as f:
                for line in f:
                    safe = line.encode("latin-1", errors="ignore").decode("latin-1")
                    pdf.multi_cell(0, 8, txt=safe.strip())

            pdf.output(str(pdf_path))

        # ── 2. All other Office docs → PDF with LibreOffice ──────────────
        else:
            # libreoffice --headless --convert-to pdf <file> --outdir <dir>
            cmd = (
                "libreoffice --headless --convert-to pdf "
                f"{shlex.quote(str(in_path))} --outdir {shlex.quote(str(out_dir))}"
            )
            result = subprocess.run(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode() or result.stdout.decode())

        # ── 3. Move PDF into ./source/  ───────────────────────────────────
        dest_dir = Path(destination_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_path = dest_dir / pdf_path.name
        shutil.move(pdf_path, final_path)

        print(f"PDF created → {final_path}")
        return final_path

    except Exception as e:
        raise RuntimeError(f"Error converting {in_path.name} → PDF: {e}") from e