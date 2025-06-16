# RAG-LLM-ARL-demo

## Run with Virtual Env (Windows / conda example)

1. Create a conda environment (Python 3.11.7 is recommended):  
   ```bash
   conda create -n achilles python=3.11.7
   ```

2. Activate the environment:  
   ```bash
   conda activate achilles
   ```

3. Install the project requirements:  
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the GUI:  
   ```bash
   python gui_beta.py
   ```

5. Add your keys:  
   * Paste the **OpenAI** key into a `.env` file (`OPENAI_API_KEY=...`).  
   * Create an `./api` folder and drop your Google credentials JSON inside.

> **Notice** – Before uploading `.docx`, `.pptx`, or `.xlsx` files, save and close any other work to avoid data‑loss.

> **Notice** - If show error: Error loading transcription into LLM: Descriptors cannot be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0. If you cannot immediately regenerate your protos, some other possible workarounds are: 1. Downgrade the protobuf package to 3.20.x or lower. 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
>```bash
>pip uninstall protobuf
>pip install protobuf==3.20.0
>```
---

## Running on Ubuntu 20.04 – 24.04 (and other Debian‑based distros)

The project is now Linux‑friendly thanks to a LibreOffice‑based Office‑to‑PDF
converter and PortAudio fixes. Follow these steps:

### 1  System packages (one‑time)

```bash
sudo apt update
sudo apt install -y   python3.11 python3.11-venv build-essential   # Python 3.11 + compiler toolchain
sudo apt install -y   portaudio19-dev libportaudio2 libportaudiocpp0  # headers for PyAudio
sudo apt install -y   libreoffice libreoffice-core fonts-liberation   # headless Office → PDF
```

*If `python3.11` isn’t in your repo, add the Deadsnakes PPA first:*  
`sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update`

### 2  Virtual‑environment

```bash
cd ~/RAG-LLM-ARL-demo          # or wherever you cloned the repo
python3.11 -m venv venv        # creates ./venv
source venv/bin/activate       # prompt shows (venv)
python -V                      # → Python 3.11.x
```

### 3  Python dependencies (note, on linux, remove win32 from requirements.txt)

```bash
pip install --upgrade pip setuptools wheel
# remove Windows‑only deps if present
sed -i '/pywin32/d' requirements.txt
pip install -r requirements.txt
```

If `pyaudio` still fails, double‑check that `portaudio19-dev` is installed.

### 4  Project setup

```bash
# OpenAI key
echo "OPENAI_API_KEY=sk-YOURKEYHERE" > .env

# Google credentials
mkdir -p api
cp ~/Downloads/your-google-creds.json api/
```
### 5  Replace the Office‑to‑PDF helper (Linux only)

win32com does not exist on Linux. Replace its usage with the LibreOffice
version below once per clone:

    Open file_conversion/office_2_pdf.py in your editor.

    Delete its contents and paste the code block below.

    Save the file – no other modules need to change.

```python
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


```

### 6  Run the GUI

```bash
python gui_beta.py
```

### 6  File‑conversion notes

* `file_conversion/office_2_pdf.py` now calls **LibreOffice** in headless mode
  – ensure the packages in step 1 are installed.
* Converted PDFs are moved to `./source/`.
* No more `win32com` / `pywin32` is required.

### 7  Playing nicely with Python 3.12+

`playsound==1.3.0` does not build on 3.12. If you upgrade, pin the maintained
fork instead:

```bash
pip install playsound@git+https://github.com/taconi/playsound
```

---

## Required tweak in `pdf_split.py`

Open `pdf_split.py`, locate the `save_output` function and replace the existing
implementation with the version below to avoid JSON metadata write errors and
ensure images are saved correctly:

```python
def save_output(rendered: BaseModel, output_dir: str, fname_base: str):
    text, ext, images = text_from_rendered(rendered)
    text = text.encode(settings.OUTPUT_ENCODING, errors='replace').decode(
        settings.OUTPUT_ENCODING)

    with open(os.path.join(output_dir, f"{fname_base}.{ext}"), "w+",
              encoding=settings.OUTPUT_ENCODING) as f:
        f.write(text)

    # Skip metadata JSON to reduce clutter
    # with open(os.path.join(output_dir, f"{fname_base}_meta.json"), "w+",
    #           encoding=settings.OUTPUT_ENCODING) as f:
    #     f.write(json.dumps(rendered.metadata, indent=2))

    for img_name, img in images.items():
        img.save(os.path.join(output_dir, f"{fname_base}{img_name}"),
                 settings.OUTPUT_IMAGE_FORMAT)

    return images
```

---

### Tested environments

* **Windows 11** (conda 23.11, Python 3.11.7)  
* **Ubuntu 22.04.4 LTS** (Python 3.11.9, LibreOffice 7.5)

Enjoy!
