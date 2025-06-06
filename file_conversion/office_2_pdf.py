from PyQt6.QtWidgets import QFileDialog

import os
import shutil

from docx2pdf import convert
from fpdf import FPDF
import win32com.client


def office_2_pdf(file_path):
    """Convert Office documents to PDF."""
    # Get the file path
    print ("come into office_2_pdf")
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)

    try:
        # Convert Word documents to PDF
        if file_extension == ".docx":
            convert(file_path)
            
        elif file_extension == ".txt":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)  # Use built-in font that supports Latin-1

            with open(file_path, "r", encoding="utf-8") as txt_file:
                for line in txt_file:
                    # Encode using Latin-1 and ignore unsupported characters, then decode back.
                    safe_line = line.encode("latin-1", errors="ignore").decode("latin-1")
                    pdf.multi_cell(0, 10, txt=safe_line.strip())

            pdf_output = os.path.splitext(file_path)[0] + ".pdf"
            pdf.output(pdf_output)
            
            

        # Convert Excel documents to PDF
        elif file_extension == ".xlsx":
            file_path = os.path.abspath(file_path)
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = False
            wb = excel.Workbooks.Open(file_path)
            wb.SaveAs(os.path.splitext(file_path)[0] + ".pdf", FileFormat=57)
            wb.Close()
            excel.Quit()

        # Convert PowerPoint documents to PDF
        elif file_extension == ".pptx":
           
            file_path = os.path.abspath(file_path)
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            
            presentation = powerpoint.Presentations.Open(file_path)
            
            pdf_output = os.path.splitext(file_path)[0] + ".pdf"
            
            presentation.SaveAs(pdf_output, 32)  # 32表示PDF格式
            presentation.Close()
            powerpoint.Quit()
            

        # Move the PDF to the 'source' directory
        pdf_file = os.path.splitext(file_path)[0] + ".pdf"
        
        if os.path.exists(pdf_file):
            destination_dir = "./source"
            os.makedirs(destination_dir, exist_ok=True)
            shutil.move(pdf_file, os.path.join(destination_dir, os.path.basename(pdf_file)))
            print(f"PDF moved to {destination_dir}")
        else:
            print(f"PDF file {pdf_file} not found.")
    
    except Exception as e:
        print(f"Error converting file to PDF: {e}")