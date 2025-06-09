import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from file_conversion.Imagett import image_to_pdf, image_to_txt
# from Imagett import image_to_pdf
def extract_text_and_images(pdf_path):
        
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(pdf_path)
    
    fname = os.path.basename(pdf_path).split(".")[0]

    images = save_output(rendered= rendered, output_dir =  "./source", fname_base=fname)
    
    for img_name, img in images.items():
        image_to_txt(os.path.join ("./source", f"{fname}" + img_name))


# extract_text_and_images("./Test_files/Memoro_Using_Large_Language_Models_to_Realize_a_Concise.pdf")
