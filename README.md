# RAG-LLM-ARL-demo

Run with Virtual Env:

Create a virtual env with Python: 
1. If using conda you can run "conda create -n "achilles" python=3.11.7"
2. Once the env is created it can be activated using "conda activate achilles"
3. run "pip install -r requirements.txt"
4. run "python gui_beta.py"


Notice: 
Before Uploading doc, pptx, xls file, please save and your other opened works for safety.


In the pdf_split.py:

go to "save_ouput" function 

Replace 

    def save_output(rendered: BaseModel, output_dir: str, fname_base: str):
        text, ext, images = text_from_rendered(rendered)
        text = text.encode(settings.OUTPUT_ENCODING, errors="replace").decode(
            settings.OUTPUT_ENCODING
        )

        with open(
            os.path.join(output_dir, f"{fname_base}.{ext}"),
            "w+",
            encoding=settings.OUTPUT_ENCODING,
        ) as f:
            f.write(text)
        with open(
            os.path.join(output_dir, f"{fname_base}_meta.json"),
            "w+",
            encoding=settings.OUTPUT_ENCODING,
        ) as f:
            f.write(json.dumps(rendered.metadata, indent=2))

        for img_name, img in images.items():
            img = convert_if_not_rgb(img)  # RGBA images can't save as JPG
            img.save(os.path.join(output_dir, img_name), settings.OUTPUT_IMAGE_FORMAT)

With

    def save_output(rendered: BaseModel, output_dir: str, fname_base: str):
        text, ext, images = text_from_rendered(rendered)
        text = text.encode(settings.OUTPUT_ENCODING, errors='replace').decode(settings.OUTPUT_ENCODING)

        with open(os.path.join(output_dir, f"{fname_base}.{ext}"), "w+", encoding=settings.OUTPUT_ENCODING) as f:
            f.write(text)
        # with open(os.path.join(output_dir, f"{fname_base}_meta.json"), "w+", encoding=settings.OUTPUT_ENCODING) as f:
        #     f.write(json.dumps(rendered.metadata, indent=2))

        for img_name, img in images.items():
            img.save(os.path.join(output_dir,f"{fname_base}" + img_name), settings.OUTPUT_IMAGE_FORMAT)
        
        return images
