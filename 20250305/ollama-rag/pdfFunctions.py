import PyPDF2
import pytesseract
from pdf2image import convert_from_path

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf_tesseract(pdf_path):
    print(f'Processing {pdf_path}')

    text = convertTesseract(pdf_path)
    return text 



def extract_text_from_pdf_pypdf(pdf_path):
    print(f'Processing {pdf_path}')
    text=""
    with open(pdf_path, "rb") as file:
      reader = PyPDF2.PdfFileReader(file)
      for page_num in range(reader.numPages):
          page = reader.getPage(page_num)
          text += page.extract_text()
    return text


def convertTesseract(path):
  # convert to image using resolution 600 dpi 
  pages = convert_from_path(path, 600)

  # extract text
  text_data = ''
  for page in pages:
    text = pytesseract.image_to_string(page)
    text_data += text + '\n'
  #print(text_data)
  return text_data
