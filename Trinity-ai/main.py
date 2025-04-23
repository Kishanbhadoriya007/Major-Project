from PyPDF2 import PdfReader
file = open("openweaver.pdf")
reader=PdfReader(file)
num_pages=reader.numPages
for p in range(num_pages):
    page=reader.getpage(p)
    text=page.extractText()
    print(text)