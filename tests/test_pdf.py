from pdf_loader import load_pdf

pages = load_pdf("sample.pdf")

print(pages[0]["page"])
print(pages[0]["text"][:500])
