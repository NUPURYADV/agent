import fitz

doc = fitz.open("SYALLABUS FIRST YEAR.pdf")

full_text = ""

for page in doc:
    full_text += page.get_text()

print(f"Total characters extracted: {len(full_text)}")
print(full_text[:50]) 
print(f"Total characters: {len(full_text)}")