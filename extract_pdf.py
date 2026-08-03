import sys
import subprocess

try:
    import fitz
except ImportError:
    print("Installing pymupdf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

pdf_path = r"C:\Users\pravin\Downloads\Telegram Desktop\LIFE PROCESSES CHAPTER 6 - FULL CHAPTER.pdf"
output_path = r"C:\Users\pravin\Downloads\DPT clinical decision support engine\extracted_text.txt"

try:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Extraction complete! Saved to {output_path}")
    print(f"Total characters extracted: {len(text)}")
    print("Preview of the first 500 characters:")
    print("-" * 50)
    print(text[:500])
    print("-" * 50)
    
    if len(text.strip()) < 100 and len(doc) > 0:
        print("Warning: Very little text was extracted. The PDF might be image-based (scanned). Real OCR might be required.")
except Exception as e:
    print(f"Error occurred: {e}")
