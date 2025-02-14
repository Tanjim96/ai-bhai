import streamlit as st
import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import re

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    return thresh

def extract_text_tesseract(image):
    try:
        custom_config = r'--oem 3 --psm 6 -l ben'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except Exception as e:
        st.error(f"Tesseract error: {str(e)}")
        return ""

def extract_text_easyocr(image):
    try:
        reader = easyocr.Reader(['bn'])
        results = reader.readtext(image)
        text = '\n'.join([result[1] for result in results])
        return text
    except Exception as e:
        st.error(f"EasyOCR error: {str(e)}")
        return ""

def format_mcq(text):
    lines = text.split('\n')
    formatted_lines = []
    current_question = []
    school_name = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for school name at the beginning
        if not school_name and re.search(r'(স্কুল|বিদ্যালয়|কলেজ)', line):
            school_name = line
            formatted_lines.append(f"স্কুল: {school_name}\n")
            continue
        
        # Check for question numbers
        if re.match(r'^[১-৯][।.]|^[1-9]\.', line):
            if current_question:
                formatted_lines.append('\n'.join(current_question))
                current_question = []
            current_question.append(line)
        # Check for options
        elif re.match(r'^[ক-ঙ][।.]', line):
            current_question.append(f"    {line}")
        else:
            current_question.append(line)
    
    if current_question:
        formatted_lines.append('\n'.join(current_question))
        
    return '\n\n'.join(formatted_lines)

def main():
    st.title("বাংলা MCQ এক্সট্র্যাক্টর")
    st.write("Upload an image containing Bengali MCQs or take a photo")
    
    # Image input options
    image_source = st.radio(
        "Choose image source:",
        ["Upload Image", "Take Photo"]
    )
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert PIL Image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_file = st.camera_input("Take a photo")
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption="Captured Photo", use_column_width=True)
            
            # Convert PIL Image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if 'image' in locals():
        # OCR method selection
        ocr_method = st.radio(
            "Choose OCR method:",
            ["Tesseract OCR", "EasyOCR"],
            help="Tesseract is faster, EasyOCR might be more accurate"
        )
        
        if st.button("Extract Text"):
            with st.spinner("Processing image..."):
                # Preprocess image
                processed_image = preprocess_image(opencv_image)
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                # Extract text
                if ocr_method == "Tesseract OCR":
                    extracted_text = extract_text_tesseract(processed_image)
                else:
                    extracted_text = extract_text_easyocr(processed_image)
                
                # Format text
                formatted_text = format_mcq(extracted_text)
                
                # Display results
                st.subheader("Extracted MCQs:")
                st.text_area(
                    label="",
                    value=formatted_text,
                    height=400
                )
                
                # Download button
                st.download_button(
                    label="Download Text",
                    data=formatted_text,
                    file_name="extracted_mcqs.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
