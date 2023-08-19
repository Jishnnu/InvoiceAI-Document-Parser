# Import necessary libraries
import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
from langchain.callbacks import StreamlitCallbackHandler
from mindee import Client, documents

# Setup PyTesseract
pytesseract.pytesseract.tesseract_cmd = "YOUR_TESSERACT_APPLICATION_FILEPATH"

# Setup Mindee API and URL
mindee_client = Client(api_key="YOUR_MINDEE_API_KEY")
url = "https://api.mindee.net/v1/products/mindee/invoices/v4/predict"


# Funtion to enhance the details in the invoice
def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


# Function to remove the noise in the invoice image
def denoise(img):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(img, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


# Function to thicken the font
def enhance_font(img):
    image = cv2.bitwise_not(img)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


# Function to remove the borders from the invoice image
def crop_borders(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    contour = sorted_contours[-1]
    x, y, w, h = cv2.boundingRect(contour)
    cropped_img = img[y : y + h, x : x + w]
    return cropped_img


# Function to exract the textual data from the invoice image
def ocr(image):
    img = image.copy()
    raw_data = pytesseract.image_to_data(img)
    for count, data in enumerate(raw_data.splitlines()):
        if count > 0:
            data = data.split()
        if len(data) == 12:
            x, y, w, h, content = (
                int(data[6]),
                int(data[7]),
                int(data[8]),
                int(data[9]),
                data[11],
            )
            cv2.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), 1)
            cv2.putText(img, content, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    return img


# Setup the Streamlit page
st.set_page_config(page_title="InvoiceAI", page_icon="🤖", layout="centered")
st.title("🤖 InvoiceAI : Smart Invoice Parser!")


# Setup the first chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get the invoice as input from the user
invoice_image = st.file_uploader("Upload your Invoice", type=["jpg", "jpeg"])

if invoice_image:
    st.image(invoice_image, "Uploaded Invoice")

    # Setup the chat for response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Parse the Invoice
        try:
            with st.spinner("Parsing the Invoice... Please wait"):
                original_image = Image.open(invoice_image)
                original_image = np.array(original_image)
                processed_image = enhance_details(original_image)

                # Invert the invoice image
                inverted_image = cv2.bitwise_not(processed_image)

                # Binarize the invoice image
                gray_scaled_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                thresh, im_bw = cv2.threshold(
                    gray_scaled_image, 210, 230, cv2.THRESH_BINARY
                )

                # Remove the noise in the invoice image
                denoised_image = denoise(im_bw)

                # Dilate the invoice image
                dialated_image = enhance_font(denoised_image)

                # Crop the borders of the invoice image
                cropped_image = crop_borders(dialated_image)
                cv2.imwrite("new_img.jpg", cropped_image)

            # Creating extracting chain
            with st.spinner("Initiating text extraction... Please wait"):
                bounded_ocr_image = ocr(cropped_image)
                st.image(
                    bounded_ocr_image,
                    "Performing OCR on the Invoice",
                    use_column_width="auto",
                )

            # Parse the Invoice and generate JSON response
            with st.spinner("Generating JSON response... Please wait"):
                input_doc = mindee_client.doc_from_path("new_img.jpg")
                result = input_doc.parse(documents.TypeInvoiceV4)
                st.json(result.http_response)

        except Exception as e:
            st.error("Error processing invoice: " + str(e))
