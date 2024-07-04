from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Blackjack Assistant!"


@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Read the image file
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Recognize the card
    card = recognize_card(img)
    return jsonify({"message": "Card recognized", "card": card}), 200


def recognize_card(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edged = cv2.Canny(blur, 50, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Initialize card variable
    card = "Unknown"

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(approx)

            # Extract the card from the image
            card_image = image[y:y + h, x:x + w]

            # Placeholder logic for card recognition
            # (Replace with actual recognition logic)
            card = "Detected Card"
            break

    return card


if __name__ == '__main__':
    app.run(debug=True)
