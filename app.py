from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("model/segmentation_model_unet_final.keras")
UPLOAD_FOLDER = "static/images/uploads"
RESULT_FOLDER = "static/images/results"
EXISTING_IMAGE = "static/images/test_image.png"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


IMG_HEIGHT, IMG_WIDTH = 256, 256
THRESHOLD = 0.15

# Preprocess the test image
def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width), color_mode="grayscale")
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]

    # Create flipped version
    flipped_img = np.fliplr(img.squeeze())

    # Apply Gaussian blur
    noise_reduced_img = cv2.GaussianBlur(img.squeeze(), (5, 5), 0)

    # Apply histogram equalization
    equalized_img = cv2.equalizeHist((img.squeeze() * 255).astype(np.uint8)) / 255.0

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(img.squeeze(), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img.squeeze(), cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_combined = (sobel_combined / sobel_combined.max()).astype(np.float32)

    # Stack all versions along the channel axis
    combined = np.stack(
        [img.squeeze(), flipped_img, noise_reduced_img, equalized_img, sobel_combined],
        axis=-1,
    )
    return combined


# Post-process the predicted mask
def postprocess_mask(mask, threshold):
    return (mask > threshold).astype(np.uint8)  # Binarize the mask

# Combine the mask with the original image
def overlay_mask_on_image(original_img, mask, color=(0, 255, 0), alpha=0.5):
    # Ensure the original image has 3 channels (grayscale to RGB)
    if len(original_img.shape) == 2:  # If single channel (grayscale)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    
    # Create a blank RGB mask
    mask_colored = np.zeros_like(original_img, dtype=np.uint8)
    
    # Apply the color to the regions where the mask is active
    mask_colored[mask.squeeze() > 0] = color
    
    # Blend the original image with the colored mask
    combined = cv2.addWeighted(original_img, 1 - alpha, mask_colored, alpha, 0)
    return combined

# Draw bounding box around the masked region
def draw_bounding_box(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_existing_image')
def get_existing_image():
    return jsonify({"image_url": f"/{EXISTING_IMAGE}"})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    print(file)
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
    else:
        file_path = EXISTING_IMAGE
 
    # Preprocess the test image
    test_image = preprocess_image(file_path, IMG_HEIGHT, IMG_WIDTH)
    original_image = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
    original_image = img_to_array(original_image).astype(np.uint8)  # Convert to uint8 for visualization
    
    # Predict the mask
    predicted_mask = model.predict(test_image[np.newaxis, ...])[0]  # Add batch dimension
    predicted_mask = postprocess_mask(predicted_mask, THRESHOLD)
    
    # Visualizations 
    overlay_image = overlay_mask_on_image(original_image.squeeze(), predicted_mask, color=(0, 255, 0))
    overlay_with_bbox = draw_bounding_box(overlay_image.copy(), predicted_mask)
    
    # Save output image
    result_path = os.path.join(RESULT_FOLDER, "predicted_result.png")
    cv2.imwrite(result_path, cv2.cvtColor(overlay_with_bbox, cv2.COLOR_BGR2RGB))

    if file_path == EXISTING_IMAGE:
        pass
    else:
        os.remove(file_path)
    return jsonify({"result_url": f"/static/images/results/predicted_result.png"})

if __name__ == '__main__':
    app.run(debug=True)

