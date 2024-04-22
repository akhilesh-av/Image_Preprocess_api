# import cv2
import numpy as np
import base64
import pywt
import urllib.request
from flask import Flask, json, request, jsonify
import os
from werkzeug.utils import secure_filename
from skimage.segmentation import felzenszwalb
from skimage import color, data, restoration
from scipy.signal import convolve2d
from cv2 import (imread,imdecode,equalizeHist,IMREAD_GRAYSCALE,
                 imencode,createCLAHE,GaussianBlur,medianBlur,Laplacian,Sobel,normalize,addWeighted,subtract,filter2D,
                 NORM_MINMAX,CV_8U,CV_64F,pyrDown,pyrUp,resize,erode,dilate,getRotationMatrix2D,warpAffine,getAffineTransform,
                 morphologyEx,MORPH_OPEN,MORPH_CLOSE,MORPH_GRADIENT,MORPH_TOPHAT,MORPH_BLACKHAT,MORPH_ELLIPSE,MORPH_RECT,
                 COLOR_BGR2RGB,COLOR_RGB2HSV,COLOR_RGB2LAB,cvtColor,IMREAD_COLOR,Canny,threshold,DIST_L2,THRESH_BINARY_INV,THRESH_OTSU,
                 connectedComponents,watershed,distanceTransform,COLOR_BGR2GRAY,split,merge,THRESH_BINARY)



app = Flask(__name__)

 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    errors = {}
    success = False   
    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 


@app.route('/hist_eq', methods=['POST'])
def hist_eq():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            # Perform histogram equalization
            equalized_image = equalizeHist(image)
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, equalized_img_encoded =imencode('.png', equalized_image)
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            equalized_img_base64 = base64.b64encode(equalized_img_encoded).decode('utf-8')
            # Append original and equalized images to results
            results.append({'original_image': original_img_base64, 'equalized_image': equalized_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    resp = jsonify(results)
    return resp




@app.route('/ahe', methods=['POST'])
def ahe():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Perform histogram equalization
            equalized_image = equalizeHist(image)
            
            # Apply adaptive histogram equalization
            clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            adaptive_equalized_image = clahe.apply(image)

            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, equalized_img_encoded = imencode('.png', equalized_image)
            _, adaptive_equalized_img_encoded = imencode('.png', adaptive_equalized_image)

            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            equalized_img_base64 = base64.b64encode(equalized_img_encoded).decode('utf-8')
            adaptive_equalized_img_base64 = base64.b64encode(adaptive_equalized_img_encoded).decode('utf-8')

            # Append original, equalized, and adaptive equalized images to results
            results.append({
                'original_image': original_img_base64,
                'equalized_image': equalized_img_base64,
                'adaptive_equalized_image': adaptive_equalized_img_base64
            })
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/clane', methods=['POST'])
def clane():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
            clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            # Apply CLAHE
            clahe_image = clahe.apply(image)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, clahe_img_encoded = imencode('.png', clahe_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            clahe_img_base64 = base64.b64encode(clahe_img_encoded).decode('utf-8')
            
            # Append original and CLAHE-enhanced images to results
            results.append({'original_image': original_img_base64, 'clahe_image': clahe_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response




@app.route('/denoise_gauss', methods=['POST'])
def denoise_gauss():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Apply Gaussian blur
            blurred_image = GaussianBlur(image, (5, 5), 0)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, blurred_img_encoded = imencode('.png', blurred_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            blurred_img_base64 = base64.b64encode(blurred_img_encoded).decode('utf-8')
            
            # Append original and blurred images to results
            results.append({'original_image': original_img_base64, 'blurred_image': blurred_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response


@app.route('/denoise_median', methods=['POST'])
def denoise_median():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Apply median blur
            filtered_image = medianBlur(image, ksize=5)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, filtered_img_encoded = imencode('.png', filtered_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            filtered_img_base64 = base64.b64encode(filtered_img_encoded).decode('utf-8')
            
            # Append original and filtered images to results
            results.append({'original_image': original_img_base64, 'filtered_image': filtered_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response




@app.route('/laplacian_edge', methods=['POST'])
def laplacian_edge():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Apply Gaussian blur to reduce noise
            blurred = GaussianBlur(image, (3, 3), 0)
            
            # Apply Laplacian filter
            laplacian = Laplacian(blurred, CV_64F)
            
            # Convert back to unsigned 8-bit integer
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, laplacian_img_encoded = imencode('.png', laplacian)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            laplacian_img_base64 = base64.b64encode(laplacian_img_encoded).decode('utf-8')
            
            # Append original and filtered images to results
            results.append({'original_image': original_img_base64, 'laplacian_image': laplacian_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/highpass_edge', methods=['POST'])
def highpass_edge():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Apply Gaussian blur to reduce noise
            blurred = GaussianBlur(image, (3, 3), 0)
            
            # Calculate the Sobel gradients
            sobelx = Sobel(blurred, CV_64F, 1, 0, ksize=3)
            sobely = Sobel(blurred, CV_64F, 0, 1, ksize=3)
            
            # Combine the gradients to obtain the magnitude
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize the magnitude image
            magnitude = normalize(magnitude, None, alpha=0, beta=255, norm_type=NORM_MINMAX, dtype=CV_8U)
            
            # Add the magnitude image to the original grayscale image
            enhanced_image = addWeighted(image, 1.5, magnitude, -0.5, 0)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, enhanced_img_encoded = imencode('.png', enhanced_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            enhanced_img_base64 = base64.b64encode(enhanced_img_encoded).decode('utf-8')
            
            # Append original and enhanced images to results
            results.append({'original_image': original_img_base64, 'enhanced_image': enhanced_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response


@app.route('/unsharp_masking_edge', methods=['POST'])
def unsharp_masking_edge():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Step 1: Apply Gaussian blur
            blurred = GaussianBlur(image, (0, 0), 1.0)
            
            # Step 2: Calculate the difference between the original and blurred image
            mask = subtract(image, blurred)
            
            # Step 3: Adjust the mask by multiplying it with a strength factor
            sharpened_image = addWeighted(image, 1.5, mask, -0.5, 0)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, sharpened_img_encoded = imencode('.png', sharpened_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            sharpened_img_base64 = base64.b64encode(sharpened_img_encoded).decode('utf-8')
            
            # Append original and sharpened images to results
            results.append({'original_image': original_img_base64, 'sharpened_image': sharpened_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response


@app.route('/wiener_deblur', methods=['POST'])
def wiener_deblur():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Define a simple blur filter
            psf = np.ones((5, 5)) / 25

            # Simulate blurred and noisy image
            blurred_noisy_img = np.copy(image)
            blurred_noisy_img = convolve2d(blurred_noisy_img, psf, 'same')
            blurred_noisy_img += 0.1 * blurred_noisy_img.std() * np.random.standard_normal(blurred_noisy_img.shape)

            # Apply Wiener filter
            deconvolved_img = restoration.wiener(blurred_noisy_img, psf, balance=0.1)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, deconvolved_img_encoded = imencode('.png', deconvolved_img)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            deconvolved_img_base64 = base64.b64encode(deconvolved_img_encoded).decode('utf-8')
            
            # Append original and deconvolved images to results
            results.append({'original_image': original_img_base64, 'deconvolved_image': deconvolved_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response


@app.route('/pyramid_multiscale', methods=['POST'])
def pyramid_multiscale():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Define the number of pyramid levels
            num_levels = 4

            # Gaussian pyramid
            gaussian_pyramid = [image]
            for i in range(1, num_levels):
                # Apply Gaussian blur
                blurred = GaussianBlur(gaussian_pyramid[-1], (5, 5), 0)
                # Downsample
                downsampled = pyrDown(blurred)
                gaussian_pyramid.append(downsampled)

            # Laplacian pyramid
            laplacian_pyramid = []
            for i in range(num_levels - 1):
                # Upsample the next level
                upsampled = pyrUp(gaussian_pyramid[i + 1])
                # Ensure the size matches the current level in the Gaussian pyramid
                upsampled = resize(upsampled, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
                # Subtract the upsampled image from the current level in the Gaussian pyramid
                laplacian = subtract(gaussian_pyramid[i], upsampled)
                laplacian_pyramid.append(laplacian)
            laplacian_pyramid.append(gaussian_pyramid[-1])  # Append the top level of the Gaussian pyramid
            
            # Convert images to bytes and encode them into base64 strings
            pyramid_images_base64 = []
            for img in gaussian_pyramid + laplacian_pyramid:
                _, img_encoded = imencode('.png', img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                pyramid_images_base64.append(img_base64)
            
            # Append base64 encoded images to results
            results.append({'pyramid_images': pyramid_images_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response




@app.route('/wavelet_multiscale', methods=['POST'])
def wavelet_multiscale():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Perform wavelet transform
            coeffs = pywt.dwt2(image, 'haar')
            
            # Reconstruction
            reconstructed_image = pywt.idwt2(coeffs, 'haar')
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, reconstructed_img_encoded = imencode('.png', reconstructed_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            reconstructed_img_base64 = base64.b64encode(reconstructed_img_encoded).decode('utf-8')
            
            # Append original and reconstructed images to results
            results.append({'original_image': original_img_base64, 'reconstructed_image': reconstructed_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



def log_compression(image):
    # Convert image to float32 for accurate calculations
    image_float = image.astype(np.float32)   
    # Add a small constant to prevent taking logarithm of zero
    constant = 1
    # Apply logarithmic transformation
    log_image = np.log1p(image_float + constant)
    # Normalize the logarithmic image to the range [0, 255]
    log_image = (log_image - np.min(log_image)) / (np.max(log_image) - np.min(log_image)) * 255
    # Convert back to uint8
    log_image = log_image.astype(np.uint8)
    return log_image

@app.route('/log_compression_dynamic_range', methods=['POST'])
def log_compression_dynamic_range():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Apply log compression
            compressed_image = log_compression(image)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, compressed_img_encoded = imencode('.png', compressed_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            compressed_img_base64 = base64.b64encode(compressed_img_encoded).decode('utf-8')
            
            # Append original and compressed images to results
            results.append({'original_image': original_img_base64, 'compressed_image': compressed_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/exponential_dynamic_range', methods=['POST'])
def exponential_dynamic_range():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Define gamma value
            gamma = 1.5

            # Apply exponential transformation
            exponential_image = np.power(image/255.0, gamma) * 255.0
            
            # Convert back to uint8
            exponential_image = exponential_image.astype(np.uint8)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, exponential_img_encoded = imencode('.png', exponential_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            exponential_img_base64 = base64.b64encode(exponential_img_encoded).decode('utf-8')
            
            # Append original and exponential images to results
            results.append({'original_image': original_img_base64, 'exponential_image': exponential_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/morphological_process', methods=['POST'])
def morphological_process():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Define a kernel (structuring element)
            kernel = np.ones((5,5), np.uint8)  # 5x5 square kernel
            
            # Perform erosion
            erosion = erode(image, kernel, iterations=1)
            
            # Perform dilation
            dilation = dilate(image, kernel, iterations=1)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, erosion_img_encoded = imencode('.png', erosion)
            _, dilation_img_encoded = imencode('.png', dilation)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            erosion_img_base64 = base64.b64encode(erosion_img_encoded).decode('utf-8')
            dilation_img_base64 = base64.b64encode(dilation_img_encoded).decode('utf-8')
            
            # Append original, erosion, and dilation images to results
            results.append({'original_image': original_img_base64, 'erosion_image': erosion_img_base64, 'dilation_image': dilation_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/geometrical_transform', methods=['POST'])
def geometrical_transform():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Scaling
            scale_factor = 0.5  # Scale factor for resizing the image
            scaled_image = resize(image, None, fx=scale_factor, fy=scale_factor)
            
            # Rotation
            rows, cols = image.shape[:2]
            rotation_angle = 45  # Rotation angle in degrees
            rotation_matrix = getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)  # 1 is the scale factor
            rotated_image = warpAffine(image, rotation_matrix, (cols, rows))
            
            # Translation
            translation_x = 50  # Translation along x-axis
            translation_y = 30  # Translation along y-axis
            translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            translated_image = warpAffine(image, translation_matrix, (cols, rows))
            
            # Warping
            # Define source and destination points for warping
            source_points = np.float32([[50, 50], [200, 50], [50, 200]])
            destination_points = np.float32([[10, 100], [200, 50], [100, 250]])
            # Compute the perspective transformation matrix
            warp_matrix = getAffineTransform(source_points, destination_points)
            # Apply the perspective transformation
            warped_image = warpAffine(image, warp_matrix, (cols, rows))
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, scaled_img_encoded = imencode('.png', scaled_image)
            _, rotated_img_encoded = imencode('.png', rotated_image)
            _, translated_img_encoded = imencode('.png', translated_image)
            _, warped_img_encoded = imencode('.png', warped_image)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            scaled_img_base64 = base64.b64encode(scaled_img_encoded).decode('utf-8')
            rotated_img_base64 = base64.b64encode(rotated_img_encoded).decode('utf-8')
            translated_img_base64 = base64.b64encode(translated_img_encoded).decode('utf-8')
            warped_img_base64 = base64.b64encode(warped_img_encoded).decode('utf-8')
            
            # Append all images to results
            results.append({'original_image': original_img_base64,
                            'scaled_image': scaled_img_base64,
                            'rotated_image': rotated_img_base64,
                            'translated_image': translated_img_base64,
                            'warped_image': warped_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/morhological_operation', methods=['POST'])
def morhological_operation():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Define structuring element (kernel)
            kernel = np.ones((5,5), np.uint8)
            
            # Perform morphological operations
            erosion = erode(image, kernel, iterations=1)
            dilation = dilate(image, kernel, iterations=1)
            opening = morphologyEx(image, MORPH_OPEN, kernel)
            closing = morphologyEx(image, MORPH_CLOSE, kernel)
            gradient = morphologyEx(image, MORPH_GRADIENT, kernel)
            tophat = morphologyEx(image, MORPH_TOPHAT, kernel)
            bottomhat = morphologyEx(image, MORPH_BLACKHAT, kernel)
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, erosion_img_encoded = imencode('.png', erosion)
            _, dilation_img_encoded = imencode('.png', dilation)
            _, opening_img_encoded = imencode('.png', opening)
            _, closing_img_encoded = imencode('.png', closing)
            _, gradient_img_encoded = imencode('.png', gradient)
            _, tophat_img_encoded = imencode('.png', tophat)
            _, bottomhat_img_encoded = imencode('.png', bottomhat)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            erosion_img_base64 = base64.b64encode(erosion_img_encoded).decode('utf-8')
            dilation_img_base64 = base64.b64encode(dilation_img_encoded).decode('utf-8')
            opening_img_base64 = base64.b64encode(opening_img_encoded).decode('utf-8')
            closing_img_base64 = base64.b64encode(closing_img_encoded).decode('utf-8')
            gradient_img_base64 = base64.b64encode(gradient_img_encoded).decode('utf-8')
            tophat_img_base64 = base64.b64encode(tophat_img_encoded).decode('utf-8')
            bottomhat_img_base64 = base64.b64encode(bottomhat_img_encoded).decode('utf-8')
            
            # Append all images to results
            results.append({'original_image': original_img_base64,
                            'erosion_image': erosion_img_base64,
                            'dilation_image': dilation_img_base64,
                            'opening_image': opening_img_base64,
                            'closing_image': closing_img_base64,
                            'gradient_image': gradient_img_base64,
                            'tophat_image': tophat_img_base64,
                            'bottomhat_image': bottomhat_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response




@app.route('/color_space_trasform', methods=['POST'])
def color_space_trasform():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_COLOR)
            
            # Convert BGR to RGB
            rgb_image = cvtColor(image, COLOR_BGR2RGB)
            
            # Convert RGB to HSV
            hsv_image = cvtColor(rgb_image, COLOR_RGB2HSV)
            
            # Convert RGB to Lab
            lab_image = cvtColor(rgb_image, COLOR_RGB2LAB)
            
            # Convert images to bytes
            _, rgb_img_encoded = imencode('.png', rgb_image)
            _, hsv_img_encoded = imencode('.png', hsv_image)
            _, lab_img_encoded = imencode('.png', lab_image)
            
            # Convert bytes to base64 strings
            rgb_img_base64 = base64.b64encode(rgb_img_encoded).decode('utf-8')
            hsv_img_base64 = base64.b64encode(hsv_img_encoded).decode('utf-8')
            lab_img_base64 = base64.b64encode(lab_img_encoded).decode('utf-8')
            
            # Append all images to results
            results.append({'rgb_image': rgb_img_base64,
                            'hsv_image': hsv_img_base64,
                            'lab_image': lab_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    return jsonify(results)



@app.route('/edge_detect', methods=['POST'])
def edge_detect():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_GRAYSCALE)
            
            # Apply Gaussian blur to reduce noise
            blurred = GaussianBlur(image, (3, 3), 0)
            
            # Apply Canny edge detector
            edges = Canny(blurred, 30, 150)  # Adjust these thresholds as needed
            
            # Convert images to bytes
            _, original_img_encoded = imencode('.png', image)
            _, edges_img_encoded = imencode('.png', edges)
            
            # Convert bytes to base64 strings
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')
            edges_img_base64 = base64.b64encode(edges_img_encoded).decode('utf-8')
            
            # Append original and edges images to results
            results.append({'original_image': original_img_base64, 'edges_image': edges_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    response = jsonify(results)
    return response



@app.route('/roi_otsu', methods=['POST'])
def roi_otsu():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_COLOR)
            
            # Convert to grayscale
            gray = cvtColor(image, COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding
            ret, thresh = threshold(gray, 0, 255, THRESH_BINARY_INV+THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((2, 2), np.uint8)
            closing = morphologyEx(thresh, MORPH_CLOSE, kernel, iterations=2)
            
            # Sure background area
            sure_bg = dilate(closing, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = distanceTransform(sure_bg, DIST_L2, 3)
            ret, sure_fg = threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = subtract(sure_bg, sure_fg)
            
            # Marker labeling
            ret, markers = connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            # Mark the region of unknown with zero
            markers[unknown == 255] = 0
            # Watershed algorithm
            markers = watershed(image, markers)
            image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries
            
            # Convert images to bytes
            _, img_encoded = imencode('.png', image)
            _, thresh_encoded = imencode('.png', thresh)
            
            # Convert bytes to base64 strings
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            thresh_base64 = base64.b64encode(thresh_encoded).decode('utf-8')
            
            # Append original and thresholded images to results
            results.append({'original_image': img_base64, 'thresholded_image': thresh_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    
    return jsonify(results)


@app.route('/roi_weatershead', methods=['POST'])
def roi_weatershead():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            img = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_COLOR)
            
            # Convert BGR to RGB
            b,g,r = split(img)
            rgb_img = merge([r,g,b])
            
            # Convert to grayscale
            gray = cvtColor(img, COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding
            ret, thresh = threshold(gray, 0, 255, THRESH_BINARY_INV+THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((2, 2), np.uint8)
            closing = morphologyEx(thresh, MORPH_CLOSE, kernel, iterations=2)
            
            # Sure background area
            sure_bg = dilate(closing, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = distanceTransform(sure_bg, DIST_L2, 3)
            ret, sure_fg = threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = subtract(sure_bg, sure_fg)
            
            # Marker labeling
            ret, markers = connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            # Mark the region of unknown with zero
            markers[unknown == 255] = 0
            # Watershed algorithm
            markers = watershed(rgb_img, markers)
            img[markers == -1] = [255, 0, 0]  # Mark watershed boundaries
            
            # Convert images to bytes
            _, img_encoded = imencode('.png', img)
            _, thresh_encoded = imencode('.png', thresh)
            _, closing_encoded = imencode('.png', closing)
            _, sure_bg_encoded = imencode('.png', sure_bg)
            _, dist_transform_encoded = imencode('.png', dist_transform)
            _, sure_fg_encoded = imencode('.png', sure_fg)
            _, unknown_encoded = imencode('.png', unknown)
            
            # Convert bytes to base64 strings
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            thresh_base64 = base64.b64encode(thresh_encoded).decode('utf-8')
            closing_base64 = base64.b64encode(closing_encoded).decode('utf-8')
            sure_bg_base64 = base64.b64encode(sure_bg_encoded).decode('utf-8')
            dist_transform_base64 = base64.b64encode(dist_transform_encoded).decode('utf-8')
            sure_fg_base64 = base64.b64encode(sure_fg_encoded).decode('utf-8')
            unknown_base64 = base64.b64encode(unknown_encoded).decode('utf-8')
            
            # Append images to results
            results.append({'original_image': img_base64,
                            'thresholded_image': thresh_base64,
                            'closing_image': closing_base64,
                            'sure_bg_image': sure_bg_base64,
                            'dist_transform_image': dist_transform_base64,
                            'sure_fg_image': sure_fg_base64,
                            'unknown_image': unknown_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    
    return jsonify(results)



@app.route('/roi_prewitt', methods=['POST'])
def roi_prewitt():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            input_image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_COLOR)
            
            # Convert to grayscale
            gray_image = cvtColor(input_image, COLOR_BGR2GRAY)
            
            # Pre-allocate the filtered_image matrix with zeros
            filtered_image = np.zeros_like(gray_image)
            
            # Prewitt Operator Mask
            Mx = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])
            My = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])
            
            # Edge Detection Process
            for i in range(gray_image.shape[0] - 2):
                for j in range(gray_image.shape[1] - 2):
                    # Gradient approximations
                    Gx = np.sum(Mx * gray_image[i:i+3, j:j+3])
                    Gy = np.sum(My * gray_image[i:i+3, j:j+3])
                    # Calculate magnitude of vector
                    filtered_image[i+1, j+1] = np.sqrt(Gx**2 + Gy**2)
            
            # Convert to uint8 for display
            filtered_image_uint8 = np.uint8(filtered_image)
            
            # Define a threshold value
            thresholdValue = 100
            
            # Thresholding
            _, output_image = threshold(filtered_image_uint8, thresholdValue, 255, THRESH_BINARY)
            
            # Convert images to bytes
            _, input_img_encoded = imencode('.png', input_image)
            _, output_img_encoded = imencode('.png', output_image)
            
            # Convert bytes to base64 strings
            input_img_base64 = base64.b64encode(input_img_encoded).decode('utf-8')
            output_img_base64 = base64.b64encode(output_img_encoded).decode('utf-8')
            
            # Append original and filtered images to results
            results.append({'original_image': input_img_base64, 'filtered_image': output_img_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    
    return jsonify(results)


@app.route('/roi_felzenswalb', methods=['POST'])
def roi_felzenswalb():
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Read the image
            input_image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_COLOR)
            
            # Convert to grayscale
            gray_image = cvtColor(input_image, COLOR_BGR2GRAY)
            
            # Apply Canny edge detector
            edges = Canny(gray_image, 30, 150)
            
            # Apply Felzenszwalb segmentation
            segments_fz = felzenszwalb(edges, scale=100, sigma=0.5, min_size=50)
            
            # Convert images to bytes
            _, input_img_encoded = imencode('.png', input_image)
            _, segments_fz_encoded = imencode('.png', segments_fz)
            
            # Convert bytes to base64 strings
            input_img_base64 = base64.b64encode(input_img_encoded).decode('utf-8')
            segments_fz_base64 = base64.b64encode(segments_fz_encoded).decode('utf-8')
            
            # Append original and segmented images to results
            results.append({'original_image': input_img_base64, 'segmented_image': segments_fz_base64})
        else:
            results.append({'error': 'File type is not allowed'})
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
