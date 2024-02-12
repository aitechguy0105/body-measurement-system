from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()
import uuid
from datetime import datetime
from src.main import body_type_measurement
import cv2 as cv
import os
import numpy as np
import boto3
from pathlib import Path
from flask import render_template
app = Flask(__name__)
engine = body_type_measurement()

@app.route('/', methods=['GET'])
def main():
    return render_template("index.html")
    # return 'hello'
@app.route('/image/<string:img_name>')
def get_image(img_name):
    # Logic to retrieve or generate the image
    image_data = open(f'{img_name}.jpg', 'rb').read()  # Read the image file as binary data
    return image_data, 200, {'Content-Type': 'image/jpeg'}
@app.route('/type', methods=['POST'])
def handle_images():

    image_front = request.files['image_front']
    image_back = request.files['image_back']
    image_left = request.files['image_left']
    image_back = request.files.get('image_back')
    height = request.form['height']
    if image_front.filename == '' or image_back.filename == '' or image_left.filename == '' or height == '':
        return 'Please provide all required image files.', 400

    height = float(height)

    image_front_data = image_front.read()
    image_front_array = np.frombuffer(image_front_data, np.uint8)
    image_front = cv.imdecode(image_front_array, cv.IMREAD_COLOR)
    height1, width1, _ = image_front.shape

    image_back_data = image_back.read()
    image_back_array = np.frombuffer(image_back_data, np.uint8)
    image_back = cv.imdecode(image_back_array, cv.IMREAD_COLOR)
    height2, width2, _ = image_back.shape

    image_left_data = image_left.read()
    image_left_array = np.frombuffer(image_left_data, np.uint8)
    image_left = cv.imdecode(image_left_array, cv.IMREAD_COLOR)
    height3, width3, _ = image_left.shape

    if height1 < 1024 or width1 < 768 or height2 < 1024 or width2 < 768 or height3 < 1024 or width3 < 768 :
        return "Image Should be at least 1024x768 pixels"

    img_f, img_s, img_b, result = engine.handle_one_user_images(img_f=image_front, img_b=image_back, img_s=image_left, height=height)
    # Perform analysis on the images
    cv.imwrite("image_front.jpg", img_f)
    # cv.waitKey(0)
    cv.imwrite("image_left.jpg", img_s)
    cv.imwrite("image_back.jpg", img_b)
    s3 = boto3.client('s3')

    # Specify the S3 bucket and the key (file name) under which the image will be stored
    bucket_name = 'aibodymeasurement'

    for img_path in Path('./').glob('*.jpg'):


        current_date_time = datetime.now()
        formatted_date_time = current_date_time.strftime("%Y%m%d%H%M%S")
        # Generate a unique identifier
        unique_id = str(uuid.uuid4())
        # Concatenate the date/time and unique identifier
        file_key = "analysis_images/" + formatted_date_time + "_" + img_path.stem + "_" + unique_id + ".jpg"
        local_file_path = str(img_path)
        # Upload the image to S3
        s3.upload_file(local_file_path, bucket_name, file_key)
    # cv.waitKey(0)
    # Replace this with your actual analysis code

    analysis_data = {
        'image1': 'analysis_result_1',
        'image_back': 'analysis_result_2'
    }
    str_fit_type = ['Regular', 'Extremly Slim', 'VerySlim', 'Slim', 'Regular Tight', 'Regular Loose', 'Loose',
                    'Very Loose', 'Extremely Loose']
    str_hip_type = ["Hip Normal", "Flat Seat", "Prominent Seat", "Drop Seat"]
    str_waist_type = ["Waist Normal", "Slightly Concative Waist", "Very Concative Waist"]
    str_chest_type = ['Normal Chest', 'Chest Out', 'Well built Chest']
    str_slope_type = ['5.0~8.0', '8.0~12.5', '12.5~16', '16~20.5', '20.5~22.0', '22.0~23.5', '23.5~24.5']
    str_shoulder_bulge_blade_type = ['Blade Normal', 'Bulge Blader', 'Bulge Top Blade ']
    str_shouler_blade_inner_type = ['Shoulder Blade Normal', 'Shoulder Blade Innder']
    str_arm_type = ["Arm_Normal", "Arm_Slight_Backward", "Arm_Very_Backward", "Arm_Very_Forward", "Arm Curved",
                    "Arm_Straight"]
    str_back_type = ["Back Normal", "Forward Head", "Humpack", " Humpack + Forward Head", "Back Style S"]
    str_body_basic_type = ["Body Normal", "Body Forward", "Body Backward", "Body Stooped"]
    str_belly_type = ["Belly Normal", "Belly-Tenesmic stomach high", "Bend forward of abdomen", "Belly Portly",
                      "Normal Fat Around"]
    result['fit'] = str_fit_type[result['fit']]
    result['hip'] = str_hip_type[result['hip']]
    result['waist'] = str_waist_type[result['waist']]
    result['chest'] = str_chest_type[result['chest']]
    result['shoulder_blade_inner_type'] = str_shouler_blade_inner_type[result['shoulder_blade_inner_type']]
    result['r_shoulder_angle'] = str_slope_type[result['r_shoulder_angle']]
    result['l_shoulder_angle'] = str_slope_type[result['l_shoulder_angle']]
    result['shoulder_buldge_blade'] = str_shoulder_bulge_blade_type[result['shoulder_buldge_blade']]
    result['arm'] = str_arm_type[result['arm']]
    result['back'] = str_back_type[result['back']]
    result['body_type'] = str_body_basic_type[result['body_type']]
    result['belly'] = str_belly_type[result['belly']]
    # Return the analysis data in JSON format
    return render_template('index.html', result = result)

if __name__ == '__main__':


    # app.run(debug=1)
    app.run(host='0.0.0.0', port=5000)
