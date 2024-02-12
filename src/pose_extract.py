import numpy as np
import cv2 as cv
import json
import platform
import argparse
from pathlib import Path
from src.util import  preprocess_image
import os
import sys

OPENPOSE_PATH = '../../openpose-build/bin/'
OPENPOSE_MODEL_PATH = '../../openpose-build/models/'


def get_points(file_name):
    f = open(file_name)

    data = json.load(f)

    for i in data['people']:
        points = i['pose_keypoints_2d']
        coordinates = []
        for point in points:
            coordinates.append(point)



    f.close()
    return coordinates

# if platform.system() == 'Linux':
#     if not os.path.isfile('../openpose/python/openpose/_openpose.so'):
#         print('openpen pose is not available: missing _openpose.so', file=sys.stderr)
# elif platform.system() == 'Windows':
#     if not os.path.isfile('../openpose/python/openpose/_openpose.dll'):
#         print('openpen pose is not available: missing _openpose.dll', file=sys.stderr)
# else:
#     print('not support os', file=sys.stderr)
#
# if not os.path.isfile('../openpose/python/openpose/openpose.py'):
#     print('openpen pose is not available: missing openpose.py', file=sys.stderr)
#
# if not os.path.isfile('../openpose/models/pose/body_25/pose_iter_584000.caffemodel'):
#     print('missing caffemodel pose_iter_584000.caffemodel', file=sys.stderr)
#
# sys.path.append(OPENPOSE_PATH)
# try:
#     from openpose import *
# except:
#     raise Exception(
#         'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

def find_pose(img):
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = OPENPOSE_MODEL_PATH
    # Construct OpenPose object allocates GPU memory
    # openpose = OpenPose(params)
    #
    # keypoints, img_pose = openpose.forward(img, True)
    #
    # return keypoints, img_pose
    print('findPose Error function invoked')
    return

class PoseExtractor:
    def __init__(self):
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["default_model_folder"] = OPENPOSE_MODEL_PATH
        # Construct OpenPose object allocates GPU memory
        print('loading Openpose model....')
        # self.openpose = OpenPose(params)
        print('loaded Openpose model sucessfully')

    def extract_pose(self, img, debug = True):
        if debug == True:
            return
            # keypoints, img_pose = self.openpose.forward(img, True)
            # return keypoints, img_pose
        else:
            # keypoints = self.openpose.forward(img, False)
            current_directory = os.getcwd()
            os.chdir("../../openpose-build")
            print(os.getcwd())
            os.system(
                'bin\OpenPoseDemo.exe --net_resolution -' + '1x368' + ' --image_dir ' + '../body_measure-master/data/resized_images' + ' --write_json '
                + '../body_measure-master/data/pose' + ' --write_images ' + '../body_measure-master/data/pose' + ' --display ' + '0')

            os.chdir(current_directory)
            print(os.getcwd())


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input_dir", required=True, help="image folder")
    # ap.add_argument("-o", "--output_dir", required=True, help='output pose dir')
    # args = vars(ap.parse_args())
    # DIR_IN = args['input_dir'] + '/'
    # DIR_OUT = args['output_dir'] + '/'
    #
    # if not os.path.exists(DIR_OUT):
    #     os.makedirs(DIR_OUT)
    #
    DIR_IN = '../data/images'
    DIR_OUT = f'../data/resized_images/'
    for f in Path(DIR_OUT).glob('*.*'):
        os.remove(f)

    # DIR_IN = '../../specification/arm'
    for img_path in Path(DIR_IN).glob('*.*'):

        img = cv.imread(str(img_path))
        img = preprocess_image(img)
        cv.imwrite(f'../data/resized_images/{img_path.stem}.jpg', img)

    extractor = PoseExtractor()
    #
    extractor.extract_pose(None, False)

    for json_path in Path('../data/pose').glob('*.json'):
        keypoints = get_points(json_path)
        keypoints = np.array(keypoints)
        keypoints.resize(1, len(keypoints) // 3, 3)
        np.save(f'../data/pose/{json_path.stem}', keypoints)


    # print(keypoints)
    # np.save('../../openpose-build/out_images/left_1', keypoints)
    # for img_path in Path(DIR_IN).glob('*.*'):
    #     print(img_path)
    #     img = cv.imread(str(img_path))
    #     img = preprocess_image(img)
    #     # keypoints, img_pose = extractor.extract_pose(img, debug=False)
    #     keypoints = extractor.extract_pose(img, debug=False)
    # keypoints = get_points(f'./out_images/front_keypoints.json')
    #
    # cv.imwrite(f'{DIR_OUT}/{img_path.stem}.png', img_pose)
    # np.save(f'{DIR_OUT}/{img_path.stem}.npy', keypoints)



