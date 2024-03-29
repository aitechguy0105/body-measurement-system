import sys
import cv2 as cv
import os
import numpy as np
import numpy.linalg as linalg
import argparse
from pathlib import Path
from src.silhouette_deeplab import DeeplabWrapper
import tensorflow as tf
import matplotlib.pyplot as plt
from src.util import  preprocess_image
from src.util import is_valid_keypoint, is_valid_keypoint_1, pair_length, pair_dir, orthor_dir, extend_segment, find_largest_contour, int_tuple
from src.util import  POSE_BODY_25_BODY_PART_IDXS
from src.pose_to_trimap import gen_fg_bg_masks, head_center_estimate

def extend_rect(rect, percent_w, percent_h):
    w_ext = percent_w * rect[2]
    x = int(rect[0] - 0.5 * w_ext)
    w = int(rect[2] + w_ext)

    h_ext = percent_h * rect[3]
    y = int(rect[1] - 0.5 * h_ext)
    h = int(rect[3] + h_ext)

    return (x, y, w, h)

#rect: x, y, w, h
def grabcut_local_window(img, sil, sure_fg_mask = None, sure_bg_mask = None, rect = None, img_viz = None):
    img_rect   = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    sil_rect   = sil[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    sil_mask_rect = sil_rect > 0

    mask = np.zeros_like(sil_rect, dtype=np.uint8)
    mask[:] = cv.GC_PR_BGD
    mask[sil_mask_rect] = cv.GC_PR_FGD
    if sure_fg_mask is not None:
        sure_fg_mask_rect = sure_fg_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        mask[sure_fg_mask_rect] = cv.GC_FGD
    if sure_bg_mask is not None:
        sure_bg_mask_rect = sure_bg_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        mask[sure_bg_mask_rect] = cv.GC_BGD

    if img_viz is not None:
        color = np.random.randint(0, 255, 3)
        cv.rectangle(img_viz, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color=int_tuple(color), thickness=3)

    for i in range(2):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        # cv.grabCut(img_rect, mask, (1, 1, img_rect.shape[0], img_rect.shape[1]), bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
        cv.grabCut(img_rect, mask, None, bgdmodel, fgdmodel, 2, cv.GC_INIT_WITH_MASK)
        sil_1 = np.where((mask == cv.GC_PR_FGD) + (mask == cv.GC_FGD), 255, 0).astype('uint8')
        # plt.imshow(sil_1)
    if img_viz is not None:
        edges = cv.Canny(sil_1, 5, 20)
        edges = cv.morphologyEx(edges, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
        rect_viz = img_viz[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
        rect_viz[edges> 0] = (0,0,255)
        # plt.imshow(img_viz[:,:,::-1])
        # plt.imshow(sure_fg_mask, alpha=0.2, cmap='Wistia')
        # plt.imshow(sure_bg_mask, alpha=0.2, cmap='cool')
        # plt.show()

    return sil_1

def rect_bounds(points, shape):
    points = np.expand_dims(points, axis=1)
    points = points.astype(np.int32)

    points[:,0,0] = points[:,0,0].clip(min=0, max= shape[1])
    points[:,0,1] = points[:,0,1].clip(min=0, max= shape[0])

    x, y, w, h = cv.boundingRect(points)
    return (x, y, w, h)

def rect_bounds_intersect(rect_0, rect_1):
    x = np.max(rect_0[0], rect_1[0])
    y = np.max(rect_0[1], rect_1[1])

    x_1 = np.min(rect_0[0] + rect_0[2], rect_1[0] + rect_1[2])
    y_1 = np.min(rect_0[1] + rect_0[3], rect_1[1] + rect_1[3])

    return (x, y, x_1 - x, y_1 - y)

def grabcut_local_window_head_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    nose = []
    if is_front == False:
        nose = (keypoints[POSE_BODY_25_BODY_PART_IDXS['REar']][:2] + keypoints[POSE_BODY_25_BODY_PART_IDXS['LEar']][:2]) / 2
    else:
        nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    dir = nose - neck
    over_head = neck + 2 * dir

    points = np.vstack([neck, over_head, lshoulder, rshoulder])
    x, y, w, h = rect_bounds(points, img.shape)
    rect = (x, y, w, h)

    #for head, we shrink the sure foreground mask because DeepLab often has error around head area
    shrink_size = int(0.2*w)
    head_sure_fg_mask = sil[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].astype(np.uint8)
    head_sure_fg_mask = cv.morphologyEx(head_sure_fg_mask, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (shrink_size, shrink_size)))
    sil[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = head_sure_fg_mask
    # plt.imshow(img[:,:,::-1])
    # plt.imshow(sil, alpha=0.3)
    # plt.show()

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part


def grabcut_local_window_shoulder_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    if is_front == False:
        neck = (keypoints[POSE_BODY_25_BODY_PART_IDXS['REar']][:2] + keypoints[POSE_BODY_25_BODY_PART_IDXS['LEar']][
                                                                     :2]) / 2
    nose = keypoints[POSE_BODY_25_BODY_PART_IDXS['Nose']][:2]

    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    lshoulder_ext, rshoulder_ext = extend_segment(lshoulder, rshoulder, 0.5)

    neck_ext_0 = neck + 0.5*(nose - neck)
    neck_ext_1 = neck + 0.5*(neck - nose)

    points = np.vstack([lshoulder_ext, rshoulder_ext, neck_ext_0, neck_ext_1])
    x, y, w, h = rect_bounds(points, img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)

    return (x, y, w, h), sil_part

def grabcut_local_window_torso_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    lshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
    rshoulder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
    #lower shoulder a bit
    lshoulder[1] += 0.2 * linalg.norm(lshoulder-rshoulder)
    rshoulder[1] += 0.2 * linalg.norm(lshoulder-rshoulder)
    lshoulder_ext, rshoulder_ext = extend_segment(lshoulder, rshoulder, 0.25)

    hip  = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    hip[1] = hip[1] + 0.2 * linalg.norm(lshoulder-rshoulder)

    points = np.vstack([lshoulder_ext, rshoulder_ext, hip])
    x, y, w, h = rect_bounds(points, img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)

    return (x, y, w, h), sil_part

def grabcut_local_window_leg_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    ltoe = []
    rtoe = []
    if is_front == True:
        ltoe = keypoints[POSE_BODY_25_BODY_PART_IDXS['LBigToe']][:2]
        rtoe = keypoints[POSE_BODY_25_BODY_PART_IDXS['RBigToe']][:2]
    else:
        ltoe = keypoints[POSE_BODY_25_BODY_PART_IDXS['LAnkle']][:2]
        rtoe = keypoints[POSE_BODY_25_BODY_PART_IDXS['RAnkle']][:2]

    x, y, w, h = rect_bounds(np.vstack([midhip, ltoe, rtoe]), img.shape)
    x, y, w, h = extend_rect((x,y,w,h), 0.6, 0.3)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def grabcut_local_window_hand_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, left_hand, img_viz):
    if left_hand:
        lshouder = keypoints[POSE_BODY_25_BODY_PART_IDXS['LShoulder']][:2]
        lwrist   = keypoints[POSE_BODY_25_BODY_PART_IDXS['LWrist']][:2]
        lwrist   = lwrist + 0.5 * (lwrist - lshouder)
        points   = np.vstack([lshouder, lwrist])
    else:
        rshouder = keypoints[POSE_BODY_25_BODY_PART_IDXS['RShoulder']][:2]
        rwrist   = keypoints[POSE_BODY_25_BODY_PART_IDXS['RWrist']][:2]
        rwrist   = rwrist + 0.5 * (rwrist - rshouder)
        points   = np.vstack([rshouder, rwrist])

    x,y,w,h = rect_bounds(points, img.shape)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def refine_silhouette_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints, img_viz):
    rect_sils = []

    pair = grabcut_local_window_head_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_shoulder_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_torso_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_leg_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_hand_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, True, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_hand_front_img(is_front, img, sil, sure_fg_mask, sure_bg_mask, keypoints, False, img_viz)
    rect_sils.append(pair)

    sil_refined = np.zeros_like(sil)
    sil_tmp= np.zeros_like(sil)
    for pair in rect_sils:
        rect = pair[0]
        sil_part = pair[1]
        sil_tmp[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = sil_part
        sil_refined = np.bitwise_or(sil_refined, sil_tmp)

    #sil_refined = np.bitwise_and(sil_refined, sil)
    return sil_refined

def grabcut_local_window_head_side_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    head_center = head_center_estimate(keypoints)
    neck_head = head_center - neck
    over_head = head_center + neck_head
    p0 = head_center + orthor_dir(neck_head)
    p1 = head_center - orthor_dir(neck_head)
    x,y,w,h = rect_bounds(np.vstack([neck, over_head, p0, p1]), img.shape)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)

    return (x, y, w, h), sil_part

def grabcut_local_window_torso_side_img(img, sil, sil_bnd_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    neck = keypoints[POSE_BODY_25_BODY_PART_IDXS['Neck']][:2]
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]

    rect_ext = extend_rect(sil_bnd_rect, 0.4, 0)

    p0 = np.array((rect_ext[0],               neck[1] - 0.05 * sil_bnd_rect[3]))
    p1 = np.array((rect_ext[0] + rect_ext[2], neck[1] - 0.05 * sil_bnd_rect[3]))

    x,y,w,h = rect_bounds(np.vstack([midhip, p0, p1]), img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def grabcut_local_window_hip_thigh_img(img, sil, sil_bnd_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    midhip = keypoints[POSE_BODY_25_BODY_PART_IDXS['MidHip']][:2]
    if is_valid_keypoint_1(keypoints, 'LKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    elif is_valid_keypoint_1(keypoints, 'RKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
    else:
        print('missing keypoints: no knee founded')
        knee = (0.5*img.shape[0], 0.7*img.shape[1])

    rect_ext = extend_rect(sil_bnd_rect, 0.4, 0)
    p0 = np.array((rect_ext[0],               midhip[1] - 0.05 * sil_bnd_rect[3]))
    p1 = np.array((rect_ext[0] + rect_ext[2], midhip[1] - 0.05 * sil_bnd_rect[3]))
    p2 = np.array((rect_ext[0],               knee[1] + 0.05 * sil_bnd_rect[3]))
    p3 = np.array((rect_ext[0] + rect_ext[2], knee[1] + 0.05 * sil_bnd_rect[3]))

    x,y,w,h = rect_bounds(np.vstack([p0, p1, p2, p3]), img.shape)
    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def grabcut_local_window_leg_side_img(img, sil, sil_bnd_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz):
    if is_valid_keypoint_1(keypoints, 'LKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['LKnee']][:2]
    elif is_valid_keypoint_1(keypoints, 'RKnee'):
        knee = keypoints[POSE_BODY_25_BODY_PART_IDXS['RKnee']][:2]
    else:
        print('missing keypoints: no knee founded')
        knee = (0.5*img.shape[0], 0.7*img.shape[1])

    rect_ext = extend_rect(sil_bnd_rect, 0.2, 0.05)
    p0 = np.array((rect_ext[0], knee[1] - 0.05*sil_bnd_rect[3]))
    p1 = np.array((rect_ext[0] + rect_ext[2], knee[1]- 0.05*sil_bnd_rect[3]))
    p2 = np.array((rect_ext[0], rect_ext[1]+rect_ext[3]))
    p3 = np.array((rect_ext[0]+rect_ext[2], rect_ext[1]+rect_ext[3]))

    x,y,w,h = rect_bounds(np.vstack([p0, p1, p2, p3]), img.shape)

    sil_part = grabcut_local_window(img, sil, sure_fg_mask, sure_bg_mask, (x, y, w, h), img_viz)
    return (x, y, w, h), sil_part

def refine_silhouette_side_img(img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints, img_viz):
    sil_rect = cv.boundingRect(contour)

    rect_sils = []
    pair = grabcut_local_window_head_side_img(img, sil, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_torso_side_img(img, sil, sil_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_leg_side_img(img, sil, sil_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    pair = grabcut_local_window_hip_thigh_img(img, sil, sil_rect, sure_fg_mask, sure_bg_mask, keypoints, img_viz)
    rect_sils.append(pair)

    sil_refined = np.zeros_like(sil)
    sil_tmp= np.zeros_like(sil)
    for pair in rect_sils:
        rect = pair[0]
        sil_part = pair[1]
        sil_tmp[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = sil_part
        sil_refined = np.bitwise_or(sil_refined, sil_tmp)

    return sil_refined

def load_silhouette(path, img):
    sil = cv.imread(path, cv.IMREAD_GRAYSCALE)
    sil = cv.resize(sil, (img.shape[1], img.shape[0]), cv.INTER_NEAREST)
    ret, sil = cv.threshold(sil, 200, maxval=255, type=cv.THRESH_BINARY)
    return sil

def fix_silhouette(sil):
    sil = cv.morphologyEx(sil, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, ksize=(3,3)))
    return sil

class SilhouetteExtractor():
    #use_mobile_mode: True => less precise but much faster on CPU.
    #use_gpu: run deeplab inference on GPU
    def __init__(self, use_mobile_model = False, use_gpu=False):
        self.deeplab_wrapper = DeeplabWrapper(use_mobile_model=use_mobile_model, use_gpu=use_gpu)

    def extract_silhouette(self, img, img_category, keypoints, img_debug=None):
        sil = self.deeplab_wrapper.extract_silhouette(img)
        # plt.imshow(sil)
        bg_mask = cv.morphologyEx(sil, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (30, 30)))

        sure_fg_mask, _ = gen_fg_bg_masks(img, keypoints, front_view=True)
        sure_fg_mask = (sure_fg_mask == 255)
        sure_bg_mask = (bg_mask != 255)
        #
        # plt.imshow(img[:,:,::-1])
        # plt.imshow(sure_fg_mask, alpha=0.4)
        # plt.imshow(sure_bg_mask, alpha=0.4)
        # plt.show()

        contour = find_largest_contour(sil, cv.CHAIN_APPROX_TC89_L1)
        if img_debug is not None:
            cv.drawContours(img_debug, [contour], -1, color=(155,155,0), thickness=2)

        # note: in case of deeplab mobile model, the silhouette returned by deeplab is often wider than real silhouette,
        # therefore, we might need to erode it a bit to have a better approximation
        if not self.deeplab_wrapper.is_precise_model():
            sil = cv.morphologyEx(sil, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (13, 13)))

        if img_category == 0:
            sil_refined = refine_silhouette_front_img(True, img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints[0, :, :],
                                                      img_debug)
        elif img_category == 1:
            sil_refined = refine_silhouette_front_img(False, img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints[0, :, :],
                                                      img_debug)
        elif img_category == 2:
            sil_refined = refine_silhouette_side_img(img, sil, sure_fg_mask, sure_bg_mask, contour, keypoints[0, :, :],
                                                     img_debug)
        else:
            print("image category error")

        sil_refined = cv.morphologyEx(sil_refined, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
        contour_refined = find_largest_contour(sil_refined, cv.CHAIN_APPROX_TC89_L1)
        sil_final = np.zeros_like(sil)
        cv.fillPoly(sil_final, pts=[contour_refined], color=(255, 255, 255))
        # sil_refined = None
        return sil, sil_refined


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image_dir", required=True, help="image folder")
    # ap.add_argument("-p", "--pose_dir", required=True,  help="pose folder")
    # ap.add_argument("-o", "--output_dir", required=True, help='output silhouette dir')
    # args = vars(ap.parse_args())
    # IMG_DIR = args['image_dir'] + '/'
    # POSE_DIR = args['pose_dir'] + '/'
    # OUT_SILHOUETTE_DIR = args['output_dir'] + '/'
    #
    # tf.logging.set_verbosity(tf.logging.WARN)
    #
    # if not os.path.exists(OUT_SILHOUETTE_DIR):
    #     os.makedirs(OUT_SILHOUETTE_DIR)
    #

    IMG_DIR = "../data/resized_images"
    POSE_DIR = "../data/pose"
    OUT_SILHOUETTE_DIR = "../data/silhouette"
    for f in Path(OUT_SILHOUETTE_DIR).glob('*.*'):
        os.remove(f)
    sil_extractor = SilhouetteExtractor(use_mobile_model = False, use_gpu=False)

    for img_path in Path(IMG_DIR).glob('*.*'):
        print(img_path)
        pose_path = f'{POSE_DIR}/{img_path.stem}_keypoints.npy'
        if not os.path.isfile(pose_path):
            print('\t missing keypoint file', file=sys.stderr)
            continue
        #
        # if 'IMG_1930' not in str(img_path):
        #     continue

        if 'left_' in str(img_path):
            is_front_img = False
        elif 'front_' in str(img_path):
            is_front_img = True
        else:
            print('not a front or side image. please attach annation: front_ or side_ to front of the image name',
                  file=sys.stderr)

        img = cv.imread(str(img_path))
        img = preprocess_image(img)
        keypoints = np.load(pose_path)
        img_debug = img.copy()

        sil_deeplab, sil_refined = sil_extractor.extract_silhouette(img, is_front_img, keypoints, img_debug=img_debug)

        if img_debug is not None:
            contour = find_largest_contour(sil_deeplab, cv.CHAIN_APPROX_TC89_L1)
            cv.drawContours(img_debug, [contour], -1, color=(0,255,255), thickness=2)
            plt.imshow(img_debug)
        cv.imwrite(f'{OUT_SILHOUETTE_DIR}/{img_path.name}', sil_refined)
        cv.imwrite(f'{OUT_SILHOUETTE_DIR}/{img_path.stem}deeplab.jpg', sil_deeplab)
        cv.imwrite(f'{OUT_SILHOUETTE_DIR}/{img_path.stem}viz.jpg', img_debug)


        # # visualization
        # img_1 = img_org.copy()
        # cv.drawContours(img_1, [contour], -1, (255, 0, 0), thickness=3)
        # cv.drawContours(img_1, [contour_refined], -1, (0, 0, 255), thickness=3)
        #
        # plt.subplot(121), plt.imshow(img[:, :, ::-1])
        # plt.subplot(122), plt.imshow(img_1[:, :, ::-1])
        # #plt.show()
        # # plt.savefig(f'{OUT_MEASUREMENT_DIR}/{img_path.name}', dpi=1000)


