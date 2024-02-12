import os

import cv2
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
import torch

from src.pytorch_openpose import util
from src.pytorch_openpose.model import bodypose_model,bodypose_25_model
import boto3
import io
session = boto3.Session(aws_access_key_id=os.getenv('aws_access_key_id'), aws_secret_access_key=os.getenv('aws_secret_access_key'))
s3 = boto3.client('s3')
bucket_name = 'aibodymeasurement'
file_key = 'body_25.pth'

model_coco = 'model/body_coco.pth'
model_body25 = 'model/body_25.pth'
# model_body25 = 's3://aibodymeasurement/body_25.pth'
class torch_openpose(object):
    def __init__(self, model_type):
        if model_type == 'body_25':
            self.model = bodypose_25_model()
            self.njoint = 26
            self.npaf = 52
            # obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            # buffer = io.BytesIO(obj['Body'].read())
            # self.model.load_state_dict(torch.load(buffer))
            # print('load model finished')

            self.model.load_state_dict(torch.load(model_body25))
        else:
            self.model = bodypose_model()
            self.njoint = 19
            self.npaf = 38
            self.model.load_state_dict(torch.load(model_coco))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        if self.njoint == 19:  #coco
            self.limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                   [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                   [0, 15], [15, 17]]
            self.mapIdx = [[12, 13],[20, 21],[14, 15],[16, 17],[22, 23],[24, 25],[0, 1],[2, 3],\
                           [4, 5],[6, 7],[8, 9],[10, 11],[28, 29],[30, 31],[34, 35],[32, 33],\
                               [36, 37]]
        elif self.njoint == 26:  #body_25
            self.limbSeq = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],\
                            [10,11],[8,12],[12,13],[13,14],[0,15],[0,16],[15,17],[16,18],\
                                [11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
            self.mapIdx = [[30, 31],[14, 15],[16, 17],[18, 19],[22, 23],[24, 25],[26, 27],[0, 1],[6, 7],\
                           [2, 3],[4, 5],  [8, 9],[10, 11],[12, 13],[32, 33],[34, 35],[36,37],[38,39],\
                               [50,51],[46,47],[44,45],[40,41],[48,49],[42,43]]
            

    def __call__(self, oriImg):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], self.njoint))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], self.npaf))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                heatmap, paf = self.model(data)

            
            heatmap = heatmap.detach().cpu().numpy()
            paf = paf.detach().cpu().numpy()

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
            heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # output 1 is heatmaps

            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # output 0 is PAFs
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += + paf / len(multiplier)

        all_peaks = []
        peak_counter = 0
        
        for part in range(self.njoint - 1):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        
        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = self.limbSeq
        # the middle joints heatmap correpondence
        mapIdx = self.mapIdx

        connection_all = []
        special_k = []
        mid_num = 10


        # return all_peaks
        for k in range(len(mapIdx)):
            # two keypoints part affinity field
            score_mid = paf_avg[:, :, mapIdx[k]]
            # first keypoint position
            candA = all_peaks[limbSeq[k][0]]
            # second keypoints position
            candB = all_peaks[limbSeq[k][1]]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        # first -> second vector
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        # vector norm
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # normalized vector
                        vec = np.divide(vec, norm)
                        # mid vector
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        # first Or X-axis keypoint PAF mid value vector
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        # second Or Y-axis keypoint PAF mid value vector
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
                        # score along the first -> second vector
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, self.njoint + 1))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        # data_viz = np.zeros((1024, 768))
        # for x in candidate:
        #     data_viz[int(x[1]) - 3:int(x[1]) + 3 , int(x[0]) - 3: int(x[0]) + 3] = int(x[2] * 255)
        #     cv2.drawMarker(oriImg, (int(x[0]), int(x[1])), color=(255,255,255), markerType=cv2.MARKER_SQUARE, markerSize=5, thickness=5)
        #     pos_keypoints = (int(x[0] - 3), int(x[1]))
        #     if (x[3] > 25):
        #         pos_keypoints = (int(x[0] - 3), int(x[1] - 3))
        #     cv2.putText(oriImg, str(x[3]), pos_keypoints, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        # mask = (data_viz > 100)
        # mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)



        # cv2.imshow('origin', oriImg)
        #
        # cv2.waitKey(0)
        # cv2.imshow('candidate visualization', data_viz)
        # cv2.waitKey(0)
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k])

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found:
                        row = -1 * np.ones(self.njoint + 1)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        
        poses = []
        for per in subset:
            pose = []
            for po in per[:-2]:
                if po >= 0:
                    joint = list(candidate[int(po)][:3])
                else:
                    joint = [0.,0.,0.]
                pose.append(joint)
            poses.append(pose)

        return poses


