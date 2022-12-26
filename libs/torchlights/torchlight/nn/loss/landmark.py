import torch.nn as nn
import torch
import torch.nn.functional as F

# Unofficial Implementation of Landmark Loss in
# CrossNet++: Cross-scale Large-parallax Warping for Reference-based Super-resolution
    

def landmark_loss(source_lm, target_lm, flow, reduce='mean'):
    """ source_lm, target_lm: [B,N,2]
        flow: offset of the flow [B,2,W,H]
    """
    # since landmark cord can be fractional, we need interpolation to
    # get the flow offset of each landmark.
    
    W, H = flow.shape[-2], flow.shape[-1]
    
    flow_x = flow[:,0:1,:,:] # [B,1,W,H]
    flow_y = flow[:,1:2,:,:]
    
    x_normalized = source_lm[:,:,0:1] / (W/2) - 1
    y_normalized = source_lm[:,:,1:2] / (H/2) - 1
    source_lm_norm = torch.cat([x_normalized, y_normalized], dim=2).unsqueeze(1)
    
    dx = F.grid_sample(flow_x, source_lm_norm, align_corners=False) # [B,1,1,N]
    dy = F.grid_sample(flow_y, source_lm_norm, align_corners=False) 
    dx = dx.squeeze(dim=1).permute(0,2,1) # [B,N,1]
    dy = dy.squeeze(dim=1).permute(0,2,1)
    d = torch.cat([dx,dy], dim=2)
    
    warp_lm = source_lm + d
    
    loss = F.mse_loss(warp_lm, target_lm, reduction=reduce)
    
    return loss


class LandmarkLoss(nn.Module):
    """ source_lm, target_lm: source/target landmarks [B,N,2]
        flow: offset of the flow [B,2,W,H]
    """

    def forward(self, source_lm, target_lm, flow):
        return landmark_loss(source_lm, target_lm, flow)


def match_landmark_sift_knn_bbs(img1, img2):
    """ Match landmarks of two images with SIFT and KNN 

        Reference: https://github.com/ayushgarg31/Feature-Matching
    """

    import cv2

    t1 = cv2.imread(img1, 0)
    t2 = cv2.imread(img2, 0)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(t1, None)
    kp2, des2 = sift.detectAndCompute(t2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good1 = []
    for m, n in matches:
        if m.distance < 0.3*n.distance:
            good1.append([m])

    matches = bf.knnMatch(des2, des1, k=2)

    good2 = []
    for m, n in matches:
        if m.distance < 0.3*n.distance:
            good2.append([m])

    good = []

    for i in good1:
        img1_id1 = i[0].queryIdx
        img2_id1 = i[0].trainIdx

        (x1, y1) = kp1[img1_id1].pt
        (x2, y2) = kp2[img2_id1].pt

        for j in good2:
            img1_id2 = j[0].queryIdx
            img2_id2 = j[0].trainIdx

            (a1, b1) = kp2[img1_id2].pt
            (a2, b2) = kp1[img2_id2].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                good.append(i)

    match_points = []
    for g in good:
        img1_id1 = g[0].queryIdx
        img2_id1 = g[0].trainIdx
        (x1, y1) = kp1[img1_id1].pt
        (x2, y2) = kp2[img2_id1].pt

        match_points.append((y1,x1,y2,x2)) # in opencv, x -> H, y -> W

    visualize = cv2.drawMatchesKnn(t1, kp1, t2, kp2, good, None, [0, 0, 255], flags=2)

    return match_points, visualize
