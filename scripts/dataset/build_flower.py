import cv2
import os

root = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/Flowers_8bit'
target_dir = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/processed'

def build_flower(viewpoint): 
    x, y = viewpoint
    save_dir = os.path.join(target_dir, '{}_{}'.format(x,y))
    os.makedirs(save_dir)
    for name in os.listdir(root):
        print(name)
        path = os.path.join(root, name)
        img = cv2.imread(path)
        demosaic = img[x:,y:,:][::14, ::14, :]
        cv2.imwrite(os.path.join(save_dir, name), demosaic)


if __name__ == '__main__':
   # Each light ﬁeld image has 376 × 541 spatial samples, and 14 × 14 angular samples. 
   # Following [47], we extract the central 8 × 8 grid of angular sample to avoid invalid images.
   # [47] Srinivasan, P.P., Wang, T., Sreelal, A., Ramamoorthi, R., Ng, R.: 
   #      Learning to synthesize a 4d rgbd light ﬁeld from a single image. In: ICCV. Volume 2. (2017) 6
   
   build_flower((3,3)) # reference
#    build_flower((4,4)) 
#    build_flower((10,10)) 
                