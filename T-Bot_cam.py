import numpy as np
import cv2

class TBotCam:
    #camera variables
    cam_mtx=None
    dist=None
    newcam_mtx=None
    roi=None
    rvec1=None
    tvec1=None
    R_mtx=None
    Rt=None
    P_mtx=None
    scalingfactor = 0
    inverse_newcam_mtx = None
    inverse_R_mtx = None

    imgdir="Captures/"
    
    #images
    img=None

    def __init__(self):
        print("Initiated!")
        

    def LoadFromFile(self):    
        savedir="/home/carlo/Documents/Visual Studio Code/TBotCamera/CameraData/"
        self.cam_mtx=np.load(savedir+'cam_mtx.npy')
        self.dist=np.load(savedir+'dist.npy')
        self.newcam_mtx=np.load(savedir+'newcam_mtx.npy')
        self.roi=np.load(savedir+'roi.npy')
        self.rvec1=np.load(savedir+'rvec1.npy')
        self.tvec1=np.load(savedir+'tvec1.npy')
        self.R_mtx=np.load(savedir+'R_mtx.npy')
        self.Rt=np.load(savedir+'Rt.npy')
        self.P_mtx=np.load(savedir+'P_mtx.npy')

        s_arr=np.load(savedir+'s_arr.npy')
        self.scalingfactor=s_arr[0]

        self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx)
        self.inverse_R_mtx = np.linalg.inv(self.R_mtx)

    def previewImage(self, text, img):
            #show full screen
            cv2.namedWindow(text, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(text,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

            cv2.imshow(text,img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def calculate_XYZ(self,u,v):
                                        
            #Solve: From Image Pixels, find World Points
            # u = pixel x
            # v = pixel y

            uv_1=np.array([[u,v,1]], dtype=np.float32)
            uv_1=uv_1.T
            suv_1=self.scalingfactor*uv_1
            xyz_c=self.inverse_newcam_mtx.dot(suv_1)
            xyz_c=xyz_c-self.tvec1
            XYZ=self.inverse_R_mtx.dot(xyz_c)

            return XYZ


cam = TBotCam
cam.LoadFromFile(cam)
xyz = cam.calculate_XYZ(cam,85, 328)

print("Resulting XYZ")
print(xyz)