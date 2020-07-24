import numpy as np
import cv2
from random import randint

class alter():
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.img = cv2.imread(self.imagePath)

    def rotate(self, angle, outfile):
        image_center = tuple(np.array(self.img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(self.img, rot_mat, self.img.shape[1::-1], flags=cv2.INTER_LINEAR)
        cv2.imwrite(outfile,result)
    
    def scale(self, outfile, fx=0.5, fy=0.5):
        result = cv2.resize(self.img, None, fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(outfile,result)

    def affineTrans(self, outfile):
        rows,cols,ch = self.img.shape
        # pts1 = np.float32([[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)]])
        # pts2 = np.float32([[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)]])
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1,pts2)

        result = cv2.warpAffine(self.img,M,(cols,rows))

        cv2.imwrite(outfile,result)

    def overlay(self, outfile, topImgPath='./hurr.png'):
        topImg = cv2.imread(topImgPath, -1)
        print(topImg.shape)
        print(self.img.shape)
        # topImg = cv2.resize(topImg, (self.img.shape[1], self.img.shape[0]))
        # result = cv2.addWeighted(self.img, 0.5, topImg, 0.5, 0)
        result = self.overlay_transparent(self.img.copy(), topImg, 0, 0)
        cv2.imwrite(outfile, result)

    def overlay_transparent(self, background, overlay, x, y):
        """
        used by overlay function
        """
        background_width = background.shape[1]
        background_height = background.shape[0]

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
                ],
                axis = 2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background


if __name__ == "__main__":
    ins = alter('./outputname-1.png')
    ins.rotate(25,'rotate.png')
    ins.overlay('o.png')
    ins.scale('s.png')
    ins.affineTrans('x.png')


