from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
from PIL import Image

from torch.autograd import Variable
import numpy as np

from kwiver.arrows.pytorch.seg_utils import *

try:
    import cv2
except ImportError:
    cv2 = None

class FCN_Segmentation(object):

    def __init__(self, model, cuda=True):
        self.cuda = cuda
        self.model = model

    def __call__(self, in_img):
        self._apply(in_img)

    def _apply(self, in_img):

        self.model.eval()

        img = transform(in_img)
        if self.cuda:
            img = img.cuda()
        v_img = Variable(img[None], volatile=True)
        score = self.model(v_img)

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

        lbl_pred = label2rgb(lbl_pred, img=in_img, n_labels=21)
        #lbl_pred = label2rgb(lbl_pred, n_labels=21)

        im = Image.fromarray(lbl_pred.squeeze())
        im.show()
        Image.fromarray(in_img).show()
        
        # get bounding boxs
        cv_in_img = cv2.cvtColor(np.array(in_img), cv2.COLOR_RGB2BGR)
        cv_pred = cv2.cvtColor(np.array(lbl_pred.squeeze()), cv2.COLOR_RGB2BGR)
        imgray = cv2.cvtColor(cv_pred, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bbox_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_list.append(tuple((x, y, w, h)))

        return bbox_list

