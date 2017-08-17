from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image as pilImage

from vital.types import BoundingBox
from kwiver.arrows.pytorch.models import Siamese


class pytorch_siamese_f_extractor(object):
    """
    Obtain the appearance features from a trained pytorch siamese
    model
    """

    def __init__(self, siamese_model_path, img_size):
        # load siamese model
        self._siamese_model = Siamese()
        self._siamese_model = torch.nn.DataParallel(self._siamese_model).cuda()

        snapshot = torch.load(siamese_model_path)
        self._siamese_model.load_state_dict(snapshot['state_dict'])
        print('Model loaded from {}'.format(siamese_model_path))
        self._siamese_model.train(False)

        self._loader = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._img_size = img_size
        self._frame = pilImage.new('RGB', (img_size, img_size))

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, val):
        self._frame = val

    def __call__(self, bbox):
        return self._obtain_feature(bbox)

    def _obtain_feature(self, bbox):

        im = self._frame.crop((float(bbox.min_x()), float(bbox.min_y()),
                      float(bbox.max_x()), float(bbox.max_y())))
        #im.show()

        # resize cropped image
        im = im.resize((self._img_size, self._img_size), pilImage.BILINEAR)
        im.convert('RGB')

        im = self._loader(im).float()

        # im[None] is for add banch dimenstion
        im = Variable(im[None], volatile=True).cuda()

        output, _, _ = self._siamese_model(im, im)

        # appearance features
        app_feature = output.data.cpu().numpy().squeeze()

        return app_feature

