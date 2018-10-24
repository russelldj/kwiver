from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image as pilImage

from vital.types import BoundingBox
from kwiver.arrows.pytorch.models import Siamese
from kwiver.arrows.pytorch.parse_gpu_list import get_device


class siameseDataLoader(data.Dataset):
    def __init__(self, bbox_list, transform, frame_img, in_size, MOT_flag):
        self._frame_img = frame_img
        self._transform = transform
        self._bbox_list = bbox_list
        self._mot_flag = MOT_flag
        self._in_size = in_size

    def __getitem__(self, index):
        if self._mot_flag is True:
            bb = self._bbox_list[index]
        else:
            bb = self._bbox_list[index].bounding_box()

        im = self._frame_img.crop((float(bb.min_x()), float(bb.min_y()),
                      float(bb.max_x()), float(bb.max_y())))

        im = im.resize((self._in_size, self._in_size), pilImage.BILINEAR)
        im.convert('RGB')

        if self._transform is not None:
            im = self._transform(im)

        return im

    def __len__(self):
        if self._mot_flag is True:
            return len(self._bbox_list)
        else:
            return self._bbox_list.size()


class pytorch_siamese_f_extractor(object):
    """
    Obtain the appearance features from a trained pytorch siamese
    model
    """

    def __init__(self, siamese_model_path, img_size, batch_size, GPU_list=None):
        self._device, use_gpu_flag = get_device(GPU_list)
        # load Siamese model
        self._siamese_model = Siamese().to(self._device)
        if use_gpu_flag:
            self._siamese_model = torch.nn.DataParallel(self._siamese_model, device_ids=GPU_list)

        snapshot = torch.load(siamese_model_path)
        self._siamese_model.load_state_dict(snapshot['state_dict'])
        print('Model loaded from {}'.format(siamese_model_path))
        self._siamese_model.train(False)

        self._transform = transforms.Compose([
            transforms.Scale(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._img_size = img_size
        self._frame = pilImage.new('RGB', (img_size, img_size))
        self._b_size = batch_size

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, val):
        self._frame = val

    def __call__(self, bbox_list, MOT_flag):
        return self._obtain_feature(bbox_list, MOT_flag)

    def _obtain_feature(self, bbox_list, MOT_flag):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        bbox_loader_class = siameseDataLoader(bbox_list, self._transform, self._frame, self._img_size, MOT_flag)
        bbox_loader = torch.utils.data.DataLoader(bbox_loader_class, batch_size=self._b_size, shuffle=False, **kwargs)

        torch.set_grad_enabled(False)
        for idx, imgs in enumerate(bbox_loader):
            v_imgs = imgs.to(self._device)
            output = self._siamese_model(v_imgs)

            if idx == 0:
                app_features = output.data
            else:
                app_features = torch.cat((app_features, output.data), dim=0)

        return app_features.cpu()