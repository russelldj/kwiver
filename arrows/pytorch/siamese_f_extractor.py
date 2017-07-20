#ckwg +28
# Copyright 2017 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import torch
import numpy as np
from torch.multiprocessing import Pool

from PIL import Image as pilImage

from darknet import Darknet19
from torch.autograd import Variable
from torchvision import transforms
import utils.yolo as yolo_utils
import utils.network as net_utils
import cfgs.config as cfg

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import Image
from vital.types import ImageContainer
from vital.types import DetectedObject
from vital.types import DetectedObjectSet
from vital.types import BoundingBox

class pytorch_detector(KwiverProcess):
    """
    This process gets an image as input, does some stuff to it and
    sends the modified version to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("model_path", "model_path", './yolo-voc.weights.h5',
          'Trained PyTorch model.')
        self.add_config_trait("model_input_size", "model_input_size", '416',
          'Model input image size' )
        self.add_config_trait("threshold", "threshold", '0.5',
          'Detection threshold')

        self.declare_config_using_trait('model_path')
        self.declare_config_using_trait('model_input_size')
        self.declare_config_using_trait('threshold')

        #self.add_port_trait('detections', 'detected_object_set', 'Output detections')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_output_port_using_trait('detected_object_set', optional)

    # ----------------------------------------------
    def _configure(self):
        self._img_size = int(self.config_value('model_input_size'))
        self._model_path = self.config_value('model_path')
        self._thr = float(self.config_value('threshold'))
        
        self._net = Darknet19
        net_utils.load_net(self._model_path, self._net)
        self._net.cuda()
        self._net.eval()
        print('load model succ...')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')

        # Get image and resize
        in_img = in_img_c.get_image().get_numpy_array()
        im = pilImage.fromarray(np.uint8(in_img))
        im = im.resize((self._img_size, self._img_size), pilImage.BILINEAR)
        loader = transforms.Compose([transforms.ToTensor])
        im.convert('RGB')
        im = loader(im).float()
        im = Variable(im, volatile=True).cuda()
        bbox_pred, iou_pred, prob_pred = self._net(im) 

        # push dummy detections object to output port
        detections = DetectedObjectSet()
        self.push_to_port_using_trait('detected_object_set', detections)

        self._base_step()


# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch_detector'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process( 'pytorch_detector', 'pytorch detector', pytorch_detector )

    process_factory.mark_process_module_as_loaded( module_name )
