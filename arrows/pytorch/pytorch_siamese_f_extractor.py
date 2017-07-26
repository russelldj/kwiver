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

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np

from PIL import Image as pilImage

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import Image
from vital.types import DetectedObject
from vital.types import DetectedObjectSet
#from .grid import Grid

class pytorch_siamese_f_extractor(KwiverProcess):
    """
    This process gets an image as input, does some stuff to it and
    sends the modified version to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("model_path", "model_path", '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/siamese/snapshot_epoch_6.pt',
          'Trained PyTorch model.')
        self.add_config_trait("model_input_size", "model_input_size", '224',
          'Model input image size' )

        self.declare_config_using_trait('model_path')
        self.declare_config_using_trait('model_input_size')

        #self.add_port_trait('detections', 'detected_object_set', 'Output detections')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('feature_set', optional)

    # ----------------------------------------------
    def _configure(self):
        self._img_size = int(self.config_value('model_input_size'))
        self._model_path = self.config_value('model_path')
        
        self._model = Siamese()
        self._model = torch.nn.DataParallel(self._model).cuda()

        snapshot = torch.load(self._model_path)
        self._model.load_state_dict(snapshot['state_dict'])
        print('Model loaded from {}'.format(self._model_path))
        self._model.train(False)
        

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')
        dos_ptr = self.grab_input_using_trait('detected_object_set')

        # Get image and resize
        im = in_img_c.get_image().get_pil_image()

        # Get detection bbox
        dos = dos_ptr.select(0.5)
        print('bbox list len is {}'.format(len(dos)))
        for item in dos:
            item_box = item.bounding_box()

            im = im.crop((float(item_box.min_x()), float(item_box.min_y()), 
                          float(item_box.max_x()), float(item_box.max_y())))
            im.show()

        # resize cropped image
        im = im.resize((self._img_size, self._img_size), pilImage.BILINEAR)
        im.convert('RGB')

        loader = transforms.Compose([
                       transforms.Scale(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ])
        im = loader(im).float()

        # im[None] is for add banch dimenstion
        im = Variable(im[None], volatile=True).cuda()

        output, _, _ = self._model(im, im)
        np_output = output.data.cpu().numpy().squeeze()
        #print(np_output)

        # push dummy detections object to output port
        #detections = DetectedObjectSet()
        #self.push_to_port_using_trait('detected_object_set', detections)

        self._base_step()


# Siamese network
# ==================================================================
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.num_fcin = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_fcin, 500)
        self.pdist = nn.PairwiseDistance(1)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        output = self.pdist(output1, output2)

        return output1, output2, output

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch_siamese_f_extractor'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process( 'pytorch_siamese_f_extractor', 'pytorch siamese feature extractor', pytorch_siamese_f_extractor )

    process_factory.mark_process_module_as_loaded( module_name )
