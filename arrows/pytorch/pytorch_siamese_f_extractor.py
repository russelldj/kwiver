# ckwg +28
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
import scipy as sp
import scipy.optimize

from PIL import Image as pilImage

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import Image
from vital.types import DetectedObject
from vital.types import DetectedObjectSet

from kwiver.arrows.pytorch.models import Siamese
from kwiver.arrows.pytorch.grid import grid
from kwiver.arrows.pytorch.track import track_state, track, track_set
from kwiver.arrows.pytorch.SRNN_matching import SRNN_matching, RnnType


class pytorch_siamese_f_extractor(KwiverProcess):
    """
    This process gets an image as input, does some stuff to it and
    sends the modified version to the output port.
    """

    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("siamese_model_path", "siamese_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/siamese/snapshot_epoch_6.pt',
                              'Trained PyTorch model.')
        self.add_config_trait("siamese_model_input_size", "siamese_model_input_size", '224',
                              'Model input image size')
        self.add_config_trait("detection_select_threshold", "detection_select_threshold", '0.0',
                              'detection select threshold')

        # target RNN full model
        self.add_config_trait("targetRNN_full_model_path", "targetRNN_full_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/targetRNN_snapshot/App_LSTM_epoch_51.pt',
                              'Trained targetRNN PyTorch model.')

        # target RNN AI model
        self.add_config_trait("targetRNN_AI_model_path", "targetRNN_AI_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/targetRNN_AI/App_LSTM_epoch_51.pt',
                              'Trained targetRNN AI PyTorch model.')

        self.add_config_trait("similarity_threshold", "similarity_threshold", '0.5',
                              'similarity threshold.')
        
        self.declare_config_using_trait('siamese_model_path')
        self.declare_config_using_trait('siamese_model_input_size')
        self.declare_config_using_trait('detection_select_threshold')
        self.declare_config_using_trait('targetRNN_full_model_path')
        self.declare_config_using_trait('targetRNN_AI_model_path')
        self.declare_config_using_trait('similarity_threshold')

        self._track_flag = False

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        # self.declare_input_port_using_trait('framestamp', optional)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('object_track_set', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('object_track_set', optional)

    # ----------------------------------------------
    def _configure(self):
        # Siamese model config
        self._img_size = int(self.config_value('siamese_model_input_size'))
        self._model_path = self.config_value('siamese_model_path')
        self._select_threshold = float(self.config_value('detection_select_threshold'))

        self._siamese_model = Siamese()
        self._siamese_model = torch.nn.DataParallel(self._siamese_model).cuda()

        snapshot = torch.load(self._model_path)
        self._siamese_model.load_state_dict(snapshot['state_dict'])
        print('Model loaded from {}'.format(self._model_path))
        self._siamese_model.train(False)
        
        # targetRNN_full model config
        targetRNN_full_model_path = self.config_value('targetRNN_full_model_path')
        targetRNN_AI_model_path = self.config_value('targetRNN_AI_model_path')
        self.SRNN_matching = SRNN_matching(targetRNN_full_model_path, targetRNN_AI_model_path)

        self._similarity_threshold = float(self.config_value('similarity_threshold'))
        self._grid = grid()

        # generated track_set
        self._track_set = track_set()

        self._loader = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')
        dos_ptr = self.grab_input_using_trait('detected_object_set')

        # Get image and resize
        im = in_img_c.get_image().get_pil_image()

        # Get detection bbox
        dos = dos_ptr.select(self._select_threshold)
        print('bbox list len is {}'.format(len(dos)))

        # interaction features
        grid_feature_list = self._grid(im.size, dos)
        print(grid_feature_list)

        track_state_list = []
        next_trackID = int(self._track_set.get_max_track_ID()) + 1
        
        # get new track state from new frame and detections
        for idx, item in enumerate(dos):
            item_box = item.bounding_box()

            # center of bbox
            center = tuple((item_box.center()))

            im = im.crop((float(item_box.min_x()), float(item_box.min_y()),
                          float(item_box.max_x()), float(item_box.max_y())))
            im.show()

            # resize cropped image
            im = im.resize((self._img_size, self._img_size), pilImage.BILINEAR)
            im.convert('RGB')

            im = self._loader(im).float()

            # im[None] is for add banch dimenstion
            im = Variable(im[None], volatile=True).cuda()

            output, _, _ = self._siamese_model(im, im)

            # appearance features
            app_feature = output.data.cpu().numpy().squeeze()

            # build track state for current bbox for matching
            cur_ts = track_state(bbox_center=center, interaction_feature=grid_feature_list[idx],
                                 app_feature=app_feature)
            track_state_list.append(cur_ts)
        
        # if there is no tracks, generate new tracks from the track_state_list
        if self._track_flag is False:
            self._track_set.add_new_track_state_list(next_trackID, track_state_list)
            self._track_flag = True

        # estimate similarity matrix
        similarity_mat, track_idx_list = self.SRNN_matching(self._track_set, track_state_list)

        # Hungarian algorithm
        row_idx, col_idx = sp.optimize.linear_sum_assignment(similarity_mat)

        for ri in row_idx:
            for ci in col_idx:
                if -similarity_mat[ri, ci] < self._similarity_threshold:
                    # initialize a new track
                    self._track_set.add_new_track_state(next_trackID, track_state_list[ci])
                    next_trackID += 1
                else:
                    # add to existing track
                    self._track_set.update_track(track_idx_list[ri], track_state_list[ci])


        # push dummy detections object to output port
        # ts = track_Set()
        #self.push_to_port_using_trait('object_track_set', ts)

        self._base_step()


# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch_siamese_f_extractor'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('pytorch_siamese_f_extractor', 'pytorch siamese feature extractor',
                                pytorch_siamese_f_extractor)

    process_factory.mark_process_module_as_loaded(module_name)

