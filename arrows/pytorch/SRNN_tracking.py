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

from vital.types import ( 
    ObjectTrackState,
    Track,
    ObjectTrackSet
)

from kwiver.arrows.pytorch.models import Siamese
from kwiver.arrows.pytorch.grid import grid
from kwiver.arrows.pytorch.track import track_state, track, track_set
from kwiver.arrows.pytorch.SRNN_matching import SRNN_matching, RnnType
from kwiver.arrows.pytorch.pytorch_siamese_f_extractor import pytorch_siamese_f_extractor

from kwiver.arrows.pytorch.MOT_bbox import MOT_bbox

def ts2ot_list(track_set):
    ot_list = [] 
    for t in track_set:
        ot = Track(id=t.id)
        ot_list.append(ot)

    for idx, t in enumerate(track_set):
        for i in range(len(t)):
            ot_state = ObjectTrackState(t[i].frame_id, t[i].detectedObj)
            if not ot_list[idx].append(ot_state):
                print('cannot add ObjectTrackState')
                exit(1)

    return ot_list


class SRNN_tracking(KwiverProcess):

    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("siamese_model_path", "siamese_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/siamese/snapshot_epoch_6.pt',
                              'Trained PyTorch model.')
        self.declare_config_using_trait('siamese_model_path')

        self.add_config_trait("siamese_model_input_size", "siamese_model_input_size", '224',
                              'Model input image size')
        self.declare_config_using_trait('siamese_model_input_size')

        # detection select threshold
        self.add_config_trait("detection_select_threshold", "detection_select_threshold", '0.0',
                              'detection select threshold')
        self.declare_config_using_trait('detection_select_threshold')

        # target RNN full model
        self.add_config_trait("targetRNN_full_model_path", "targetRNN_full_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/targetRNN_snapshot/App_LSTM_epoch_51.pt',
                              'Trained targetRNN PyTorch model.')
        self.declare_config_using_trait('targetRNN_full_model_path')

        # target RNN AI model
        self.add_config_trait("targetRNN_AI_model_path", "targetRNN_AI_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/targetRNN_AI/App_LSTM_epoch_51.pt',
                              'Trained targetRNN AI PyTorch model.')
        self.declare_config_using_trait('targetRNN_AI_model_path')

        # matching similarity threshold
        self.add_config_trait("similarity_threshold", "similarity_threshold", '0.5',
                              'similarity threshold.')
        self.declare_config_using_trait('similarity_threshold')

        # MOT gt detection
        #-------------------------------------------------------------------
        self.add_config_trait("MOT_Testing_flag", "MOT_Testing_flag", 'False', 'MOT testing flag')
        self.declare_config_using_trait('MOT_Testing_flag')
        
        self.add_config_trait("MOT_GT_det_file_path", "MOT_GT_det_file_path", 
                             '', 'MOT ground truth detection file targetRNN_full_model_path for testing')
        self.declare_config_using_trait('MOT_GT_det_file_path')
        #-------------------------------------------------------------------

        self._track_flag = False
        self._step_id = 1

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
        self._select_threshold = float(self.config_value('detection_select_threshold'))

        # Siamese model config
        siamese_img_size = int(self.config_value('siamese_model_input_size'))
        siamese_model_path = self.config_value('siamese_model_path')
        self._app_feature_extractor = pytorch_siamese_f_extractor(siamese_model_path, siamese_img_size)

        # targetRNN_full model config
        targetRNN_full_model_path = self.config_value('targetRNN_full_model_path')
        targetRNN_AI_model_path = self.config_value('targetRNN_AI_model_path')
        self._SRNN_matching = SRNN_matching(targetRNN_full_model_path, targetRNN_AI_model_path)

        # MOT related
        MOT_file_path = self.config_value('MOT_GT_det_file_path')
        self._m_bbox = MOT_bbox(MOT_file_path)

        MOT_flag = self.config_value('MOT_Testing_flag')
        self._mot_flag = (MOT_flag == 'True')

        self._similarity_threshold = float(self.config_value('similarity_threshold'))
        self._grid = grid()

        # generated track_set
        self._track_set = track_set()

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print('step {}'.format(self._step_id))

        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')
        dos_ptr = self.grab_input_using_trait('detected_object_set')

        # Get current frame and give it to app feature extractor
        im = in_img_c.get_image().get_pil_image()
        self._app_feature_extractor.frame = im

        # TODO: replace the dos with MOT ground truth bbox for testing
        # Get detection bbox
        if self._mot_flag is True:
            dos = self._m_bbox[self._step_id] 
        else:
            dos = dos_ptr.select(self._select_threshold)
        #print('bbox list len is {}'.format(len(dos)))

        # interaction features
        grid_feature_list = self._grid(im.size, dos, self._mot_flag)
        #print(grid_feature_list)

        track_state_list = []
        next_trackID = int(self._track_set.get_max_track_ID()) + 1
        
        # get new track state from new frame and detections
        for idx, item in enumerate(dos):
            if self._mot_flag is True:
                bbox = item
                d_obj = DetectedObject(bbox=item , confid=1.0)
            else:
                bbox = item.bounding_box()
                d_obj = item

            # center of bbox
            center = tuple((bbox.center()))

            # appearance features
            app_feature = self._app_feature_extractor(bbox)

            # build track state for current bbox for matching
            cur_ts = track_state(frame_id=self._step_id, bbox_center=center, interaction_feature=grid_feature_list[idx],
                                 app_feature=app_feature, bbox=[int(bbox.min_x()), int(bbox.min_y()), 
                                                                int(bbox.width()), int(bbox.height())],
                                 detectedObject=d_obj)
            track_state_list.append(cur_ts)
            
        # if there is no tracks, generate new tracks from the track_state_list
        if self._track_flag is False:
            next_trackID = self._track_set.add_new_track_state_list(next_trackID, track_state_list)
            self._track_flag = True
        else:

            print('track_set len {}'.format(len(self._track_set)))
            print('track_state_list len {}'.format(len(track_state_list)))

            # estimate similarity matrix
            similarity_mat, track_idx_list = self._SRNN_matching(self._track_set, track_state_list)

            # Hungarian algorithm
            row_idx_list, col_idx_list = sp.optimize.linear_sum_assignment(similarity_mat)
            
            for i in range(len(row_idx_list)):
                r = row_idx_list[i]
                c = col_idx_list[i]

                if -similarity_mat[r, c] < self._similarity_threshold:
                    # initialize a new track
                    self._track_set.add_new_track_state(next_trackID, track_state_list[c])
                    next_trackID += 1
                else:
                    # add to existing track
                    self._track_set.update_track(track_idx_list[r], track_state_list[c])
            
            # for rest unmatched track_state, we initialize new tracks
            if len(track_state_list) - len(col_idx_list) > 0:
                for i in range(len(track_state_list)):
                    if i not in col_idx_list:
                        self._track_set.add_new_track_state(next_trackID, track_state_list[i])
                        next_trackID += 1

        print('total tracks {}'.format(len(self._track_set)))

        # push track set to output port
        ot_list = ts2ot_list(self._track_set)
        ots = ObjectTrackSet(ot_list)

        self.push_to_port_using_trait('object_track_set', ots)

        self._step_id += 1

        self._base_step()


# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.SRNN_tracking'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('SRNN_tracking', 'Structural RNN based tracking',
                                SRNN_tracking)

    process_factory.mark_process_module_as_loaded(module_name)

