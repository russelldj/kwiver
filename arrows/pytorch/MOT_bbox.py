from vital.types import BoundingBox

class MOT_bbox(object):
    def __init__(self, gt_file_path):
        self._frame_track_dict = self._process_gt_file(gt_file_path) 

    def _process_gt_file(self, gt_file_path):
        r"""Process MOT gt file
            The output of the function is a dictionary with following format
            [frame_num : [(id_num, bb_left, bb_top, bb_width, bb_height)]]
        """
        frame_track_dict = {}
        with open(gt_file_path, 'r') as f:
            for line in f:
                cur_line_list = line.rstrip('\n').split(',')
                if bool(frame_track_dict.get(int(cur_line_list[0]))):
                    frame_track_dict[int(cur_line_list[0])].extend([tuple(cur_line_list[1:6])])
                else:
                    frame_track_dict[int(cur_line_list[0])] = [tuple(cur_line_list[1:6])]
    
        return frame_track_dict

    def __getitem__(self, f_id):
        try:
            bb_info = self._frame_track_dict[f_id]
        except KeyError:
            print('frame id: {} does not exist!'.format(f_id))
            exit(0)
        
        bb_list = []
        
        for item in bb_info:
            x = float(item[1])
            y = float(item[2])
            w = float(item[3])
            h = float(item[4])

            bb_list.append(BoundingBox(x, y, x + w, y + h))

        return bb_list

