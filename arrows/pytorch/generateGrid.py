import cv2
import numpy as np

def generateGrid(img, grid_rows=15, grid_cols=15):
    r"""
            The output of the function is a dictionary (frame_num: 2D list) with following format
            [frame_num : [[grid_11, grid_12, ... , grid_1n], [grid_21, grid_22, ..., grid_2n,] ... grid_nn]]
            grid_mn = 1: one or more bboxs' center fall in the cell(mn)
            grid_mn = 0: none of bboxs' center falls in the cell(mn)
    """
    temp_list = gt_file_path.split('/')
    img_path = '/'.join(temp_list[:-2]) + '/img1'

    # get img height and width
    for (_, _, filename) in walk(img_path):
        img = cv2.imread('{}/{}'.format(img_path, filename[0]))
        img_h, img_w = img.shape[:2]
        break

    # calculate grid cell height and width
    cell_h = img_h / grid_rows
    cell_w = img_w / grid_cols

    frame_track_dict = process_gt_file(gt_file_path)

    frame_grid_dict = {}
    for frame_id, bb_info in frame_track_dict.items():

        # initial all gridcell to 0
        grid_list = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        for bb in bb_info:
            x, y, w, h = max(0, int(float(bb[1]))), max(0, int(float(bb[2]))), \
                         int(float(bb[3])), int(float(bb[4]))
            # bbox center
            c_w = min(x + w / 2, img_w - 1)
            c_h = min(y + h / 2, img_h - 1)

            # cell idxs
            row_idx = int(c_h // cell_h)
            col_idx = int(c_w // cell_w)

            try:
                grid_list[row_idx, col_idx] = 1
            except IndexError:
                print('c_h:{}, c_w:{}, row_idx:{}, col_idx:{}'.format(c_h, c_w, row_idx, col_idx))

        frame_grid_dict[frame_id] = grid_list

    return frame_track_dict, frame_grid_dict, cell_h, cell_w

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.generateGrid'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process( 'generateGrid', 'generateGrid', grid )

    process_factory.mark_process_module_as_loaded( module_name )
