import os


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_tile_xy_in_fn(file_path):
    fn_base = file_path.split('/')[-1].split('.')[0]
    x, y = fn_base.split('_')[:2]
    return (int(x), int(y))


def get_tile_wh_in_fn(file_path):
    fn_base = file_path.split('/')[-1].split('.')[0]
    w, h = fn_base.split('_')[2:4]
    return (int(w), int(h))


def get_max_xy_in_paths(paths):
    xys = [list(map(int, p.split('/')[-1].split('.')[0].split('_')[:2])) for p in paths]
    return max(xys)


def find_str_in_list(str_list, substr, substr_1=None):
    if substr_1 is None:
        return [s for s in str_list if substr in s]
    else:
        return [s for s in str_list if substr in s and substr_1 in s]


def compute_tile_xys(xy, tile_size):
    # compute the start (x, y)s of the tiles that contain the patch to be returned.
    # note that the xs and ys are start from 1
    x_left, x_right, y_top, y_bottom = xy
    tile_col_L = (x_left - 1) // tile_size[0] + 1  # start from 1
    tile_col_R = (x_right - 1) // tile_size[0] + 1  # start from 1
    tile_row_T = (y_top - 1) // tile_size[1] + 1  # start from 1 
    tile_row_B = (y_bottom - 1) // tile_size[1] + 1  # start from 1

    x_00 = (tile_col_L - 1) * tile_size[0] + 1  # top left corner
    y_00 = (tile_row_T - 1) * tile_size[1] + 1  # top left corner

    x_01 = (tile_col_R - 1) * tile_size[0] + 1
    y_01 = (tile_row_T - 1) * tile_size[1] + 1

    x_10 = (tile_col_L - 1) * tile_size[0] + 1
    y_10 = (tile_row_B - 1) * tile_size[1] + 1

    x_11 = (tile_col_R - 1) * tile_size[0] + 1
    y_11 = (tile_row_B - 1) * tile_size[1] + 1

    return (x_00, y_00), (x_01, y_01), (x_10, y_10), (x_11, y_11)


def compute_patch_xys(xy, patch_size, tile_size):
    # get the (x, y)s from each quadrant, and get the cross_status
    # cross_status:
    #     "00": patch only in top-left tile
    #     "01": patch corsses the top-left and top-right tiles
    #     "10": patch crosses the top-left and bottom-left tiles
    #     "10": patch crosses all the four quadrants
    # (x, y)s will be the (x_left, x_right, y_top, y_bottom) tuples in the four quadrants
    #     the (x, y)s will be -1 if the patch does not corss that quadrant

    x_left, x_right, y_top, y_bottom = xy

    tile_col_L = (x_left - 1) // tile_size[0] + 1  # start from 1
    tile_col_R = (x_right - 1) // tile_size[0] + 1  # start from 1
    tile_row_T = (y_top - 1) // tile_size[1] + 1  # start from 1 
    tile_row_B = (y_bottom - 1) // tile_size[1] + 1  # start from 1

    # initialize the coordinates 
    x_left_00, x_right_00, y_top_00, y_bottom_00 = -1, -1, -1, -1
    x_left_01, x_right_01, y_top_01, y_bottom_01 = -1, -1, -1, -1
    x_left_10, x_right_10, y_top_10, y_bottom_10 = -1, -1, -1, -1
    x_left_11, x_right_11, y_top_11, y_bottom_11 = -1, -1, -1, -1        

    x_left_00 = x_left % tile_size[0]
    y_top_00 = y_top % tile_size[1]
    cross_status = "No"

    if tile_col_L == tile_col_R:
        if tile_row_T == tile_row_B:
            x_right_00 = x_left_00 + patch_size[0] - 1
            y_bottom_00 = y_top_00 + patch_size[1] - 1
            cross_status = "00"
        else:
            x_right_00 = x_left_00 + patch_size[0] - 1
            y_bottom_00 = tile_size[1]
            x_left_10 = x_left_00
            x_right_10 = x_right_00
            y_top_10 = 1
            y_bottom_10 = y_bottom % tile_size[1]
            cross_status = "10"
    else:
        if tile_row_T == tile_row_B:
            x_right_00 = tile_size[0]
            y_bottom_00 = y_top_00 + patch_size[1] - 1
            x_left_01 = 1
            x_right_01 = x_right % tile_size[0]
            y_top_01 = y_top_00
            y_bottom_01 = y_bottom_00
            cross_status = "01"
        else:
            x_right_00 = tile_size[0]
            y_bottom_00 = tile_size[1]
            x_left_01 = 1
            x_right_01 = x_right % tile_size[1]
            y_top_01 = y_top_00
            y_bottom_01 = tile_size[1]
            x_left_10 = x_left_00
            x_right_10 = x_right_00
            y_top_10 = 1
            y_bottom_10 = y_bottom % tile_size[1]
            x_left_11 = 1
            x_right_11 = x_right_01
            y_top_11 = 1
            y_bottom_11 = y_bottom_10
            cross_status = "11"

    return cross_status, \
        (x_left_00, x_right_00, y_top_00, y_bottom_00), \
        (x_left_01, x_right_01, y_top_01, y_bottom_01), \
        (x_left_10, x_right_10, y_top_10, y_bottom_10), \
        (x_left_11, x_right_11, y_top_11, y_bottom_11)

