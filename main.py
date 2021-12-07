import argparse
import cv2
import os
import numpy as np
import numba as nb
from font_mask import create_mask
from synthesis import *
from tracking import *

def parse_args():
    parser = argparse.ArgumentParser(description='Smart Erasor')
    parser.add_argument('--template_path', type=str, required=False, default='./target.png', help='Path to the template sample')
    parser.add_argument('--in_path', type=str, required=False, default='./video/book.avi', help='the path of input video')
    parser.add_argument('--out_path', type=str, required=False, default='./video_out/output_book.avi', help='the path of erased video')

    # parameters for texture impainting
    parser.add_argument('--kernel_size', type=int, required=False, default=25, help='One dimension of the square synthesis kernel')
    parser.add_argument('--visualize', required=False, action='store_true', help='Visualize the synthesis process')


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    template = cv2.imread(args.template_path)
    if template is None:
        raise ValueError('Unable to read image from template_path.')
    path = args.template_path
    image_parameter = {'./target.png' : [[0, 0, 0], [25, 255, 255], 278, 292, 0, 50],
                       './blanket.png' : [[0, 0, 120], [180, 15, 255], 52, 62, 200, 239]
                       }
    low_bound, high_bound = np.array(image_parameter[path][0]), np.array(image_parameter[path][1])
    width_left, width_right = image_parameter[path][2], image_parameter[path][3]
    height_top, height_bottom = image_parameter[path][4], image_parameter[path][5]

    template_mask = create_mask(template, low_bound, high_bound, np.ones((3, 3), np.uint8), np.ones((3, 3), np.uint8), 1)
    hole_index = np.where(template_mask == 255)

    for hole_row, hole_col in zip(hole_index[0], hole_index[1]):
        template[hole_row, hole_col] = 255
        # for blanket, the row needs to be set to 255 is 80:160
        # template[80:160, hole_col] = 255
        # as for newspaper, the row needs to be set to 255 is 70:120
        template[70:120, hole_col] = 255


    target = np.copy(template)[height_top:height_bottom, width_left:width_right, :]
    H, W = template.shape[0], template.shape[1]
    height, width = target.shape[0], target.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(args.in_path)
    out = cv2.VideoWriter(args.out_path, fourcc, 15.0, (320, 240))

    template[height_top:height_bottom, width_left:width_right, :] = 255
    # for blanket, the template should be parametered on [:, :70, :] = 255
    # template[:, :70, :] = 255

    # for newspaper, the template should be parametered on the following parameters
    template[H-20:, :, :] = 255
    template[150:, 200:, :] = 255
    template[:, :15, :] = 255

    start_flag = 0

    while(start_flag <= 227):
        print(start_flag)
        ret, frame = cap.read()
        target_col, frame_img = ncc_track(H, W, height, width, ret, frame, target)
        frame_mask = np.copy(template_mask)

        for hole_row, hole_col in zip(hole_index[0], hole_index[1]):
            # the condition of hole_col < target_col is for blanket
            # if hole_col < target_col:
            if hole_col > target_col:
                frame_img[hole_row, hole_col] = 255
                frame_img[70:110, hole_col] = 255
                frame_mask[70:110, hole_col] = 255
                # frame_img[80:160, hole_col] = 255
                # frame_mask[80:160, hole_col] = 255
            else:
                frame_mask[hole_row, hole_col] = 0
                frame_mask[70:110, hole_col] = 0
                # frame_mask[80:160, hole_col] = 0

        if start_flag > 0 and len(hole_index_last[0]) != 0:
            for hole_row, hole_col in zip(hole_index_last[0], hole_index_last[1]):
                # the condition of hole_col < target_col is for blanket.
                # if hole_col < target_col:
                if hole_col > target_col:
                    frame_img[hole_row, hole_col] = frame_last[hole_row, hole_col]

        hole_index_last = np.where(frame_mask == 255)
        template_temp = np.copy(template)

        # the following operation of target_col larger than 160 is only for newspaper, not for blanket. and this operation is for light correction
        #############################
        if target_col > 160:
            template_temp[:, int(target_col):, :] = 255
        else:
            template_temp[:, 160:, :] = 255
        #############################
        
        frame_out = synthesize_texture(original_sample=frame_img, sample=template_temp, kernel_size=args.kernel_size, visualize=args.visualize)
        cv2.imwrite('./image_out_book/frame_out_{}.jpg'.format(start_flag), frame_out)
        frame_last = np.copy(frame_out)

        start_flag += 1

        out.write(frame_out)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    out.release()

if __name__ == '__main__':
    main()