
# coding: UTF-8

import os
import os.path
import glob
import argparse
import cv2
import numpy as np
import json

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='''Calibrate pro-cam system using chessboard and structured light projection
Place captured images as:
    ./ --- capture_1/ --- graycode_00.png
    |              |- graycode_01.png
    |              |        .
    |              |        .
    |              |- graycode_??.png
    |- capture_2/ --- graycode_00.png
    |              |- graycode_01.png
    |      .       |        .
    |      .       |        .
    ''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('proj_height', type=int, help='projector pixel height')
    parser.add_argument('proj_width', type=int, help='projector pixel width')
    parser.add_argument('chess_vert', type=int,
                        help='number of cross points of chessboard in vertical direction')
    parser.add_argument('chess_hori', type=int,
                        help='number of cross points of chessboard in horizontal direction')
    parser.add_argument('chess_block_size', type=float,
                        help='size of blocks of chessboard (mm or cm or m)')
    parser.add_argument('graycode_step', type=int,
                        default=1, help='step size of graycode')
    parser.add_argument('-black_thr', type=int, default=40,
                        help='threshold to determine whether a camera pixel captures projected area or not (default : 40)')
    parser.add_argument('-white_thr', type=int, default=5,
                        help='threshold to specify robustness of graycode decoding (default : 5)')
    parser.add_argument('-camera', type=str, default=str(),
                        help='camera internal parameters (fx,fy,cx,cy)')
    parser.add_argument('-projector', type=str, default=str(),
                        help='projector internal parameters (fx,fy,cx,cy)')

    return parser.parse_args()

def load_graycode_images(proj_height, proj_width, graycode_step, black_thr, white_thr):
    """
    Load and decode graycode images from the capture directories.
    
    Parameters:
        proj_height (int): Projector pixel height.
        proj_width (int): Projector pixel width.
        graycode_step (int): Step size of graycode.
        black_thr (int): Threshold to determine if a camera pixel captures projected area.
        white_thr (int): Threshold to specify robustness of graycode decoding.
        
    Returns:
        tuple: Lists containing valid camera corners, projector corners, and object points.
    """
    if not os.path.exists('./capture_1'):
        raise FileNotFoundError("The required 'capture_1' directory was not found.")
    
    # Create GraycodePattern instance
    graycode = cv2.structured_light_GrayCodePattern_create(
        proj_width, proj_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    # Capture checkerboard corners and graycode patterns from images
    cam_corners_list = []
    proj_corners_list = []
    proj_objps_list = []
    path_list = sorted(glob.glob('./capture_*'))
    if not path_list:
        raise ValueError("No capture directories found.")
    for path in path_list:
        imgs = []
        for i in range(graycode.getNumberOfPatternImages() + 2):
            img_path = os.path.join(path, 'graycode_{:02d}.png'.format(i))
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} not found.")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            imgs.append(img)
        cam_corners, proj_corners, proj_objps = decode_graycode_pattern(
            graycode, imgs, (proj_width, proj_height), graycode_step)
        if cam_corners is not None:
            cam_corners_list.append(cam_corners)
            proj_corners_list.append(proj_corners)
            proj_objps_list.append(proj_objps)

    return cam_corners_list, proj_corners_list, proj_objps_list
def decode_graycode_pattern(graycode, imgs, shape, step):
    """
    Decode the graycode pattern from the captured images.
    
    Parameters:
        graycode (cv2.structured_light_GrayCodePattern): Graycode pattern instance.
        imgs (list): List of captured images.
        shape (tuple): Shape of the projector (width, height).
        step (int): Step size of graycode.
        
    Returns:
        tuple: Camera corners, projector corners, and projector object points.
    """
        # Decode graycode pattern
    res, cam_corners = cv2.findChessboardCorners(imgs[0], shape)
    if not res:
        return None, None, None

    _, proj_pix = graycode.decode(imgs[1:-1], step)
    if proj_pix is None:
        return None, None, None

    proj_corners = np.column_stack(
        (np.ravel(proj_pix[:, :, 0]), np.ravel(proj_pix[:, :, 1])))
    proj_objps = np.column_stack(
        (np.ravel(proj_pix[:, :, 0]), np.ravel(proj_pix[:, :, 1]), np.zeros(proj_pix.shape[0] * proj_pix.shape[1])))

    return cam_corners, proj_corners, proj_objps

def calibrate_system(cam_corners_list, proj_corners_list, proj_objps_list, cam_shape, proj_shape):
    """
    Calibrate the pro-cam system using the decoded patterns and chessboard corners.
    
    Parameters:
        cam_corners_list (list): List of camera corners.
        proj_corners_list (list): List of projector corners.
        proj_objps_list (list): List of projector object points.
        cam_shape (tuple): Shape of the camera image (height, width).
        proj_shape (tuple): Shape of the projector (width, height).
        
    Returns:
        tuple: Calibration results including camera and projector intrinsics, distortion parameters, and transformation matrices.
    """
    if not cam_corners_list or not proj_corners_list or not proj_objps_list:
        raise ValueError("Lists for calibration are empty. Calibration cannot proceed.")
    
    # Calibrate camera
    ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
        proj_objps_list, cam_corners_list, cam_shape, None, None, None, None)
    
    # Initial solution of projector's parameters
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, proj_shape, None, None, None, None)
    
    # Stereo calibration
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, _, _ = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None)
    
    return ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec
def print_numpy_with_indent(arr, indent=""):
    """
    Helper function to print numpy arrays with a given indent.
    """
    print(indent + str(arr).replace('\n', '\n' + indent))
def display_results(cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec):
    """
    Display the calibration results in a formatted manner.
    
    Parameters:
        cam_int (numpy.ndarray): Camera intrinsic parameters.
        cam_dist (numpy.ndarray): Camera distortion parameters.
        proj_int (numpy.ndarray): Projector intrinsic parameters.
        proj_dist (numpy.ndarray): Projector distortion parameters.
        cam_proj_rmat (numpy.ndarray): Rotation matrix from camera to projector.
        cam_proj_tvec (numpy.ndarray): Translation vector from camera to projector.
    """
    print('=== Result ===')
    
    print('Camera intrinsic parameters:')
    print_numpy_with_indent(cam_int, '    ')
    
    print('Camera distortion parameters:')
    print_numpy_with_indent(cam_dist, '    ')
    
    print('Projector intrinsic parameters:')
    print_numpy_with_indent(proj_int, '    ')
    
    print('Projector distortion parameters:')
    print_numpy_with_indent(proj_dist, '    ')
    
    print('Rotation matrix / translation vector from camera to projector')
    print('(they translate points from camera coord to projector coord):')
    print_numpy_with_indent(cam_proj_rmat, '    ')
    print_numpy_with_indent(cam_proj_tvec, '    ')

    print()  # Empty line for separation
def save_results_to_file(cam_shape, ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec):
    """
    Save the calibration results to an XML file.
    
    Parameters:
        cam_shape (tuple): Shape of the camera image (height, width).
        ret (float): Root mean square error of the calibration.
        cam_int (numpy.ndarray): Camera intrinsic parameters.
        cam_dist (numpy.ndarray): Camera distortion parameters.
        proj_int (numpy.ndarray): Projector intrinsic parameters.
        proj_dist (numpy.ndarray): Projector distortion parameters.
        cam_proj_rmat (numpy.ndarray): Rotation matrix from camera to projector.
        cam_proj_tvec (numpy.ndarray): Translation vector from camera to projector.
    """
    fs = cv2.FileStorage('calibration_result.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('img_shape', cam_shape)
    fs.write('rms', ret)
    fs.write('cam_int', cam_int)
    fs.write('cam_dist', cam_dist)
    fs.write('proj_int', proj_int)
    fs.write('proj_dist', proj_dist)
    fs.write('rotation', cam_proj_rmat)
    fs.write('translation', cam_proj_tvec)
    fs.release()

    
def main_refactored():
    args = parse_arguments()
    
    cam_corners_list, proj_corners_list, proj_objps_list = load_graycode_images(
        args.proj_height, args.proj_width, args.graycode_step, args.black_thr, args.white_thr)
    
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec = calibrate_system(
        cam_corners_list, proj_corners_list, proj_objps_list, (args.proj_height, args.proj_width), 
        (args.chess_hori, args.chess_vert))
    
    display_results(cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec)
    save_results_to_file((args.proj_height, args.proj_width), ret, cam_int, cam_dist, 
                         proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec)



if __name__ == '__main__':
    main_refactored()
