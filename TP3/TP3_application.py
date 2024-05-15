import cv2, os
import os.path as osp
import numpy as np

ESC_KEY = 27
Q_KEY = 113

class DisplayAPP():
    def __init__(self, 
                 output_folder,
                 camera_id=0, 
                 coin_l=9, 
                 coin_h=6, 
                 num_img=25,
                 criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                 flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS
                 ):
        self.save_dir = output_folder
        self.criteria = criteria
        self.pattern_size = (coin_l, coin_h)
        self.flags = flags
        self.camera_id = camera_id
        self.num_img = num_img

    def find_corners(self, img, pattern_size=None):
        pattern_size = self.pattern_size if pattern_size == None else pattern_size

        if len(img.shape) == 3 and img.shape[2] == 3: 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, self.flags)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            img = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        else:
            cv2.putText(img, "Unable to Detect Chessboard", (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        return img, ret, corners
    
    def calibrate_camera(self, img_frames, pattern_size=None):
        h, v = self.pattern_size if pattern_size == None else pattern_size
        objp = np.zeros((v*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:h,0:v].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane. 
        for img in img_frames:
            img, ret_f, corners = self.find_corners(img)
            if ret_f:
                objpoints.append(objp)
                imgpoints.append(corners)
        gray = cv2.cvtColor(img_frames[-1], cv2.COLOR_BGR2GRAY)

        return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def display_camera_or_video(self):
        while True:
            mode = input("Dispaly video in loop (Enter a) OR Open camera (ENTER b)?\n")
            
            if mode == 'a':
                video_path = input("Enter video path:\n")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Cannot open {video_path}")
                    continue
                else:
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        # if frame is read correctly ret is True
                        if not ret:
                            break
                        frames.append(frame)
                    return self.display_images_in_loop(frames)
                
            elif mode == 'b':
                # camera_id = 0
                # while True:
                #     cap = cv2.VideoCapture(camera_id)
                #     if not cap.isOpened():
                #         print("Cannot open camera")
                #         camera_id = int(input("Please enter your camera number:\n"))
                #     elif camera_id == -1:
                #         exit()
                #     else:
                #         print("Successfully open your camera!")
                return self.display_camera()
            
            else:
                print("Please Press A or B!")
                # exit()

    def display_camera(self):
        """
        Help:
        - Press "g" to convert gray and color images
        - Press "s" to save the current frame
        - Press "f" to start and exit chessboard detection
        - Press "q" or "esc" to exit
        - After saving “num_img” images, calibration will automatically begin,
          and a new display window will show the camera image with the distortion calibrated.
        """
        cap = cv2.VideoCapture(self.camera_id)
        grayscale_mode = False
        detect_chessboard_mode = False
        calibrate_camera_mode = False
        save_press = 0

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if grayscale_mode:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Camera', frame)
            else:
                cv2.imshow('Camera', frame)

            if detect_chessboard_mode:
                frame, _, _ = self.find_corners(frame)
                cv2.imshow('Camera', frame)
            # else:
            #     cv2.imshow('Camera', frame)

            if save_press >= self.num_img:
                save_press = 0 # We can collecte images again.
                print("Image acquisition completed. Starting Camera Calibration!")
                img_frames = read_images_from_folder(self.save_dir)
                ret, mtx, dist, rvecs, tvecs = self.calibrate_camera(img_frames)
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                print("The rest of the image has been calibrated for distortion.")
                calibrate_camera_mode = True # In order not to repeat the print.

            if calibrate_camera_mode: 
                frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                cv2.imshow('Calibrated Camera', frame)
            else:
                cv2.imshow('Camera', frame)

            key = cv2.waitKey(1)
            if key == ord('g'):
                grayscale_mode = not grayscale_mode
            if key == ESC_KEY or key == Q_KEY:
                break
            if key == ord('s') and save_press < self.num_img:
                cv2.imwrite(osp.join(self.save_dir, str(save_press) +'.png'), frame)
                save_press += 1
            if key == ord('f'):
                detect_chessboard_mode = not detect_chessboard_mode

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def display_images_in_loop(self, frames):
        """
        Help:
        - Press "g" to convert gray and color images
        - Press "s" to save the current frame
        - Press "f" to start chessboard detection (cannot exit detection)
        - Press "q" or "esc" to exit
        """
        current_index = 0
        save_press = 0
        grayscale_mode = False
        detect_chessboard_mode = False

        while True:
            image = frames[current_index]

            if image is None:
                print(f"Cannot read image current_index:{current_index}")
                continue
            if grayscale_mode:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Image', image)
            else:
                cv2.imshow('Image', image)

            if detect_chessboard_mode:
                image, _, _ = self.find_corners(image)
                cv2.imshow('Image', image)
            else:
                cv2.imshow('Image', image)

            key = cv2.waitKey(1)
            if key == ESC_KEY or key == Q_KEY: 
                break
            if key == ord('g'):
                grayscale_mode = not grayscale_mode
            if key == ord('s') and save_press < self.num_img:
                cv2.imwrite(osp.join(self.save_dir, str(save_press) +'.png'), image)
                save_press += 1
            if key == ord('f'):
                detect_chessboard_mode = not detect_chessboard_mode
            
            current_index = (current_index + 1) % len(frames)
        
        cv2.destroyAllWindows()

def read_images_from_folder(frame_folder):
    frame_names = sorted([f for f in os.listdir(frame_folder) 
                        if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))])
    frames = [cv2.imread(osp.join(frame_folder, f)) for f in frame_names]
    return frames

def print_help():
    print()

def user_api():
    while True:
        camera_id = input("L'identifiant de la caméra utilisée:\n")
        if camera_id == '-1':
            exit()

        try:
            camera_id = int(camera_id)
        except ValueError:
            print("Veuillez entrer un nombre entier!")
            continue
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ID de caméra incorrect")
            continue
        else:
            print("Successfully open your camera!")
            break

    while True:
        coin_l = input("Le nombre de coin intérieur à l'échiquier dans la largeur:\n")
        try:
            coin_l = int(coin_l)
            break
        except ValueError:
            print("Veuillez entrer un nombre entier!")
            continue
    while True:
        coin_h = input("Le nombre de coin intérieur à l'échiquier dans la hauteur:\n")
        try:
            coin_h = int(coin_h)
            break
        except ValueError:
            print("Veuillez entrer un nombre entier!")
            continue
    while True:
        num_img = input("Le nombre d'images pour calculer la matrice intrinsèque et les coefficients de distorsion:\n")
        try:
            num_img = int(num_img)
            break
        except ValueError:
            print("Veuillez entrer un nombre entier!")
            continue

    return camera_id, coin_l, coin_h, num_img

def main():
    save_dir = 'my_camera'
    os.makedirs(save_dir, exist_ok=True)
    camera_id, coin_l, coin_h, num_img = user_api()

    app = DisplayAPP(save_dir, 
                     camera_id=camera_id,
                     coin_l=coin_l, 
                     coin_h=coin_h, 
                     num_img=num_img,)

    # Choose Display a video in loop or Display your camera fig
    # If want dispaly a video, please enter "a" and enter a video path
    # If want dispaly your camera fig, please enter "b"
    app.display_camera_or_video()

    # 3 Calibration d’une caméra en OpenCV à partir d’un ensemble d’images
    # Display images in image_folder in loop:
    file_dir = "calib_gopro"
    img_frames = read_images_from_folder(file_dir)
    # app.display_images_in_loop(img_frames)
    ret, mtx, dist, rvecs, tvecs = app.calibrate_camera(img_frames)
    print(mtx,"\n", dist)

    test_image_path = "calib_gopro/GOPR8420.JPG"
    test_img = cv2.imread(test_image_path)
    # Redressement de l’image
    h, w = test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
    cv2.imshow('Redressement image', dst)
    cv2.waitKey(5000)
    cv2.imwrite('calibresult.png', dst)

if __name__ == "__main__":
    main()
