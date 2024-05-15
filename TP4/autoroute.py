# openCV import
import cv2 as cv
import sys, os
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

def read_img_from_npy(npyfile_path, read_mode):
    image = np.load(npyfile_path)
    if read_mode == 'color':
        return image
    elif read_mode == 'gray':
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
def moyenne_images(images, M):
    # Initialiser une variable pour la somme des images
    somme_images = np.zeros_like(images[0], dtype=np.float32)
    # Somme des M premières images
    for i in range(M):
        img = images[i].astype(np.float32)  # Convertir l'image en float32 pour éviter un débordement
        somme_images += img
    # Calculer la moyenne en divisant la somme par le nombre d'images
    moyenne = somme_images / M
    # Convertir en type uint8 si nécessaire
    moyenne_uint8 = np.array(moyenne, dtype=np.uint8)
    return moyenne_uint8

def filter_mask(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv.dilate(opening, kernel, iterations=2)
    # threshold
    # th = dilation[dilation < 240] = 0
    return dilation

def flux_lumineux(ref_gray, frame, frame_gray, rout_mask):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Find corners in gray image
    p0 = cv.goodFeaturesToTrack(ref_gray, mask=rout_mask, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(ref_gray)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(ref_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

    return img

def main():
    # Define variables
    filename = sys.argv[1] if len(sys.argv) > 1 else 'TP4/autoroute3.mp4'
    save_folder = os.path.splitext(filename)[0]
    os.makedirs(save_folder, exist_ok=True)

    # Reading the image (and forcing it to grayscale)
    cap = cv.VideoCapture(filename)

    run = cap.isOpened()
   # Making sure the capture has opened successfully
    if not run:
        # capture opening has failed we cannot do anything :'(
        print("capture opening has failed we cannot do anything :'(")
        sys.exit()

    # Get video wide and high (resolution)
    video_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    # le nombre d’images qui compose la séquence vidéo
    total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    # Frame rate de la vidéo.
    fps = cap.get(cv.CAP_PROP_FPS)
    # La durée d’une vidéo est le nombre total d’images divisé par le taux d’images en secondes.
    total_time= total_frame/fps

    frames = []
    gray_frames = []
    while True:
        ret, im = cap.read()
        if not ret:
            break
        frames.append(im)
        imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        gray_frames.append(imGray)

    M = 500
    moyenne_img = moyenne_images(gray_frames, M)
    # cv.imshow("Moyenne image", moyenne_img)

    _, remove_sky = cv.threshold(moyenne_img, 170, 255, cv.THRESH_TOZERO_INV)
    cv.imshow("Remove sky", remove_sky)
    _, rout_mask = cv.threshold(remove_sky, 120, 255, cv.THRESH_BINARY)
    # cv.imshow("bilinairy image", rout_mask)
    kernel_open = np.ones((15,13), np.uint8)
    rout_mask = cv.morphologyEx(rout_mask, cv.MORPH_OPEN, kernel_open)
    # cv.imshow("morphologyEx Open", rout_mask)
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,25))
    rout_mask = cv.morphologyEx(rout_mask, cv.MORPH_CLOSE, kernel_close)
    cv.imshow("rout_mask", rout_mask)
    moyenne_result = cv.bitwise_and(moyenne_img, rout_mask)
    # cv.imshow("moyenne_image_result", result)

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    key = None
    kernel_open = np.ones((5,5), np.uint8)
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,15))
    current_index = 0

    # Initialize for flow track
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take the frame interesting
    current_index = 22
    old_gray = gray_frames[current_index]
    cv.imshow("frame interasting", old_gray)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=rout_mask, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frames[0])

    while key != ESC_KEY and key!= Q_KEY:
        im = frames[current_index]
        imGray = gray_frames[current_index]
        rout_imGray = cv.bitwise_and(imGray, rout_mask)

        cars = cv.absdiff(imGray, moyenne_img)
        cars = cv.bitwise_and(cars, rout_mask)
        _, car_mask = cv.threshold(cars, 70, 255, cv.THRESH_BINARY)

        # Count cars by connectedComponents
        blank_cars = cv.morphologyEx(car_mask, cv.MORPH_OPEN, kernel_open)
        blank_cars = cv.morphologyEx(blank_cars, cv.MORPH_CLOSE, kernel_close)
        num_cars, _ = cv.connectedComponents(blank_cars)
        num_cars1 = num_cars - 1

        # Count cars by findContours
        line_cars = cv.morphologyEx(car_mask, cv.MORPH_CLOSE, kernel_close)
        contours, _ = cv.findContours(line_cars, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_cars2 = len(contours)

        # Count cars by other method
        edges = cv.Canny(cars, threshold1=30, threshold2=100)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel_close)
        contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_cars3 = 0
        for contour in contours:
            # polygonal approximation
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            # Check if it is close to a quadrilateral
            if len(approx) == 4:
                cv.drawContours(cars, [approx], 0, (0, 255, 0), 2)
                num_cars3 += 1

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, rout_imGray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            im = cv.circle(im, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv.add(im, mask)
            cv.imshow("flow", img)
            # Now update the previous frame and previous points
            old_gray = rout_imGray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cv.putText(rout_imGray, 
                   f"Count1: {num_cars1}", 
                   (5, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        cv.putText(rout_imGray, 
                   f"Count2: {num_cars2}", 
                   (5, 100), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                   
        cv.putText(rout_imGray, 
                   f"Count3: {num_cars3}", 
                   (5, 150), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        # with_flux = flux_lumineux(gray_frames[0], im, imGray, mask)
            
        cv.imshow("Car masks", car_mask)
        cv.imshow("Route Gray video", rout_imGray)
        cv.imshow("Voitures", cars)
        
        # cv.imshow("Flux", with_flux)
        # cv.imshow("Blank cars", blank_cars)
        # cv.imshow("Line cars", line_cars)
        # cv.imshow("edges cars", edges)

        current_index = (current_index + 1) % len(frames)

        key = cv.waitKey(30)
        # key = cv.pollKey()

    # release cap
    cap.release()
    # Destroying all OpenCV windows
    key = cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
