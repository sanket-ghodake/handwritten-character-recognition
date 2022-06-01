import cv2
import numpy as np
import math

'''Useful Points ->
1. Its important to show frame always after we edited something to image. Keep winnname same as before it will 
    replaced old frame with new, so that we can see changes made in code eaily.
    
    Useful links ->
1. waitkey() - https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7 
2.TODO: Drawing a line on an image using mouse clicks with Python OpenCV library - 
    https://stackoverflow.com/questions/60587273/drawing-a-line-on-an-image-using-mouse-clicks-with-python-opencv-library 
3. Combining Two Images with OpenCV - https://stackoverflow.com/questions/7589012/combining-two-images-with-opencv 
4. cv2.VideoCapture(index + cv2.CAP_DSHOW) - https://stackoverflow.com/a/52130912/16040502 
5. cv2.destroyWindow(winname=name_for_frame) # winnname should be the name you given for showing the frame and 
    not the captured frame name 
6. cv2.putText() - https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
7. cv2.resize() - https://www.geeksforgeeks.org/image-resizing-using-opencv-python/
8. cv2.setMouseCallBack() - https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga89e7806b0a616f6f1d502bd8c183ad3e 
9. Color picker W3School - https://www.w3schools.com/colors/colors_picker.asp 
10. cv2.circle() - https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670 
11. All drawing functions - https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html 
12. NumPy arrat initialization - https://www.pluralsight.com/guides/different-ways-create-numpy-arrays
13. 2D array all corner point's value - https://stackoverflow.com/questions/49060724/select-corner-elements-of-a-2d-numpy-array 
14. Swapping column of nd array - https://www.geeksforgeeks.org/how-to-swap-columns-of-a-given-numpy-array/ 
15. No constant in python - https://stackoverflow.com/questions/2682745/how-do-i-create-a-constant-in-python 
16.TODO conditional in python based on datatypes - https://stackoverflow.com/questions/14113187/how-do-you-set-a-conditional-in-python-based-on-datatypes/49067320
        https://stackoverflow.com/questions/1835018/how-to-check-if-an-object-is-a-list-or-tuple-but-not-string?rq=1 
17. Iterate over multiple lists simultaneously - https://www.geeksforgeeks.org/python-iterate-multiple-lists-simultaneously/
18. Tuple of length 1 - https://stackoverflow.com/questions/31293862/why-is-a-tuple-of-tuples-of-length-1-not-actually-a-tuple-unless-i-add-a-comma 

'''
########################################### GLOBAL DECLERATIONS ###########################################
# Very common aspect ration is 4X3, 16X9
# Laptop display resolution - 768X1366 horizontal X vertical / width X height
LAPTOP_RESOLUTION = (720, 1280)
IMAGE = "IMAGE"

""" ALL EVENTS form OpenCV: 
    [    
        'EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 
        'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 
        'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 
        'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP'
    ] 
"""
EVENTS = [i for i in dir(cv2) if 'EVENT' in i]  # All EVENTS from OpenCV
IMAGE_FRAME_NAME = IMAGE  # Global name for frame

"""
    Initializing 4 corner points of page/block to be selected. In clockwise fashion(0-top left 1-top right 2-bottom
    right 3-bottom left). Points are in the format (y-coordinate,x-coordinate). Data type should be 32 bits.
    TODO why 32bits? - https://stackoverflow.com/questions/17241830/opencv-polylines-function-in-python-throws-exception 
"""
page_corner_points = np.zeros((4, 2), dtype=np.int32)
dragging_point = -1  # No point is selected for dragging. When selected gets number between 0,1,2,3.
flag_point_move = False  # Is currently any point is moving
original_image = None  # If we have to save changes(Especially drawings) then writing to this variable
secondary_image = None  # For logical part to be done. If we have to save our changes to image.


########################################### FUNCTION DEFINATION ###########################################

def updating_corner_of_page(point_numbers=None, co_ordinates=None, img=None):
    """
    Updates the corners of page to be selected. (By contour or by manually selecting points).
    All arguments by default are none. Points are in the format (y-coordinate,x-coordinate)
    :param point_numbers: 0-top left 1-top right 2-bottom right 3-bottom left or tuple/list/int of numbers(number)
    :param co_ordinates: tuple(y coordinate,x coordinate) or tuple/list/int of co-ordinates(coordinate)
    :param img: page corners are set as per image dimentsion. Img nd-array to be passed as argument.
    :return: 1- if points updated 0-if wrong arguments given. Updates global array - 'page_corner_points'
    """
    global page_corner_points
    if img is not None and isinstance(img, np.ndarray):
        # Setting four pts according to img size.
        page_corner_points[0] = [0, 0]
        page_corner_points[1] = [img.shape[1], 0]
        page_corner_points[2] = [img.shape[1], img.shape[0]]
        page_corner_points[3] = [0, img.shape[0]]
        # setting pts 'p' pixels inside the image
        p = 5
        temp = np.array([[p, p], [-p, p], [-p, -p], [p, -p]])
        page_corner_points = page_corner_points + temp
        return 1
    elif point_numbers is not None and co_ordinates is not None and isinstance(point_numbers, int) and isinstance(
            co_ordinates, (list, tuple)):
        page_corner_points[point_numbers] = co_ordinates
        return 1
    elif point_numbers is not None and co_ordinates is not None and isinstance(point_numbers,
                                                                               (list, tuple)) and isinstance(
        co_ordinates, (list, tuple)):
        for (point_number, co_ordinate) in zip(point_numbers, co_ordinates):
            page_corner_points[point_number] = co_ordinate
        return 1
    return 0


def draw_quadrilateral(img, original=False):
    """
    Drawing quadrilateral on image from points in page_corner_points array.
    IMPORTANT - we are reshaping matrix from (4,2) to (4,1,2)
    Code refer from - https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    https://stackoverflow.com/questions/17241830/opencv-polylines-function-in-python-throws-exception
    https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
    :param img: Image on which shape to be drawn
    :parameter original : True - drawing another circle ,False - Moving existing circle
    :return: Modified image - img or new_image
    """
    global original_image, secondary_image
    global page_corner_points

    # Reshaping matrix - drawing polygon - https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    pts = page_corner_points.reshape(-1, 1, 2)  # TODO - what is transformation from this?
    if original:  # Writing to original image, no need to use global variable, because numpy array is passed by pointer
        # If this line exeute quadrilaterla will displayed.
        cv2.polylines(img=img, pts=[pts], isClosed=True, color=(255, 187, 51), thickness=5, lineType=cv2.LINE_AA)

        # TODO -If this line execute, only four points of quadrilateral will displayed.(Filled circle) Why?
        # cv2.polylines(img=img, pts=pts, isClosed=True, color=(255, 187, 51), thickness=10, lineType=cv2.LINE_AA)

        cv2.imshow(IMAGE_FRAME_NAME, img)
        # original_image = img # TODO - check this line
        return img
    else:  # Showing only copy image, not writing to original image
        if original_image is not None:
            new_image = original_image.copy()
            cv2.polylines(img=new_image, pts=[pts], isClosed=True, color=(255, 187, 51), thickness=5,
                          lineType=cv2.LINE_AA)
            cv2.imshow(IMAGE_FRAME_NAME, new_image)
            secondary_image = new_image
            return new_image


def draw_circle(x, y, img, original=False):
    """
    Draw's circle around centre. It shows how to use cv2.circle() function.
    Code refer from - https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
    :parameter x: x-coordinate centre
    :parameter y: y-coordinate centre
    :parameter img: image nd-array
    :parameter original : True - drawing another circle ,False - Moving existing circle
    :return: Modified image - img or new_image
    """
    global original_image, secondary_image
    if original:  # Writing to original image, no need to use global variable, because numpy array is passed by pointer
        cv2.circle(img=img, center=(y, x), radius=3, color=(255, 187, 51), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(IMAGE_FRAME_NAME, img)
        # original_image = img # TODO - check this line
        return img
    else:  # Showing only copy image, not writing to original image
        if original_image is not None:
            new_image = original_image.copy()
            cv2.circle(img=new_image, center=(y, x), radius=3, color=(255, 187, 51), thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow(IMAGE_FRAME_NAME, new_image)
            secondary_image = new_image
            return new_image


def detect_mouse_click(event, x, y, flags, param):
    """
    Provides basic logic to do with various mouse EVENTS.
    Code refer - https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
    Varios mouse EVENTS are -
    How to use - cv2.setMouseCallback('winname',detect_mouse_click)
        'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP',
        'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP',
        'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
        'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP'
    :param event: which event occur TODO
    :param x: y-coordinate
    :param y: x-coordinate
    :param flags: TODO how to pass this parameters, how to use them
    :param param: TODO
    :return: Depends on logic written in function
    """
    global flag_point_move, page_corner_points, dragging_point, original_image, secondary_image
    x, y = y, x  # Swapping for our code convineance because x is y-coordinate and vice-versa

    if event == cv2.EVENT_LBUTTONDOWN:
        for pts in range(4):
            # Here dragging points accuracy(radius) is set to 5 pixel
            if point_in_circle(page_corner_points[pts], (y, x), r=5):
                dragging_point = pts
                flag_point_move = True
                updating_corner_of_page(point_numbers=dragging_point, co_ordinates=(y, x))
                break
        # print(x, y) # Print mouse location
    if event == cv2.EVENT_MOUSEMOVE:
        if flag_point_move:
            updating_corner_of_page(point_numbers=dragging_point, co_ordinates=(y, x))

            draw_quadrilateral(param)
        # print(x, y) # Print mouse location
    if event == cv2.EVENT_LBUTTONUP:
        if flag_point_move:
            flag_point_move = False
            dragging_point = -1
    if event == cv2.EVENT_LBUTTONDBLCLK:
        original_image = secondary_image
        updating_corner_of_page(img=original_image)
        # TODO After above line corner are reset to img size , but when we clicking corners new quadrilaterl is generating.


def keyboard_press_key_check(key_name="ESC"):
    """ Function check for waitkey number after key pressed and return tuple with key_name and number
    By default it expect 'ESC' key is pressed.
    To use function press key and then ESC. The interrupt number for last key pressed before ESC will return.
    It prints all numbers for which you pressed the key.
    :param key_name: key name for which press number is to be checked.
    :return: (key_name,number to generate interrupt)
    """

    array_created = np.full((100, 100, 3), 255, dtype=np.uint8)
    # displaying the image
    cv2.imshow("image", array_created)
    temp = 27  # Temp variable for logic
    while True:
        w = cv2.waitKey(0) & 0xFF
        if w == -1:
            continue
        elif w == 27:
            return key_name, temp
        else:
            print(w)
            temp = w


def keyboard_interrupt(key='ESC', wait_key_number=27):  # TODO - Make it as a callback function
    """ Function set to check whether given key is pressed. By default it expect 'ESC' key is pressed.
        :parameter key - Key Name
        :parameter wait_key_number - It can get from 'keyboard_press_key_check' function
        :return True or False"""
    if cv2.waitKey(
            0) & 0xFF == wait_key_number:  # Link refer - <https://stackoverflow.com/questions/60307472/how-to-keep
        # -an-image-open-in-opencv-even-if-u-press-a-key>
        print("Key Pressed :", key)
        return True
    return False


def webcam_list():
    """
    Returns all options available for webcams / Gives list of webcams connected with there
        numbers to use in OpenCV
        code refer from - https://stackoverflow.com/a/53310665/16040502
    :return: arr list with numbers of webcam to use in openCV.
    """
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index + cv2.CAP_DSHOW)  # link - https://stackoverflow.com/a/52130912/16040502
        if not cap.read()[0]:
            # index-=1 # REDUNDANT -subtracting index that we are checking for
            break
        else:
            arr.append(index)  # appending index to arr
        cap.release()  # release object
        index += 1  # going to check next index
    return arr


def read_webcam_video(webcam_index, name_for_frame):
    """ To read webcam as video and displaying it.
        webcam_index == camera number [0 or -1 -> integrated webcam ,2 3 4 ...-> other external webcam connected]
        Code refer from - https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
        Refer logic written in it. It has no direct use in comple codes.
        Press ESC to stop webcam.
    :parameter webcam_index: Index of webcam for which video has to be displayed
    :parameter name_for_frame: Name to be given to frame
    :return Output webcam, no return, press ESC to stop function/webcam"""

    # cap = cv2.VideoCapture(webcam_index)
    cap = cv2.VideoCapture(webcam_index + cv2.CAP_DSHOW)  # https://stackoverflow.com/a/52130912/16040502
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()  # false if no frames has been grabbed
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow(name_for_frame, frame)
        if keyboard_interrupt():
            break
    # When everything done, release the capture
    cv2.destroyWindow(
        winname=name_for_frame)  # winnname should be the name you given for showing the frame and not the captured
    # frame name
    cap.release()


def stack_images(scale, img_array):  # TODO - Put text on every image in stack
    """ Stack images together as given in img_array .
        Works well with images with different dimensions.
    Code taken from - https://www.computervision.zone/topic/chapter-6-joining-images/ and
    https://www.youtube.com/watch?v=Wv0PSs0dmVI
     How to use ->
        imgStack = stack_images(0.25, ([img, img, img], [img, img, img]))
        cv2.imshow("ImageStack", imgStack)
    :parameter scale: Scalling factor
    :parameter img_array : Images given in matrix form, same matrix get displayed with block replaced with images.
    :return Image window of all stacked images
    """
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def scaling_image_by_factor(img, factor=1):  # TODO
    """
    Scaling Factor or Scale Factor is usually a number that scales or multiplies some quantity, in our case the width
    and height of the image. It helps keep the aspect ratio intact and preserves the display quality. So the image
    does not appear distorted, while you are upscaling or downscaling it.
    Choice of Interpolation Method for Resizing –
    cv2.INTER_AREA: This is used when we need to shrink an image.
    cv2.INTER_CUBIC: This is slow but more efficient.
    cv2.INTER_LINEAR: This is primarily used when zooming is required. This is the default interpolation technique in OpenCV.
    :param factor: factor for scaling
    :return: image after scaling
    """
    size = img[0].shape
    if len(img) == 1:
        return img[0]
    if len(img) == 2:
        if size[0] > LAPTOP_RESOLUTION[0]:
            scale = LAPTOP_RESOLUTION[0] / size[0]
    pass


def point_in_circle(center, point, r):
    """
    Find the distance between the center of the circle and the points given.
    If the distance between them is less than the radius then the point is inside
    the circle. if the distance between them is equal to the radius of the circle
    then the point is on the circumference of the circle.
    :param center: co-ordinate of center (y,x)
    :param point: co-ordiante of point(y,x)
    :param r: radius of circle
    :return: True/False if point lies in circle or not.
    """
    if r >= (math.sqrt(math.pow(center[0] - point[0], 2) +
                       math.pow(center[1] - point[1], 2))):
        return True
    return False


def stack_images_version_2(img_array):  # TODO - Rectangular packing algorithm in this function
    """
    *** Code is tested only for images with same dimensions.
    :param img_array: All images that have to display
    :return: Output a stacked images window
    """

    # vis = np.concatenate((original_image, img2), axis=1) # axis = 1 ->Horizontal


def put_text(img, text="Text_Here", line=cv2.LINE_AA):  # TODO
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 500), font, 4, (255, 255, 255), 2, line)


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print("Error: ", e)
