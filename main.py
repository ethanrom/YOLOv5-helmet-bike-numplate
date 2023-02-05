from functions import *
source = 0

show_video = True  # set true when using video file
save_img = False  # set true when using only image file to save the image
font = cv2.FONT_HERSHEY_DUPLEX

cap = cv2.VideoCapture(source)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, frame_size)  # resize image
        detections = detect_objects(frame)  # get object detections

        fps = 1 / (time.time() - start_time)
        start_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if show_video:
            cv2.imshow('frame', frame)
        if save_img:
            cv2.imwrite('output.jpg', frame)

        # Check if user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
