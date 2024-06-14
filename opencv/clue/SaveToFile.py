# import the opencv library
import cv2

# define a video capture object
cap = cv2.VideoCapture(0)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w,h))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# After the loop release the cap and out objects
cap.release()
out.release()

# Destroy all the windows
cv2.destroyAllWindows()
