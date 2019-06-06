import cv2
import sys
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.40,
    'gpu': 1.0
}
tfnet = TFNet(option)

lol = sys.argv[1:2]
str1 = ''.join(lol).strip('[]')
print(lol)
print(str(str1))

capture = cv2.VideoCapture(str1)
colors = [tuple(255 * np.random.rand(3)) for i in range(100)]
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#Coordinates
st_co_x = 0
st_co_y = 0
w = 700
h = 70
en_co_x = st_co_x + w
en_co_y = st_co_y + h

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        tbb=0
        objs={}
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'],result['bottomright']['y'])
            label = result['label']
            if label not in objs:
                objs.update({label:1})
            else:
                objs[label]+=1
            objs_values_str = str(list(objs.values()))
            objs_keys_str = str(list(objs.keys()))
            objs_str = str(objs)
            obst = ' '.join('{}: {} '.format(key, val) for key, val in sorted(objs.items()))
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)

            frame = cv2.rectangle(frame, tl, br, color, 2)
            frame = cv2.rectangle(frame, (st_co_x,st_co_y), (en_co_x,en_co_y), (255,255,255), -1)
            frame = cv2.putText(frame, obst,(st_co_x,st_co_y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,0,0) , 1)
            tbb=tbb+1
            total_objs='Total objects in a frame : {}'.format(tbb)
            frame = cv2.putText(frame, text , tl, cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,0,0) , 2)
            fps = 'FPS {:.1f}'.format(1 / (time.time() - stime))
            frame = cv2.putText(frame, fps,(st_co_x,st_co_y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,0,0) , 1)
            frame = cv2.putText(frame, total_objs ,(st_co_x,st_co_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,0,0) , 1)
        cv2.imshow('Object Detection', frame)
        
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print('Total objects in a frame :',tbb)
        print(obst)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
