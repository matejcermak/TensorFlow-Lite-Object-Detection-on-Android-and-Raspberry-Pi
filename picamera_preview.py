from picamera2 import Picamera2, Preview
from libcamera import controls, Transform
import time
from pprint import *

picam2 = Picamera2()


config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    transform=Transform(hflip=1, vflip=1),
    raw=picam2.sensor_modes[1] # 60 fps, wide
)
#config = picam2.create_preview_configuration()
picam2.configure(config)

picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual, # do not use AF
    "LensPosition": 0.0, # focus to infinity
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off, # no noise reduction
    "AeExposureMode": controls.AeExposureModeEnum.Short, # short exposure preffered
    #"AeConstraintMode": controls.AeConstraintModeEnum.Shadows  # shadows? 
})

picam2.start_preview(Preview.QTGL)
picam2.start()

pprint(picam2.capture_metadata())
pprint(picam2.sensor_modes)
pprint(picam2.camera_controls)

while True:
    buffer = picam2.capture_buffer()
    #print(buffer.shape)
    time.sleep(10)

