from datetime import datetime
from picamera2 import Picamera2, Preview
from picamera2.encoders import Quality, H264Encoder
from picamera2.outputs import FfmpegOutput
from libcamera import controls, Transform
import time
from pprint import *

picam2 = Picamera2()

config = picam2.create_video_configuration(
    #main={"size": (1280, 720)},
    transform=Transform(hflip=1, vflip=1),
    raw=picam2.sensor_modes[1] # 60 fps, wide
)
picam2.configure(config)

picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual, # do not use AF
    "LensPosition": 0.0, # focus to infinity
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off, # no noise reduction
    "AeExposureMode": controls.AeExposureModeEnum.Short # short exposure preffered
})

#pprint(picam2.capture_metadata())
#pprint(picam2.sensor_modes)
#pprint(picam2.camera_controls)

# start recording #

HOURS_RECORDING = 5
DESTINATION = '/home/cermatej/Desktop/recordings'
ENCODER = H264Encoder()

for i in range(HOURS_RECORDING * 60):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f"{DESTINATION}/{timestamp}_{i}.mp4"
    print(f'Recording {filename}')

    picam2.start_recording(encoder=ENCODER, output=FfmpegOutput(filename), quality=Quality.MEDIUM)
    time.sleep(60)
    picam2.stop_recording()

