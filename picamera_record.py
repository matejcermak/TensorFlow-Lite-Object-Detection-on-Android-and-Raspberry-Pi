from picamera2 import Picamera2
from datetime import datetime
from picamera2.encoders import Quality
import time
from libcamera import controls

picam2 = Picamera2()
picam2.set_controls(
    {"AfMode": controls.AfModeEnum.Continuous}
)
HOURS_RECORDING = 7
DESTINATION = '/home/cermatej/Desktop/recordings'

for i in range(HOURS_RECORDING * 60):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    picam2.start_and_record_video(
            f"{DESTINATION}/{timestamp}_{i}.mp4", 
            duration=60,
            #quality=Quality.VERY_HIGH,
            #show_preview=True
    )


#while True:
#    buffer = picam2.capture_buffer()
#    print(type(buffer))
#    time.sleep(2)

