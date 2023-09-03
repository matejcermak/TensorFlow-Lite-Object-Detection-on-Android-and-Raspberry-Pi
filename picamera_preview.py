from picamera2 import Picamera2, Preview
from libcamera import controls
import time

picam2 = Picamera2()

#config = picam2.create_preview_configuration(main={"size": (1600, 1200)})
config = picam2.create_preview_configuration()
picam2.configure(config)

picam2.set_controls(
    {"AfMode": controls.AfModeEnum.Continuous}
)

picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)

#while True:
#    buffer = picam2.capture_buffer()
#    print(type(buffer))
#    time.sleep(2)

