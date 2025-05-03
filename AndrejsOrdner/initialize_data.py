import dill
import time
from rpi_ws281x import Adafruit_NeoPixel, Color
import cv2
import RPi.GPIO as GPIO
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import ast
from skimage import img_as_ubyte
from skimage.color import rgb2gray

from CameraServerClass import CameraServer
from TRSensorClass import TRSensor
from ServoControllerClass import ServoController
from CNNClass import CNN_NET

# LED strip configuration constants:
LED_COUNT      = 4      # Number of LED pixels.
LED_PIN        = 18     # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ    = 800000 # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 5      # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = 255    # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0

# Global flag to assist with shutdown (still useful even in serial operation)
stop_event = False

# ----------------------------------------------------------------------------
# AlphaBot Class with LED, Buzzer, and Recognition Integration
# ----------------------------------------------------------------------------
class AlphaBot(object):
    def __init__(self):
        # Motor pins and parameters
        self.AIN1 = 12
        self.AIN2 = 13
        self.BIN1 = 20
        self.BIN2 = 21
        self.ENA = 6
        self.ENB = 26
        self.PA = 50
        self.PB = 50
        self.integral = 0
        self.last_proportional = 0
        self.maximum = 35
        self.DR = 16
        self.DL = 19
        self.CS = 5
        self.Clock = 25
        self.Address = 24
        self.DataOut = 23
        self.Buzzer = 4
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        motor_pins = [self.AIN1, self.AIN2, self.BIN1, self.BIN2, self.ENA, self.ENB]
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.Clock, GPIO.OUT)
        GPIO.setup(self.CS, GPIO.OUT)
        GPIO.setup(self.Address, GPIO.OUT)
        GPIO.setup(self.DataOut, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.Buzzer, GPIO.OUT)
        GPIO.setup(self.DR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.DL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.PWMA = GPIO.PWM(self.ENA, 500)
        self.PWMB = GPIO.PWM(self.ENB, 500)
        self.PWMA.start(self.PA)
        self.PWMB.start(self.PB)
        self.stop()  # Ensure motors are off initially

        # Initialize distance sensors
        self.DR_status = 1
        self.DL_status = 1

        # Initialize additional components
        self.tr_sensor = TRSensor(self.CS, self.Clock, self.Address, self.DataOut)
        self.servo = ServoController()
        self.camera_server = CameraServer()

        # ----------------------------
        # LED Strip Initialization
        # ----------------------------
        self.led_strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ,
                                            LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.led_strip.begin()

        # Load the CNN model for digit recognition
        try:
            self.cnn_model = torch.load("weights.h5", weights_only=False)
            print("CNN model loaded successfully.")
        except Exception as e:
            print("Error loading CNN model:", e)
            self.cnn_model = None

        # Initialize object recognition model and labels
        self.object_model = None
        self.imagenet_classes = None
        self.load_object_recognition_model()

        # A flag to track whether line following should happen
        self.line_following_active = True

    def load_object_recognition_model(self):
        try:
            self.object_model = models.quantization.mobilenet_v2(
                weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,
                quantize=True
            )
            self.object_model.eval()
            with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
                labels_dict = ast.literal_eval(f.read())
                self.imagenet_classes = [labels_dict[i] for i in range(len(labels_dict))]
            print("Object recognition model loaded successfully.")
        except Exception as e:
            print("Error loading object recognition model:", e)
            self.object_model = None
            self.imagenet_classes = None

    # ----------------------------
    # LED Utility Methods
    # ----------------------------
    def set_led(self, index, r, g, b):
        """Set a single LED's color."""
        if 0 <= index < LED_COUNT:
            self.led_strip.setPixelColor(index, Color(r, g, b))

    def update_leds(self):
        """Update the LED strip to show the current colors."""
        self.led_strip.show()

    def clear_leds(self):
        """Turn off all LEDs."""
        for i in range(LED_COUNT):
            self.led_strip.setPixelColor(i, Color(0, 0, 0))
        self.led_strip.show()

    def set_leds_default(self):
        """Set a default pattern on the LED strip."""
        self.set_led(0, 255, 0, 0)    # Red
        self.set_led(1, 0, 255, 0)    # Green
        self.set_led(2, 0, 0, 255)    # Blue
        self.set_led(3, 255, 255, 0)  # Yellow
        self.update_leds()
        time.sleep(2)
        self.clear_leds()

    def setPWMA(self, value):
        self.PA = value
        self.PWMA.ChangeDutyCycle(self.PA)

    def setPWMB(self, value):
        self.PB = value
        self.PWMB.ChangeDutyCycle(self.PB)

    def calibrate_sensors(self):
        print("Calibrating sensors...")
        for i in range(0, 100):
            if i < 25 or i >= 75:
                self.right()
            else:
                self.left()
            self.setPWMA(30)
            self.setPWMB(30)
            self.tr_sensor.calibrate()
        self.stop()
        print("Calibrated Min:", self.tr_sensor.calibratedMin)
        print("Calibrated Max:", self.tr_sensor.calibratedMax)

    def stop_line_follow(self):
        self.line_following_active = False
        self.stop()

      # ----------------------------
    # Motor Control Methods
    # ----------------------------
    def forward(self):
        self.PWMA.ChangeDutyCycle(self.PA)
        self.PWMB.ChangeDutyCycle(self.PB)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)

    def stop(self):
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(0)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.LOW)

    def backward(self):
        self.PWMA.ChangeDutyCycle(self.PA)
        self.PWMB.ChangeDutyCycle(self.PB)
        GPIO.output(self.AIN1, GPIO.HIGH)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.HIGH)
        GPIO.output(self.BIN2, GPIO.LOW)

    def left(self):
        self.PWMA.ChangeDutyCycle(30)
        self.PWMB.ChangeDutyCycle(30)
        GPIO.output(self.AIN1, GPIO.HIGH)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)

    def right(self):
        self.PWMA.ChangeDutyCycle(30)
        self.PWMB.ChangeDutyCycle(30)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.HIGH)
        GPIO.output(self.BIN2, GPIO.LOW) 
    # ----------------------------
    # Camera and Recognition Methods
    # ----------------------------
    def start_camera(self):
        self.camera_server.start_server()

    def stop_camera(self):
        self.camera_server.stop_server()

    # ----------------------------
    # Motor Control Methods
    # ----------------------------
    def forward(self):
        self.PWMA.ChangeDutyCycle(self.PA)
        self.PWMB.ChangeDutyCycle(self.PB)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)

    def stop(self):
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(0)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.LOW)

    def backward(self):
        self.PWMA.ChangeDutyCycle(self.PA)
        self.PWMB.ChangeDutyCycle(self.PB)
        GPIO.output(self.AIN1, GPIO.HIGH)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.HIGH)
        GPIO.output(self.BIN2, GPIO.LOW)

    def left(self):
        self.PWMA.ChangeDutyCycle(30)
        self.PWMB.ChangeDutyCycle(30)
        GPIO.output(self.AIN1, GPIO.HIGH)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)

    def right(self):
        self.PWMA.ChangeDutyCycle(30)
        self.PWMB.ChangeDutyCycle(30)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.HIGH)
        GPIO.output(self.BIN2, GPIO.LOW)

def is_pickleable(obj):
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError):
        return False
    return True

# ----------------------------------------------------------------------------
# Main Serial Loop
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    bot = AlphaBot()
    bot.clear_leds()
    bot.start_camera()
    print("Camera server started. Visit http://<your_pi_ip>:5000/ in your browser.")
    time.sleep(2)
    bot.calibrate_sensors()
    bot.servo.middle()
    bot.servo.center()

with open('initialized_data.pkl', 'wb') as f:
    pickable_globals = {k: v for k, v in globals().items() if is_pickleable(v)}    
    dill.dump(pickable_globals,f)
