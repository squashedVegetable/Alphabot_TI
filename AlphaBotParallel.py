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

import threading

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
        self.maximum = 60
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

        # Obeject counter
        self.counter = 1

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
            self.setPWMA(10)
            self.setPWMB(10)
            self.tr_sensor.calibrate()
        self.stop()
        print("Calibrated Min:", self.tr_sensor.calibratedMin)
        print("Calibrated Max:", self.tr_sensor.calibratedMax)

    def stop_line_follow(self):
        self.line_following_active = False
        self.stop()

    # ----------------------------
    # Sensor and Buzzer Methods
    # ----------------------------
    def infrared_obstacle_check(self):
        self.DR_status = GPIO.input(self.DR)
        self.DL_status = GPIO.input(self.DL)
        return self.DL_status == 0 or self.DR_status == 0

    def buzzer_on(self):
        GPIO.output(self.Buzzer, GPIO.HIGH)

    def buzzer_off(self):
        GPIO.output(self.Buzzer, GPIO.LOW)

    # ----------------------------
    # Camera and Recognition Methods
    # ----------------------------
    def start_camera(self):
        self.camera_server.start_server()

    def stop_camera(self):
        self.camera_server.stop_server()

    def recognize_digit(self):
        if not hasattr(self, 'cnn_model') or self.cnn_model is None:
            print("CNN model not loaded. Cannot recognize digit.")
            return None
        try:
            frame = self.camera_server.picam2.capture_array()
            if frame is None:
                print("No frame captured for digit recognition.")
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            im_gray = rgb2gray(frame)
            img_gray_u8 = img_as_ubyte(im_gray)
            (_, im_bw) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_resized = cv2.resize(im_bw, (28, 28))
            im_gray_invert = 255 - img_resized
            im_final = im_gray_invert.reshape(1, 1, 28, 28)
            im_final = torch.from_numpy(im_final).type(torch.FloatTensor)
            ans = self.cnn_model(im_final)
            ans_list = ans[0].tolist()
            predicted_digit = ans_list.index(max(ans_list))
            return predicted_digit
        except Exception as e:
            print(f"Error during digit recognition: {e}")
            return None

    def recognize_object(self):
        if self.object_model is None or self.imagenet_classes is None:
            print("Object recognition model not loaded. Cannot recognize object.")
            return
        if not hasattr(self, 'camera_server') or not hasattr(self.camera_server, 'picam2'):
            print("Camera server or PiCamera2 not initialized for object recognition.")
            return
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Ensure correct input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        try:
            with torch.no_grad():
                frame = self.camera_server.picam2.capture_array()
                if frame is None:
                    print("No frame captured for object recognition.")
                    return
                input_tensor = preprocess(frame)
                input_batch = input_tensor.unsqueeze(0)
                output = self.object_model(input_batch)
                probs = output[0].softmax(dim=0)
                top_prob, top_idx = torch.max(probs, dim=0)
                print(f"Object Recognition: {top_prob.item() * 100:.2f}% {self.imagenet_classes[top_idx.item()]}")
                if top_idx.item() == 761:       # remote control
                    self.set_led(0, 255, 0, 0)  # LED 1 red
                elif top_idx.item() == 784:     # screwdriver
                    self.set_led(1, 255, 255, 0)  # LED 2 yellow
                elif top_idx.item() == 504:     # coffee mug
                    self.set_led(2, 0, 255, 0)  # LED 3 green
                self.update_leds()
        except Exception as e:
            print(f"Error during object recognition: {e}")

    # ----------------------------
    # Line Following Update
    # ----------------------------
    def update_line_follow(self):
        """Perform one cycle of the line following logic."""
        # Only run if the line following flag is set.
        if not self.line_following_active:
            return

        self.forward()
        position, sensors = self.tr_sensor.readLine()

        if all(sensor > 900 for sensor in sensors):
            self.setPWMA(0)
            self.setPWMB(0)
        else:
            proportional = position - 2000
            derivative = proportional - self.last_proportional
            self.integral += proportional
            self.last_proportional = proportional
            power_difference = proportional / 30 + self.integral / 10000 + derivative * 2
            if power_difference > self.maximum:
                power_difference = self.maximum
            if power_difference < -self.maximum:
                power_difference = -self.maximum
            if power_difference < 0:
                self.setPWMA(self.maximum + power_difference)
                self.setPWMB(self.maximum)
            else:
                self.setPWMA(self.maximum)
                self.setPWMB(self.maximum - power_difference)

    def obstacle_detection(self):
        while not stop_event:
            if self.infrared_obstacle_check():
                print("Obstacle detected!")
                self.stop_line_follow()
                self.stop()
                time.sleep(2)
                self.clear_leds()
                self.buzz_n_times()
                print("Resuming line following.")
                self.counter += 1
                self.line_following_active = True
            time.sleep(0.02)

    def line_following(self):
        while not stop_event:
            self.update_line_follow()

    def object_detection(self):
        object_detection_state = "left"
        last_object_detection_time = time.time()
        object_detection_interval = 1

        while not stop_event:
            current_time = time.time()
            if current_time - last_object_detection_time >= object_detection_interval:
                if object_detection_state == "left":
                    self.servo.move_left()
                    object_detection_state = "center1"
                elif object_detection_state == "center1":
                    self.servo.center()
                    object_detection_state = "right"
                elif object_detection_state == "right":
                    self.servo.move_right()
                    object_detection_state = "center2"
                elif object_detection_state == "center2":
                    self.servo.center()
                    object_detection_state = "left"

                self.recognize_object()
                last_object_detection_time = current_time
            time.sleep(0.02)

    def buzz_n_times(self):
        for i in range(self.counter):
            self.buzzer_on()
            time.sleep(0.02)
            self.buzzer_off()
            time.sleep(0.2)

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

    obstacle_detection_thread = threading.Thread(target=bot.obstacle_detection)
    line_following_thread = threading.Thread(target=bot.line_following)
    # object_recognition_thread = threading.Thread(target=bot.object_detection)

    obstacle_detection_thread.start()
    line_following_thread.start()
    # object_recognition_thread.start()
    try:
        while not stop_event:
            time.sleep(1)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping execution.")
        stop_event = True
        obstacle_detection_thread.join()
        line_following_thread.join()
        # object_recognition_thread.join()
    finally:
        bot.stop_line_follow()
        bot.stop_camera()
        bot.servo.stop()
        GPIO.cleanup()
        print("All operations stopped. Exiting program.")
