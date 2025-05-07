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
from concurrent.futures import ProcessPoolExecutor
from VisionWorker import recognition_worker, load_object_recognition_model

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
        self.maximum = 40
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

        # A flag to track whether line following should happen
        self.line_following_active = True

        # Obeject counter
        self.counter = 0
        # self.buzz_lock = threading.Lock()

        # Use on extra core for object recognition
        # self.executor = ProcessPoolExecutor(max_workers=1, initializer=load_object_recognition_model)
        # self._pending = set() 
        # Load label list
        with open("imagenet1000_clsidx_to_labels.txt") as f:
            labels_dict = ast.literal_eval(f.read())
        self.imagenet_classes = [labels_dict[i] for i in range(len(labels_dict))]

        self.IR_DEBOUNCE = 2
        self.IR_COOLDOWN_SEC = 1.5
        self.ir_hist = 0
        self.ir_ignore_until = 0.0

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
        # for i in range(0, 100):
        #     if i < 25 or i >= 75:
        #         self.right()
        #     else:
        #         self.left()
        #     self.setPWMA(15)
        #     self.setPWMB(15)
        #     self.tr_sensor.calibrate()
        time.sleep(0.1)
        self.stop()
        self.tr_sensor.calibrate()
        self.stop()
        time.sleep(0.1)
        print("Calibrated Min:", self.tr_sensor.calibratedMin)
        print("Calibrated Max:", self.tr_sensor.calibratedMax)

    def stop_line_follow(self):
        self.line_following_active = False
        self.stop()

    # ----------------------------
    # Sensor and Buzzer Methods
    # ----------------------------
    def infrared_obstacle_check(self):
        now = time.monotonic()

        if now < self.ir_ignore_until:
            return False
        
        hit_now = GPIO.input(self.DR) == 0 or GPIO.input(self.DL) == 0
        self.ir_hist = ((self.ir_hist << 1) | hit_now) & 0xFF

        mask = (1 << self.IR_DEBOUNCE) - 1
        debounced_hit = (self.ir_hist & mask) == mask
        if debounced_hit:
            self.ir_ignore_until = now + self.IR_COOLDOWN_SEC
            return True
        return False

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

    def handle_recognition_result(self, future):
        print("try to recognize object")
        try:
            idx, prob = future.result()
            label = self.imagenet_classes[idx]
            print(f"[Vision] {prob*100:5.1f}%  {label}", flush=True)

        except Exception as e:
            print("Object inference failed:", e, flush=True)
            return
        finally:
            self._pending.discard(future)
        

        if idx == 440 or idx == 720 or idx == 737 or idx == 898:
            self.set_led(0, 255, 0, 0)
        elif idx == 784:
            self.set_led(1, 255, 255, 0)
        elif idx == 504:
            self.set_led(2, 0, 255, 0)
        self.update_leds()

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
            power_difference = proportional / 30 + self.integral / 10000 + derivative * 2 #TODO: jemand *4 vorgeschlagen 
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
                self.counter += 1
                self.buzz_n_times()
                self.line_following_active = True
                time.sleep(1)
                print("Resuming line following.")
            time.sleep(0.01)

    def line_following(self):
        while not stop_event:
            self.update_line_follow()
            time.sleep(0.01)

    def buzz_n_times(self):
        for i in range(self.counter):
            light_counter = self.counter % 4
            self.buzzer_on()
            time.sleep(0.02)
            self.buzzer_off()
            time.sleep(0.1)
            if (light_counter == 1):
                self.set_led(0, 255, 0, 0) # Red
            elif (light_counter == 2):
                self.set_led(1, 0, 255, 0) # Green
            elif (light_counter == 3):
                self.set_led(2, 0, 0, 255) # Blue
            elif (light_counter == 4):
                self.set_led(3, 255, 255, 0) # Yellow
            else:
                self.clear_leds()
            self.update_leds()


    def vision_worker(self):
        while not stop_event:
            self.stop_line_follow()
            self.stop()
            time.sleep(0.5)

            try:
                with torch.no_grad():
                    frame = self.camera_server.picam2.capture_array()
                    if frame is None:
                        print("No frame captured for object recognition.")
                        return
            except Exception as e:
                print(f"Error during object recognition: {e}")
            fut = self.executor.submit(recognition_worker, frame)
            self._pending.add(fut)
            fut.add_done_callback(self.handle_recognition_result)
            self.clear_leds()
            self.line_following_active = True
            
            time.sleep(2)        

# ----------------------------------------------------------------------------
# Main Parallel Loop
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    bot = AlphaBot()
    bot.clear_leds()
    bot.start_camera()
    print("Camera server started. Visit http://<your_pi_ip>:5000/ in your browser.")
    time.sleep(5)
    bot.calibrate_sensors()
    bot.servo.middle()
    bot.servo.center()

    obstacle_detection_thread = threading.Thread(target=bot.obstacle_detection)
    line_following_thread = threading.Thread(target=bot.line_following)
    # vision_worker_thread = threading.Thread(target=bot.vision_worker)

    obstacle_detection_thread.start()
    line_following_thread.start()
    # vision_worker_thread.start()

    try:
        while not stop_event:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping execution.")
        stop_event = True
        obstacle_detection_thread.join()
        line_following_thread.join()
        # vision_worker_thread.join()
        bot.executor.shutdown(wait=False)
    finally:
        bot.stop_line_follow()
        bot.stop_camera()
        bot.servo.stop()
        bot.executor.shutdown(wait=True)
        GPIO.cleanup()
        print("All operations stopped. Exiting program.")
