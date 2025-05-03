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


# ----------------------------------------------------------------------------
# AlphaBot Class with LED, Buzzer, and Recognition Integration
# ----------------------------------------------------------------------------
class AlphaBot(object):
    def __init__(self):
        with open('initialized_data.pkl', 'rb') as f:
            loaded_func = dill.load(f)

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

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

# ----------------------------------------------------------------------------
# Main Serial Loop
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    bot = AlphaBot()
    bot.clear_leds()
    #bot.start_camera()
    #print("Camera server started. Visit http://<your_pi_ip>:5000/ in your browser.")
    time.sleep(2)
    #bot.calibrate_sensors()

    # Initialize a state variable for object detection servo movements.
    # The state cycles through: 'left', 'center1', 'right', 'center2'
    object_detection_state = "left"
    last_object_detection_time = time.time()
    object_detection_interval = 1.0  # seconds between object detection updates

    try:
        while not stop_event:
            bot.forward()        


            time.sleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping execution.")
        stop_event = True
    finally:
        bot.stop_line_follow()
        bot.stop_camera()
        bot.servo.stop()
        GPIO.cleanup()
        print("All operations stopped. Exiting program.")
