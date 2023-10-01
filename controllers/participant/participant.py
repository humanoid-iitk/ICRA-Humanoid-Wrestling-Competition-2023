

from controller import Robot
import sys
#sys.path.append('..').
from utils.motion_library import MotionLibrary

# Eve's locate_opponent() is implemented in this module:.,,,
from utils.image_processing import ImageProcessing as IP
from utils.fall_detection import FallDetection
from utils.gait_manager import GaitManager
from utils.camera import Camera
from utils.finite_state_machine import FiniteStateMachine


class Sultaan (Robot):
    SMALLEST_TURNING_RADIUS = 0.1
    SAFE_ZONE = 0.75
    TIME_BEFORE_DIRECTION_CHANGE = 80  # 8000 ms / 40 ms/.

    def __init__(self):
        Robot.__init__(self)
        self.fall = False
        self.time_step = int(self.getBasicTimeStep())
        self.library = MotionLibrary()

        self.camera = Camera(self)
        self.fall_detector = FallDetection(self.time_step, self)
        self.gait_manager = GaitManager(self, self.time_step)
        self.heading_angle = 3.14 / 2
        self.counter = 0
        self.library.add('New', './Khushi2.motion', loop = True)
        self.leds = {
            'rightf': self.getDevice('Face/Led/Right'), 
            'leftf': self.getDevice('Face/Led/Left'), 
            'righte': self.getDevice('Ears/Led/Right'), 
            'lefte': self.getDevice('Ears/Led/Left'), 
            'chest': self.getDevice('ChestBoard/Led'), 
        }


        # for locking motor
        joints = ['HipYawPitch', 'HipRoll', 'HipPitch', 'KneePitch', 'AnklePitch', 'AnkleRoll']
        self.L_leg_motors = []
        for joint in joints:
            motor = self.getDevice(f'L{joint}')
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(1)
            self.L_leg_motors.append(motor)

        self.R_leg_motors = []
        for joint in joints:
            motor = self.getDevice(f'R{joint}')
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(1)
            self.R_leg_motors.append(motor)

    def run(self):
        while self.step(self.time_step) != -1:
            # We need to update the internal theta value of the gait manager at every step:
            t = self.getTime()
            self.leds['rightf'].set(0xff0000)
            self.leds['leftf'].set(0xff0000)
            self.leds['righte'].set(0xff0000)
            self.leds['lefte'].set(0xff0000)
            self.leds['chest'].set(0xff0000)
            self.gait_manager.update_theta()
            if(self.fall_detector.detect_fall()): 
                self.fall = True
            if 0.3 < t < 2:
                self.start_sequence()
            elif t > 2:
                self.fall_detector.check()
                if(not self.fall):
                    self.walk()

    def start_sequence(self):
        """At the beginning of the match, the robot walks forwards to move away from the edges."""
        self.gait_manager.command_to_motors(heading_angle=0)

    def near_edge(self):
        image = self.camera2.get_image()
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (len(contours) > 0):
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            height, width    = image.shape[:2]
            bottom_threshold = 0.92 * height

            points_below_threshold = sum(point[1] >= bottom_threshold for point in box)
            percentage_below_threshold = points_below_threshold / len(box)
            if percentage_below_threshold >= 0.5 and cv2.contourArea(largest_contour) >= 180:
                return True
            
        return False

    
     def on_ring(self):
        image = self.camera2.get_image()
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        img1 = hsv_image.copy()
        img2 = hsv_image.copy()
        
        colorr_low = np.array([193,62,35])
        colorr_high = np.array([205,107,65])
        colorf_low = np.array([83,62,42])
        colorf_high = np.array([154,110,70])
        mask1 = cv2.inRange(img1, colorr_low, colorr_high)
        mask2 = cv2.inRange(img2, colorf_low, colorf_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        res1 = cv2.bitwise_and(img1,img1,mask1)
        res2 =  cv2.bitwise_and(img2,img2,mask2)
        gray1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        contours1, _ = cv2.findContours(gray1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(gray2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        # Check if contours2 is non-zero before calculating its centroid
        
        cy1, cx1 = None, None
        if len(contours1) > 0:
            contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
            cy1, cx1 = IP.get_contour_centroid(contours1[0])
        # Check if contours2 is non-zero before calculating its centroid
        cy2, cx2 = None, None
        if len(contours2) > 0:
            contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
            cy2, cx2 = IP.get_contour_centroid(contours2[0])

        print("cy1 = ", cy1, ", cy2 = ", cy2)
        if len(contours1) > 0 and len(contours2) > 0:
            if cy1 > cy2:
                return False
            else:
                return True


    
    def walk(self):
        normalized_x = self._get_normalized_opponent_x() 
        desired_radius = (self.SMALLEST_TURNING_RADIUS / normalized_x) if abs(normalized_x) > 1e-3 else None
        if(normalized_x > 0): 
            self.heading_angle = 3.14/4
            self.counter = 0;  
        elif(normalized_x < 0): 
            self.heading_angle = -(3.14/4)
            self.counter = 0 
        elif(normalized_x == 0): 
            return  
        self.counter += 1
        self.gait_manager.command_to_motors(desired_radius=desired_radius, heading_angle=self.heading_angle)
        self.library.play('New')

    def _get_normalized_opponent_x(self):
        """Locate the opponent in the image and return its horizontal position in the range [-1, 1]."""
        img = self.camera.get_image()
        _, _, horizontal_coordinate = IP.locate_opponent(img)
        if horizontal_coordinate is None:
            return 0
        return horizontal_coordinate * 2 / img.shape[1] - 1

# create the Robot instance and run main loop
wrestler = Sultaan()
wrestler.run()






