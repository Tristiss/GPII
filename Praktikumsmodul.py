import time
import math
import locale
import serial
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_analog_in_v3 import BrickletAnalogInV3
from tinkerforge.brick_silent_stepper import BrickSilentStepper
from tinkerforge.bricklet_color_v2 import BrickletColorV2
from pynput import keyboard
import numpy as np
import matplotlib as mpl

class hardware():
    @staticmethod
    def setup(**kwargs):
        brick_lets = { # add more if more are needed
            "analog_in" : BrickletAnalogInV3,
            "silent_stepper" : BrickSilentStepper,
            "colour" : BrickletColorV2
        }

        # sets up IP connections to the stepper and analog in bricklet and sets default configurations
        HOST = "localhost"
        PORT = 4223

        # Create Tinkerforge objects
        ipcon = IPConnection()

        bricks = []

        for name, uid in kwargs:
            if name not in brick_lets:
                hardware.shut_down(ipcon)
                raise KeyError("Uknown Brick(let)")
            
            bricks.append(brick_lets[name](uid, ipcon))

        ipcon.connect(HOST, PORT)
        # Don't use device before ipcon is connected

        return ipcon, tuple(bricks)
    
    @staticmethod
    def shut_down(ipcon, ser:list=[]):
        ipcon.disconnect() # disconnect IP Connection
        for i in ser: # disconnect any open serial connections
            hardware.send_ser_msg(i, b"SOUT0\r") # specific to Manson
            i.close()

    @staticmethod
    def send_ser_msg(ser, msg): # this works for any device that returns a OK\r after a command 
        # DO NOT USE THIS FUNCTION IF YOU NEED THE RECEIVED DATA
        rec_count = 0
        ok_received = False

        ser.write(msg) # send serial command
        while ok_received == False: 
            # this loop checks for the OK\r that in this case the manson returns
            # this breaks for other devices that don't send OK\r as an answer or if they send data, that
            # data will be lost when using this function
            start = time.perf_counter()
            returned_msg = []
            while returned_msg[-1:] != [b"\r"] and time.perf_counter() - start < 10:
                # reads serial receive buffer (bad implementation of serial.readline())
                returned_msg.append(ser.read())
                t = time.perf_counter()
            
            if b"O" in returned_msg and b"K" in returned_msg: # checks if the received message is an ok
                print("OK received!")
                ok_received = True
                return None
            returned_msg = []
            rec_count += 1

            if rec_count > 3: # raise an error if the three lines after a command don't inlcude an OK\r
                raise TimeoutError("No OK received")
            
    @staticmethod
    def manson_init(ser): # send necessary commands to initiate communication with Manson SSP 8160/8162
        hardware.send_ser_msg(ser, b"SABC3\r")
        hardware.send_ser_msg(ser, b"ENDS\r")
        hardware.send_ser_msg(ser, b"SOCP0100\r")
        hardware.send_ser_msg(ser, b"SOUT1\r")

    @staticmethod
    def monitor(flag):
        # this function is a daemon thread that raises a flag that the measurment and calibration loops look out for
        pressed_keys = set()
        def on_press(key):
            # checks if the pressed key is the Escape button
            pressed_keys.add(key)
            if keyboard.Key.esc in pressed_keys:
                print("Esc detected! Stopping measurement...")
                flag.set()  # Signal measurement thread to stop

        def on_release(key):
            # released keys are being deleted from the set of pressed keys
            pressed_keys.discard(key)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()  # Blocks until shutdown
    
    @staticmethod
    def set_voltage(ser:serial.Serial, volt:float): # volt [V]; specific to the Manson SSP 8160/8162
        volt = int(volt * 100) # convert V to mV
        msg = f"VOLT3{volt:04d}\r"
        hardware.send_ser_msg(ser, msg.encode())
        print(msg[5:])
        time.sleep(0.5)
        if volt == 0:
            time.sleep(9)

class eval():
    @staticmethod
    def k(n:int) -> float:
        k_li = {
            1   :   13.97,
            2   :   4.527,
            3   :   3.306,
            4   :   2.87,
            5   :   2.649,
            6   :   2.517,
            7   :   2.429,
            8   :   2.366,
            9   :   2.32,
            10  :   2.284,
            11  :   2.255,
            12  :   2.231,
            13  :   2.212,
            14  :   2.195,
            15  :   2.181,
            16  :   2.169,
            17  :   2.158,
            18  :   2.149,
            19  :   2.14,
            20  :   2.133,
            21  :   2.126,
            22  :   2.12,
            23  :   2.115,
            24  :   2.11,
            25  :   2.105,
            26  :   2.101,
            27  :   2.097,
            28  :   2.093,
            29  :   2.09,
            30  :   2.087,
            31  :   2.084,
            32  :   2.081,
            33  :   2.079,
            34  :   2.076,
            35  :   2.074,
            36  :   2.072,
            37  :   2.07,
            38  :   2.068,
            39  :   2.066,
            40  :   2.064
        }
        return k_li[n]

    @staticmethod
    def u_voltage(volt): # this is specific to the Manson SSP 8160 / 8162
        return volt * 0.002 + 0.05

    @staticmethod
    def weighted_mean(values:list[float], uncertainties:list[float]) ->float:
        if len(values) != len(uncertainties):
            raise ArithmeticError("Miss match in val and unc length")
        upper = lower = 0
        for i in range(len(values)):
            upper += values[i] / np.square(uncertainties[i])
            lower += 1 / np.square(uncertainties[i])

        return upper / lower

    @staticmethod
    def unc_sum(uncertainties:list[float]) -> float:
        lower = 0
        for i in uncertainties:
            lower += 1 / np.square(i)
        return lower

    @staticmethod
    def internal_unc_type_a(uncertainties:list[float]) -> float:
        lower = eval.unc_sum(uncertainties)
        n = len(uncertainties)
        return eval.k(n - 1) * np.sqrt(1 / lower) / np.sqrt(n)

    @staticmethod
    def external_unc_type_a(values:list[float], uncertainties:list[float], weighted_mean:float) -> float:
        n = len(uncertainties)
        if len(values) != n:
            raise ArithmeticError("Miss match in val and unc length")
        upper = 0
        lower = eval.unc_sum(uncertainties)
        for i in range(len(values)):
            upper += np.square(values[i] - weighted_mean) / uncertainties[i]
        return eval.k(n - 1) / np.sqrt(n) * np.sqrt(upper / ((len(values) - 1) * lower))

    @staticmethod
    def weigted_type_a_unc(values:list[float], uncertainties:list[float]) -> list[float]:
        mean = eval.weighted_mean(values, uncertainties)
        internal = eval.internal_unc_type_a(uncertainties)
        external = eval.external_unc_type_a(values, uncertainties, mean)
        return [mean, max(internal, external)]

    @staticmethod
    def normal_type_a_unc(values:list[float]) -> float:
        n = len(values)
        return eval.k(n - 1) * np.std(values) / np.sqrt(n)

    # Source - https://realpython.com/python-rounding/#rounding-up
    # By DevCademy Media Inc. DBA Real Python
    # Retrieved 2026-01-14, usage is allowed only non commercially
    #import math
    # ...
    #def round_up(n, decimals=0):
    #   multiplier = 10**decimals
    #   return math.ceil(n * multiplier) / multiplier

    @staticmethod
    def round_up(n, decimals = 0):
        multiplier = 10**decimals
        return math.ceil(n * multiplier) / multiplier

    @staticmethod
    def eval_start_up():
        # enable latex in plots
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams.update(mpl.rcParamsDefault)

        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')