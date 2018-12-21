import numpy as np
import tensorflow as tf
import time, sys, os, math


gravity = 9.81 #m/s**2

'''Start with just gravity, then add air resistance
Simulate the motors on each arm'''
# Air resistance:
#=========================================
#((air density)(drag)(area))
#--------------------------- (velocity)**2
#            2
#=========================================

# q or 0.5*(density)*(velocity**2)
# Drag coefficient:
#=========================================
#
#Cd = D/(q*A)

#Drone datasheet for now, might include everything eventually
class Drone:
    def __init__(self):
        maxspeed=15 #m/s
        maxacceleration = 4 #m/s**2
        directionX = 1 #0-1 is north
        directionY = 1 #0-1 is west
        #Change the values above to simulate a realistic scenario

    def steering(self):
        pass
        #there should be a vector which represents the direction of the drone,
        # (N/S)(W/E) as an example with 1 being north...
        #Then there needs to be a drive forward, backward, sideways
        #As the drone drives in these directions, there should be "hardware"
        # which stabilizes it, as if the drone keeps driving forward, it will crash into the ground
        #This would allow the user to know where it is going and by releasing the controls, the drone will stabilize
        #Steering will include both directions, and tilting


class environment:
    def __init__(self):
        surrounding_area=[[0]*8]*8 #8 by 8 of values 0, meaning nothing around
        #This should be 3D but start with 2D
        Windspeed = 1 #m/s this is implemented later
        winddirection = (1,0) #From north, to south


