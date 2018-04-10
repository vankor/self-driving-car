# PID control for seld-driving car

PID is a kind of controllers being used in robotics. It collects error signal (cross track error) to know the difference between desired and measurd values and sets robot's behaviour to be closer to desired points. In current project I am using cross-track error (cte) - difference between actual car positions and reference trajectory. PID controller should fing steering angles of a car to minimize cte.

## PID components

### P component (proportional gain)

Proporional gain component causes the car to steer in oposite to the error value direction. For example if reference trajectory line is left then car should turn right proportionally to excess value. 

### D component (differential gain)

D component is puprposed to avoid overshooting and ringing of center line. A properly tuned D parameter will cause the car to approach the center line smoothly without ringing.

### I (integral gain)

Integral gain aggregates the cross-track error over time. In current case integral gain can reduce car oscilations for hight speed values. Biases can be mitigated, for instance if a zero steering angle does not correspond to a straight trajectory. The corresponding contribution to the steering angle is given by -K_i sum(cte). 

## Parameters tuning
Parametes were initialy tuned manualy trying zero values for D and I components. Also, Twidle algorithm was applied to find optimal parameter values.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 



