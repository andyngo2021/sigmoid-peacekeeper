# Peacekeeper

<p align="center"><b>[WORK IN PROGRESS]</b></p>

Computer vision program to determine if someone is in danger of falling asleep at the wheel with the power of OpenCV

<h3>Instructions</h3>

First, clone the GitHub repository.
```
git clone https://github.com/andyngo2021/sigmoid-peacekeeper.git
```
Then navigate to the directory
```
cd sigmoid-peacekeeper
```

Pip install the necessary libraries
```
pip install -r requirements.txt
```

Then calibrate the system by running calibrate.py
```
python calibrate.py
```
1. When the prompt appears, close your eyes.
2. When you hear a beep sound, you can open your eyes and then close the program by hitting the ESC key.

Run the actual program, main.py
```
python main.py
```

NOTE: You can quit out of the main program by also using the ESC key.

<br>

<h3>Features (so far)</h3>

I intend to keep working on this project even after Sigmoid Hacks to make it as good as possible but here's what I was able to accomplish so far by relying on online tutorials:


- Will give an alert if eyes are closed for an extended period of time
- Will give an alert if you look in the corner of your eyes for a long time
- Will alert you if your head isn't pointed to the center of the road 

<br>

<h3>Goals for the future</h3>

- I'm kinda broke rn and don't have a Raspberry Pi but in the future I'd like to get one and make it possible for my code to 
work on one
- Maybe a phone app might be better instead so no one has to buy anything to get this to work
- Yawning detection
- Analysis of the movement of the car itself to determine if the driver is incapable of driving in a straight line

