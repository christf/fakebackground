# FakeBackground

This software can produce a virtual background on webcams via v4l2 loopback 
device on Linux.
The idea and much of the code was borrowed from: https://github.com/fangfufu/Linux-Fake-Background-Webcam

Unfortunately running tensorflow on my laptop cooks the device while generating 
2 frames per second output with a 4 second delay. Also a bit of the background 
  was still revealed.
It might be how I chose the parameters for bodypix but since my machine clearly 
was not up to the task I abandoned that approach and started using a green 
screen.
That way my machine creates 15FPS with sub second latency, which is fast enough for video calls.

# Prerequisites

* bright Green Cloth
* depending on the results, Lighting


# Things to Do

## Automatic Initialization

It would be nice if the fake background detector would initialize the correct 
color space. We could use bodypix to take a few analysis frames and then have 
the resulting image analyzed for hsv values which then are used during the 
run-time of the tool.

It would be nice if this could be achieved without running an external node 
application.

## Remove wobbling

Webcam images contain quite a bit of noise. The image processing removes the 
background based on HSV values. While this works, the noise in the images 
causes wobbling edges.

It would be interesting to reduce that. Possibly by not calculating an entirely 
new mask for every single frame to smoothen out the noise a bit.


## Getting rid of the green screen

Ideally we could get rid of the green screen entirely - at home it is just a 
nuisance having to have it around.
