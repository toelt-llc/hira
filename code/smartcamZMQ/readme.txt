====================================================
   QUICK TUTORIAL FOR THE NEXT2U SMARTCAM PACKAGE
====================================================

1. Introduction
---------------

This folder contains the code able to read, and show in a graphical user interface, 
the videos obtained by writing on file the frames captured through a thermal camera 
and/or a visible/nir camera, which have been co-registered and optically calibrated.
The instructions for using this package are given in what follows, both as a simple 
user and then by interacting with the internal functions to programmatically access 
to the information stored in the videos themselves.

2. Installation and execution
-----------------------------

The SmartCam tool requires a Python interpreter, version 3.8 or higher, but still 
compatible with the libraries exploited by the software. The list of necessary 
libraries is given in the "requirements.txt" file: all this libraries must be 
installed in the system (to this respect, several methods are possible: typically, 
such libraries are installed in a "virtual environment" using a tool like pip or 
conda, but simply choose your preferred installation method). Once the libraries 
have been installed (and the virtual environment, if any was created, has been 
activated), just execute the python interpreter on the main package file, which 
is smartcamPlayer.py. Then, a GUI should appear, through which it is possible to 
open and navigate the provided recordings. 
NOTICE: according to the format in which the videos have been saved (that is, just 
a binary file, or a file only for the meta-data plus a couple of mpeg recordings),
and then the size of the the recording itself (which of course increases with the
video duration), the tool may require a long time to open a recording! Nonetheless,
don't be afraid: just give the software enough time, in the end the images will 
appear in the GUI! Finally, once a recording has been opened, you can easily surf
though the frames by using the buttons on the left of the GUI (or even the slider
on the bottom of the images).

3. Programmatic interaction
---------------------------

Accessing the information stored in the recordings (images, landmarks if available,
and timestamps) is quite easy as well. In this case, you do not really need to know
how such information has been stored: it is indeed enough to interact (though a small 
bunch of methods) with a class called SmartCamDecoder (obviously implemented in the 
smartcamDecoder.py script), which, as the name suggests, is in charge of properly 
decoding the information saved in the involved files. The way this class must be 
used is explained in what follows:

a) create the object:

	decoder = SmartCamDecoder()

b) open the recording you are interested in:

	num_frames, total_time, video_rate = decoder.on_file(name)

   where the name parameter is the full filename (with the relative or absolute path) 
   of the recording you want to open. Upon a successful operation, this method returns
   non-zero values, representing the number of available frames in the recording, its
   duration (float value, in seconds) and the video rate of the recording itself.
   Notice that if the name parameter is None, then the recording is closed: after such 
   an operation the decoder must be discarded, and a new instance must be issued. In 
   general, use a fresh new decoder instance for each recording you want to analyze.

c) let the decoder decode the information for the desired frame:

	decoder.read_frame(index)

   where the index parameter must be an integer in the range [0, num_frames), num_frames
   being the value obtained when opening the recording. It is not necessary to read the
   various frames in sequence, you can jump from a frame to any other one, with the only 
   limitation that the index value must be valid. This method returns some results, but 
   they are not useful in this context (so, simply discard them).

d) get the decoded images for the frame which has been just read in:

	vImg, vT, tImg, tT = decoder.get_images()

   the function returns, in order, the visible/nir image, its timestamp, then the same
   fields for the thermal case. All the timestamps are given as unix-times, expressed in 
   micro-seconds.

e) get the decoded landmarks (if available) for the frame which has been just read in:

	vPts, vVis, tPts, tVis, score, lands = decoder.get_points()

   where the results represent, respectively:
    - the visible/nir image landmarks
    - the visibility of each previous point (array of booleans, where true indicates
      that the point is indeed visible in the image)
    - the thermal image landmarks
    - the visibility of each previous point (like the previous visible/nir case, but now 
      for the thermal one)
    - a global interger score, representing the fact that the landmarks have been found 
      or not (they have been detected if this score is larger than 0)
    - the raw mediapipe (the exploited tracking library) landmarks


4. Image subscriber
-------------------
The Smartcam package includes a pretty small script which implements a ZMQ subscriber:
besides its normal job, indeed, the player sends each thermal image decoded from file 
through a ZMQ publisher, bound on port 9999. Thus, the subscriber connects to this port, 
and then decodes (and shows) each image sent by the player on an opencv window.
Its usage is trivial, but it should be noticed that to run this script a "standard" 
(that is, non headless) installation of the opencv library is necessary. This means 
that a different virtual environment must be used from the one created for the player
(for which, instead, the standard opencv installation does not work)...



That's all!

For any question/problem/request/clarification or other stuff, do not hesitate to write 
to s.nocco@next2u-solutions.com






