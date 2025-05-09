# Upscaler with ZMQ subscriber

This folder contains the script used to read, upscale and publish frames to a ZMQ subscriber. 

## Scripts

 - [1.upscaler_zmq](./1_upscaler_zmq.py) : Launches the listening to a subscriber. The model is initialized and ready for frame inference. After upscaling the model outputs and save frames and publishes to an other ZMQ adress (10000).  
 - [2.publisher_zm](./2_publisher_zmq.py) : ZMQ publisher demo to test the subscriber. Reads a .zsc 60x80 lep2 file. (currently processing only the first 2000 frames).  
 - [3.subscriber_zm](./3_subscriber_zmq.py): Subscribes to the output address of the upscaler and check the received frames. Options to read as binary or numpy.  
 - [4.reader](./4_reader_bin.py): Reads the save binary file from the upscaler.  

## Usage

Starting the 1 for the first time will download model weights (defaults). By default the script looks for GPU, can be changed to CPU for testing on other machines.  
The scripts listens to the specified ZMQ socket (default 9999) and outputs to an other one (10000).  
```bash 
python 1_upscaler_zmq.py
```
Once the upscaler is started one can launch the 2 in parallel for a demo publisher of frames. OPTIONAL : The frames are published with a sligth delay to test processing speed.  
```bash 
python 2_publisher_zmq.py
```
During the processing of 1 of frames published by 2, 3 can be launched for live subscribing to upscaled frames.  
```bash 
python 3_subscriber_zmq.py
```
Since the 1 also saves the files, one can read/reconstruct the frames file with 4 .  
```bash 
python 4_reader_bin.py
```

## Queue integration 
To implement the queue system a secondary queue thread is created.  
_QueueWorker_ : A new Class is introduced to handle ZMQ receiving in a separate thread.
The QueueWorker Class is instantiated and started so the ZMQUpscaler's run can function normally.

**Queue functions** :   
_self.queue_in.put(frame)_: puts the received frame into the queue.  
_self.queue_in.get(timeout=0.1)_: is used when the main thread retrieves frames from the queue. The timeout prevents the get() call from blocking indefinitely if the queue is empty. Some tests to be done on the timeout option.  
_self.queue_in.task_done()_: acknowledges to queue that processing of a given frame is finished.  
