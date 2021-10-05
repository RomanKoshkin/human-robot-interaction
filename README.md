# Tutoring Object Manipulation Skills in a Human-Robot Interaction Paradigm

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
![GitHub](https://img.shields.io/github/license/RomanKoshkin/ema_x_bot)
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This project explored the ability of recurrent neural networks trained on a small set of motor prim- itives to generate meaningful novel patterns with limited corrective feedback from the experimenter.

<img src="assets/Det_training.pdf" width=25% height=25%>

## RECORD_SEPARATE_trajectories_2.ipynb

- BUILDS SEVERAL TRAJECTORIES BASED ON CAPTURED WAYPOINTS, SAVES THEM INTO A RADA FILE
- REPLAYS ALL THE TRAJECTORIES (WITH SHORT PAUSES TO REPOSITION THE CUBE), WHILE DENSELY RECORDING 
  JOINT ANGLES AND VIDEO FRAMES. RECORDS INTO 'RAD0.dat', 'RAD1.dat' FILES
      DICTIONARIES WITH KEYS ('joints', 'igm', 't')
- ALLOWS ONE TO SEE VIDEO AND JOINT ANGLES FROM RECORDED TRAJECTORIES

## VAE_RNN (vid_motor) TRAIN.ipynb

- A VAE ENCODES VIDEO AND JOINT ANGLES TOGETHER INTO A LATENT REPRESENTATION OF SIZE 20 (16 JOINTS, NO COMPRESSION, VIDEO 64X64 INTO A 4D REPRESENTATION)
- USES THE LATENT REPRESENTATIONS OBTAINED BY FORWARDING THE FIRST FRAME THROUGH THE VAE
- IN A SEQUENCE AND THE JOINT ANGLES AT TIME 0, AS __ THE HIDDEN STATES __ FOR THE RNN
- THE (DETERMNISTIC) RNN SIMILARLY TO THE THE VAE TAKES VISUAL AND MOTOR INPUT, BUT INSTEAD AT ITS BOTTLENECK IT HAS A RECURRENT
LAYER.

## VAE_RNN (vid_motor) TEST.ipynb

- trained on three similar but separate trajectories (not concatenated)
- THE VAE (ENCODES THE MOTOR+VISUAL INTO A LATENT STATE) AND RNN (USES THE LATENT STATE
  AS THE INITIAL HIDDEN STATE) MODELS
- WEIGHTS TRAINED BY VAE_encode_vid_motor_TEST
- USES THESE MODELS TO CONTROLS THE PHYSICAL ROBOTA VAE ENCODES VIDEO AND JOINT ANGLES TOGETHER INTO A LATENT REPRESENTATION OF SIZE 20 (16 JOINTS, NO COMPRESSION + VIDEO 64X64 INTO A 4D REPRESENTATION)


## RECORD_SEPARATE_trajectories_3_w_head.ipynb

Same as RECORD_SEPARATE_trajectories_2.ipynb but also records data for the training of PV-RNN/YOLOv3.	

## Train RNN.ipynb

This notebook trains a deterministic RNN.

## Test RNN-Original_Firmware.ipynb

This notebook tests a deterministic RNN (trained in Train RNN.ipynb) and calls/implements all the necessary control functions to run the cube grab-and-lift experiment.

## PVRNN Prior Generation-with_head_tracking.ipynb

loads a pre-trained PV-RNN/YOLOv3 model and demonstrates PV-RNN in the prior generation mode. I.e. generates a primitive seen during training.

## ERROR REGRESSION_with_head_tracking.ipynb

loads a pre-trained PV-RNN/YOLOv3 model and demonstrates PV-RNN in the error regression mode.


checkpont_ww [BEST] - this checkpoint reproduces the motions WELL
	here we allow the robot's head to track the object. no concatenations

checkpoint_ww1 - here we object tracking is OFF. Dataset IS concatenated

checkpoint_ss - object tracking is on, datasets concatenated, pred 0.1, gt 0.9

# the best model

checkpoint_overnight_90_10 - object tracking is on, datasets concatenated, pred 0.1, gt 0.9 (OVERNIGHT TRAINING) {BEST}
