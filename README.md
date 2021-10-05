# Tutoring Object Manipulation Skills in a Human-Robot Interaction Paradigm

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
![GitHub](https://img.shields.io/github/license/RomanKoshkin/ema_x_bot)
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This project explored the ability of recurrent neural networks trained on a small set of motor prim- itives to generate meaningful novel patterns with limited corrective feedback from the experimenter.

## Introduction

According to one hypothesis, complex motor behaviors are compositions of a limited set of innate and/or learned motor primitives. We attempted to test this hypothesis in a humanoid robot controlled by a recurrent neural network (RNN). Specifically, our robot was trained to grasp a target object in three different locations. If the hypothesis is true, the robot should be able grasp the target not only in the trained, but also in other locations. 

Aware that model architecture might lead to vastly different results, we explored two alternative architectures: a simpler deterministic RNN (with and without a separately trained variational autoencoder) and a more advanced variational Bayes RNN. In the Methods section below we limit our account to the most important aspects, for full details please see the Git repository ¥cite{humrobint}.

## Methods

We used a 16-joint humanoid robot (Torobo Humanoid, Tokyo Robotics). The models were trained on a dataset of three primitives either  representing separate sequences each or stacked along the time axis into one long trajectory. A primitive was defined as a sequence of joint angle vectors of size 16 comprising a motion trajectory. Each primitive began at the starting position, continued smoothly until the object -- a soft toy die (approx. 13 x 13 x 13 cm) lying on a white plastic surface -- was grasped, (for the deterministic model -- also lifted, put back on the table), released, and terminated back at the starting position. Complete with motor data (joint angles comprising the movement trajectory) the datasets contained temporally synchronized visual inputs (grayscale images of size 64 x 64 pixels). The training data were recorded using a separate Python script at 10 Hz as the robot's arms were moving along a trajectory of seven waypoints, while the head was fixated on the object by an independently running object detection model (YOLO v3  ¥cite{yolov3, ErikLinderNoren2020}). Note that with the deterministic models YOLO was only used during training, while during testing the RNN controlled all the joints (including neck). The deterministic models were were implemented in PyTorch and trained with two loss functions (for motor and visual loss). For the stochastic model we used the NRL C++ library; for details please refer to ¥cite{ahmadi2019novel, chame2020towards}

## Deterministic models

We used two slightly different deterministic models. The first and simpler one (Figs. ¥ref{fig:Det_training}, ¥ref{fig:Det_testing}) was trained on a dataset of three primitives each representing a separate sequence; the primitives were stacked along the time axis into one long trajectory.

<img src="assets/Det_training.png" width=80%>

However, the downside of this approach is that the model may simply learn one long trajectory and be 'reluctant' to switch/interpolate between primitives in a way that is not consistent with the order of points in the training trajectory. In other words, it is unclear whether such a model can in principle handle cases where the die is located off the spots seen during training.

<img src="assets/Det_testing.png" width=80%>

Training the RNN on separate sequences seemed a reasonable next step. Our second model (Fig. ¥ref{fig:VAE}) was trained on separate primitives. During training, initial hidden states corresponding to each trajectory were taken from the bottleneck of a separate variational autoencoder pre-trained on the same dataset. These hidden states provided low-dimensional representations of the future trajectory the RNN should generate. With only three primitives three different arbitrary hidden variables would work equally well, but with a larger number of primitives the corresponding initial hidden states should retain the relative structure of the primitives they encode.

<img src="assets/VAE.png" width=80%>

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
