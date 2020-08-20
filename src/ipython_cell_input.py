set_position_control(torobo)
torobo.move(ToroboOperator.TORSO_HEAD, positions=[0, 0.3, 0.0, 0.5], duration=3)
time.sleep(3.1)
image_topic = "/camera/color/image_raw"
msg = rospy.wait_for_message(image_topic, Image)
cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
plt.figure(figsize=(10,10))
plt.imshow(cv2_img)
plt.savefig('/home/torobo/catkin_ws/src/tutorial/PyTorch-YOLOv3/data/samples/dice.jpg', dpi=300)