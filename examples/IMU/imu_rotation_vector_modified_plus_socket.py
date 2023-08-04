#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation as R



device = dai.Device()


imuType = device.getConnectedIMU()
imuFirmwareVersion = device.getIMUFirmwareVersion()
print(f"IMU type: {imuType}, firmware version: {imuFirmwareVersion}")

if imuType != "BNO086":
    print("Rotation vector output is supported only by BNO086!")
    exit(1)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920, 1080)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)


xlinkOut.setStreamName("imu")

# enable ROTATION_VECTOR at 400 hz rate
imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)
# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(1)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)

# Link plugins IMU -> XLINK
imu.out.link(xlinkOut.input)



def quaternion_to_euler(x, y, z, w):
    r = R.from_quat([x, y, z, w])
    # return r.as_euler('zyx', degrees=True)
    return r.as_euler('xyz', degrees=True)




# Pipeline is defined, now we can connect to the device
with device:
    device.startPipeline(pipeline)

    def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds()*1000

    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    baseTs = None
    while True:
        imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived
        videoIn = video.get()
        imuPackets = imuData.packets
        frame_disp = videoIn.getCvFrame()
        true_roll = []
        true_pitch = []
        true_yaw = []
        for imuPacket in imuPackets:
            rVvalues = imuPacket.rotationVector

            rvTs = rVvalues.getTimestampDevice()
            if baseTs is None:
                baseTs = rvTs
            rvTs = rvTs - baseTs

            imuF = "{:.06f}"
            tsF  = "{:.03f}"

            # print(f"Rotation vector timestamp: {tsF.format(timeDeltaToMilliS(rvTs))} ms")
            # print(f"Quaternion: i: {imuF.format(rVvalues.i)} j: {imuF.format(rVvalues.j)} "
            #     f"k: {imuF.format(rVvalues.k)} real: {imuF.format(rVvalues.real)}")
            # print(f"Accuracy (rad): {imuF.format(rVvalues.rotationVectorAccuracy)}")

            euler_angles = quaternion_to_euler(imuF.format(rVvalues.i), imuF.format(rVvalues.j), imuF.format(rVvalues.k), imuF.format(rVvalues.real))
            print("euler_angles ", euler_angles)

            euler_radian = np.radians(euler_angles)

            roll, heading, pitch = euler_radian
            true_roll.append(roll)
            true_pitch.append(pitch - 90)
            true_yaw.append(heading)

        roll = np.average(true_roll)
        pitch = np.average(true_pitch)
        heading = np.average(true_yaw)

        width = 1920
        height = 1080
        height_camera = 400
        distance_to_horizon = 3570 * height_camera ** (1 / 2)
        alpha = np.arcsin(height_camera / distance_to_horizon)
        #TODO: change focal length
        offset = int(np.tan(abs(alpha - pitch)) * 2000 * np.sign(alpha - pitch))
        offset_roll = 0
        include_roll = False
        if include_roll:
            if roll > 0:
                offset_roll = int(np.tan(roll) * width // 2)
            else:
                offset_roll = -int(np.tan(-roll) * width // 2)

        left_horizon = height // 2 + offset + offset_roll
        right_horizon = height // 2 + offset - offset_roll
        # left_horizon = 500
        # right_horizon = 600

        # print("left_horizon ", left_horizon)
        cv2.line(frame_disp, (0, height - left_horizon), (width - 1, height - right_horizon),
                 (255, 255, 255), 2)
        cv2.imshow("video", frame_disp)




        if cv2.waitKey(1) == ord('q'):
            break
