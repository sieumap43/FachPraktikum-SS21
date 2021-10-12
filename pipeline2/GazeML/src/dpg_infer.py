#!/usr/bin/env python3
"""
Author: An Trieu
Date: 17.07.2021
"""
import argparse
import os
import queue
import threading
import time

import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf

from datasources import Video, Webcam
from models import DPG, ELG
import util.gaze
import util.gazemap as gm

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 16
        eye_image_shape = (90, 150)

        # Define webcam stream data source
        # Change data_format='NHWC' if not using CUDA
        data_source = Video(args.from_video,
                            tensorflow_session=session, batch_size=batch_size,
                            data_format='NCHW' if gpu_available else 'NHWC',
                            eye_image_shape=eye_image_shape)

        # Define model
        model = DPG(
            session, train_data={'videostream': data_source},
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                },
            ],
        )


        # Begin visualization thread
        inferred_stuff_queue = queue.Queue()

        def _visualize_output():
            last_frame_index = 0
            last_frame_time = time.time()
            fps_history = []
            all_gaze_histories = []

            if args.fullscreen:
                cv.namedWindow('vis', cv.WND_PROP_FULLSCREEN)
                cv.setWindowProperty('vis', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

            while True:
                # If no output to visualize, show unannotated frame
                if inferred_stuff_queue.empty():
                    next_frame_index = last_frame_index + 1
                    if next_frame_index in data_source._frames:
                        next_frame = data_source._frames[next_frame_index]
                        if 'faces' in next_frame and len(next_frame['faces']) == 0:
                            if not args.headless:
                                cv.imshow('vis', next_frame['bgr'])
                            if args.record_video:
                                video_out_queue.put_nowait(next_frame_index)
                            last_frame_index = next_frame_index
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        return
                    continue

                # Get output from neural network and visualize
                output = inferred_stuff_queue.get()
                # print("HAHAHAHAHA") # DEBUG
                # print(output.keys()) # DEBUG
                # print("gaze", output['gaze'].shape) # DEBUG
                # print("gaze", output['gaze'])
                # print("eye", output['eye'].shape) # DEBUG
                # print("gazemaps", output['gazemaps'].shape)
                bgr = None
                for j in range(batch_size):
                    frame_index = output['frame_index'][j]
                    if frame_index not in data_source._frames:
                        continue
                    frame = data_source._frames[frame_index]
                    # Decide which gazemaps are usable
                    # gazemaps_amax = np.amax(output['gazemaps'][j, :].reshape(-1, 18), axis=0)
                    # can_use_eye = np.all(gazemaps_amax > 0.7)
                    can_use_eye = 1

                    start_time = time.time()
                    eye_index = output['eye_index'][j]
                    bgr = frame['bgr']
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    # eye_gazemaps = output['gazemaps'][j, :]
                    if eye_side == 'left':
                        eye_image = np.fliplr(eye_image)
                    # Embed eye image and gazemaps for picture-in-picture
                    eye_upscale = 1
                    eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
                    eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
                    eye_image_annotated = np.copy(eye_image_raw)

                    # gmap_iris = output['gazemaps'][j, :, :, 0]
                    # gmap_eyeball = output['gazemaps'][j, :, :, 1]
                    # print("gmap_iris------")
                    # print("shape", gmap_iris.shape)
                    # print(np.max(gmap_iris), np.min(gmap_iris))
                    # print("------------------")

                    # gmap_iris = np.float32(eye_gazemaps[j, :, :, 0])
                    # gmap_eyeball = np.float32(eye_gazemaps[j, :, :, 1])
                    gaze = output['gaze'][j,:]
                    gmap_iris, gmap_eyeball =  gm.from_gaze2d(gaze, eye_image_shape)
                    gmap = np.clip(gmap_iris + 0.5 * gmap_eyeball, a_min=0, a_max=1)
                    gmap = cv.cvtColor(gmap, cv.COLOR_GRAY2BGR) * 255

                    print('bgr min max', bgr.min(), bgr.max())
                    face_index = int(eye_index / 2)
                    eh, ew, _ = eye_image_raw.shape
                    gmap_h, gmap_w, _ = gmap.shape

                    # Segmented eyes
                    v0 = face_index * 2 * eh
                    v1 = v0 + eh
                    # v2 = v1 + eh
                    u0 = 0 if eye_side == 'left' else ew
                    u1 = u0 + ew
                    # Gazemaps
                    v2 = v1 + gmap_h
                    ueb = gmap_w
                    uir = ueb + gmap_w

                    bgr[v0:v1, u0:u1] = eye_image_raw
                    bgr[v1:v2, 0:ueb] = gmap
                    # bgr[v1:v2, ueb:uir] = gmap

                    # Visualize preprocessing results
                    for f, face in enumerate(frame['faces']):
                        cv.rectangle(
                            bgr, tuple(np.round(face[:2]).astype(np.int32)),
                            tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                            color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                        )

                    # Smooth and visualize gaze direction
                    num_total_eyes_in_frame = len(frame['eyes'])
                    if len(all_gaze_histories) != num_total_eyes_in_frame:
                        all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
                    gaze_history = all_gaze_histories[eye_index]
                    if can_use_eye:
                        # i_x0, i_y0 = iris_centre
                        # e_x0, e_y0 = eyeball_centre
                        # theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                        # phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                        #                         -1.0, 1.0))
                        # current_gaze = np.array([theta, phi])
                        # gaze_history.append(current_gaze)
                        # gaze_history_max_len = 10
                        # if len(gaze_history) > gaze_history_max_len:
                        #     gaze_history = gaze_history[-gaze_history_max_len:]
                        # util.gaze.draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
                        #                     length=120.0, thickness=1)
                        pass
                    else:
                        gaze_history.clear()

                    dtime = 1e3*(time.time() - start_time)
                    if 'visualization' not in frame['time']:
                        frame['time']['visualization'] = dtime
                    else:
                        frame['time']['visualization'] += dtime

                    def _dtime(before_id, after_id):
                        return int(1e3 * (frame['time'][after_id] - frame['time'][before_id]))

                    def _dstr(title, before_id, after_id):
                        return '%s: %dms' % (title, _dtime(before_id, after_id))

                    if eye_index == len(frame['eyes']) - 1:
                        # Calculate timings
                        frame['time']['after_visualization'] = time.time()
                        fps = int(np.round(1.0 / (time.time() - last_frame_time)))
                        fps_history.append(fps)
                        if len(fps_history) > 60:
                            fps_history = fps_history[-60:]
                        fps_str = '%d FPS' % np.mean(fps_history)
                        last_frame_time = time.time()
                        fh, fw, _ = bgr.shape
                        cv.putText(bgr, fps_str, org=(fw - 110, fh - 20),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                                   color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)
                        cv.putText(bgr, fps_str, org=(fw - 111, fh - 21),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.79,
                                   color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
                        if not args.headless:
                            cv.imshow('vis', bgr)
                        last_frame_index = frame_index

                        # Record frame?
                        if args.record_video:
                            video_out_queue.put_nowait(frame_index)

                        # Quit?
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            return

                        # Print timings
                        if frame_index % 60 == 0:
                            latency = _dtime('before_frame_read', 'after_visualization')
                            processing = _dtime('after_frame_read', 'after_visualization')
                            timing_string = ', '.join([
                                _dstr('read', 'before_frame_read', 'after_frame_read'),
                                _dstr('preproc', 'after_frame_read', 'after_preprocessing'),
                                'infer: %dms' % int(frame['time']['inference']),
                                'vis: %dms' % int(frame['time']['visualization']),
                                'proc: %dms' % processing,
                                'latency: %dms' % latency,
                            ])
                            print('%08d [%s] %s' % (frame_index, fps_str, timing_string))

        visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
        visualize_thread.daemon = True
        visualize_thread.start()

        # Do inference forever
        infer = model.inference_generator()
        while True:
            output = next(infer)
            for frame_index in np.unique(output['frame_index']):
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                if 'inference' in frame['time']:
                    frame['time']['inference'] += output['inference_time']
                else:
                    frame['time']['inference'] = output['inference_time']
            inferred_stuff_queue.put_nowait(output)

            if not visualize_thread.isAlive():
                break

            if not data_source._open:
                break