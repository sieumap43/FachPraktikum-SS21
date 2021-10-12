#!/bin/bash

################################################################
# Prepare dataset
################################################################
cd ../../GazeHub
wget -c http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz -O - | tar -xz
cd MPIIGaze_3D/
python data_processing_mpii.py
python convert_to_gaze_redirection.py
cp -r mpiigaze ../../gaze_redirection/dataset # to be augmented by the GR
cp -r mpiigaze ../../GazeML/datasets # to train the GE
cd ../../GazeML/my_src
python convert_img2h5.py --src_dir datasets/mpiigaze --dst_file datasets/mpiigaze.h5


################################################################
# Train Gaze Redirector
################################################################
# Train on Columbia
cd ../../gaze_redirection
mv src/data_loader_orig.py sr/data_loader.py
python main.py --mode train --data_path ./dataset/all --log_dir ./log/ --batch_size 32 --vgg_path ./vgg_16.ckpt
mv src/data_loader.py src/data_loader_orig

# Augment MPIIGaze
mv src/data_loader_exp_3.py src/data_loader.py
python main.py --mode eval --data_path ./dataset/mpiigaze --log_dir ./log/ --batch_size 32
mv src/data_loader.py src/data_loader_exp_3.py
mv log/eval/genes ../GazeML/datasets/mpiigaze_gen

################################################################
# Train Gaze Estimator
################################################################
# Prepare data
cd ../GazeML/datasets
mkdir mpiigaze_aug
cp mpiigaze/* mpiigaze_aug/ # copy original data
cp mpiigaze_gen/* mpiigaze_aug # copy generated data
cd ../my_src
python convert_img2h5.py --src_dir mpiigaze_aug --test_dir mpiigaze --dst_file MPIIGaze_aug.h5

# Train
python dpg_train_exp_3.py