import cv2
import glob
import numpy as np

# Not used during data generation ################################
# You will need to do the calculations yourself using the test data
MEAN = np.asarray([[[[0.485, 0.456, 0.406]]]], dtype=np.float32) # [1,1,1,3]
STD = np.asarray([[[[0.229, 0.224, 0.225]]]], dtype=np.float32) # [1,1,1,3]
# Not used during data generation ################################

files = glob.glob("/home/toby/WinDatasets/betrayal/calibration-events/*.png")
print(f"num files: {len(files)}")
img_datas = []
for idx, file in enumerate(files):
    bgr_img = cv2.imread(file)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, dsize=(320,576))
    extend_batch_size_img = resized_img[np.newaxis, :]
    normalized_img = extend_batch_size_img / 255 # 0.0 - 1.0
    floats = np.array(normalized_img).astype(np.float32)
    print(
        f'{str(idx+1).zfill(2)}. extend_batch_size_img.shape: {extend_batch_size_img.shape}'
    ) # [1,112,200,3]
    img_datas.append(floats)
    break

calib_data = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_data.shape}') # [10,112,200,3]

mean = np.mean(calib_data, axis=(0,1,2))
print(f"mean: {mean}")

std = np.std(calib_data, axis=(0,1,2))
print(f"std: {std}")


np.save(file='data/calibdata.npy', arr=calib_data)

loaded_data = np.load('data/calibdata.npy')
print(f'loaded_data.shape: {loaded_data.shape}') # [10,112,200,3]

"""
-cind INPUT_NAME NUMPY_FILE_PATH MEAN STD
int8_calib_data = (loaded_data - MEAN) / STD # -1.0 - 1.0

e.g. How to specify calibration data in CLI or Script respectively.
1. CLI
  -cind "pc_dep" "data/calibdata.npy" "[[[[0.485,0.456,0.406]]]]" "[[[[0.229,0.224,0.225]]]]"
  -cind "feat" "data/calibdata2.npy" "[[[[0.123,...,0.321]]]]" "[[[[0.112,...,0.451]]]]"

2. Script
  custom_input_op_name_np_data_path=[
      ["pc_dep", "data/calibdata.npy", [[[[0.485,0.456,0.406]]]], [[[[0.229,0.224,0.225]]]]],
      ["feat", "data/calibdata2.npy", [[[[0.123,...,0.321]]]], [[[[0.112,...,0.451]]]],
  ]
"""