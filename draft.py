# import sys
# print("Python executable:", sys.executable)
# print("Python path:", sys.path)

# import soundata

# dataset = soundata.initialize('urbansound8k')
# dataset.download()  # download the dataset
# dataset.validate()  # validate that all the expected files are there

# example_clip = dataset.choice_clip()  # choose a random example clip
# print(example_clip)  # see the available data

import shutil

# After finding the correct source path from above
source_path = r"tmp\sound_datasets\urbansound8k"  # Use r prefix for raw string
dest_path = r"C:/Users/leona/pyproj/main_pro/TESI/simple_audio_netwrok/datasets"

# Move the files
shutil.move(source_path, dest_path)