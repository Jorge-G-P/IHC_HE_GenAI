import os

def replace_np_bool_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_data = file.read()

    file_data = file_data.replace('dtype=np.bool', 'dtype=bool')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(file_data)

# Path to the specific file
file_path = r'C:\Users\amaia\anaconda3\envs\IHC_HE_GenAI\lib\site-packages\skimage\morphology\_skeletonize.py'
replace_np_bool_in_file(file_path)



def replace_np_bool_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_data = file.read()

    file_data = file_data.replace('dtype=np.bool', 'dtype=bool')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(file_data)

# Path to the imgaug directory
imgaug_path = r'C:\Users\amaia\anaconda3\envs\IHC_HE_GenAI\lib\site-packages\imgaug'

# Recursively replace np.bool in all Python files in the imgaug directory
for root, _, files in os.walk(imgaug_path):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            replace_np_bool_in_file(file_path)



def replace_antialias_with_lanczos(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_data = file.read()

    file_data = file_data.replace('Image.ANTIALIAS', 'Image.Resampling.LANCZOS')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(file_data)

# Path to the tensorboardX summary file
file_path = r'C:\Users\amaia\anaconda3\envs\IHC_HE_GenAI\lib\site-packages\tensorboardX\summary.py'
replace_antialias_with_lanczos(file_path)