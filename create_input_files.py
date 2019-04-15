import os

from image_captioning.utils import create_input_files

datadir = os.path.split(os.path.split(__file__)[0])[0]
output_folder = os.path.join(datadir, 'data/MSCOCO14/')

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path=os.path.join(datadir,  'data/MSCOCO14/dataset_coco.json'),
                       image_folder=os.path.join(datadir, 'data/MSCOCO14/'),
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=output_folder,
                       max_len=50)
