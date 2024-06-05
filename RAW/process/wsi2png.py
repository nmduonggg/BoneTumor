import os
import numpy as np
import glob
from PIL import Image
import PIL
import platform

import argparse
import openslide
import sys

def _load_image(buf, size):
    '''buf must be a buffer.'''

    # Load entire buffer at once if possible
    MAX_PIXELS_PER_LOAD = (1 << 29) - 1
    # Otherwise, use chunks smaller than the maximum to reduce memory
    # requirements
    PIXELS_PER_LOAD = 1 << 26

    def do_load(buf, size):
        '''buf can be a string, but should be a ctypes buffer to avoid an
        extra copy in the caller.'''
        # First reorder the bytes in a pixel from native-endian aRGB to
        # big-endian RGBa to work around limitations in RGBa loader
        rawmode = (sys.byteorder == 'little') and 'BGRA' or 'ARGB'
        buf = PIL.Image.frombuffer('RGBA', size, buf, 'raw', rawmode, 0, 1)
        # Image.tobytes() is named tostring() in Pillow 1.x and PIL
        buf = (getattr(buf, 'tobytes', None) or buf.tostring)()
        # Now load the image as RGBA, undoing premultiplication
        return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBa', 0, 1)

    # Fast path for small buffers
    w, h = size
    if w * h <= MAX_PIXELS_PER_LOAD:
        return do_load(buf, size)

    # Load in chunks to avoid OverflowError in PIL.Image.frombuffer()
    # https://github.com/python-pillow/Pillow/issues/1475
    if w > PIXELS_PER_LOAD:
        # We could support this, but it seems like overkill
        raise ValueError('Width %d is too large (maximum %d)' %
                         (w, PIXELS_PER_LOAD))
    rows_per_load = PIXELS_PER_LOAD // w
    img = PIL.Image.new('RGBA', (w, h))
    for y in range(0, h, rows_per_load):
        rows = min(h - y, rows_per_load)
        if sys.version[0] == '2':
            chunk = buffer(buf, 4 * y * w, 4 * rows * w)
        else:
            # PIL.Image.frombuffer() won't take a memoryview or
            # bytearray, so we can't avoid copying
            chunk = memoryview(buf)[y * w:(y + rows) * w].tobytes()
        img.paste(do_load(chunk, (w, rows)), (0, y))
    return img

openslide.lowlevel._load_image = _load_image

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", help="file path of WSI (.mrxs)")
    parser.add_argument("--output-path", help="output file for saving")
    parser.add_argument("--level", 
                        help="downscale level | The level specified downsamples the images as follows: [1, 2, 4, 8, 16, 32, 64, 128, 256],\
                            where the level is the index in the list. E.g. level = 0 is the original image size, level = 3 downsamples the image by a factor of 4.", 
                            type=int, default=0)
    parser.add_argument("--overwrite", help="override if the output path existed", type=bool, default=False)
    
    return parser.parse_args()


class PngExtractor:
    """
    This Object extracts a whole mrxs file to a png format.

    :param file_path: string
        path to the mrxs single file or folder of files.
    :param output_path: string
        path to the output folder. The output format is the same name as the mrxs file,
        with an appendix if multiple patches are extracted.
    :param staining: Staining identifier, that would be specified right before .mrxs (e.g. CD8) (optional)
    :param level: int (optional)
        Level of the mrxs file that should be used for the conversion (default is 0).
    :param overwrite: overides exisiting extracted patches (default is False)

    """

    def __init__(self, file_path: str, output_path: str, staining: str = '', level: int = 0, overwrite: bool = False):
        # initiate the mandatory elements
        self.file_path = file_path
        self.output_path = output_path
        # instantiate optional parameters
        self.staining = staining
        self.level = level
        self.overwrite = overwrite

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, output_path):
        # make the output folder if it does not exist
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        self._output_path = output_path

    @property
    def wsi_files(self):
        if os.path.isfile(self.file_path):
            files = [self.file_path]
        else:
            files = glob.glob(os.path.join(self.file_path, f'*{self.staining}.mrxs'))
            files.extend(glob.glob(os.path.join(self.file_path, f'*{self.staining}.ndpi')))
        return files

    @property
    def files_to_process(self):
        # we only have one file to process
        if len(self.wsi_files) == 1:
            filename = os.path.splitext(os.path.basename(self.file_path))[0]
            output_file_name = os.path.join(self.output_path, f'{filename}-level{self.level}')
            # skip existing files, if overwrite = False
            if not self.overwrite and os.path.isfile(f'{output_file_name}.png'):
                print(f'File {output_file_name} already exists. Output saving is skipped. To overwrite add --overwrite.')
            else:
                return [(output_file_name, self.file_path)]

        # we have multiple files to process
        else:
            files_to_process = []
            for wsi_path in self.wsi_files:
                filename = os.path.splitext(os.path.basename(wsi_path))[0]
                output_file_name = os.path.join(self.output_path, f'{filename}-level{self.level}')
                # skip existing files, if overwrite = False
                if not self.overwrite and os.path.isfile(f'{output_file_name}.png'):
                    print(f'File {output_file_name} already exists. Output saving is skipped. To overwrite add --overwrite.')
                    continue
                files_to_process.append((output_file_name, wsi_path))

            return files_to_process

    def process_files(self):
        # process the full image
        if os.path.isfile(self.file_path) or os.path.isdir(self.file_path):
            for output_file_path, wsi_path in self.files_to_process:
                assert os.path.isfile(wsi_path)
                wsi_img = openslide.open_slide(wsi_path)
                # extract the patch
                png = self.extract_crop(wsi_img)
                # save the image
                print(f'Saving image {output_file_path}.png')
                Image.fromarray(png[:, :, :3]).save(f'{output_file_path}.png')

        else:
            # Something went wrong
            print('mrxs paths are invalid.')

    def extract_crop(self, wsi_img, coord=None):
        # coordinates have to be in format [tl, tr, br, bl] ((0,0) is top-left)
        # crop the region of interest from the mrxs file on the specified level
        # get the level and the dimensions
        # id_level = np.argmax(np.array(wsi_img.level_downsamples) == self.level)
        id_level = self.level
        dims = wsi_img.level_dimensions[id_level]

        if coord:
            top_left_coord = [int(i) for i in coord[0]]
            width = coord[2][0] - coord[0][0]
            height = coord[2][1] - coord[0][1]
            size = (int(width), int(height))
            # make sure the dimension we want to crop are within the image dimensions
            assert coord[3][0] <= dims[0] and coord[3][1] <= dims[1]
        else:
            # if no coordinates are specified, the whole image is exported
            top_left_coord = [0, 0]
            size = dims

        # extract the region of interest
        img = wsi_img.read_region(location=top_left_coord, level=id_level, size=size)

        # Convert to img
        img = np.array(img)
        img[img[:, :, 3] != 255] = 255
        return img


def extract_whole_slide(file_path: str, output_path: str, staining: str = '', level: int = 0, overwrite: bool = False):
    png_extractor = PngExtractor(file_path=file_path, output_path=output_path, staining=staining,
                                 level=level, overwrite=overwrite)

    # process the files
    png_extractor.process_files()


if __name__ == '__main__':
    args = vars(create_args())
    extract_whole_slide(**args)