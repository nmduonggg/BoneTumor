import os
from openslide import open_slide
import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tqdm

def check_blank(img_np):
    image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edge = cv2.Laplacian(image, cv2.CV_64F)
    return edge.mean()

if __name__=='__main__':
    # load the slide file into an object
    slide = open_slide("/home/admin/duongnguyen/BoneTumor/RAW/REAL_WSIs/bulk/images/2024/08/02/03/4cy4dr/slide-2024-07-25T16-20-48-R1-S15.mrxs")
    tile_dir = '/home/admin/duongnguyen/BoneTumor/RAW/REAL_WSIs/tiles'
    os.makedirs(tile_dir, exist_ok=True)
    
    
    slide_props = slide.properties
    print(slide_props.keys())

    print("Vendor is: ", slide_props['openslide.vendor'])
    print("Pixel size of X in um is: ", slide_props['openslide.mpp-x'])
    print("Pixel size of Y in um is: ", slide_props['openslide.mpp-y'])

    # Objective used to capture the image -> resolution
    objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    print("The objective power is: ", objective)

    # get slide dimensions for the level 0 - max resolution level
    slide_dims = slide.dimensions
    print(slide_dims)

    # Get a thumbnail of the image and visualize
    # thumbnail is a resized image
    slide_thumb_600 = slide.get_thumbnail(size=(512, 512))
    slide_thumb_600_np = np.array(slide_thumb_600)
    plt.imsave('./openslide_image.jpg', slide_thumb_600_np)

    ### Resolution
    # level 0: native resolution
    # get slide dims at each level. Remember that whole slide images store information as pyramid at various levels
    dims = slide.level_dimensions
    num_levels = len(dims)

    print("Number of levels in this image are: ", num_levels)
    print("Dimensions of various levels in this image are: ", dims)

    # By how much are levels downsampled from the original
    factors = slide.level_downsamples
    print("Each level is downsampled by an amount of: ", factors)


    # Copy an image from a level
    slide_dim = dims[2]    # level 1: 20x -> 10x
    # Give pixel coordinates (top left pixel in the original large image)
    # Also give the level number 
    # Size of your output image
    # Remember that the output would be a RGBA not RGB

    level1_image = slide.read_region((0, 0), 2, slide_dim) # -> Pillow, RGBA
    print("Read image done")
    level1_image = level1_image.convert('RGB')
    level1_image_np = np.array(level1_image)
    plt.imsave('./x20_to_smaller.jpg', level1_image_np)

    from openslide.deepzoom import DeepZoomGenerator

    # Generate object for tiles using the DeepZoomGenerator -> Different levels again
    tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
    print("The number of levels in the tiles object are: ", tiles.level_count)

    print("The dimensions of data in each level are: ", tiles.level_dimensions)

    # Total number of tiles in the tiles object
    print("Total number of tiles = ", tiles.tile_count)

    # How many tiles at a specific level 
    level_num = 18
    print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])    # shape of level is rowxcolumn grid = (slide_row // tile_row) x (slide_col // slide_col)
    print("This means there are ", tiles.level_tiles[level_num][0] * tiles.level_tiles[level_num][1], " total tiles in this level")

    tile_dims = tiles.get_tile_dimensions(18, (0, 0))
    print(tile_dims)

    # Tile count at the highest resolution level (19)
    # Note that: last tile may not be exactly the tile size we desired

    # Extract tile
    single_tile = tiles.get_tile(18, (16, 27))  # level, address
    single_tile_RGB = single_tile.convert('RGB')

    cols, rows = tiles.level_tiles[18]

    # cut WSIs into tiles
    # cnt = 0
    # with open(os.path.join(tile_dir, '_size.txt'), 'w') as output:
    #     output.write(f"{rows} {cols} {slide_dim[1]} {slide_dim[0]}")
    # for row in tqdm.tqdm(range(rows), total=rows):
    #     for col in range(cols):
    #         tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
    #         # print("Now saving tile: ", tile_name)
    #         tmp_tile = tiles.get_tile(18, (col, row)).convert('RGB')
    #         tmp_tile_np = np.array(tmp_tile)
    #         # if check_blank(tmp_tile_np) < 5.0: continue
            
    #         np.save(tile_name + '.npy', tmp_tile_np)
            
    #         cnt += 1
    
    # print(f"Done extract {cnt} tiles")
            
