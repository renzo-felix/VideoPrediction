import numpy as np
import cv2
import tempfile
import os
import pybullet as p

def create_checkerboard(size=256, squares=8):
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    square_size = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                texture[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = [255, 255, 255]
    return texture

def create_stripes(size=256, stripes=16):
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    stripe_width = size // stripes
    for i in range(stripes):
        if i % 2 == 0:
            texture[:, i*stripe_width:(i+1)*stripe_width] = [255, 255, 255]
    return texture

def load_texture(texture_type):
    if texture_type == "checkerboard":
        tex_data = create_checkerboard()
    elif texture_type == "stripes":
        tex_data = create_stripes()
    else:
        return -1

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    cv2.imwrite(temp_path, tex_data)
    tex_id = p.loadTexture(temp_path)

    try:
        os.remove(temp_path)
    except OSError:
        pass

    return tex_id
