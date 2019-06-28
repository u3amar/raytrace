from PIL import Image
import numpy as np

if __name__ == '__main__':
    im_width = 200
    im_height = 100
    im_arr = np.zeros((im_height, im_width, 3), dtype=np.uint8)

    for j in range(im_height):
        for i in range(im_width):
            r = int(255.0 * float(i + 1) / im_width)
            g = int(255.0 * float(im_height - (j + 1)) / im_height)
            b = int(255.0 * .2)
            im_arr[j, i] = [r, g, b]

    im = Image.fromarray(im_arr, 'RGB')
    im.show()
