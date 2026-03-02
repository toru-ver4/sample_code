import numpy as np
from OpenImageIO import (
    ImageSpec,
    ImageOutput,
    UINT16
)

width = 1920
height = 1080
output_filename = "./P3D65_ST2084_with_CICP.png"

# create ramp image
line = np.linspace(0.0, 1.0, width, dtype=np.float32)
img = line.reshape(1, -1, 1).repeat(3, axis=2).repeat(height, axis=0)

# output png via oiio
output = ImageOutput.create(filename=output_filename)
yres, xres, channels = img.shape
image_spec = ImageSpec(xres, yres, channels, UINT16)
image_spec.attribute("CICP", "int[4]", [12, 16, 0, 1])
output.open(filename=output_filename, spec=image_spec)
output.write_image(img)
output.close()
