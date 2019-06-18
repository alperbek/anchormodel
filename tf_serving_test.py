import PIL.Image
import numpy
import requests
from pprint import pprint
import time

image = PIL.Image.open("img.jpg")  # Change dog.jpg with your image
image_np = numpy.array(image)


payload = {"instances": [image_np.tolist()]}
start = time.perf_counter()
res = requests.post(
    "http://localhost:8080/v1/models/default:predict", json=payload)
print(f"Took {time.perf_counter()-start:.2f}s")
pprint(res.json())
