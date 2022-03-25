import onnxruntime as nxrun
import numpy as np
import time


def mills():
    return int(round(time.time() * 1000))


## start inference session
sess = nxrun.InferenceSession("models\yolov3_captcha_20210624.onnx")

## input, output shape
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_shape = sess.get_inputs()[0].shape
output_shape = sess.get_outputs()[0].shape
print(
    "input layer name: {}\ninput layer shape: {}\n"
    "output layer name: {}\noutput layer shape: {}\n".format(
        input_name, input_shape, output_name, output_shape
    )
)

start_time = mills()
# dummy_input shape should be equal to input_shape
dummy_input = np.ones([1, 416, 416, 3], dtype=np.float32)
## run onnx model with onnx runtime python
result = sess.run(None, {input_name: dummy_input})

print("model single inference in milliSeconds on onnxruntime: ", mills() - start_time)
print("Output", result)
