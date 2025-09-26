import tensorflow as tf
from tensorflow.python.client import device_lib

# Get a list of all physical GPU devices visible to TensorFlow
gpu_devices = tf.config.list_physical_devices('GPU')

print(f"Number of GPUs Available: {len(gpu_devices)}")

if gpu_devices:
    print("GPU is available and TensorFlow is using it!")
    print("Details:", gpu_devices)
else:
    print("GPU not found. TensorFlow is using the CPU.")

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())