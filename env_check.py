import torch, numpy, os
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("torch path:", os.path.dirname(torch.__file__))
print("numpy:", numpy.__version__)
print("numpy path:", os.path.dirname(numpy.__file__))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
