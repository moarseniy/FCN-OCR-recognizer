import torch
import pprint
import sys

path = sys.argv[1]

obj = torch.load(path, map_location="cpu")

print("TYPE:", type(obj))
print()

pprint.pp(obj)