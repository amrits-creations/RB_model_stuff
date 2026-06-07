from .single_layer import SingleLayerPCN

MODELS = {
    "single": SingleLayerPCN,
}

# I dont know why I keep forgetting this one - this file is to ensure that: 1. the __init__ file is to ensure that this folder is treated like a python package, and 2. to have a central place to import all the models from, so that we can do "from models import SingleLayerPCN" instead of "from models.single_layer import SingleLayerPCN".
