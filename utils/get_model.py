import timm
from model.vitfs import *


def get_model(model_name, image_size):

    if any(
        kw in model_name
        for kw in [
            "vit_tiny",
            "mobilevit",
            "vitfs"
        ]
    ) :
        model = timm.create_model(
            model_name,
            pretrained=False,
            img_size=image_size,
        )
        print(f"Created model {model_name} with img_size={image_size}")
    else:
        model = timm.create_model(model_name, pretrained=False)
        print(f"Created model {model_name} with img_size=None")
    
    return model