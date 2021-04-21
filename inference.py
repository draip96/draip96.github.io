import json

from commons import get_model, transform_image




imagenet_class_index = json.load(open('_static/imagenet_class_index.json'))

def get_prediction(model_name, image_bytes):

    model = get_model(model_name)
    
#    try:
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)

#    except Exception:
#        return 0, 'error'

    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
