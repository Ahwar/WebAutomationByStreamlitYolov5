import pandas
import torch
import time


def load_model(model_name, pred_conf):
    """Load the yolov5 model using torch.hub"""
    s_t = time.time()
    # Model
    model = torch.hub.load("ultralytics/yolov5", "custom", model_name)  # force reload
    model.conf = pred_conf
    print("inference time: {:.2f} milliseconds".format((time.time() - s_t) * 1000))
    return model


def predict_captcha(model, img_path, input_max_size):
    """run inference on loaded model, format and print output"""
    success, pred = False, False

    s_t = time.time()
    # Inference
    results = model(img_path, size=input_max_size)

    # Results
    # list of all class names in prediction dataframe sorted by xmin
    result = results.pandas().xyxy[0].sort_values(by="xmin")["name"].to_list()
    result = [str(i) for i in result]
    pred = "".join(result)
    results.print()
    success = True if len(pred) == 6 else False
    print("inference time: {:.2f} milliseconds".format((time.time() - s_t) * 1000))
    return success, pred


# load model
model = load_model("models/best.pt", 0.25)
# make prediction
predicted, pred = predict_captcha(model, "24C2Q2.png", 320)
print("prediction: ", pred)

# results.print()
# results.save()
