import cv2



def letterbox(
    image=None,
    new_shape=(640, 640),
    auto=False,
    scaleFill=False,
    scaleup=True,
    center=True,
    stride=32,
    labels=None
):
    labels = {}
    img = image
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = labels.pop("rect_shape", new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    if labels.get("ratio_pad"):
        labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

    return img


def create_warmup_file(save_path, np_array=None, string=False, string_list=None, img_path=None, dtype='float32'):
    # For Triton model warmup, typical model input is a np.array, string, or image
    # The warmup files must be in bytes, so use this function to create them.
    if hasattr(np_array, 'shape'):
        np_array = np_array.astype(dtype)
        np_array.tofile(save_path)
    elif isinstance(img_path, str):
        import cv2
        img = cv2.imread(img_path).astype(dtype)
        img.tofile(save_path)
    elif isinstance(string, str):
        from tritonclient.utils import serialize_byte_tensor
        serialized = serialize_byte_tensor(np.array([string.encode("utf-8")], dtype=object))
        with open(save_path, "wb") as f:
            f.write(serialized.item())
    elif isinstance(string_list, list):
        from tritonclient.utils import serialize_byte_tensor
        output = []
        for item in string_list:
            output.append(item.encode("utf-8"))
        
        serialized = serialize_byte_tensor(np.array(output, dtype=object))
        with open(save_path, "wb") as f:
            f.write(serialized.item())
    else:
        print("Invalid input. Input a numpy array, string, string list, or image path.")



if __name__ == "__main__":
    import numpy as np

    # preprocess
    create_warmup_file(
        "./preprocess/warmup/INPUT_0",
        img_path="../app/images/pic_228.png",
        dtype='int8'
    )

    # object_detection
    img = cv2.imread("../app/images/pic_228.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox(
        img,
        (640, 640)
    )
    img = np.stack(img)
    img = img[..., ::-1].transpose((2, 0, 1))
    img = np.ascontiguousarray(img).astype('float16')
    img /= 255
    create_warmup_file(
        "./object_detection/warmup/images",
        np_array=img[None],
        dtype='float16'
    )

    # postprocess
    create_warmup_file(
        "./postprocess/warmup/INPUT_1",
        np_array=np.array([1280, 720], dtype='int16')
    )