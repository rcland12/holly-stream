import os
import cv2


def check_save_path(save_path: str, start_index: int) -> int:
    """
    Function to check if save path exists, and create the directory if not.
    It also will check if the start_index is correct, else will correct it.

    Args:
        save_path (str): Path where images will be saved to.
        start_index (int): Index the image numbering starts at.

    Returns:
        start_index (int): Unless there are already images present, it will return the start_index that was input.
    """
    if not os.path.isdir(save_path):
        os.system(f'mkdir {save_path}')

    else:
        files = os.listdir(save_path)

        if files:
            numbers = [
                int(file.split("_")[1].split(".")[0]) if "png" in file else None
                for file in files
            ]
            max_number = max(list(filter(None, numbers)))

            if max_number >= start_index:
                user_input = input(
                    f"""\
                    \n  Detected a maximum of {max_number} images already saved at {save_path}.\
                    \n  Do you want to start the indexing at {max_number + 1}? (y/n)\
                    \n"""
                )
                match user_input:
                    case "y":
                        start_index = max_number + 1
                    case "n":
                        pass
                    case _ :
                        print("Invalid user input. Enter 'y' or 'n'.")
                        exit(1)

    return start_index


def main(
    save_path: str,
    number_to_save: int = 100,
    start_index: int = 0,
    period: int = 5,
    camera_index: int = 0,
    camera_width: int = 1280,
    camera_height: int = 720
) -> None:
    """
    This function will take pictures from your camera source every x seconds to collect data for training.

    Args:
        save_path (str): Path where images will be saved. Named as pic_0.png, pic_1.png, ...
        number_to_save (int): Number of images to collect.
        start_index (int): Image number to start on. Useful if you have already collected 50 images, you can set start_index to 51.
        period (int): Save image every period, in seconds.
        camera_index (int): Index of camera device. Most linux devices are found at /dev/video0, hence index 0.
        camera_width (int): Width of input image.
        camera_height (int): Height of input image.
        camera_fps (int): Frames per second to capture with camera device.

    Returns:
        None
    """
    start_index = check_save_path(save_path, start_index)

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    camera_fps = int(camera.get(cv2.CAP_PROP_FPS))

    frame_index = 0
    number_to_save = number_to_save + start_index
    take_picture = camera_fps * period

    try:
        while camera.isOpened():
            ret, frame = camera.read()

            if not ret:
                print("Frame failed to load...")
                break

            if frame_index % take_picture == 0:
                cv2.imwrite(save_path + f"pic_{start_index}.png", frame)
                print(f"Saving image {start_index}/{number_to_save}")
                start_index += 1

                if start_index == number_to_save:
                    break

            frame_index += 1

        print(f"Saved {number_to_save} picture to `{save_path}` successfully.")
    
    finally:
        camera.release()
    
    return


if __name__ == "__main__":
    main(
        save_path="../data/",
        number_to_save=200,
        start_index=0,
        period=5,
        camera_index=0,
        camera_width=1280,
        camera_height=720
    )
