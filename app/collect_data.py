import cv2
from nanocamera import Camera



def main(
    save_path,
    number_to_save=100,
    start_index=0,
    period=5,
    camera_index=0,
    camera_width=1280,
    camera_height=720,
    camera_fps=30
):
    camera = Camera(
        device_id=camera_index,
        flip=0,
        width=camera_width,
        height=camera_height,
        fps=camera_fps
	)

    frame_index = 0
    number_to_save = number_to_save + start_index
    take_picture = camera_fps * period

    while camera.isReady():
        frame = camera.read()

        if frame_index % take_picture == 0:
            cv2.imwrite(save_path + f"pic_{start_index}.png", frame)
            print(f"Saving image {start_index}/{number_to_save}")
            start_index += 1

            if start_index == number_to_save:
                break
        
        frame_index += 1
    
    print(f"Saved {number_to_save} picture to `{save_path}` successfully.")
    return



if __name__ == "__main__":
    main(
        save_path="../data/",
        number_to_save=100,
        start_index=236,
        period=5,
        camera_index=0,
        camera_width=1280,
        camera_height=720,
        camera_fps=30
    )