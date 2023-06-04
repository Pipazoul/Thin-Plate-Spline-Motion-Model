from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
import os
from demo import load_checkpoints, make_animation, find_best_frame
from skimage import img_as_ubyte
import urllib.request
from PIL import Image
#import cv2
import urllib



warnings.filterwarnings("ignore")

app = FastAPI()

class AnimationRequest(BaseModel):
    source_image: str
    driving_video: str
    dataset_name: str

@app.post("/predictions")
async def generate_animation(request: AnimationRequest):
    try:
        source_image_path = "source_image.jpg"
        driving_video_path = "driving_video.mp4"

        # Download the source image
        urllib.request.urlretrieve(request.source_image, source_image_path)
        print("Downloaded source image", flush=True)

        # save base64 encoded image to file
        #image = Image.open(source_image_path)
        #image.save(source_image_path)

        print("Wrote source image", flush=True)
        

        
        print("Downloaded source image", flush=True)

        # Download the driving video
        urllib.request.urlretrieve(request.driving_video, driving_video_path)

        # save base64 encoded video to file
        #video_file = open(driving_video_path, "wb")
        #video_file.write(request.driving_video)


        reader = imageio.get_reader(driving_video_path)
        fps = reader.get_meta_data()["fps"]
        print(fps)

        print("Wrote driving video", flush=True)
        
        # Load checkpoints
        config_path = "config/vox-256.yaml"
        checkpoint_path = "checkpoints/vox.pth.tar"
        device = "cuda"  # or "cuda"
        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path, checkpoint_path, device)
        
        print("Loaded checkpoints", flush=True)

        # Load and resize source image
        source_image = imageio.imread(source_image_path)
        pixel = 256
        source_image = resize(source_image, (pixel, pixel))[..., :3]

        print("Loaded and resized source image", flush=True)
        # Load and resize driving video frames all frames
        driving_video = []
        # resize video to 256x256 
        for im in reader:
            driving_video.append(im)
        
        
        driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]
        reader.close()

        

        print("Resized driving video frames", flush=True)
        
        # Generate animation predictions more than 1 second long
        #predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = "relative")
        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = "relative")

        print("Generated animation", flush=True)
        # Save animation
        output_video_path = "output.mp4"
        imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        print("Saved animation", flush=True)
        # Return animation base64 encoded
        # Return animation as base64 encoded
        with open(output_video_path, "rb") as video_file:
            video_bytes = video_file.read()
        import base64
        # also add string data:video/mp4;base64
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")

        encoded_video = "data:video/mp4;base64," + encoded_video

        return {"animation": encoded_video}
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    '''finally:
        # Clean up downloaded files
        if os.path.exists(source_image_path):
            os.remove(source_image_path)
        if os.path.exists(driving_video_path):
            os.remove(driving_video_path)
        #if os.path.exists(output_video_path):
        #    os.remove(output_video_path)
        '''
