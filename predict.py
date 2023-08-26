# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
import os
from demo import load_checkpoints, make_animation, find_best_frame
from skimage import img_as_ubyte
import urllib.request
from PIL import Image
#import cv2
import urllib
import boto3
from botocore.client import Config


from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # Move checkpoint folder /checkpoints/ to /src/

        # remove /src/checkpoints/ if it exists
        if os.path.exists("/src/checkpoints/"):
            os.system("rm -rf /src/checkpoints/")

        os.system("mv /checkpoints/ /src/")
        # Clean workspace
        



    def predict(
        self,
        source_image: Path = Path(description="Source Image", default=None),
        driving_video: Path = Path(description="Driving Video", default=None),
        s3_bucket: str = Input(description="S3 Bucket", default=None),
        s3_region: str = Input(description="S3 Region", default=None),
        s3_access_key: str = Input(description="S3 Access Key", default=None),
        s3_secret_key: str = Input(description="S3 Secret Key", default=None),
        s3_endpoint_url: str = Input(description="S3 Endpoint URL", default=None),
        s3_use_ssl: bool = Input(description="S3 Use SSL", default=True),
        s3_path: str = Input(description="S3 Path", default=None)
    ) -> str:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        source_image_path = str(source_image)
        driving_video_path = str(driving_video)

        reader = imageio.get_reader(driving_video_path)
        fps = reader.get_meta_data()["fps"]
        print(fps)

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
        

        # Load checkpoints
        config_path = "config/vox-256.yaml"
        checkpoint_path = "checkpoints/vox.pth.tar"
        device = "cuda"  # or "cuda"
        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path, checkpoint_path, device)
        
        print("Loaded checkpoints", flush=True)

        # Generate animation predictions more than 1 second long
        #predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = "relative")
        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = "relative")

        print("Generated animation", flush=True)
        # Save animation
        output_video_path = "output.mp4"
        imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        print("Saved animation", flush=True)

        # extract audio from driving video using imageio ffmpeg
        
        #detect audio codec
        cmd = "ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 " + driving_video_path
        audio_codec = os.popen(cmd).read()
        audio_codec = audio_codec.strip()
        print(audio_codec)

        audio_path = "audio." + audio_codec

        cmd = "ffmpeg -i " + driving_video_path + " -vn -acodec copy " + audio_path
        os.system(cmd)

        print("Extracted audio", flush=True)

        # merge audio and video using ffmpeg 
        merged_video_path = "merged.mp4"
        #imageio_ffmpeg.write_frames(output_video_path, predictions, fps=fps, audio_path=audio_path)
        # input is output_video_path and audio_path and output is merged_video_path
        cmd = "ffmpeg -i " + output_video_path + " -i " + audio_path + " -c:v copy -c:a aac -strict experimental " + merged_video_path
        os.system(cmd)

        print("Merged audio and video", flush=True)

    
        try:
            ####################################### S3 Upload Video #######################################
            print("Uploading Video to S3...") 
            # use pathstyle because minio does not support virtual host style
            s3 = boto3.client(
                's3',
                region_name=s3_region,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                endpoint_url="https://" + s3_endpoint_url,
                use_ssl=s3_use_ssl,
                config=boto3.session.Config(signature_version='s3v4'),
                verify=False
            )
            
            import datetime
            # create a timespamp for video name ex 5345446354.mp4
            timestamp = datetime.datetime.now().timestamp()

            s3.upload_file(
                Filename=f'{merged_video_path}',
                Bucket=s3_bucket,
                Key=f'{s3_path}/{timestamp}.mp4',
            )

            print("Done Uploading Video to S3")

            # get the url of the uploaded video for 7 days
            url = s3.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': s3_bucket,
                    'Key': f'{s3_path}/{timestamp}.mp4'
                },
                ExpiresIn=604800
            )

            return url
        except(Exception):
            print("Error: Could not create video")
            raise Exception("Error: Could not create video")
        finally:
            # Clean up downloaded files
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(merged_video_path):
                os.remove(merged_video_path)


