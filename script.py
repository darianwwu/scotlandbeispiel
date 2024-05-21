from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

controlnet = ControlNetModel.from_pretrained("mespinosami/controlearth", torch_dtype=torch.float16).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
    safety_checker = None, requires_safety_checker = False
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

control_image_1 = Image.open('./image1.png').convert("RGB")
control_image_2 = Image.open('./image2.png').convert("RGB")
prompt = "convert this openstreetmap into its satellite view"

num_images = 5
for i in range(num_images):
    image = pipe(prompt, num_inference_steps=50, image=control_image_1).images[0].save(f'img1-generated-{i}.png')
for i in range(num_images):
    image = pipe(prompt, num_inference_steps=50, image=control_image_2).images[0].save(f'img2-generated-{i}.png')