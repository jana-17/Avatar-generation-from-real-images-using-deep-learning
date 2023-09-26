# Avatar-generation-from-real-images-using-deep-learning
# InstructPix2Pix Cartoonization with Stable Diffusion

This Python script uses the `StableDiffusionInstructPix2PixPipeline` model to cartoonize an input image based on a given text prompt. It leverages the Tim Brooks' "instruct-pix2pix" model, which translates text instructions into images using the Pix2Pix architecture with stable diffusion.

## Prerequisites

Before you can run this script, ensure you have the following prerequisites installed:

- Python 3.6+
- PyTorch
- PIL (Python Imaging Library)
- Matplotlib

You can install the required Python packages using pip:


## Usage

1. Clone or download this repository to your local machine.

2. Replace `"path/to/your/image.jpg"` in the `IMAGE_PATH` variable with the path to your input image.

3. Customize the `PROMPT` variable with the text instruction you want to use for cartoonization. For example:

4. Open a terminal and navigate to the directory containing the script.

5. Run the script using the following command:

6. The cartoonized image will be displayed using Matplotlib.

## Model

The script uses the "instruct-pix2pix" model by Tim Brooks, which has been fine-tuned for text-to-image translation.

## Configuration

- You can adjust the `num_inference_steps` and `image_guidance_scale` parameters in the `pipe` object to control the cartoonization process.

## License

This script is provided under the MIT License. Refer to the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Tim Brooks](https://github.com/timbrooks/instruct-pix2pix) for the "instruct-pix2pix" model.
- [OpenAI](https://openai.com) for the GPT-3.5 model that powers this assistant.
