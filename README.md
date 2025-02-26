# SigLIP2 LitServe

[![Open In Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/sitammeur/studios/deploy-paligemma-2-mix-vision-language-model)

[SigLIP 2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) extends the pretraining objective of SigLIP with prior, independently developed techniques into a unified recipe, for improved semantic understanding, localization, and dense features. This project shows how to create a self-hosted, private API that deploys [SigLIP 2 model variant](https://huggingface.co/google/siglip2-so400m-patch14-384) with LitServe, an easy-to-use, flexible serving engine for AI models built on FastAPI.

## Project Structure

The project is structured as follows:

- `server.py`: The file containing the main code for the web server.
- `client.py`: The file containing the code for client-side requests.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder containing screenshots for working on the application.
- `images`: The folder containing images for testing purposes.
- `.gitignore`: The file containing the list of files and directories to be ignored by Git.

## Tech Stack

- Python (for the programming language)
- PyTorch (for the deep learning framework)
- Hugging Face Transformers Library (for the model)
- LitServe (for the serving engine)

## Getting Started

To get started with this project, follow the steps below:

1. Run the server: `python server.py`
2. Upon running the server successfully, you will see uvicorn running on port 8000.
3. Open a new terminal window.
4. Run the client: `python client.py`

Now, you can see the model's output based on the input request. The model classifies the image labels and returns the output.

## Usage

The project can be used to serve the SigLIP 2 family of models using LitServe. It allows you to input an image and candidate labels to classify the image based on the labels provided. The model can be used for various tasks, suggesting potential use cases such as zero-shot image classification and image-text retrieval or as a vision encoder for VLMs (and other vision tasks).

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Contact

If you have any questions or suggestions about the project, feel free to contact me on my GitHub profile.

Happy coding! 🚀
