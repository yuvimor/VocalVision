# VocalVision

**A Smart Vision Application for Assistive Technology**

---

## About the Project

This project is part of a broader initiative to integrate advanced AI capabilities into smart glasses. The ultimate goal is to empower these glasses with the ability to:

1. Capture images using an integrated camera.
2. Send the images to an AI model.
3. Use audio prompts to provide detailed and contextually aware descriptions of the images.

The application, **VocalVision**, serves as a live demonstration of this concept.

## Features

### Key Components

- **Audio Transcription**: Users can input audio in either English or Hindi. The audio is transcribed using:
  - OpenAI Whisper model (for English)
  - Wav2Vec2 model (for Hindi)

- **Translation**: Transcriptions in Hindi are translated to English using the Helsinki-NLP translation pipeline.

- **Visual-Language Model Tasks**: The transcribed and/or translated text is used as a prompt for the LLaVA model to process input images. Supported tasks include:
  - Image description
  - Arithmetic computation based on image content

- **User Interface**: A Gradio-based interface enhances usability, enabling streamlined interaction.

---

## Technology Stack

The project leverages the following tools and libraries:

- **[Gradio](https://gradio.app/)**: Simplified interface creation for machine learning applications.
- **[OpenAI Whisper](https://github.com/openai/whisper)**: Cutting-edge model for English audio transcription.
- **[Wav2Vec2](https://huggingface.co/models)**: Pretrained model for Hindi audio transcription.
- **[Helsinki-NLP](https://huggingface.co/Helsinki-NLP)**: Neural machine translation pipeline.
- **[LLaVA](https://github.com/haotian-liu/LLaVA)**: Multimodal model for visual-language tasks.
- **[PIL](https://pillow.readthedocs.io/)**: Image processing library.
- **[librosa](https://librosa.org/)**: Audio processing library.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vocalvision.git
   cd vocalvision
   ```

2. Install the required libraries:
   ```bash
   pip install gradio
   pip install -q -U transformers==4.37.2 bitsandbytes==0.41.3 accelerate==0.25.0
   pip install openai-whisper
   ```

---

## Usage

### Scenario: Describe an Image Through an Audio Prompt

1. **Run the Application**:
   - Open `gradio_app.ipynb` in Jupyter Notebook.
   - Run all cells sequentially. At the end, a Gradio interface will be deployed.

2. **Upload Inputs on the Gradio Interface**:
   - Provide an audio file (English or Hindi).
   - Provide an image file.
   - Specify the language of the audio ("Hindi" or "English").

3. **Output**:
   - The application will transcribe and/or translate the audio.
   - It will then process the image and provide a detailed description based on the audio prompt.

---

## Workflow

### Flowchart and Demo

To understand the working process, refer to the following resources included in the repository:

- **Flowchart**: Visualizes the step-by-step workflow.
- **Demo Video**: Demonstrates the live execution of the application.

---

## Future Scope

- **Smart Glasses Integration**: Embed this application into hardware to enable real-time processing.
- **Enhanced Multimodal Capabilities**: Expand the range of tasks beyond image description.
- **Language Expansion**: Support for more languages.

---

## Acknowledgements

Special thanks to the developers and maintainers of the open-source models and libraries used in this project.

---
