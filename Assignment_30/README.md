
# ERA V2 Multi-Modal Large Language Model (LLM)

Welcome to the ERA V2 Multi-Modal Large Language Model (LLM) project! This model is designed to handle various types of input data — text, images, and audio — to provide contextual and informative responses. Built upon Microsoft’s Phi-2 model and enhanced with fine-tuning on the LLaVA Instruct-150k dataset, this multi-modal LLM brings together the power of language and visual comprehension to respond to complex queries in diverse formats

### Features
Text Input: Traditional text-based question answering and text generation.

Image Input: Visual question answering, where the model can interpret and respond to image-based queries.

Audio Input: Speech-to-text functionality that transcribes audio files and uses the transcription as input for generating responses.

Fine-Tuned with LLaVA: Trained on the LLaVA Instruct-150k dataset, enhancing visual and conversational understanding.

### Setup
#### Prerequisites
Python 3.8+

Hugging Face Transformers

Gradio

Whisper for speech-to-text

Additional dependencies can be installed from requirements.txt

### Installation
```
git clone https://github.com/dusaurabh/Saurabh-ERA-V2-Master/tree/main
cd Assignment_30
pip install -r requirements.txt
```

### Usage
#### Running the Gradio App
```
python app.py
```

### Model Inference
The model accepts three input types:

1. Text: Enter your question in the textbox.
2. Image: Upload an image for visual question answering.
3. Audio: Upload an audio file for transcription and further processing.
   
Example usage in Gradio app:

1. Input Text: "What is a large language model?"
2. Upload Image: Picture of a landmark (to ask questions about the landmark)
3. Upload Audio: Recording of a question (the audio will be transcribed and processed)

### Some sample outputs

#### Audio Input

![Audio Input](outputs_images/audio_output.png)

#### Text Input

![Text Input](outputs_images/output_1.png)

#### Text + Image Input

![Text Input](outputs_images/image_output_1.png)

![Text Input](outputs_images/image_output_3.png)

![Text Input](outputs_images/image_output_4.png)


### Training
This project uses the LLaVA Instruct-150k dataset for multi-modal learning.

Dataset Preprocessing
Text and Image Tokenization: Tokenize questions, answers, and process images to embeddings.
Batch Processing: Process images, questions, and answers in batches, utilizing an A100 GPU for efficiency.  Due to budget constraint, i trained this model for 1 epoch only and due to this you can see the inaccurate results some of the time

For Training - I have used A100 90 GB server from paperspace and it took 24 hours for 1 epoch training

### Training logs

```
True
0 processed
10000 processed
20000 processed
30000 processed
40000 processed
50000 processed
60000 processed
70000 processed
80000 processed
90000 processed
100000 processed
110000 processed
120000 processed
130000 processed
140000 processed
150000 processed
Downloading images: 100%|██████████| 199770/199770 [00:00<00:00, 316668.19it/s]
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]

I  12485
Epoch 1/3, Average Loss: 2.4875
Model checkpoint saved at model_checkpoints/model_checkpoint_epoch_1.pt

```

After 1 epoch i stop the training due to increase cost of A100 GPU

### Future Work
Expand Dataset: Incorporate additional multi-modal datasets for broader knowledge and versatility.

Enhanced Visual Capabilities: Integrate advanced visual question answering datasets.
