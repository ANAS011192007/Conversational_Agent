import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
from transformers import pipeline
import librosa
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, AutoModelForCTC,AutoFeatureExtractor
processor = Wav2Vec2Processor.from_pretrained("BangAsr007/repo")
model = AutoModelForCTC.from_pretrained("BangAsr007/repo")
model_name="BangAsr007/repo"
tokenizer="BangAsr007/repo"
feature_extractor = AutoFeatureExtractor.from_pretrained("BangAsr007/repo")
class Automatic_Speech_Recognition:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        # self.processor = Wav2Vec2Processor.from_pretrained("D:\asr\Latest_Conversational Agent\BangAsr007\repo")
        # self.model = AutoModelForCTC.from_pretrained("D:\asr\Latest_Conversational Agent\BangAsr007\repo")

    # def Audio(self, audio_path):
    #     arrays = []
    #     array, sampling_rate = librosa.load(self.audio_path)
    #     # resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    #     # array = resampler(array)
    #     input_values = processor(array, sampling_rate=16_000).input_values[0]
    #     arrays.append({"input_values":input_values[0]})

    #     batch = processor.pad(
    #         arrays,
    #         return_tensors="pt",
    #     )

    #     return batch['input_values']
    #     # return input_values

    # def infer(self, values):
    #     audio=values
    #     with torch.no_grad():
    #         logits = model(input_values=audio).logits
    #         transcription = processor.batch_decode(logits=logits.cpu().numpy()).text[0]
    #         return transcription\
    def Audio(self, audio_path):
        asr = pipeline("automatic-speech-recognition", model=model_name, device=0,tokenizer=tokenizer)
        feature_extractor = feature_extractor
        speech, sr = librosa.load(self.audio_path, sr=feature_extractor.sampling_rate)
        prediction = asr(
                    speech, chunk_length_s=112, stride_length_s=None
                )
        pred = prediction["text"]
        return pred