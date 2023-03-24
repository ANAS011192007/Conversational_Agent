import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import librosa
import os

class AudioSimilarityChecker:
    def __init__(self, audio_folder, threshold):
        self.audio_folder = audio_folder
        self.audio_list = []
        self.threshold = threshold
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

    def load_audio_files(self):
        try:
            # Iterate over all files in the audio folder
            for filename in os.listdir(self.audio_folder):
                if filename.endswith(".wav"):
                    file_path = os.path.join(self.audio_folder, filename)
                    y, sr = librosa.load(file_path)
                    self.audio_list.append(y)
        except Exception as e:
            print("error getting numpy array: ", e)

    def compare_audio(self, audio_path):
        # Load the audio file to be checked
        y, sr = librosa.load(audio_path)
        self.audio_list.append(y)

        # audio files are decoded on the fly
        inputs = self.feature_extractor(self.audio_list, padding=True, return_tensors="pt")
        embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

        # the resulting embeddings can be used for cosine similarity-based retrieval
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)

        similarities = []
        for i in range(len(self.audio_list)-1):
            similarity = cosine_sim(embeddings[-1], embeddings[i])
            similarities.append(similarity)

        # access the similarity scores using the list
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        print(
            f'The maximum similarity is {max_similarity} and it is between {self.audio_list[max_index]} and {audio_path}')

        if max_similarity < self.threshold:
           return False
        else:
            return True

# Usage example
# audio_folder = "audio_record"
# threshold = 0.7
# audio_path = "audio/audio.wav"

# audio_similarity_checker = AudioSimilarityChecker(audio_folder, threshold)
# audio_similarity_checker.load_audio_files()
# audio_similarity_checker.compare_audio(audio_path)
