try:
    from AudioSimilarityChecker import AudioSimilarityChecker
    from asr import Automatic_Speech_Recognition
    from flask import Flask, render_template, request, redirect, url_for
    import json
    import requests
    from flask_sqlalchemy import SQLAlchemy
    import cv2
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    import librosa
    import os
    import shutil
    print("importing dependencies successful")
except Exception as e:
    print("error dependencies:", e)
    exit()


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///asr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
db.init_app(app)

with app.app_context():
    db.create_all()


class ASR(db.Model):
    No = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(200), nullable=False)
    def __repr__(self) -> str:
        return f"{self.No} - {self.question}"

class Ques_Ans(db.Model):
    __tablename__ = 'Ques_Ans'
    ID = db.Column(db.Integer, primary_key=True)
    Question = db.Column(db.String(500), nullable=False)
    Answer = db.Column(db.String(1000), nullable=False)

    def __repr__(self) -> str:
        return f"{self.ID} - {self.Question} - {self.Answer}"

class Q_A(db.Model):
    __tablename__ = 'Q_A'
    No = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(200), nullable=False)
    answer = db.Column(db.String(200), nullable=False)

    def __repr__(self) -> str:
        return f"{self.No} - {self.Question} - {self.Answer}"


app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# user_identified = False


@app.route('/')
def index():
    last_ques=ASR.query.order_by(ASR.No.desc()).first()
    # all_ques=Ques_Ans.query.all()
    all_ques=Q_A.query.all()
    m_s_q=list()
    import requests

    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    for a in all_ques:
        output = query({
            "inputs": {
                "source_sentence": str(last_ques.question),
                "sentences": [
                    str(a.question)
                ]
            },
        })
        m_s_q.append(output)
    flat_list = []
    for sublist in m_s_q:
        for item in sublist:
            flat_list.append(item)
    max_value = max(flat_list)
    print(max_value)
    index = flat_list.index(max_value)
    # ans=Ques_Ans.query.filter_by(ID=index+1).first()
    ans=Q_A.query.filter_by(No=index+1).first()
    if max_value < 0.7:
        # ques = ASR.query.all()
        return render_template('index.html', ques=str(last_ques.question) , ans="Sorry, I don't know the answer to your question.")
    else:
        # ques = ASR.query.all()
        return render_template('index.html', ques=str(last_ques.question), ans=str(ans.answer))



@app.route('/upload', methods=['POST'])
def upload_file():
    audio = request.files['audio']
    video = request.files['video']
    if audio and video:
        # specify the location of the 'audio', and 'video' folder
        audio_folder = os.path.join(app.root_path, 'audio')
        video_folder = os.path.join(app.root_path, 'video')

        # if the 'audio' folder does not exist, create it
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)

        # if the 'video' folder does not exist, create it
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # save the audio file to the 'audio' folder
        audio_path = os.path.join(audio_folder, audio.filename)
        audio.save(audio_path)
        # print(audio_path)

        # save the video file to the 'video' folder
        video_path = os.path.join(video_folder, video.filename)
        video.save(video_path)
        # print(video_path)

        ''' 
        get frame from video

        '''
        # # Open the video using cv2
        # cap = cv2.VideoCapture(video_path)

        # # Create a folder to save the extracted frames
        # if not os.path.exists('image'):
        #     os.mkdir('image')

        # # Set the initial frame number
        # frame_number = 0

        # # Extract frames from the video and save them in the 'image' folder
        # while True:
        #     # Read the current frame
        #     ret, frame = cap.read()

        #     # Break the loop if the video has ended
        #     if not ret:
        #         break

        #     # Save the frame to the 'image' folder
        #     cv2.imwrite('image/frame_{}.jpg'.format(frame_number), frame)

        #     # Increment the frame number
        #     frame_number += 1

        # # Release the video capture
        # cap.release()
        # #

        # check user identification
        audio_similarity_checker = AudioSimilarityChecker(
            audio_folder="audio_record", threshold=0.7)
        audio_similarity_checker.load_audio_files()
        matchFound = audio_similarity_checker.compare_audio(audio_path)
        audio_record="D:/asr/Latest_Conversational Agent/audio_record/A1.wav"

        if not matchFound:
            shutil.copy(audio_path, audio_record)

        # Automatic Speech Recognition
        asr= Automatic_Speech_Recognition(
            audio_path=audio_path)
        str_text=asr.Audio(audio_path)
        # str_text=asr.infer(logits)
        # API_URL = "https://api-inference.huggingface.co/models/Anas00/abcd"
        # headers = {
        #     "Authorization": "Bearer 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"}

        # def query(filename):
        #     with open(filename, "rb") as f:
        #         data = f.read()
        #     response = requests.request(
        #         "POST", API_URL, headers=headers, data=data)
        #     return json.loads(response.content.decode("utf-8"))

        # text = query(audio_path)
        # str_text = str()
        # for key in text:
        #     str_text += str(text[key])
        # print(str_text)

        asr = ASR(question=str_text)
        db.session.add(asr)
        db.session.commit()

        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)