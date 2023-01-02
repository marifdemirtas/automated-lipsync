import os
import chunk
from urllib import response
from flask import Flask, render_template, Response, jsonify, request, g, make_response, session
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import platform

import subprocess
import cv2
import numpy as np
import torch
import time

from media import VideoCamera, generate_audio, generate_video, times
from config import Config

CAMERAS = {}

app = Flask(__name__)
app.config.from_object(Config)


def allowed_file(filename, type):
    if filename == '':
        return False
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'][type]


@app.route('/', methods=['GET', 'POST'])
def index():
    session.clear()
    if request.method == 'POST':
        visual_choice = request.form['source']

        if visual_choice == 'webcam':
            return redirect(url_for('webcam'), code=307)

        # check if the post request has the file part
        if 'audio' not in request.files or request.files['audio'].filename == '':
            flash('Sound source not uploaded')
            return redirect(url_for('index'))
        
        if visual_choice not in request.files or request.files[visual_choice].filename == '':
            flash('Visual source not selected')
            return redirect(url_for('index'))
        
        audio = request.files['audio']
        visual = request.files[visual_choice]
    
        id = str(uuid.uuid4())
        session["file_id"] = id
        if audio and allowed_file(audio.filename, 'audio'):
            session['audio_filename'] = os.path.join(app.config['UPLOAD_FOLDER'], 'A' + id + '.' + audio.filename.split('.')[-1])
            audio.save(session['audio_filename'])
            if audio.filename.split('.')[-1] == 'mp4':
                subprocess.call(f"ffmpeg -i {session['audio_filename']} {session['audio_filename'].replace('mp4', 'wav')}", shell=platform.system() != 'Windows')
                session['audio_filename'] = session['audio_filename'].replace('mp4', 'wav')
        else:
            return render_template('index.html') 

        if visual and allowed_file(visual.filename, visual_choice):
            session['visual_filename'] = os.path.join(app.config['UPLOAD_FOLDER'], id + '.' + visual.filename.split('.')[-1])
            visual.save(session['visual_filename'])
        else:   
            return render_template('index.html') 
        
        return redirect(url_for('camera'))
    return render_template('index.html')


@app.route('/webcam', methods=['POST'])
def webcam():
    if request.method == 'POST':
        if 'audio' not in request.files or request.files['audio'].filename == '':
            flash('Sound source not uploaded')
            return redirect(url_for('index'))

        audio = request.files['audio']
        id = str(uuid.uuid4())
        session["file_id"] = id
        if audio and allowed_file(audio.filename, 'audio'):
            session['audio_filename'] = os.path.join(app.config['UPLOAD_FOLDER'], id + '.' + audio.filename.split('.')[-1])
            audio.save(session['audio_filename'])
            if audio.filename.split('.')[-1] == 'mp4':
                subprocess.call(f"ffmpeg -i {session['audio_filename']} {session['audio_filename'].replace('mp4', 'wav')}", shell=platform.system() != 'Windows')
                session['audio_filename'] = session['audio_filename'].replace('mp4', 'wav')
    return render_template('camera.html', stream=True)

@app.route('/camera', methods=['GET'])
def camera():
    return render_template('camera.html', stream=app.config['RUNTIME_MODE'] != 'offline')


@app.route('/video_feed')
def video_feed():
    src = session.get('visual_filename')
    if src is not None:
        src = os.path.join(app.config['UPLOAD_FOLDER'], src)
    
    video_stream = VideoCamera(src, session.get('audio_filename'), app_config=app.config)
    CAMERAS[session.get('file_id')] = video_stream
    session['video_index'] = -10
    print("***", video_stream.result)
    if video_stream.result is None:
        return Response(generate_video(video_stream),
            mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return redirect(url_for('static', filename=video_stream.result), code=301)


@app.route('/audio_feed')
def audio_feed():
    src = session.get('audio_filename')
    return Response(generate_audio(src), mimetype="audio/x-wav")


@app.route('/time')
def rettime():
    print(times)
    return str(np.mean(times)) + " " + str(np.std(times))


@app.route('/status', methods=['POST'])
def video_action():
    if request.form['action'] == 'play':
        CAMERAS[session['file_id']].i = 0 if session.get('video_index', -1) < 0 else session['video_index'] 
    else:
        session['video_index'] = CAMERAS[session['file_id']].i
        CAMERAS[session['file_id']].i = -10
    return "OK"