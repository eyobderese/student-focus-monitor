# Step 1 — one-time setup

pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

python generate_[assets.py](http://assets.py)

# Step 2 — one-time training (needs MRL Eye Dataset in data/mrl_eyes/)

python models/train_eye_[cnn.py](http://cnn.py)

python models/train_svm_[baseline.py](http://baseline.py)

# Step 3 — evaluate (optional, for report metrics)

python eval/eval_ear_[mar.py](http://mar.py)        # needs YawDD frames

python eval/eval_head_[pose.py](http://pose.py)      # needs 300W-LP + angles.json

python eval/eval_[attention.py](http://attention.py)      # needs trained models + MRL

python eval/eval_[latency.py](http://latency.py)        # runs on live webcam

# Step 4 — run system

python [main.py](http://main.py)                     # webcam

streamlit run dashboard/[app.py](http://app.py)     # dashboard