import fasttext
import os
import multiprocessing



path = os.path.abspath(os.path.dirname(__file__))

model = fasttext.train_unsupervised(
    os.path.join(path, "..", "data", "interviews_train.txt"), 
    model='skipgram',
    dim=100,
    epoch=50,
    neg=10,
    wordNgrams=3,
    thread=multiprocessing.cpu_count(),
    )

model.save_model(os.path.join(path, "..", "models", "model_interviews.bin"))