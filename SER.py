'''from emotion_recognition import EmotionRecognizer

# initialize instance, this will take a bit the first time executed
# as it'll extract the features and calls determine_best_model() automatically
# to load the best performing model on the picked dataset
rec = EmotionRecognizer(emotions=["angry", "neutral", "sad", "happy", "calm", "fear", "disgust", "ps", "boredom"], balance=False, verbose=1, custom_db=False)
# it will be trained, so no need to train this time
# get the accuracy on the test set
print(rec.confusion_matrix())
# predict angry audio sample
prediction = rec.predict('test/angry.wav')
print(f"Prediction: {prediction}")'''

from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
# init a model, let's use SVC
my_model = SVC()
# pass my model to EmotionRecognizer instance
# and balance the dataset
rec = EmotionRecognizer(model=my_model, emotions=["angry", "neutral", "sad", "happy", "calm", "fear", "disgust", "ps", "boredom"], balance=True, verbose=0)
# train the model
rec.train()
# check the test accuracy for that model
print("Test score:", rec.test_score())
# check the train accuracy for that model
print("Train score:", rec.train_score())