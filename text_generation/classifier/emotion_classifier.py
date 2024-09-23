import warnings

from transformers import pipeline
#from utils import emotion_classifier, emotion_classifier2, EmoAccuracyrequired

emotion_classifier ="j-hartmann/emotion-english-distilroberta-base"

emotion_classifier2 = "michellejieli/emotion_text_classifier"

EmoAccuracyrequired = 0.2

def getclassifier():
    classifier = pipeline("text-classification", model = emotion_classifier, return_all_scores=True)
    return classifier

def check_possible_emotions_linebyline(text, classifier):
    emo = {'fear':0.0, 'anger':0.0, 'joy':0.0, 'sadness':0.0, 'surprise':0.0, 'neutral':0.0, 'disgust':0.0}
    lines = text.split(".")
    for line in lines:
        result = classifier(line)[0]
        for emotion in result:
            emo[emotion['label']] += emotion['score']
    n = len(lines)
    possible_emo = {}
    for label in emo:
        emo[label] = emo[label]/n
        if(emo[label]>EmoAccuracyrequired):
            possible_emo[label] = emo[label]
    return possible_emo


def get_evaluated_response(sentences):
    # for sentence in sentences:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = getclassifier()
    sentences = sentences.rstrip()
    sentences = sentences.split("\n\n")
    for i in range(len(sentences)):
        sentence = sentences[i]
        print(sentence)
        emotion = sentence.split("]")[0][1:]
        line = sentence.split("]")[1][1:]
        possible_emo = check_possible_emotions_linebyline(line, classifier)
        #make it to smallcase
        og_emotion = emotion
        emotion = emotion.lower()
        #print(emotion, " --> ", possible_emo)
        if(emotion == "sad"):
            emotion == "sadness"
        if(emotion == "calm"):
            emotion == "neutral"
        if(emotion == "angry"):
            emotion == "anger"
        if(emotion not in possible_emo.keys()):
            #replace the 1st instance of ["emotion"] with ["best_possible_emotion"] in sentence
            #best emo is the greatest emotion by value
            best_emo = max(possible_emo, key=possible_emo.get)
            best_emo = best_emo.capitalize()
            sentence = sentence.replace(og_emotion, best_emo, 1)
        sentences[i] = sentence
    return sentences




    

    

    

