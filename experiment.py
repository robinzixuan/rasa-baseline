from urllib.parse import parse_qs
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.core.channels.channel import OutputChannel, UserMessage
import asyncio
import rasa.core.agent
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.core.agent import Agent
from rasa.utils.endpoints import EndpointConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt



class Model:
    def __init__(self, path= 'test_data.csv', model_dir = './models'):
        self.model_dir = model_dir

        self.action_endpoint = EndpointConfig(url='http://localhost:5055/webhook')

        
        self.encoder = LabelEncoder()
        self.load_data(path)
        
        

    def load_data(self, path):
        data = pd.read_csv(path)
        data = data.rename(columns={'Unnamed: 0': 'message'})
        self.message = data['message']
        self.intent = data['intent']
        uniq = np.unique(self.intent)
        uniq = np.append(uniq, 'nlu_fallback')
        uniq = np.append(uniq, '')
        self.encoder.fit(uniq)
        self.label = self.encoder.transform(self.intent)
        print(uniq)
        print('Done')

        

    def predict(self):
        messages = []
        count = 0
        #diet_classifier = DIETClassifier()
        predictions = []
        agent = Agent.load(self.model_dir, action_endpoint = self.action_endpoint)
        a = np.random.permutation(self.message.shape[0])
        self.message = self.message[a]
        self.label = self.label[a]
        for inp in self.message:
            try:
                message = Message.build(inp)
                messages.append(message)
                res = asyncio.run(agent.parse_message(inp))
            
                if res:
                    
                    intent = res['intent_ranking'][0]['name']
                    confidence = res['intent_ranking'][0]['confidence']
                    responseSelector = res['response_selector']
                    #print(intent)
                    
                    if intent == 'chitchat':
                        intent = responseSelector['chitchat']['ranking'][0]['intent_response_key']
                        confidence *= responseSelector['chitchat']['ranking'][0]['confidence']
                    elif intent == 'faq':
                        intent = responseSelector['faq']['ranking'][0]['intent_response_key']
                        confidence *= responseSelector['faq']['ranking'][0]['confidence']
                    
                    predictions.append(intent)
                else:
                    predictions.append('')

            except:
                predictions.append('')
        print(predictions)
        predictions = self.encoder.transform(predictions)
        return predictions

    def accuracy(self, predictions):
        print(self.label)
        print(predictions)
        accuracy = np.sum(predictions == self.label)/(len(predictions))
        cm = confusion_matrix(self.label, predictions)
        precision = precision_score(self.label, predictions,average='weighted')
        recall = recall_score(self.label, predictions,average='weighted')
        f1 = precision * recall * 2/(precision + recall)
        return accuracy, precision, recall, f1,cm
       

model = Model(path= 'test_data2.csv')
#model = Model(path= 'test_data.csv')
predictions = model.predict()
accuracy, precision, recall, f1, cm = model.accuracy(predictions)
print("accuracy: %.4f" % accuracy)
print("precision: %.4f" % precision)
print("recall: %.4f"% recall)
print("f1: %.4f"%f1)

print(cm)
sns.heatmap(cm, cmap="YlGnBu")
plt.show()









