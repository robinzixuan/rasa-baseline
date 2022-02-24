import yaml
import pandas as pd

result = dict()


with open("train_test_split/training_data.yml", "r") as stream:
    try:
        data = yaml.safe_load(stream)['nlu']
    except yaml.YAMLError as exc:
        print(exc)


for i in range(len(data)):
    d = data[i]
    #intent = d['intent'].split("/")[0]
    intent = d['intent']
    example = d['examples'].strip('').split('\n')
    for i in example:
        i = i[2:]
        result[i] = intent
    
result = pd.DataFrame.from_dict(result, orient='index', columns=['intent'])

result.to_csv('train_data2.csv')


result = dict()


with open("train_test_split/test_data.yml", "r") as stream:
    try:
        data = yaml.safe_load(stream)['nlu']
    except yaml.YAMLError as exc:
        print(exc)


for i in range(len(data)):
    d = data[i]
    #intent = d['intent'].split("/")[0]
    intent = d['intent']
    example = d['examples'].strip('').split('\n')
    for i in example:
        i = i[2:]
        result[i] = intent
    
result = pd.DataFrame.from_dict(result, orient='index', columns=['intent'])

result.to_csv('test_data2.csv')