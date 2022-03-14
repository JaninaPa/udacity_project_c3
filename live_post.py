import requests
import json

post_data = {
    "age": ["19"],
    "workclass": ["Private"],
    "fnlgt": ["301606"],
    "education": ["Some-college"],
    "education-num": ["10"],
    "marital-status": ["Never-married"],
    "occupation": ["Other-service"],
    "relationship": ["Own-child"],
    "race": ["Black"],
    "sex": ["Male"],
    "capital-gain": ["0"],
    "capital-loss": ["0"],
    "hours-per-week": ["35"],
    "native-country": ["United-States"],
}
response = requests.post('https://udacity-project-c3.herokuapp.com/model_inference/', data=json.dumps(post_data))
if response.status_code == 200:
    print("Request successful. Status Code: 200")
    print("Response body:")
    print(response.json())
