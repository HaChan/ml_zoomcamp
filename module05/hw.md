# Question 1 Version of Pipenv

```
pipenv --version

# pipenv, version 2024.2.0
```

# Question 2 The first hash for scikit-learn

```
jq -r '.default["scikit-learn"].hashes[0] // .develop["scikit-learn"].hashes[0]' Pipfile.lock
```

# Question 3 Probability of subscription

client:

```
{"job": "management", "duration": 400, "poutcome": "success"}
```

```python
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)


features = dv.transform(data)

probabilities = model.predict_proba(features)
print(probabilities[:,1])
```

# Question 4

Written in the [predict_service.py](predict_service.py) file

# Question 5

```
docker image ls | grep zoomcamp-model

# svizor/zoomcamp-model   3.11.5-slim   975e7bdca086   9 days ago       130MB
```

# Question 6

[Dockerfile](Dockerfile)
