```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
py train.py
```

(on Windows, you will need instead to run `source venv/Scripts/activate`)

Let this train long enough to save a model.

Thenceforth you can just load the model and get a new generated batch with:

```
py main.py
```