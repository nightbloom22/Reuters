# Document Classification
Classify documents on topics/places, using Reuters-21578 data.


### Requirements
Please see requirements.txt.
<br />
To install these packages, use the following command in a <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/" target="_blank"> virtualenv</a>.

```bash
$ pip install -r requirements.txt
```

### Training data
Based on Reuters-21578 files.
<br />
Available in sgm format on data/ 

Trained data's topics can be found in
data/all-topics-strings.lc.txt

Trained data's places can be found in
data/all-places-strings.lc.txt

To train and test, run the following 

### Train 
```bash
$ python main.py 
```

### Classifier 
Multi-class classification: perceptron and transformer. 

Multi-label classification: MultiOutputClassifier
