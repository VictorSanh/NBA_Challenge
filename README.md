# NBA Challenge
 MVA - S. Mallat - Lesson at Collège de France - Data Challenge

The main goal of the challenge is to predict the winning NBA team at the end of a game when we only have information about the first two quarters of the game. This is a binary classification problem which will be based on the evolution of scores per second and some key events from the first second of the game until the end of the second quarter. You can find more details [here](https://challengedata.ens.fr).

We achieve a 75,01% submission accuracy which ranks us in the top 20% of the challengers.

### How to install the requirements ?

Python 2:
```python
pip install -r requirements.txt
```

Python 3:
```python
pip3 install -r requirements.txt
```


### Best submissions

Our best submission achieves 75.01% in accuracy (test). It builds an XGBoost classifier with moderately fine-tuned parameters on top of the raw dataset extended with some engineered features. It can be run using the command `python3 run_best_submission.py`.

The second best submission is a Bi-directional LSTM completed with two dense layers on which we trained a Random Forest classifier (achieving 74.48% submission accuracy). The third best submission is a Random Classifier heavily with heavily fine-tuned parameters which achieves a 73.77% submission accuracy.

For more details and comments, please refer to the [experiment report](https://github.com/VictorSanh/NBA_Challenge/blob/master/Report.pdf).
