# Email Spam Detection with Machine Learning

**Natural Language Processing · Binary Text Classification**

A binary text classifier that separates spam from legitimate (ham) messages using TF-IDF features and three scikit-learn models.

## Dataset
UCI Machine Learning Repository spam collection: 5,572 labelled text messages. After removing duplicates, 5,169 remain (87.4% ham, 12.6% spam). The same approach applies directly to email text.

## Approach
1. Load the data and check the class distribution
2. Preprocess text: lowercase, remove HTML codes, URLs, digits and punctuation, remove stopwords
3. TF-IDF features (unigrams and bigrams)
4. Stratified 80/20 train/test split
5. Train Multinomial Naive Bayes, Logistic Regression and Linear SVM
6. Evaluate with accuracy, precision, recall, F1-score and confusion matrices
7. Discussion of why recall matters for spam detection
8. Bonus: word clouds for spam and ham

## Results (test set, spam class)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.9816 | 0.9590 | 0.8931 | 0.9249 |
| Logistic Regression | 0.9739 | 0.8824 | 0.9160 | 0.8989 |
| Linear SVM | 0.9816 | 0.9308 | 0.9237 | 0.9272 |

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Email_Spam_Detection.ipynb
```
Then choose **Kernel → Restart & Run All**.

## Files
- `Email_Spam_Detection.ipynb`: the full commented notebook
- `spam.csv`: dataset
- `requirements.txt`: dependencies

**Tech stack:** Python, pandas, scikit-learn, seaborn, matplotlib, WordCloud
