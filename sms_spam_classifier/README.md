# SMS Spam Classifier 📱✉️

This project builds a machine learning classifier to detect SMS spam messages using **Logistic Regression** and **Gradient Boosting**, trained on the SMS Spam Collection dataset.

## 📊 Dataset
- Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) or [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 SMS messages (ham/spam)
- Features extracted using **TF-IDF (unigrams + bigrams, max 3000 features)**
- Class balance: 86.6% ham, 13.4% spam

## ⚙️ Preprocessing
- Encoded labels: `ham=0`, `spam=1`
- Text converted into TF-IDF vectors
- Stratified train/test split (75/25)

## 🤖 Models
- Logistic Regression (baseline, max_iter=500)
- Gradient Boosting (200 estimators, learning rate=0.05, depth=3)

## ✅ Results
| Model                | Accuracy | Precision (macro) |
|-----------------------|----------|-------------------|
| Logistic Regression   | 0.9727   | 0.9847            |
| Gradient Boosting     | 0.9591   | 0.9707            |

## 📈 Loss Curves
Gradient Boosting train/test log-loss curves show:
- Both losses decrease and stabilize after ~50–100 iterations
- No severe overfitting observed

## 📂 Files
- `sms_spam_classifier.py` → Python script with models & plots
- `spam.csv` → dataset (not uploaded; download from Kaggle/UCI)
- `SMS_Spam_Classifier_Report.docx` → one-page report
- `Gradient_Boosting_Loss_Curves.png` → loss curve plot
- `requirements.txt` → dependencies

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Place `spam.csv` in the project folder (download from Kaggle/UCI).

4. Run the script:
   ```bash
   python sms_spam_classifier.py
   ```

## 🔮 Future Work
- Handle class imbalance with SMOTE / class weighting
- Try deep learning models (LSTM, BERT)
- Evaluate with recall/F1 for spam detection priority
