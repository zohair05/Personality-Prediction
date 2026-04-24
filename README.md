# 🧠 Personality Prediction System (Introvert vs Extrovert)

A machine learning-based text classification system that predicts whether a person is an **Introvert** or **Extrovert** based on user input. The system also improves over time by learning from new user-provided examples.


## 📌 Features

* Text preprocessing using **NLTK (tokenization + lemmatization)**
* Feature extraction using **TF-IDF**
* Classification using **Logistic Regression**
* **Cosine similarity matching** for faster predictions
* Confidence-based predictions with fallback learning
* Continuous learning (model retrains with new user data)
* Model persistence using **Pickle**


## 📂 Project Structure
---
```
personality_project/
│
├── dataset.csv                  # Training dataset
├── personality_pipeline.pkl     # Saved ML model
├── Personality Prediction.py    # Main script
└── README.md                   # Project documentation
```
---
## ⚙️ Installation

### 1. Clone the repository (or download files)

```bash
git clone <your-repo-link>
cd personality_project
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn nltk
```

### 3. Download NLTK resources (auto-handled in code)

```python
nltk.download('punkt')
nltk.download('wordnet')
```

## ▶️ How to Run
```bash
python "Personality Prediction.py"
```

## 🧪 How It Works

### 1. Preprocessing

* Converts text to lowercase
* Removes punctuation
* Applies **lemmatization**

### 2. Model Pipeline

* TF-IDF vectorization
* Logistic Regression classifier

### 3. Prediction Flow

* User enters a sentence
* System checks similarity with existing data
* If similarity > 0.6 → direct label
* Else:

  * Uses model prediction with probability
  * If confidence < 0.6 → asks user for label
  * Retrains model with new data


## 📊 Model Evaluation

* Accuracy score printed after training
* Classification report included
* 5-fold cross-validation used for stability


## 🔄 Continuous Learning

If the model is unsure:
* It asks the user to label input
* Adds new data to dataset
* Retrains model
* Saves updated model and dataset


## 💾 Model Saving

* Model saved as: `personality_pipeline.pkl`
* Dataset updated automatically
* Loads existing model if available


## 🧠 Example Usage

Tell us what you enjoy doing:
> I like reading books alone and writing journals

```Output:
Predicted Personality: Introvert (Confidence: 0.82)
```

## 🚀 Future Improvements

* Add more personality classes (MBTI types)
* Build a web app using Flask or Streamlit
* Improve dataset size and quality
* Add deep learning models (LSTM/BERT)
* Enhance UI/UX
