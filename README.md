# Instagram Review Analysis using an AI Agent

### Sentiment Analysis on Instagram Comments using Fine-Tuned BERT

This project leverages a custom AI agent to perform **sentiment-based review analysis** on any **public Instagram profile** — requiring only the username. The system fetches recent posts, scrapes user comments, and runs a full **NLP pipeline** to determine how people feel about the account.

---

## 📌 Project Highlights

### Model Development & Fine-Tuning
- Utilized [`callmesan/indic-bert-roman-urdu-fine-grained`](https://huggingface.co/callmesan/indic-bert-roman-urdu-fine-grained) from Hugging Face for **Roman Urdu sentiment classification**.
- Fine-tuned the model on **two separate datasets** with preprocessing.
- Re-labeled extreme sentiment scores (`very positive`/`very negative`) into **3 simplified categories**: `positive`, `negative`, and `neutral` using **SVM** from `scikit-learn`.
- Trained with **Hugging Face Transformers** on **PyTorch** backend for flexibility and performance.

### 📊 Data Collection & Processing
- Scraped latest **30 Instagram posts** and their comments using **Apify's Instagram scraper**.
- Organized data into a clean **Pandas DataFrame** for further processing.

### 📈 Exploratory Data Analysis (EDA)
- Used **Matplotlib**, **Seaborn**, and **Pandas** for insightful visualizations.
- Analyzed sentiment trends, frequency distributions, and user perception patterns.

### Deployment & Interface
- Built an **interactive Streamlit app** where users input an Instagram username.
- Instantly shows:
  - Comment sentiment breakdown
  - Visualizations
  - Overall audience perception

---

##  Setup Instructions

### Step 1: Clone the repository or copy files

Save all project files into a **separate directory** on your machine.

### Step 2: Navigate to the directory in your terminal

###  Step 3: Install all dependencies

Make sure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt

```
###  Step 4: Run the Streamlit interface

Launch the app using:

```bash
streamlit run interface.py
``` 
## 🔗 Tech Stack
#### Hugging Face Transformers
#### IndicBERT (fine-tuned)
#### scikit-learn (SVM)
#### PyTorch
#### Apify Scraper
#### Pandas / Matplotlib / Seaborn
#### Streamlit

