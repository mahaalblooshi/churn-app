# Customer Churn Prediction (E-commerce)

**Live app:** https://churn-app-cgvbf33lx2pwabamvyefmz.streamlit.app

In this project I built a machine learning model that predicts if an online shop customer will **churn** (stop buying). Then I made a web app so anyone can try it.

I followed the **CRISP-DM** steps, which is a standard way to organize a data science project:

| CRISP-DM step | What I did |
|---|---|
| 1. Business understanding | Decided what problem to solve |
| 2. Data understanding | Looked at the data and made charts |
| 3. Data preparation | Cleaned the data and got it ready for the model |
| 4. Modeling | Trained 4 models and picked the best |
| 5. Evaluation | Checked how good the model is |
| 6. Deployment | Built the web app and put it online |

---

## 1. Business understanding

**The problem:** When customers stop buying from a shop, the shop loses money. It is also cheaper to keep a customer than to find a new one.

**My goal:** Build a model that looks at how a customer behaved recently and predicts if they will leave. The shop can then act early, for example by sending an offer.

**What I predict:** the `Churn` column. 1 means the customer will churn and 0 means they will not. This is a **classification** problem with two classes.

## 2. Data understanding

I used a file called `ecommerce.csv` with information about customers. First I checked the data with `head()`, `info()`, `isna().sum()` and `describe()` to see the columns, the data types, the missing values and basic statistics.

Then I picked the columns I needed:

| Type | Columns |
|---|---|
| How recently and how often they shop | `Days_Since_Last_Order`, `Orders_Last_90_Days`, `Website_Visits_Last_30_Days`, `App_Sessions_Last_30_Days` |
| How much they spend | `Total_Spend_AED` |
| Problems and support | `Complaint_Flag`, `Support_Tickets_Last_90_Days`, `Returns_Last_90_Days` |
| Engagement | `Cart_Abandonment_Rate`, `Email_Open_Rate` |
| Customer type | `Membership_Level`, `Subscription_Status` |
| What I want to predict | `Churn` |

I also made some charts to understand the data:
- A pair plot of the numbers, coloured by `Churn`
- Bar charts of `Membership_Level` and `Subscription_Status` by `Churn`
- A histogram of `Total_Spend_AED` by `Churn`

## 3. Data preparation

These are the steps I did to get the data ready:

1. **Missing values:** `Email_Open_Rate` had some empty values, so I filled them with the **median**.
2. **Input and output:** I put all the columns except `Churn` in `X` (input) and `Churn` in `y` (output).
3. **Text to numbers:** The model only understands numbers, so I used one-hot encoding on `Membership_Level` and `Subscription_Status`.
4. **Train and test split:** I used 80% of the data to train the model and 20% to test it (`random_state=42`).
5. **Scaling:** I used `StandardScaler` so all numbers are on a similar scale. I fitted it on the training data only, so the test data stays unseen.

## 4. Modeling

I trained four different models and compared them:

| Model | Settings |
|---|---|
| Logistic Regression | default settings |
| K-Nearest Neighbors | `n_neighbors=5` |
| Decision Tree | `max_depth=5` |
| Random Forest | `n_estimators=100`, `max_depth=5` |

I chose **Random Forest** as my final model.

## 5. Evaluation

To check how good the models are, I did the following:

- I printed a **classification report** (precision, recall, F1-score, accuracy) for the training data and the test data. Comparing them shows if the model is overfitting (doing well on training but badly on new data).
- I used **10-fold cross-validation** on the Random Forest to check that the results are stable.
- I used **GridSearchCV** to try different settings: `n_estimators` of 50, 100 and 200, and `max_depth` of 3, 5 and 7.
- I trained the final Random Forest with `n_estimators=100` and `max_depth=5` and checked it again on the test set.

**My results on the test set** (threshold 0.5, churn class):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.81 | 0.65 | 0.23 | 0.33 |
| K-Nearest Neighbors | 0.78 | 0.47 | 0.26 | 0.33 |
| Decision Tree | 0.80 | 0.61 | 0.20 | 0.30 |
| **Random Forest (final)** | 0.79 | 0.79 | 0.06 | 0.11 |

**What these numbers mean:** only about 21% of customers churn. A model that always says "will not churn" would already get about 79% accuracy, so accuracy alone is not a good measure here. Recall (how many real churners the model finds) is more important.

At the default threshold of 0.5 the Random Forest finds very few churners (recall 0.06), because its churn probabilities are usually below 0.5. To fix this in the app, I added a **churn threshold** slider. When the threshold is lower, the model flags more customers, so recall goes up and precision goes down. The *Model performance* page shows this trade-off live.

## 6. Deployment

### 6.1 Making the pipeline (in `deployment.ipynb`)

In the first part of the project I cleaned and scaled the data by hand. For the app that would be hard to repeat, so I put everything into one **scikit-learn Pipeline**. This way the app does all the steps automatically.

The pipeline does this:
- **Number columns:** fill missing values with the median, then scale them
- **Text columns:** fill missing values with the most common value, then one-hot encode them
- **Model:** Random Forest (`n_estimators=100`, `max_depth=5`)

I trained the pipeline, tested it, and saved it as **`churn_model.pkl`** using `joblib`. Then I loaded it again and tested one example customer to make sure it works.

I also saved the test-set results of all four models to **`test_predictions.csv`**. The file has the real answer (`y_true`) and each model's churn probability for every test customer. The app uses it to draw the performance charts without needing the full dataset.

### 6.2 Making the Streamlit app (`app.py`)

The app loads `churn_model.pkl` and has a **sidebar** and **four tabs**.

**Sidebar**
- **Churn threshold** slider: customers with a churn probability at or above this value are marked as **High** risk. It also changes the numbers on the *Model performance* tab.
- Shows the risk bands: Low < half the threshold ≤ Medium < threshold ≤ High

**🔮 Predict**
- Boxes, sliders and dropdowns for the 12 customer details
- **Quick examples** ("Loyal customer" and "At-risk customer") to fill the form in one click
- When you click **Predict Churn**, it shows:
  - **Will Churn** or **Will Not Churn**
  - The churn **probability** (for example 73%) with a gauge chart
  - The **risk level** (Low, Medium or High)
  - **Suggested actions**, for example "send a 'we miss you' discount". These come from simple business rules, not from the model.

**📈 Model performance**
- Recall, precision, accuracy and F1-score at the chosen threshold
- **Confusion matrix**: how many customers the model got right and wrong
- **ROC curve** with the AUC score
- **Feature importance**: which customer details matter most to the model
- **Recall vs precision by threshold**: how moving the threshold changes the results
- **Model comparison** table for all four models, with the best value in each column highlighted

**🕘 History**
- Every prediction is saved automatically with the time
- Shows the number of predictions, how many are high risk, and the average churn probability
- A bar chart of the churn probability of each prediction
- Download the history as a CSV file, or clear it
- The history is only kept while the page is open. Refreshing the page clears it.

**ℹ️ About**
- Explains the project and how to use the app

To run it on your own computer:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### 6.3 Problems I had and how I fixed them

| Problem | How I fixed it |
|---|---|
| `sns.load_dataset` error | That function only loads Seaborn's example data. I used `pd.read_csv` for my own file. |
| `streamlit` command not found | I ran it with `python -m streamlit run app.py`. |
| `No module named sklearn` | My terminal used a different Python version than my notebook. I used the same Python as the notebook. |
| `No module named seaborn` online | I removed the unused imports from `app.py`. |

### 6.4 Putting the app online

1. I made a public GitHub repository called `churn-app`.
2. I uploaded `app.py`, `churn_model.pkl`, `test_predictions.csv` and `requirements.txt`. I did not upload the dataset.
3. I put the same scikit-learn version in `requirements.txt` that I used to train the model, so the model loads without errors.
4. I connected the repository on **Streamlit Community Cloud**, chose `app.py` as the main file and Python 3.12.
5. I clicked Deploy, and now the app is public at the link at the top of this file.
6. When I change a file on GitHub, Streamlit Community Cloud updates the app automatically.

## Project files

```
churn-app/
├── app.py              # the Streamlit web app
├── churn_model.pkl     # the saved pipeline (data prep + Random Forest)
├── test_predictions.csv # test-set answers and probabilities for the performance page
├── requirements.txt    # the libraries the app needs (streamlit, pandas, joblib, scikit-learn, plotly)
└── README.md           # this file
```

All the analysis and model training is in `deployment.ipynb`. The dataset `ecommerce.csv` is not in the repository.

## Tools I used

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, joblib, Streamlit, GitHub, Streamlit Community Cloud

## What I could improve

Already done in the app: showing the churn probability, the risk level, and which features matter most.

Still to do:
- **Improve recall:** train the Random Forest with `class_weight='balanced'` so it finds more churners.
- **Clean the membership levels:** the data has the same level written in different ways (`Basic` and `basic`, `Silver` and `SILVER`). Making them all the same before training would help.
- Try other models, like gradient boosting.
- Save the prediction history in a database so it is not lost when the page is refreshed.
- Retrain the model when new customer data comes in.
