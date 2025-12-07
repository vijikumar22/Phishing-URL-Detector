🛡️ Phishing URL Detector

Machine Learning + Streamlit App

This project detects phishing URLs using machine learning and URL-based feature engineering.
It provides:

✔️ Real-time URL scanner
✔️ ML model trained on 1000 synthetic + real phishing patterns
✔️ Risk explanations
✔️ Probability scores
✔️ Adjustable detection threshold
✔️ Bulk URL scanning (CSV upload)
✔️ History & analytics dashboard

🚀 Features
🔍 Single URL Detection

Enter any URL

Get risk score + prediction

ML explanation about why URL is risky

📁 Bulk URL Scanning

Upload CSV with url column

Predicts phishing / legitimate

Shows analytics charts

Download results as CSV

📊 History & Analytics

Stores session predictions

Graphs phishing vs legitimate

Export history

🧠 Machine Learning

Model: RandomForestClassifier
Features extracted from URLs include:

Length

Subdirectory count

Suspicious keywords

Dot count

HTTPS presence

Query parameters

IP-based domain usage

