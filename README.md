# NSL-KDD Feature Input UI

Simple Flask app where the user uploads a CSV file containing NSL-KDD features instead of filling out a form.
The header row of the CSV must list all feature names expected by the model; one or more records may follow. Three of the columns are categorical (<code>protocol_type</code>, <code>service</code>, <code>flag</code>) and the remaining fields are numeric. Categorical values may be strings; numeric values must be parseable as floats.

Run locally:

```bash
python -m pip install -r requirements.txt
python app.py
```

Point your browser at http://127.0.0.1:5000/ and choose a CSV file for prediction.