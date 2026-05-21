from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
state = {}

def load_and_train():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "earthquake1.csv"))
    df = df.drop('id', axis=1)

    timestamp = []
    for d, t in zip(df['date'], df['time']):
        ts = datetime.datetime.strptime(d + ' ' + t, '%Y.%m.%d %I:%M:%S %p')
        timestamp.append(time.mktime(ts.timetuple()))
    df['Timestamp'] = pd.Series(timestamp).values
    df = df.drop(['date', 'time'], axis=1)

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = preprocessing.LabelEncoder()
            df[col] = df[col].fillna('unknown')
            le.fit(df[col])
            df[col] = le.transform(df[col]).astype(float)
            label_encoders[col] = le

    si = SimpleImputer(missing_values=np.nan, strategy="mean")
    si.fit(df[["dist", "mw"]])
    df[["dist", "mw"]] = si.transform(df[["dist", "mw"]])

    scaler = preprocessing.MinMaxScaler()
    d = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(d, columns=df.columns)

    y = np.array(df_scaled['xm'])
    X = np.array(df_scaled.drop('xm', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    dt = DecisionTreeRegressor(random_state=40)
    dt.fit(X_train, y_train)

    knn = KNeighborsRegressor(n_neighbors=6)
    knn.fit(X_train, y_train)

    state['models'] = {'lr': lr, 'dt': dt, 'knn': knn}
    state['scaler'] = scaler
    state['label_encoders'] = label_encoders
    state['all_cols'] = df_scaled.columns.tolist()
    state['accuracies'] = {
        'lr': round(lr.score(X_test, y_test) * 100, 2),
        'dt': round(dt.score(X_test, y_test) * 100, 2),
        'knn': round(knn.score(X_test, y_test) * 100, 2),
    }
    state['xm_min'] = float(df['xm'].min() * (scaler.data_max_[df_scaled.columns.tolist().index('xm')] - scaler.data_min_[df_scaled.columns.tolist().index('xm')]) + scaler.data_min_[df_scaled.columns.tolist().index('xm')])
    # Store actual xm range for de-normalization
    xm_idx = df_scaled.columns.tolist().index('xm')
    state['xm_scale_min'] = scaler.data_min_[xm_idx]
    state['xm_scale_max'] = scaler.data_max_[xm_idx]

startup_error = None
try:
    print("Training models...")
    load_and_train()
    print("Done. Models ready.")
except Exception as e:
    import traceback
    startup_error = traceback.format_exc()

@app.route('/')
def index():
    if startup_error:
        return f"<pre>{startup_error}</pre>", 500
    le = state['label_encoders']
    countries = sorted(le['country'].classes_.tolist())
    directions = sorted(le['direction'].classes_.tolist())
    return render_template('index.html',
                           countries=countries,
                           directions=directions,
                           accuracies=state['accuracies'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    le = state['label_encoders']
    all_cols = state['all_cols']

    direction = data.get('direction', 'west')
    country = data.get('country', 'turkey')
    dir_enc = float(le['direction'].transform([direction])[0]) if direction in le['direction'].classes_ else 0.0
    country_enc = float(le['country'].transform([country])[0]) if country in le['country'].classes_ else 0.0

    city_enc = float(np.median(list(range(len(le['city'].classes_)))))
    area_enc = float(np.median(list(range(len(le['area'].classes_)))))

    row_map = {
        'lat': float(data.get('lat', 38.2)),
        'long': float(data.get('long', 28.35)),
        'country': country_enc,
        'city': city_enc,
        'area': area_enc,
        'direction': dir_enc,
        'dist': float(data.get('dist', 2.3)),
        'depth': float(data.get('depth', 10.0)),
        'md': float(data.get('md', 0.0)),
        'richter': float(data.get('richter', 3.5)),
        'mw': float(data.get('mw', 4.5)),
        'ms': float(data.get('ms', 0.0)),
        'mb': float(data.get('mb', 0.0)),
        'Timestamp': float(time.mktime(datetime.datetime.now().timetuple())),
        'xm': 0.0,
    }

    row = np.array([[row_map[c] for c in all_cols]])
    row_scaled = state['scaler'].transform(row)
    feat_idx = [i for i, c in enumerate(all_cols) if c != 'xm']
    X_input = row_scaled[:, feat_idx]

    xm_min = state['xm_scale_min']
    xm_max = state['xm_scale_max']

    results = {}
    for key, model in state['models'].items():
        norm = float(np.clip(model.predict(X_input)[0], 0, 1))
        actual = round(norm * (xm_max - xm_min) + xm_min, 3)
        results[key] = {'normalized': round(norm, 4), 'xm': actual}

    return jsonify({'results': results, 'accuracies': state['accuracies']})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
