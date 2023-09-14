from flask import Flask, render_template, request
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64, io
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


app = Flask(__name__)
# Get the path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_model():
    # Construct the path to the model.h5 file
    # model_file_path = os.path.join(current_dir, '\interfaz\DataSets')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model.h5 file
    train_file_path = os.path.join(current_dir, 'DataSets/train.csv')
    holidays_file_path = os.path.join(current_dir, 'DataSets/holidays_events.csv')
    test_file_path = os.path.join(current_dir, 'DataSets/test.csv')
    # # return load_model(model_file_path)
    # return model_file_path
    df_train = pd.read_csv(train_file_path,
                            usecols=['store_nbr', 'family', 'date', 'sales'],
                            dtype={
                                'store_nbr': 'category',
                                'family': 'category',
                                'sales': 'float32',
                            },
                            parse_dates=['date'],
                            infer_datetime_format=True,
                        )

    df_holidays = pd.read_csv(holidays_file_path,
                                dtype={
                                    'type': 'category',
                                    'locale': 'category',
                                    'locale_name': 'category',
                                    'description': 'category',
                                    'transferred': 'bool',
                                },
                                parse_dates=['date'],
                                infer_datetime_format=True,
                            )
    df_holidays = df_holidays.set_index('date').to_period('D')

    df_test = pd.read_csv(test_file_path,
                            dtype={
                                'store_nbr': 'category',
                                'family': 'category',
                                'onpromotion': 'uint32',
                            },
                            parse_dates=['date'],
                            infer_datetime_format=True,
                        )
    df_test['date'] = df_test.date.dt.to_period('D')
    df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

    df_clean = df_train[df_train['sales'] < 60000]
    df_train['sales'] = df_train['sales'].apply(lambda x : df_clean['sales'].mean() if x > 60000 else x)

    df_train['date'] = df_train.date.dt.to_period('D')
    df_train = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()

    average_sales = (
        df_train
        .groupby('date').mean()
        .squeeze()
        .loc['2016-08-15':'2017-08-15']
    )

    holidays = (
        df_holidays
            .query("locale in ['National', 'Regional']")
            .loc['2016-08-15':'2017-08-15', ['description']]
            .assign(description=lambda x: x.description.cat.remove_unused_categories())
        )
    holidays = holidays.drop(holidays[holidays['description'] == "Navidad"].index)


    y = df_train.unstack(['store_nbr', 'family']).loc['2016-08-15':'2017-08-15']

    fourier = CalendarFourier(freq="M", order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal= True,
        additional_terms=[fourier],
        drop=True,
    )
    X = dp.in_sample()

    model = LinearRegression()
    model.fit(X, y)

    X_days = y.copy()
    X_days = X_days.index.dayofweek
    X_onehotdays = pd.get_dummies(X_days)

    X_holidays = pd.get_dummies(holidays)

    X.insert(0, 'reference', range(0, len(X)))

    X = X.merge(X_onehotdays, left_on='reference', right_on=X_onehotdays.index).set_index(X.index)

    X = X.drop('reference', axis=1)

    X = X.drop(0, axis=1)

    X2 = X.join(X_holidays, on='date').fillna(0.0)

    X2 = X2.rename(columns={1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})

    model = LinearRegression()
    model.fit(X2, y)
    y_pred = pd.DataFrame(
        model.predict(X2),
        index=X2.index,
        columns=y.columns
    )

    X_test = dp.out_of_sample(steps=16)
    X_test.index.name = 'date'

    X_days = X_test.index.dayofweek
    X_onehotdays = pd.get_dummies(X_days)

    X_holidays = pd.get_dummies(holidays)

    X_test.insert(0, 'reference', range(0, len(X_test)))

    X_test = X_test.merge(X_onehotdays, left_on='reference', right_on=X_onehotdays.index).set_index(X_test.index)

    X_test = X_test.drop('reference', axis=1)

    X_test = X_test.drop(0, axis=1)

    X_test = X_test.join(X_holidays, on='date').fillna(0.0)
    X_test = X_test.rename(columns={1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})

    y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
    y_submit = y_submit.stack(['store_nbr', 'family'])
    y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])

    return y_submit

def graph_predictions(store_nbr, family, y_frame):
    print('FAMILIA = ', family)
    print('STORE NBR = ', store_nbr)
    y_frame = y_frame[(y_frame.index.get_level_values('store_nbr') == str(store_nbr)) & (y_frame.index.get_level_values('family') == family)]
    print(y_frame)
    y_frame.reset_index().plot(x= 'date', y= 'sales', title= (f'Ventas de la tienda {store_nbr} de la familia {family}'))

    # TESTEAR EL PLOT A VER SI SE MUESTRA O NO
    # Generate random data
    # x = np.linspace(0, 10, 100)
    # y = np.random.rand(100)
    # # Create a DataFrame
    # df = pd.DataFrame({'x': x, 'y': y})
    # # Create a simple scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(df['x'], df['y'])
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Random Scatter Plot')

    # Guardar el plot en una imagen
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.read()).decode()
    return img_base64

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_sales():
    if request.method == 'POST':
        store_nbr = int(request.form.get('store_nbr'))
        family = str(request.form.get('family'))
        y_frame = load_model()
        print(y_frame)
        img_base64 = graph_predictions(store_nbr, family, y_frame)

        return render_template('result.html', img=img_base64)

if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)