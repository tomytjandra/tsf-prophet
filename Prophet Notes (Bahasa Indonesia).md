# Time Series Forecasting: Prophet

- Documentation: https://facebook.github.io/prophet/docs/quick_start.html
- Paper: https://peerj.com/preprints/3190/

## Quick Start

- Cara penggunaan mirip dengan `sklearn`:
    - class: `Prophet()`
    - method: `.fit()` dan `.predict()`

- Input: DataFrame dengan dua kolom:
    1. `ds` (datestamp), format YYYY-MM-DD atau YYYY-MM-DD HH:MM:SS
    2. `y` (numeric)

- Simple Workflow:
    1. Import libraries dan dataset
    2. Modeling
        ```
        m = Prophet()
        m.fit(df)
        ```
    3. Prediction
        ```
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        ```
    4. Visualize fitting and components
        ```
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)
        ```

        Optional: `plotly` (versi di atas 4.0)
        ```
        from fbprophet.plot import plot_plotly, plot_components_plotly

        plot_plotly(m, forecast)
        plot_components_plotly(m, forecast)
        ```

## Saturating Forecast

### Forecasting Growth

- Secara default, Prophet menggunakan linear growth

- Logistic growth: digunakan ketika terdapat suatu titik maksimum yang dicapai oleh nilai forecast, nilai ini disebut sebagai carrying capacity

- Cara:
    1. Menambahkan kolom `cap` pada data training
        ```
        df['cap'] = 8.5
        ```
        Note: nilai carrying capacity tidak harus sebuah konstan
    2. Gunakan parameter `growth` saat fitting model
        ```
        m = Prophet(growth='logistic')
        m.fit(df)
        ```
    3. Menambahkan kolom `cap` pada data tanggal yang akan di-forecast
        ```
        future = m.make_future_dataframe(periods=1826)
        future['cap'] = 8.5
        fcst = m.predict(future)
        ```

### Saturating Minimum

- Secara default, logistic growth memiliki minimum = 0

- Nilai minimum ini dapat dispecify dengan menambahkan kolom `floor` bersamaan pada penambahan kolom `cap`, seperti pada langkah 1 dan 3 sebelumnya

- Note: Kolom `cap` wajib dispecify apabila ingin menggunakan saturating minimum

## Trend Changepoints

### Automatic Changepoint Detection

- Secara default, Prophet secara otomatis mendeteksi titik perubahan dan mengadaptasi nilai trendnya, dengan cara meletakkan 25 changepoints potensial yang diletakkan secara seragam di 80% awal data time series.

- Visualisasi changepoint:
    ```
    from fbprophet.plot import add_changepoints_to_plot
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    ```

- Parameter yang dapat diubah pada `Prophet()`:
    - `n_changepoints` (default = 25): mengubah banyaknya changepoint potensial di awal. Namun tidak disarankan, lebih baik melakukan regularisasi
    - `changepoint_range` (default = 0.8): mengubah persentase panjang time series untuk diletakkan changepoint potensial. Apabila nilainya terlalu besar, maka cenderung overfitting terhadap fluktuasi data di akhir waktu

### Adjusting Trend Flexibility (Regularisasi)

- Apabila trend terlalu overfit atau underfit, maka gunakan parameter `changepoint_prior_scale` pada `Prophet()` untuk mengatur fleksibilitas trend. Secara default nilainya 0.05.
    - Semakin besar nilai `changepoint_prior_scale`, maka semakin fleksibel. Artinya semakin rentan overfitting.

### Specifying the locations of the changepoints

- Secara manual menspesifikan tanggal yang diperbolehkan sebagai changepoint **potensial**. Namun, tanggal ini akan dicek lagi berdasarkan nilai `changepoint_prior_scale`
    ```
    m = Prophet(changepoints=['2014-01-01'])
    forecast = m.fit(df).predict(future)
    ```

## Seasonality, Holiday Effects, and Regressors

### Modeling Holidays and Special Events

- Model Prophet dengan efek holiday atau event penting yang dapat mempengaruhi secara signifikan nilai `y`
- Cara:
    1. Buat DataFrame dengan kolom:
        - `holiday`: nama event
        - `ds`: tanggal event
        - `lower_window` (optional): batas bawah interval holiday (berapa hari sebelum tanggal event)
        - `upper_window` (optional): batas atas interval holiday (berapa hari setelah tanggal event)

        ```
        playoffs = pd.DataFrame({
            'holiday': 'playoff',
            'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                                    '2010-01-24', '2010-02-07', '2011-01-08',
                                    '2013-01-12', '2014-01-12', '2014-01-19',
                                    '2014-02-02', '2015-01-11', '2016-01-17',
                                    '2016-01-24', '2016-02-07']),
            'lower_window': 0,
            'upper_window': 1,
        })
        superbowls = pd.DataFrame({
            'holiday': 'superbowl',
            'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
            'lower_window': 0,
            'upper_window': 1,
        })
        holidays = pd.concat((playoffs, superbowls))
        ```

    2. Input objek `holidays` sebagai parameter

        ```
        m = Prophet(holidays=holidays)
        forecast = m.fit(df).predict(future)
        ```

    3. Efek holiday akan tampil pada plot components menggunakan `plot_forecast_component()`

### Built-in holidays

- Menambahkan hari libur berdasarkan negara dengan method `add_country_holidays()`
    ```
    m = Prophet(holidays=holidays)
    m.add_country_holidays(country_name='US')
    m.fit(df)
    ```

- Cara melihat hari libur yang ditambahkan, dengan attribute `train_holiday_names`
    ```
    m.train_holiday_names
    ```

- Additional: secara manual melihat list holiday di Indonesia
    ```
    # https://github.com/facebook/prophet/blob/master/python/fbprophet/hdays.py
    from fbprophet import hdays
    holidays_indo = hdays.Indonesia()
    holidays_indo._populate(2021)
    pd.DataFrame([holidays_indo], index=['holiday']).T.rename_axis('ds').reset_index()
    ```

### Fourier Order for Seasonalities

- Seasonality diestimasi menggunakan Fourier Series, yaitu pendekatan untuk merepresentasikan fungsi periodik sebagai penjumlahan dari fungsi sinus dan cosinus.

- Order pada Fourier series dapat diatur untuk masing-masing seasonality. Secara default, order `yearly_seasonality=10` dan `weekly_seasonality=3`. Apabila ingin diubah, contohnya:
    ```
    m = Prophet(yearly_seasonality=20).fit(df)
    ```

- Semakin besar order pada Fourier Series, maka semakin kompleks fungsi yang diestimasi, sehingga semakin rentan overfitting.

### Specifying Custom Seasonalities

- Secara default, Prophet mencari weekly dan yearly seasonalities. Kita dapat menambahkan seasonality sendiri sesuai kebutuhan dengan method `add_seasonality()`. Parameter:
    - `name`: nama dari seasonality
    - `period`: banyaknya data sehingga dianggap sebagai satu periode
    - `fourier_order`: order pada Fourier Series (lihat bagian sebelumnya)

    ```
    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    ```

### Seasonalities that depend on other factors

- Secara default, Prophet mengasumsikan seasonality untuk seluruh periodenya sama.

- Pada contoh berikut, misal weekly seasonality pada NFL season dengan non-NFL season berbeda
    1. Buat kondisi (True or False) pada data
    ```
    def is_nfl_season(ds):
        date = pd.to_datetime(ds)
        return (date.month > 8 or date.month < 2)

    df['on_season'] = df['ds'].apply(is_nfl_season)
    df['off_season'] = ~df['ds'].apply(is_nfl_season)
    ```

    2. Menambahkan parameter `condition_name` sesuai nama kolom yang didefinisikan sebelumnya
    ```
    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
    m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')
    ```

    3. Hasilnya, seasonality pada on dengan off season berbeda

### Prior scale for holidays and seasonality

- Terdapat parameter tambahan untuk mengatur seberapa kuat efek holiday dan seasonalitynya pada model:
    - `holidays_prior_scale` (default=10): smoothing parameter untuk efek holiday
    - `seasonality_prior_scale` (default=10): smoothing parameter untuk seluruh efek seasonal

- Apabila kekuatan tiap seasonality ingin diatur secara terpisah, maka tambahkan parameter `prior_scale` dalam method `add_seasonality()`:
    ```
    m = Prophet()
    m.add_seasonality(
        name='weekly', period=7, fourier_order=3, prior_scale=0.1)
    ```

### Additional regressors

- Menambahkan efek regressor (predictor) pada model, menggunakan method `add_regressor()`
- Nilai extra regressor harus sudah diketahui untuk data historis maupun data future. Sehingga masing-masing regressor harus dilakukan forecasting secara terpisah terlebih dahulu.
- Hati-hati, dengan cara seperti ini error pada forecasting regressor akan berkontribusi terhadap error hasil forecasting nilai `y`
- Referensi: https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb#we-first-instantiates-a-new-fbprophet-model,-using-the-exact-same-prior-scales-and-parameters-as-before

## Multiplicative Seasonality

- Secara default, Prophet fit additive seasonality:
$y = Trend + Seasonal$

- Ketika seasonalnya "grows" dengan trend, maka kita dapat set `seasonality_mode='multiplicative'` pada `Prophet()`
    ```
    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df)
    ```
    Note: pada plot components, seasonality akan berupa persentase

- Terkadang tidak semua komponen (seasonality, regressor, atau holiday effect) berupa multiplicative. Kita dapat set parameter `mode='additive'` atau `mode='multiplicative'` untuk masing-masing komponen
    ```
    m = Prophet(seasonality_mode='multiplicative')
    m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
    m.add_regressor('regressor', mode='additive')
    ```
    Note: seasonality biasanya hanya 1 tipe: additive semua atau multiplicative semua, kecuali kalau ada alasan yang kuat untuk menggunakan mode yang berbeda

## Uncertainty Intervals

- Secara default, Prophet mengembalikan uncertainty interval [`yhat_lower`, `yhat_upper`] yang bersumber dari:
    - Uncertainty trend
    - Uncertainty seasonality
    - Noise / error

### Uncertainty in the trend

- Asumsi: nilai future akan memiliki perubahan trend yang sama dengan data historis

- Dengan meningkatkan `changepoint_prior_scale`, maka akan meningkatkan uncertainty pada nilai forecast. Hal ini karena model akan lebih memperhitungkan fluktuasi trend pada data historis, yang membuat model lebih tidak pasti dalam memprediksi nilai.

- Lebar uncertainty interval dapat diatur melalui parameter `interval_width`. Default = 0.8
    ```
    forecast = Prophet(interval_width=0.95).fit(df).predict(future)
    ```
    - Semakin besar nilai `interval_width`, maka interval akan semakin lebar

### Uncertainty in seasonality

- Secara default, tidak ada uncertainty interval pada seasonality.

- Agar ada uncertainty pada seasonality, harus dilakukan Bayesian sampling dengan menyertakan parameter `mcmc.samples` (Markov Chain Monte Carlo). Semakin besar nilainya, semakin lama komputasi, namun uncertainty inteval semakin baik.

## Outliers

- Problem outliers pada data TS:
    - Membuat uncertainty interval pada forecast semakin lebar
    - Niali Trend dan Seasonality akan tertarik ke arah outlier (high influence outlier)

- Handle outlier: remove secara manual, yaitu jadikan `None` pada nilai `y` dengan conditional subsetting. Hal ini karena Prophet tidak masalah dengan missing data.
    ```
    df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
    model = Prophet().fit(df)
    ```

## Non-Daily Data

### Sub-daily Data

- Ketika data yang dimiliki adalah sub-daily (hourly, minutely, secondly) dengan format YYYY-MM-DD HH:MM:SS, maka Prophet otomatis mencari daily seasonality.

- Pastikan gunakan parameter `freq` saat menggunakan method `make_future_dataframe()`

    ```
    future = m.make_future_dataframe(periods=300, freq='H')
    ```

### Data with regular gaps

- Misal data time series kita memiliki gap, kasus:
    1. Restoran hanya buka jam 9 pagi sampai 9 malam (gap 12 jam)
    2. Perkantoran hanya buka Senin-Jumat (gap 2 hari)

- Problem: nilai forecast pada waktu di gap akan buruk karena tidak ada di data historis.

- Solusi: pada dataframe `future` harus dilakukan conditional subsetting terhadap waktu yang ada di data historis saja. Misal kasus 1: subset waktu antara jam 9 - 21, kasus 2: subset hari antara Senin-Jumat saja.

### Monthly data

- Secara default, Prophet akan forecast daily data. Agar hasil forecast hanya pada frekuensi bulanan, maka atur parameter `freq` di method `make_future_dataframe`
    ```
    future = m.make_future_dataframe(periods=120, freq='MS')
    ```

- Referensi TS offset: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

- Untuk data bulanan, yearly seasonality dapat dimodelkan melalui extra regressor: `is_jan`, `is_feb`, dll dan menyertakan `yearly_seasonality=False`

### Holidays with aggregated data

- Efek holiday pada data mingguan/bulanan akan diabaikan apabila tanggal holiday tidak jatuh pada data. Misal: data kita mingguan di hari Minggu, maka holiday di hari Senin akan diabaikan

- Untuk menyertakan efek holiday, tanggal harus digeser secara manual ke hari (atau bulan) yang berdekatan

## Diagnostics

![](https://facebook.github.io/prophet/static/diagnostics_files/diagnostics_3_0.png)

- Fitur cross validation untuk menghitung error dengan method `cross_validation()`. Parameter:
    - `horizon`: banyak data setelah cut-off yang akan diforecast, sebagai data test
    - `initial`: banyak data yang digunakan untuk training, pada awal forecast (default=3*`horizon`)
    - `period`: jeda antara cut-off, yaitu tanggal pisah data train dan test (default=0.5*`horizon`)

    ```
    from fbprophet.diagnostics import cross_validation
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
    ```

- Cut-off dapat disertakan secara manual
    ```
    cutoffs = pd.to_datetime(['2013-02-15', '2013-08-15', '2014-02-15'])
    df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='365 days')
    ```

- Untuk mengukur error, gunakan method `performance_metrics()`. Parameter:
    - `rolling_window` (default=0.1): persentase nilai forecasting yang digunakan untuk menghitung error di tiap horizon nya
        - `rolling_window=0` maka tiap horizon akan memiliki nilai error
        - `rolling_window=1` maka akan memunculkan nilai error secara keseluruhan
    ```
    from fbprophet.diagnostics import performance_metrics
    df_p = performance_metrics(df_cv)
    ```

- Jenis error (selisih `yhat` dan `y`)
    - MSE: Mean Squared Error; error kuadrat yang dirata-ratakan
    - RMSE: Root Mean Squared Error; akar dari MSE
    - MAE: Mean Absolute Error; error absolute yang dirata-ratakan
    - MAPE: Mean Absolute Percentage Error; persentase error absolute yang dirata-ratakan
    - MDAPE: Median Absolute Percentage Error; persentase error absolute yang dimediankan
    - Coverage: Persentase hasil prediksi `y` yang masuk ke dalam interval [`yhat_lower`, `yhat_upper`]

- Visualisasi error dengan method `plot_cross_validation_metric()`. Terdapat parameter `rolling_window` sama persis dengan method `performance_metrics()`
    ```
    from fbprophet.plot import plot_cross_validation_metric
    fig = plot_cross_validation_metric(df_cv, metric='mape')
    ```
    - Scatter plot: error tiap observasi untuk tiap horizon
    - Garis biru: rolling window (smoothing) dari scatter plot

### Parallelizing cross validation

- Untuk komputasi paralel saat cross validation
    - `parallel=None` (default)
    - `parallel="processes"` (rekomendasi untuk small data)
    - `parallel="threads"`
    - `parallel="dask"` (rekomendasi untuk big data, namun perlu install package Dask cluster)

### Hyperparameter tuning

- Cara hyperparameter tuning di Prophet: for-looping untuk Grid search. Langkah:
    1. Membuat `param_grid` untuk kemungkinan nilai-nilai argument
    2. Generate kombinasi model pada `all_params`
    3. Fitting model dalam for-looping
    4. Cross validation, menampung nilai error yang ingin dilihat. Rekomendasi `rolling_window=1`
    5. Mencari model dengan nilai error terkecil

- Rekomendasi parameter yang dituning:
    - `changepoint_prior_scale` [0.001, 0.5], default = 0.05
    - `seasonality_prior_scale` [0.01, 10], default = 10
    - `holidays_prior_scale` [0.01, 10], default = 10
    - `seasonality_mode` ['additive', 'multiplicative'], default = 'additive'

- Mungkin yang dapat dituning:
    - `changepoint_range` [0.8, 0.95], default = 0.8

- Parameter yang tidak direkomendasikan untuk dituning:
    - `growth`: linear atau logistic tergantung business
    - `changepoints`: lokasi tergantung exploratory
    - `n_changepoints`, penggantinya `changepoint_prior_scale`
    - `yearly_seasonality`, `weekly_seasonality`, `daily_seasonality`
    - `holidays`
    - `mcmc_samples`
    - `interval_width`: hanya berefek ke uncertainty interval
    - `uncertainty_samples`
    - `stan_backend`

## Additional Topics

### Saving models

- Di Python, menyimpan model **tidak boleh** ke pickle ataupun joblib, melainkan simpan ke file json dengan method `model_to_json` dan load dengan `model_from_json`

    ```
    import json
    from fbprophet.serialize import model_to_json, model_from_json

    with open('serialized_model.json', 'w') as fout:
        json.dump(model_to_json(m), fout)  # Save model

    with open('serialized_model.json', 'r') as fin:
        m = model_from_json(json.load(fin))  # Load model
    ```

### Flat trend and custom trends

- Untuk time series yang hanya ada pola seasonal, tidak ada trend, gunakan `growth='flat'`

- Custom trends: https://github.com/facebook/prophet/pull/1466/file

### Updating fitted models

- Prophet hanya dapat di fit sekali, model harus ditrain ulang ketika ada data baru. Tidak menjadi sebuah masalah karena fitting model Prophet cukup cepat.

- Untuk mempercepat refitting model, bisa menggunakan warm start pada parameter `init`. Namun cara ini tidak direkomendasi apabila penambahan data sangat banyak.