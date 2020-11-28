#library ini berguna untuk train dan test data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#numpy digunakan untuk mengelola data numeric
import numpy as np
#inisialisasi library yang digunakan
import pandas as pd    #pandas digunakan untuk membaca dataset
from flask import render_template #flask digunakan untuk menampilkan hasil output kedalam web
from flask import Flask, flash , redirect, request
import pygal #library pygal digunakan untuk membuat radar chart
import cairo #library ini digunakan untuk konversi ke svg lalu ditampilkan dalam flask
#menghilangkan error chained assigment untuk IPython kernel
pd.options.mode.chained_assignment = None
app = Flask(__name__)

#app route ini berguna untuk menentukan url halaman yang ada isinya / atau dapat di akses / dituju
@app.route("/")
#parameter yang di dalamnya berguna untuk men-get data dari website
@app.route('/index', methods=['GET', 'POST'])
#inisialisasi function index akan menampung code-code yang akan digunakan dalam web
def index():

    # Membandingkan 2 tim yang akan bertanding
    select = request.form.get('comp_select')
    select2 = request.form.get('comp2_select')

    #read data dari dataset
    base = pd.read_csv("baseball.csv")
    #inisialisasi variable tim
    team1 = select or 'ARI'
    team2 = select2 or 'ATL'
    #mengambil data set dengan key tertentu
    dpyears_team1 = base[base.Team == team1]
    dpyears_team2 = base[base.Team == team2]
    OBP_team1 = dpyears_team1['OBP'].mean()
    SLG_team1 = dpyears_team1['SLG'].mean()
    BA_team1 = dpyears_team1['BA'].mean()
    OOBP_team1 = dpyears_team1['OOBP'].mean()
    OSLG_team1 = dpyears_team1['OSLG'].mean()
    OBP_team2 = dpyears_team2['OBP'].mean()
    SLG_team2 = dpyears_team2['SLG'].mean()
    BA_team2 = dpyears_team2['BA'].mean()
    OOBP_team2 = dpyears_team2['OOBP'].mean()
    OSLG_team2 = dpyears_team2['OSLG'].mean()

    #menempatkan data tim dalam list yang digunakan untuk melakukan prediksi
    team1_data = [OBP_team1, SLG_team1, BA_team1, OOBP_team1, OSLG_team1]
    team2_data = [OBP_team2, SLG_team2, BA_team2, OOBP_team2, OSLG_team2]

    #digunakan untuk mengambil dataset dengan tahun dibawah 2002
    runs_team1 = dpyears_team1[dpyears_team1.Year < 2002]
    runs_team2 = dpyears_team2[dpyears_team2.Year < 2002]

    #memperoleh RD dari tim 1
    runs_team1['RD'] = runs_team1['RS'] - runs_team1['RA']

    #melakukan split untuk train dan test data tim 1
    X_team1 = np.array(runs_team1['RD']).reshape(-1, 1)
    y_team1 = np.array(runs_team1['W']).reshape(-1, 1)
    X_train_team1, X_test_team1, y_train_team1, y_test_team1 = train_test_split(X_team1, y_team1, test_size=0.25)

    #digunakan untuk mencek korelasi/keterhubungan antara 2 variable (RD dan W) tim 1
    regr_team1 = LinearRegression()
    regr_team1.fit(X_train_team1, y_train_team1)

    #memperoleh nilai prediksi kemenangan pada 1 musim(tim 1)
    win_team1_predict = round(regr_team1.predict(X_test_team1).mean())
    label1 = "{} pred win:{}".format(team1, win_team1_predict)

    #menghitung rd dari tim 2
    runs_team2['RD'] = runs_team2['RS'] - runs_team2['RA']

    #melakukan split untuk train dan test data tim 2
    X_team2 = np.array(runs_team2['RD']).reshape(-1, 1)
    y_team2 = np.array(runs_team2['W']).reshape(-1, 1)
    X_train_team2, X_test_team2, y_train_team2, y_test_team2 = train_test_split(X_team2, y_team2, test_size=0.25)

    # digunakan untuk mencek korelasi/keterhubungan antara 2 variable (RD dan W) tim 2
    regr_team2 = LinearRegression()
    regr_team2.fit(X_train_team2, y_train_team2)

    # memperoleh nilai prediksi kemenangan pada 1 musim (tim 2)
    win_team2_predict = round(regr_team2.predict(X_test_team2).mean())
    label2 = "{} pred win:{}".format(team2, win_team2_predict)

    #mengatur element yang terdapar pada radar chart
    radar_chart = pygal.Radar(width=400, height=300)
    radar_chart.title = 'Compare Team'
    radar_chart.x_labels = ['OBP', 'SLG', 'BA', 'OOBP', 'OSLG']
    #membentuk radar chat , dengan membandingkan 2 team yaitu *pemenang dan tim tebakan user
    radar_chart.add(label1, team1_data)
    radar_chart.add(label2, team2_data)
    radar_chart = radar_chart.render_data_uri()

    #merender html untuk ditampilkan di dalam web
    #variable temp untuk get data lalu di lempar pada html
    #variable data untuk mengisi dropdown pilihan tim
    return render_template('index.html',chart = radar_chart ,
                           data=[{'name':'ARI'},{'name':'ATL'},{'name':'BAL'},{'name':'BOS'},{'name':'CHC'},{'name':'CHW'}
                                 ,{'name':'CIN'},{'name':'CLE'},{'name':'COL'},{'name':'DET'},{'name':'HOU'},{'name':'KCR'}
                                 ,{'name':'LAA'},{'name':'LAD'},{'name':'MIA'},{'name':'MIL'},{'name':'MIN'},{'name':'NYM'}
                                 ,{'name':'NYY'},{'name':'OAK'},{'name':'PHI'},{'name':'PIT'},{'name':'SDP'},{'name':'SEA'}
                                 ,{'name':'SFG'},{'name':'STL'},{'name':'TBR'},{'name':'TEX'},{'name':'TOR'},{'name':'WSN'}],
                           temp=str(select),temp2=str(select2))

#untuk menjalankan flask , naun fitur debug akan diset ke True
if __name__ == "__main__":
    app.run(debug=True)