from flask import Flask, render_template, request

from sklearn.externals import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def main():
   if request.method == 'GET':
      templateData = {
         'title' : 'Prever',
         'preco': '',
      }
      return render_template('index.html', **templateData)

   if request.method == 'POST':
      house = request.form.values()
      house = [float(i) for i in house]
      classifier = joblib.load('../ml_algorithm/classifier.pkl')
      answer = classifier.predict(np.asarray(house).reshape(1, -1))
      templateData = {
         'title' : 'Prever',
         'preco': 'Preço previsto para esta casa: $' + str(answer[0]),
      }
      return render_template('index.html', **templateData)

@app.route('/sobre/')
def sobre():
   templateData = {
      'title' : 'Sobre',
   }
   return render_template('sobre.html', **templateData)

@app.route('/comparacao/')
def comparacao():
   templateData = {
      'title' : 'Comparação',
   }
   return render_template('comparacao.html', **templateData)


if __name__ == "__main__":
   app.run()
