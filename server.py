import os
import flask
from flask import Flask, request, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import copy
import math
import joblib
import itertools
from matplotlib import pyplot as plt
from scipy import ndimage
#from umap import UMAP
from sklearn.decomposition import PCA
from collections import Counter, OrderedDict

import shap

from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# make_regression
X, y = make_regression(n_samples=500, n_features=12, noise=1, random_state=42)


from file_functions import verifyDir, get_current_path

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pca = PCA(n_components=2)
pca.fit(X_train)
explainer = shap

absolute_current_path = get_current_path()

# create Flask app
verifyDir(absolute_current_path+'/static')
app = Flask(__name__)#, static_folder=absolute_current_path+'/static')
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)#, origins="*")


'''
Do not cache images on browser, see: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
'''
@app.after_request
def add_header(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def get_train_test(X, y):
    return X_train, X_test, y_train, y_test

def get_2D_projection(X):
    return pca.transform(X)

@app.route('/pca/', methods=['GET'])
def get_projection():
    xx = get_2D_projection(X_test).tolist()
    #print(xx)
    ret = [{'id':i,'x':x[0],'y':x[1]} for i,x in enumerate(xx)]

    return flask.jsonify(ret)

@app.route('/explanation/', methods=['GET','POST'])
def get_explanation():
    selected_ids = None
    if request.method == 'POST':
        try:
           selected_ids = request.get_json()['selected_ids']
           if(len(selected_ids) == 0):
               selected_ids=None
        except Exception as e:
            print("ERROR", e)
    
    if selected_ids is not None:
        shap_values = explainer.shap_values(X_test[selected_ids])
    else:
        shap_values = explainer.shap_values(X_test)
    
    total_importance_per_feature = np.mean(shap_values, axis=0)
    
    ret = [ {"feature_name": f"x{i}", "importance": value}  for i, value in enumerate(total_importance_per_feature.tolist())]
        
    return flask.jsonify(ret)


@app.route('/data/', methods=['GET','POST'])
def get_data():
    return flask.jsonify({"X_train": X_train.tolist(), "X_test": X_test.tolist(), 
                          "y_train": y_train.tolist(), "y_test": y_test.tolist()})

'''
In the main, before running the server, run clustering, store results in 
variables a2_clustering, a3_clustering, a4_clustering
'''
if __name__=='__main__':

    # split train test
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    # calculat eprojection
    projection = get_2D_projection(X_test)
    
    # train your model
    clf = Ridge(alpha=1.0)
    clf.fit(X_test, y_test)
    
    # instance your explainer
    explainer = shap.LinearExplainer(clf, X_test)
    
    print("Initializing server ...")
    app.run(port=8080, debug=True, use_reloader=True, threaded=True)

