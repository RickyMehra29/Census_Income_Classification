import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)


scalar = pickle.load(open("Census_Income_scaler.pkl", "rb"))
income_model = pickle.load(open("Census_Income_RFC.pkl", "rb"))

workclass = {'Private': 2, 'Self-emp-not-inc': 4, 'Local-gov': 1, 'State-gov': 5, 'Self-emp-inc': 3, 'Federal-gov': 0, 'Without-pay': 6}

education = {'HS-grad': 4, 'Some-college': 8, 'Bachelors': 2, 'School': 7, 'Masters': 5, 'Assoc-voc': 1, 'Assoc-acdm': 0, 
                'Prof-school': 6, 'Doctorate': 3}

marital_status = {'Married-civ-spouse': 2, 'Never-married': 4, 'Divorced': 0, 'Separated': 5, 'Widowed': 6, 
                    'Married-spouse-absent': 3, 'Married-AF-spouse': 1}

occupation = {'Prof-specialty': 9, 'Craft-repair': 2, 'Exec-managerial': 3, 'Adm-clerical': 0, 'Sales': 11, 'Other-service': 7, 
                'Machine-op-inspct': 6, 'Transport-moving': 13, 'Handlers-cleaners': 5, 'Farming-fishing': 4, 'Tech-support': 12, 
                'Protective-serv': 10, 'Priv-house-serv': 8, 'Armed-Forces': 1}

relationship =  {'Husband': 0, 'Not-in-family': 1, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5, 'Other-relative': 2}

race = {'White': 2, 'Black': 0, 'Other': 1}

sex = {'Male': 1, 'Female': 0}

def map_dict(encoder_dict, x):
    for key,value in encoder_dict.items():
        if key==x:
            return value

def validate_type(input_value):
    try:
        return int(input_value)
    except (ValueError, TypeError):
        return input_value

# Load the model
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = [validate_type(x) for x in request.form.values()]
    data[1] = map_dict(workclass,data[1])
    data[3] = map_dict(education,data[3])
    data[4] = map_dict(marital_status,data[4])
    data[5] = map_dict(occupation,data[5])
    data[6] = map_dict(relationship,data[6])
    data[7] = map_dict(race,data[7])
    data[8] = map_dict(sex,data[8])

    print(data)

    final_input = scalar.transform(np.array(data).reshape(1,-1))
    output = income_model.predict(final_input)[0]
    print(output)

    if output == 0:
       result = '<=50K'
    else:
        result = '>50K'    
    return render_template('home.html', output_text="The Income of the person is {}.".format(result))



if __name__ == '__main__':
    app.run(debug=True)
