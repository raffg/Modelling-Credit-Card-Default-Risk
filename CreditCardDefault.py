# -*- coding: utf-8 -*-
"""
This program evaluates the risk of default for credit card loan applicants. It 
uses logistic regression to determine model coefficients and then feeds those 
coefficients into a binary classification model. The binary classification 
model has also bee built in Excel and can be viewed here: https://goo.gl/PNTDd8
"""

import sys
import os
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

import HelperFunctions as hf

sys.path.append(os.path.join(sys.path[0], 'Credit Card Default'))

training_raw = pd.read_csv('training.csv', low_memory=False)
#training_raw = pd.read_csv('data.csv', low_memory=False)
test_raw = pd.read_csv('test.csv', low_memory=False)

cost_per_false_positive = 2500
cost_per_false_negative = 5000

training_raw['outcomes_rev'] = (training_raw.outcomes != 1).astype('int')
test_raw['outcomes_rev'] = (test_raw.outcomes != 1).astype('int')

training_raw['debt_to_income'] = (training_raw['credit_card_debt'] + 
                              training_raw['automobile_debt']) / training_raw['income']
test_raw['debt_to_income'] = (test_raw['credit_card_debt'] + 
                              test_raw['automobile_debt']) / test_raw['income']

training_square = pd.DataFrame()
test_square = pd.DataFrame()

cols_to_square = ['age','years_employment','years_at_address','income',
                  'credit_card_debt','automobile_debt','debt_to_income']

training_square[cols_to_square] = training_raw[cols_to_square].apply(lambda x: x**2)
test_square[cols_to_square] = test_raw[cols_to_square].apply(lambda x: x**2)

training_square.rename(columns={'age':'age2','years_employment':'years_employment2',
                                'years_at_address':'years_at_address2','income':'income2',
                                'credit_card_debt':'credit_card_debt2',
                                'automobile_debt':'automobile_debt2',
                                'debt_to_income':'debt_to_income2'}, inplace=True)
test_square.rename(columns={'age':'age2','years_employment':'years_employment2',
                            'years_at_address':'years_at_address2','income':'income2',
                            'credit_card_debt':'credit_card_debt2',
                            'automobile_debt':'automobile_debt2',
                            'debt_to_income':'debt_to_income2'}, inplace=True)
training_raw = pd.concat([training_raw, training_square], axis=1)
test_raw = pd.concat([test_raw, test_square], axis=1)

training_norm = pd.DataFrame()
test_norm = pd.DataFrame()

cols_to_norm = ['age','years_employment','years_at_address','income',
                'credit_card_debt','automobile_debt', 'debt_to_income', 
                'age2','years_employment2','years_at_address2','income2',
                'credit_card_debt2','automobile_debt2', 'debt_to_income2']
training_norm[cols_to_norm] = training_raw[cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
test_norm[cols_to_norm] = test_raw[cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))

training_norm.rename(columns={'age':'age_norm','years_employment':'years_employment_norm',
                'years_at_address':'years_at_address_norm','income':'income_norm',
                'credit_card_debt':'credit_card_debt_norm',
                'automobile_debt':'automobile_debt_norm',
                'debt_to_income':'debt_to_income_norm', 
                'age2':'age2_norm','years_employment2':'years_employment2_norm',
                'years_at_address2':'years_at_address2_norm','income2':'income2_norm',
                'credit_card_debt2':'credit_card_debt2_norm',
                'automobile_debt2':'automobile_debt2_norm',
                'debt_to_income2':'debt_to_income2_norm'}, inplace=True)
test_norm.rename(columns={'age':'age_norm','years_employment':'years_employment_norm',
                'years_at_address':'years_at_address_norm','income':'income_norm',
                'credit_card_debt':'credit_card_debt_norm',
                'automobile_debt':'automobile_debt_norm',
                'debt_to_income':'debt_to_income_norm', 
                'age2':'age2_norm','years_employment2':'years_employment2_norm',
                'years_at_address2':'years_at_address2_norm','income2':'income2_norm',
                'credit_card_debt2':'credit_card_debt2_norm',
                'automobile_debt2':'automobile_debt2_norm',
                'debt_to_income2':'debt_to_income2_norm'}, inplace=True)

training_raw = pd.concat([training_raw, training_norm], axis=1)
test_raw = pd.concat([test_raw, test_norm], axis=1)

training_raw['intercept'] = 1
test_raw['intercept'] = 1

def run_model(features, cost_per_false_positive=None, cost_per_false_negative=None):
    training = training_raw.copy()
    test = test_raw.copy()
    
    y_values = 'outcomes ~ ' + ' + '.join(features)

    features.insert(0, 'intercept')
    
    y, X = dmatrices(y_values,training,return_type='dataframe')
    
    model = LogisticRegression(fit_intercept = False).fit(X,y.values.ravel())
    
    model_scores = hf.scores(training[features],model.coef_[0])

    model_scores['outcomes'] = training['outcomes']
    model_scores['outcomes_rev'] = training['outcomes_rev']
    
    total_outcomes, total_outcomes_1, total_outcomes_0, condition_incidence = hf.calc_condition_incidence(training)
    
    model_thresholds = hf.calc_thresholds(model_scores, 200)
    
    TP_model_scores = hf.true_positive_analysis(model_scores, model_thresholds)
    FP_model_scores = hf.false_positive_analysis(model_scores, model_thresholds)
    
    autc_matrix, autc = hf.area_under_the_curve(TP_model_scores, FP_model_scores,model_thresholds, 
                            total_outcomes, total_outcomes_1, total_outcomes_0)
    
    min_cost_per_event = None
    
    if cost_per_false_positive != None and cost_per_false_negative != None:
        min_cost_per_event, min_threshold, min_cost_matrix, TP_rate, FP_rate = hf.minimum_cost_per_event(autc_matrix, 
                              cost_per_false_positive, cost_per_false_negative)
        
        prob_TP = TP_rate * condition_incidence
        prob_FP = FP_rate * condition_incidence
        classification_incidence = prob_TP + prob_FP
    else:
        min_cost_per_event = None
        min_threshold = None
        TP_rate = None
        FP_rate = None
        prob_TP = None
        prob_FP = None
        classification_incidence = None
    
    test_scores = hf.scores(test[features],model.coef_[0])
    test_scores['outcomes'] = test['outcomes']
    test_scores['outcomes_rev'] = test['outcomes_rev']
    test_outcomes, test_outcomes_1, test_outcomes_0, test_condition_incidence = hf.calc_condition_incidence(training)
    
    TP_test_scores = hf.true_positive_analysis(test_scores, model_thresholds)
    FP_test_scores = hf.false_positive_analysis(test_scores, model_thresholds)
    test_autc_matrix, test_autc = hf.area_under_the_curve(TP_test_scores, FP_test_scores, model_thresholds, 
                            test_outcomes, test_outcomes_1, test_outcomes_0)
    min_cost_per_event_test = None
    if cost_per_false_positive != None and cost_per_false_negative != None:
        a,b,cost_per_event_matrix, test_TP_rate, test_FP_rate = hf.minimum_cost_per_event(test_autc_matrix, cost_per_false_positive, cost_per_false_negative)
        min_cost_per_event_test = cost_per_event_matrix.get_value(min_threshold, 'Cost Per Event')
        test_prob_TP = test_TP_rate * test_condition_incidence
        test_prob_FP = test_FP_rate * test_condition_incidence
        test_classification_incidence = test_prob_TP + test_prob_FP
    else:
        cost_per_event_matrix = None
        test_TP_rate = None
        test_FP_rate = None
        min_cost_per_event_test = None
        test_prob_TP = None
        test_prob_FP = None
        test_classification_incidence = None
    
    return (features, model.coef_[0], autc, test_autc, min_threshold, min_cost_per_event, 
            min_cost_per_event_test, condition_incidence, test_condition_incidence, 
            prob_TP, test_prob_TP, classification_incidence, test_classification_incidence)




features1 = ['age','years_employment','years_at_address','income',
             'credit_card_debt','automobile_debt']
model1 = run_model(features1, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 1===========')
hf.print_results(model1)

features2 = ['age','years_employment','years_at_address','debt_to_income']
model2 = run_model(features2, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 2===========')
hf.print_results(model2)

features3 = ['age','years_employment','years_at_address','income',
             'credit_card_debt','automobile_debt', 'debt_to_income']
model3 = run_model(features3, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 3===========')
hf.print_results(model3)

features4 = ['age','age2','years_employment','years_employment2',
             'years_at_address','years_at_address2','income','income2',
             'credit_card_debt','credit_card_debt2','automobile_debt',
             'automobile_debt2','debt_to_income','debt_to_income2']
model4 = run_model(features4, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 4===========')
hf.print_results(model4)

features5 = ['age','age2','years_employment','years_employment2',
             'years_at_address','years_at_address2','debt_to_income','debt_to_income2']
model5 = run_model(features5, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 5===========')
hf.print_results(model5)

features6 = ['age_norm','years_employment_norm','years_at_address_norm','income_norm',
             'credit_card_debt_norm','automobile_debt_norm']
model6 = run_model(features6, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 6===========')
hf.print_results(model6)

features7 = ['age_norm','years_employment_norm','years_at_address_norm','debt_to_income']
model7 = run_model(features7, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 7===========')
hf.print_results(model7)

features8 = ['age_norm','years_employment_norm','years_at_address_norm','debt_to_income_norm']
model8 = run_model(features8, cost_per_false_positive, cost_per_false_negative)
print ('===========Model 8===========')
hf.print_results(model8)
