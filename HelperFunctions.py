'''
This file contains helper functions for CreditCardDefault.py
'''

import pandas as pd
import numpy as np


def calc_thresholds(df, num_thresholds):
    '''
    helper function to calculate the threshold values
    '''
    threshold_step = (df['scores'].max() - df['scores'].min()) / (num_thresholds - 1)
    threshold = df['scores'].min()
    thresholds = [threshold]
    while threshold < df['scores'].max():
        threshold += threshold_step
        thresholds.append(threshold)
    return list(reversed(thresholds))


def scores(df, coefficients):
    '''
    helper function to multiply a DataFrame by coefficients
    '''
    scores = (df * coefficients).sum(axis=1)
    scores = pd.DataFrame(scores)
    scores.columns = ['scores']
    
    return scores


def calc_condition_incidence(df):
    total_outcomes = len(df)
    total_outcomes_1 = len(df[df['outcomes'] == 1])
    total_outcomes_0 = total_outcomes - total_outcomes_1
    
    condition_incidence = total_outcomes_1 / total_outcomes
    
    return (total_outcomes, total_outcomes_1, total_outcomes_0, condition_incidence)


def true_positive_analysis(df, thresholds):
    '''
    helper function to calculate the true positives at each threshold
    '''
    df = df.copy()
    for threshold in thresholds:
        df[str(threshold)] = df[['scores','outcomes']].apply(lambda x: 
            x['outcomes'] if x['scores']>threshold else 0, axis=1)

    return df
    
    
def true_positive_analysis2(df, thresholds):
    df = df.copy()
    thresholds_col = ['{:.16f}'.format(e) for e in thresholds]
    data = df.outcomes[:,np.newaxis] * ((df.scores[:,np.newaxis] - thresholds > 0))
    df = df.join(pd.DataFrame(data=data, columns=thresholds_col))
    
    return df


def false_positive_analysis(df, thresholds):
    '''
    helper function to calculate the false positives at each threshold
    '''
    df = df.copy()
    for threshold in thresholds:
        df[str(threshold)] = df[['scores','outcomes_rev']].apply(lambda x: 
            x['outcomes_rev'] if x['scores']>threshold else 0, axis=1)

    return df


def false_positive_analysis2(df, thresholds):
    '''
    helper function to calculate the false positives at each threshold
    '''
    df = df.copy()
    thresholds_col = ['{:.16f}'.format(e) for e in thresholds]
    data = df.outcomes_rev[:,np.newaxis] * ((df.scores[:,np.newaxis] - thresholds > 0))
    df = df.join(pd.DataFrame(data=data, columns = thresholds_col))
    
    return df
    
    
def area_under_the_curve(df_positive, df_negative, thresholds, total_outcomes, total_outcomes_1, total_outcomes_0):
    '''
    helper function to calculate the area under the receiver operating 
    characteristic (ROC) curve and true/false positive rates
    '''
    
    autc = pd.DataFrame(index=('TPs at threshold','TP rate','FPs at threshold',
             'FP rate','Area of Rectangle','FNs at threshold','Cost Per Event'))
    tp_rate_prev = 0
    fp_rate_prev = 0

    for threshold in thresholds:
        threshold = str(threshold)
        true_positives = df_positive[threshold].sum(axis=0)
        autc.set_value('TPs at threshold', threshold, true_positives)
        tp_rate = true_positives / total_outcomes_1
        autc.set_value('TP rate', threshold, tp_rate)
        false_positives = df_negative[threshold].sum(axis=0)
        autc.set_value('FPs at threshold', threshold, false_positives)
        fp_rate = false_positives / total_outcomes_0
        autc.set_value('FP rate', threshold, fp_rate)
        if threshold != thresholds[0]:
            area_of_rectangle = ((tp_rate + tp_rate_prev) * 
                                (fp_rate - fp_rate_prev)) / 2
        tp_rate_prev = tp_rate
        fp_rate_prev = fp_rate
        autc.set_value('Area of Rectangle', threshold, area_of_rectangle)
        autc.set_value('FNs at threshold', threshold, total_outcomes_1 - true_positives)
        total_area = autc.sum(axis=1)['Area of Rectangle']
        if total_area < .5:
            total_area = 1 - total_area
        
    return autc, total_area


def minimum_cost_per_event(autc, cost_per_false_positive, cost_per_false_negative):
    '''
    helper function to calculate the minimum cost per event for a given matrix
    '''

    total_outcomes = (len(autc.columns))
    autc = autc.T
    autc['Cost Per Event'] = (cost_per_false_negative * autc['FNs at threshold'] + 
        cost_per_false_positive * autc['FPs at threshold']) / total_outcomes
    min_cost_per_event = autc['Cost Per Event'].min(axis=0)
    min_threshold = autc['Cost Per Event'].idxmin(axis=0)
    TP_rate = autc.get_value(min_threshold, 'TP rate')
    FP_rate = autc.get_value(min_threshold, 'FP rate')
    
    return min_cost_per_event, min_threshold, autc, TP_rate, FP_rate


def print_results(model):
    print ('|=================================================================|')
    print ('|                   Feature     Coefficient                       |')
    print ('|                   -------     -----------                       |')
    for idx in range(len(model[0])):
        print ('|', '{:>25}'.format(model[0][idx]), ' : ', '{:<25}'.format(
                model[1][idx]), '        |')
    print ('|                                                                 |')
    print ('|                                   Training       Test           |')
    print ('|         Area Under the ROC Curve:', '{:<10}'.format(
            round(float(model[2]), 5)), '   ', '{:<10}'.format(
                    round(float(model[3]), 5)), '    |')
    if model[4] != None:
        print ('|           Minimum Cost Per Event:', '{:<10}'.format(
                round(float(model[5]), 5)), '   ', '{:<10}'.format
            (round(float(model[6]), 5)), '    |')
        print ('|                     at threshold:', '{:<20}'.format
               (round(float(model[4]), 10)), '         |')
        print ('|              Condition Incidence:', '{:<10}'.format(
                round(float(model[7]), 5)), '   ', '{:<10}'.format(
                        round(float(model[8]), 5)), '    |')
        print ('|    Probability of True Positives:', '{:<10}'.format(
                round(float(model[9]), 5)), '   ', '{:<10}'.format(
                        round(float(model[10]), 5)), '    |')
        print ('|  Test (Classification) Indidence:', '{:<10}'.format
               (round(float(model[11]), 5)), '   ', '{:<10}'.format(
                       round(float(model[12]), 5)), '    |')
    print ('|=================================================================|')
    print ('')
    
    return


def confusion_matrix(condition_incidence, probability_of_true_positive, classification_incidence):
    a = condition_incidence
    b = 1 - a
    c = classification_incidence
    d = 1 - c
    e = probability_of_true_positive
    f = a - e
    g = c - e
    h = b - g
    
    return (a, b, c, d, e, f, g, h)
