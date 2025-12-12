import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import preprocess
import sys

files = []
file = sys.argv[2]
horizon = sys.argv[1]
HORIZON = int(horizon)
files.append(file)
preprocess_class_instantiation = preprocess.preprocess(files, HORIZON)

datasets = preprocess_class_instantiation.generate_data()

file_for_given_row = {}
for index, crypto_name in enumerate(files):
    file_for_given_row[crypto_name] = index
    # if crypto_name == '1INCH.csv':
    # print(f'index : {index} -- crypto_name : {crypto_name}')

# for key, val in file_for_given_row.items():
#     print(key, val)

# print(file_for_given_row['1INCH.csv'])

# for f in files:
#     print(f"https://www.kaggle.com/datasets/svaningelgem/crypto-currencies-daily-prices?resource=download&select={f}")

# print(datasets)
##### NESTED LIST RETRIEVAL OF X FOR GIVEN CURRENCY ##################################################################################
# filename = '1INCH.csv'

def get_X_or_y_for_given_currency(currency, x_or_y):
        processed_file_dict = {}
        if (x_or_y == 'X') or (x_or_y == 'x'):
            x_or_y = 0
        elif (x_or_y == 'Y') or (x_or_y == 'y'):
            x_or_y = 1
        else:
            print(f'x_or_y needs to be x or y')
        row = file_for_given_row[currency]
        
        for csv_name, row in file_for_given_row.items():
            if currency == csv_name:
                processed_file_dict[csv_name] = datasets[row][x_or_y]
                return processed_file_dict[csv_name]
    
# get_X_or_y_for_given_currency(filename, 'X')
# get_X_or_y_for_given_currency(filename, 'y')
def the_currency(currency_we_want_to_look_at):
	

	X = get_X_or_y_for_given_currency(currency_we_want_to_look_at, 'X')
	y = get_X_or_y_for_given_currency(currency_we_want_to_look_at, 'y')

	return X, y

currency_we_want_to_look_at = '1INCH.csv'
# currency_we_want_to_look_at = 'AAVE.csv'
# currency_we_want_to_look_at = 'ADA.csv'
# currency_we_want_to_look_at = 'ALGO.csv'
# currency_we_want_to_look_at = 'AMP.csv'
X, y = the_currency(currency_we_want_to_look_at)

# print("first row of X:", X[0])
# print("first 10 closes from X[:,4]:", X[:10, 4])
# print("first 10 y values:", y[:10])
# # ## number of days since start | open | high | low | close | sma(10 days) | golden cross | death cross
# # ##          0                 |  1   |  2   |  3  |   4   |      5       |      6       |      7
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X0 = X[:, 0]

 
def get_prediction_at_given_x(pipeln, feature_indecies, prediction_points, ax):
    preds = []
 
    last_day = X[:, 0].max()
 
    for days_ahead in prediction_points:
        future_day = last_day + days_ahead
        
        last_row_features = X[-1, feature_indecies].copy()
        
        last_row_features[0] = future_day
        
        x_row = last_row_features.reshape(1, -1)
 
        y_pred = pipeln.predict(x_row)[0]
        preds.append(y_pred)
        
        ax.plot(future_day, y_pred, marker='x', color='purple', label=f'prediction at day {int(future_day)}: {y_pred:.4f}', markersize=25)
    return preds, future_day

def predictionVals_Plot_DictOfErrors(prediction_points):
    train_and_test_errors_at_given_degree = {}
    polynomial_degrees = [1,2,3,4,5,6,7,8]
    feature_indecies = [0,1,2,3,4,5,6,7]
    plt.figure(figsize=(20,10))
    
    alphas_to_test = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    for i in range(len(polynomial_degrees)):
        
        ### NOTE: make the line eq a mutlti-var lin eq
        polynomial_features_train = PolynomialFeatures(degree=polynomial_degrees[i], include_bias=False)
        
        ### NOTE: apply lin reg to the new mutlti-var lin eq
        linear_regression_model_train = RidgeCV(alphas=alphas_to_test)# ridge adds l2 to regression to shrink the coefficients of the loss funciton to prevent overfitting
        
        ### NOTE: instantiate the pipline
        pipeline_on_training_data = Pipeline([("polynomial_features", polynomial_features_train), ("linear_regression", linear_regression_model_train)])
        
        ### NOTE: apply the pipline
        
        ##spilt
        #                                                                                                                       
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X[:, feature_indecies], y, test_size=0.3, random_state=67, shuffle=True)

        ##fit 
        pipeline_on_training_data.fit(X_train2, y_train2)
        
        ##predict on train and test
        y_pred_train2 = pipeline_on_training_data.predict(X_train2)
        y_pred_test2 = pipeline_on_training_data.predict(X_test2)
        
        train_MAE2 = mean_absolute_error(y_train2, y_pred_train2)
        test_MAE2 = mean_absolute_error(y_test2, y_pred_test2)
        
        train_MSE2 = mean_squared_error(y_train2, y_pred_train2)
        test_MSE2 = mean_squared_error(y_test2, y_pred_test2)

        train_RMSE2 = np.sqrt(train_MSE2)
        test_RMSE2 = np.sqrt(test_MSE2)
        
        naive_MAE2 = mean_absolute_error(y_test2, np.full(y_test2.shape, np.mean(y_train2)))
        
        train_and_test_errors_at_given_degree[i] = {
            
            'train' : {
                'MAE' : train_MAE2,
                'MSE' : train_MSE2,
                'RMSE' : train_RMSE2
            },
            'test' : {
                'MAE' : test_MAE2,
                'MSE' : test_MSE2,
                'RMSE' : test_RMSE2
            },
            'naive_MAE' : naive_MAE2
        }
        
    def barchart_and_graph():
        rows = []
        for i, deg in enumerate(polynomial_degrees):
            m = train_and_test_errors_at_given_degree[i]
            rows.append({
                'deg' : deg,
                'train_RMSE' : m['train']['RMSE'],
                'test_RMSE' : m['test']['RMSE'],
                'train_MAE' : m['train']['MAE'],
                'test_MAE' : m['test']['MAE'],
                'naiveMAE' : m['naive_MAE']

            })
        # make a dataframe of the rows list using .sort_values by degree
        metrics_df = pd.DataFrame(rows).sort_values(['test_RMSE', 'test_MAE'], ascending=True)

        # print(metrics_df)
        # pick best degree (row) by the lowest test RMSE key then test MAE key then initialize the 
        best_row = metrics_df.sort_values(['test_RMSE', 'test_MAE'], ascending=True).iloc[0]  ## used AI to learn about the approach of the tie break (since RMSE punishes )
        # print(best_row)
        best_degree = int(best_row['deg'])
        # print(best_degree)
        
        poly_features_best = PolynomialFeatures(degree=best_degree, include_bias=False)
        linear_reg_best = RidgeCV(alphas=alphas_to_test)
        pipeline_best = Pipeline([("polynomial_features", poly_features_best), ("linear_regression", linear_reg_best)])
        
        # Re-split with same random_state to ensure consistency
        X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X[:, feature_indecies], y, test_size=0.3, random_state=67, shuffle=True)
        
        pipeline_best.fit(X_train_best, y_train_best)
    
        plt.figure(figsize=(20,10))
        
        ax = plt.subplot(1,1,1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(X_test_best[:,0], y_test_best, color="red", label='test')
        
        # LOBF_train_set = pipeline_on_training_data.predict(X[:, feature_indecies])
        LOBF_best = pipeline_best.predict(X[:, feature_indecies])
        
        # ax.plot(X0, LOBF_X, label="model trained on all data", color="purple", linewidth=2) data leakage 
        ax.plot(X0, LOBF_best, label="model trained on training data", color="green", linewidth=3)
        
        # prediction_vals = get_prediction_at_given_x(pipeline_on_training_data, feature_indecies, prediction_points, ax)
        prediction_vals, future_day = get_prediction_at_given_x(pipeline_best, feature_indecies, prediction_points, ax)
        
        max_x = max(np.max(X0), future_day) + 5
        min_x = np.min(X0)
        y_min = min(np.min(y), np.min(LOBF_best), np.min(y_test_best), np.min(prediction_vals))
        y_max = max(np.max(y), np.max(LOBF_best), np.max(y_test_best), np.max(prediction_vals))
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel(f'days in the future')
        ax.set_ylabel('value of the target')
        ax.legend()
        ax.set_title(currency_we_want_to_look_at)
        ax.grid() 
        
        best_fit_plot = plt.gcf()
        fig_rmse, ax_rmse = plt.subplots(1, 1, figsize=(9, 4)) # Single subplot
        ax_rmse.bar(best_row['deg'], best_row['train_RMSE'], label='train_RMSE', alpha=1)
        ax_rmse.bar(best_row['deg'], best_row['test_RMSE'], label='test_RMSE', alpha=0.5)
        ax_rmse.bar(best_row['deg'], best_row['naiveMAE'], label='naiveMAE', alpha=0.25)
        ax_rmse.set_title(f'RMSE at best degree {best_degree}')
        ax_rmse.set_xlabel(f"{best_degree}\nbest RMSE error on the test set\n{best_row['test_RMSE']:.4f}")
        ax_rmse.set_xticks([])
        ax_rmse.set_ylabel('RMSE value')
        ax_rmse.legend()

        # 2. MAE Bar Chart
        fig_mae, ax_mae = plt.subplots(1, 1, figsize=(9, 4)) # Single subplot
        ax_mae.bar(best_row['deg'], best_row['train_MAE'], label='train_MAE', alpha=1)
        ax_mae.bar(best_row['deg'], best_row['test_MAE'], label='test_MAE', alpha=0.5)
        ax_mae.bar(best_row['deg'], best_row['naiveMAE'], label='naiveMAE', alpha=0.25)
        ax_mae.set_title(f'MAE at best degree {best_degree}')
        ax_mae.set_xlabel(f"{best_degree}\nbest MAE error on the test set\n{best_row['test_MAE']:.4f}")
        ax_mae.set_xticks([])
        ax_mae.set_ylabel('MAE value')
        ax_mae.legend()

        # Return the Axes objects from the new separate figures
        return best_fit_plot, fig_rmse, fig_mae, prediction_vals

        
    best_graph, RMSE_bar_chart, MAE_bar_chart, prediction_vals = barchart_and_graph()
        
    return prediction_vals, train_and_test_errors_at_given_degree, best_graph, RMSE_bar_chart, MAE_bar_chart

pred_x_at_this_point = [13]

prediction_vals, train_and_test_errors_at_given_degree, best_graph, RMSE_bar_chart, MAE_bar_chart =  predictionVals_Plot_DictOfErrors(pred_x_at_this_point)
print(f"\nPredicted closing price in {HORIZON} days: {float(prediction_vals[0]):.6f}")
with open("prediction.txt", "w") as f:
    f.write(f'{prediction_vals[0]:.6f}')
best_graph.savefig("static/plots/best_graph.png")
RMSE_bar_chart.savefig("static/plots/RSME_Bar_Chart")
MAE_bar_chart.savefig("static/plots/MAE_Bar_Chart")



