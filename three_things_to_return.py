import lin_reg_file

pred_x_at_this_point = [20, 21, 99, 58, 67, 55, 33, 34, 35, 36, 37, 39]

prediction_vals, best_fit_plot, train_and_test_errors_at_given_degree = lin_reg_file.predictionVals_Plot_DictOfErrors(pred_x_at_this_point)

bar_chart1, bar_chart2 = lin_reg_file.bar_chart()

