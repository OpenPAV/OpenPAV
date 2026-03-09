from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import geatpy as ea
import pandas as pd
import numpy as np


def IDM(arg, delta_d, v, delta_v):
    """ Calculate the Intelligent Driver Model (IDM) acceleration. """
    v0, T, a, b, s0 = arg
    s_star = s0 + max(0, v * T + (v * delta_v) / (2 * ((a * b) ** 0.5)))
    small_value = 1e-5  # To avoid division by zero
    return a * (1 - (v / v0) ** 4 - (s_star / (delta_d + small_value)) ** 2)


def FVD(arg, delta_d, v, delta_v):
    """ Calculate the Full Velocity Difference Model (FVD) acceleration. """
    alpha, lamda, v_0, b, beta = arg
    V_star = v_0 * (np.tanh(delta_d / b - beta) - np.tanh(-beta))
    ahat = alpha * (V_star - v) + lamda * delta_v
    return ahat


class MyProblem(ea.Problem):
    """ A class to define optimization problems for evolutionary algorithms. """

    def __init__(self, df, lb, ub):
        M = 1  # Number of objectives
        maxOrMin = [1]  # 1 for minimization
        Dim = 5  # Number of decision variables
        varTypes = [0] * Dim  # 0 for continuous variables
        lbin = [1] * Dim  # 1 to include the lower bound
        ubin = [1] * Dim  # 1 to include the upper bound
        ea.Problem.__init__(self, "", M, maxOrMin, Dim, varTypes, lb, ub, lbin, ubin)
        self.df = df

    def aimFunc(self, pop):
        """ Objective function for optimization. """
        x = pop.Phen
        results = []
        for parameters in x:
            arg = tuple(round(param, 3) for param in parameters[:5])
            self.df['a_hat'] = self.df.apply(lambda row: IDM(arg, row['Spatial_Gap'],
                                                             row['Speed_FAV'],
                                                             -row['Speed_Diff']), axis=1)
            results.append(mean_squared_error(self.df['Acc_FAV'], self.df['a_hat']))
        pop.ObjV = np.vstack(results)  # Assign objective values to the population


class CFModelRegress:
    """ A class for regression analysis of car-following models. """

    def __init__(self, path, timestep, delay=0):
        df = pd.read_csv(path)
        self.dfs = {name: group for name, group in df.groupby('ID_FAV')}
        self.timestep = timestep
        self.delay = delay

    def addressOneTra(self, oneTraj):
        """ Prepare independent and dependent variables from trajectory data. """
        xData = []
        yData = []
        delay_steps = int(self.delay / self.timestep)
        for t in range(delay_steps, oneTraj.shape[0]):
            xData.append([oneTraj['Spatial_Gap'].iloc[t - delay_steps],
                          oneTraj['Speed_FAV'].iloc[t - delay_steps],
                          oneTraj['Speed_Diff'].iloc[t - delay_steps]])
            yData.append([oneTraj['Acc_FAV'].iloc[t]])
        return xData, yData

    def reorganizeDataIndividualVeh(self):
        """ Reorganize data by vehicle for further analysis. """
        reorganizedData = {}
        for veh, oneVehData in self.dfs.items():
            sampleData = {"x": [], "y": []}
            for Trajectory_ID, group in oneVehData.groupby('Trajectory_ID'):
                x_oneTra, y_oneTra = self.addressOneTra(group)
                sampleData["x"] += x_oneTra
                sampleData["y"] += y_oneTra
            reorganizedData[veh] = sampleData
        return reorganizedData

    def linearRegression(self, veh, xData, yData):
        """ Perform linear regression analysis. """
        xData_np = np.array(xData).reshape(len(xData), -1)
        yData_np = np.array(yData)
        model = LinearRegression()
        model.fit(xData_np, yData_np)
        y_pred = model.predict(xData_np)
        mse = mean_squared_error(yData_np, y_pred)
        coefficients = model.coef_.flatten()
        intercept = model.intercept_
        r_squared = model.score(xData_np, yData_np)
        n_temp = xData_np.shape[0]
        k_temp = xData_np.shape[1]
        adjusted_r_squared = 1 - (1 - r_squared) * (n_temp - 1) / (n_temp - k_temp - 1)
        data = {'Vehicle': [veh], 'R2': [r_squared], 'RMSE': [np.sqrt(mse)]}
        for i, coef in enumerate(coefficients):
            data[f'Coef_{i}'] = coef
        data['Intercept'] = intercept
        return pd.DataFrame(data)

    def IDM_regression(self, veh, xData, yData):
        """ Perform regression using the IDM car-following model. """
        xData_df = pd.DataFrame(xData)
        yData_df = pd.DataFrame(yData)
        df = pd.concat([xData_df, yData_df], axis=1)
        df.columns = ['Spatial_Gap', 'Speed_FAV', 'Speed_Diff', 'Acc_FAV']
        problem = MyProblem(df, [0.1, 0.1, 20, 0.1, 0.1], [10, 10, 40, 10, 10])  # Parameters for FVD model
        Encoding = 'RI'
        NIND = 25  # Population size
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        myAlgorithm = ea.soea_SEGA_templet(problem, population)
        myAlgorithm.MAXGEN = 100  # Max generations
        myAlgorithm.verbose = True
        myAlgorithm.drawing = 1
        BestIndi, population = myAlgorithm.run()
        arg = tuple(round(param, 3) for param in BestIndi.Phen[0, :5])
        df['a_hat'] = df.apply(lambda row: IDM(arg, row['Spatial_Gap'],
                                               row['Speed_FAV'],
                                               -row['Speed_Diff']), axis=1)
        r_squared = r2_score(df['Acc_FAV'], df['a_hat'])
        mse = mean_squared_error(df['Acc_FAV'], df['a_hat'])
        results = {'Vehicle': [veh], 'R2': [r_squared], 'RMSE': [np.sqrt(mse)]}
        for i in range(5):
            results[f'Coef_{i}'] = BestIndi.Phen[0, i]
        return pd.DataFrame(results)

    def main(self, output_path, model):
        """ Main function to run the regression analysis. """
        allData = self.reorganizeDataIndividualVeh()
        df_list = []
        for veh, data in allData.items():
            if model == "linear":
                df_list.append(self.linearRegression(veh, data['x'], data['y']))
            elif model == "IDM":
                df_list.append(self.IDM_regression(veh, data['x'], data['y']))
        merged_df = pd.concat(df_list)
        merged_df.to_csv(output_path, index=False)
        return merged_df
