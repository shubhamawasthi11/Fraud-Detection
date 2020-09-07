#!/home/ubuntu/anaconda3/envs/py363/bin/python3.6

from datetime import date
import os
import pandas as pd
from dateutil.relativedelta import relativedelta
from PyAstronomy import pyasl
import numpy as np
import time
from scipy import stats


start_time = time.time() 


#set directory
os.chdir('C:\\Users\\your-folder\\Downloads')


#function for writing data to excel
def write_excel(dataframe, file_name):
    writer = pd.ExcelWriter(file_name+'.xlsx', engine='xlsxwriter')
    dataframe.to_excel(writer, index=False)
    writer.save()


#function for calculating standard deviation
def std_dev(dataframe, metric):
    mean = round(dataframe[metric].mean())
    std = round(dataframe[metric].std())
    std_1 = mean - (1 * std)
    std_minus_1 = mean + (1 * std)
    return std_1, std_minus_1


#function to create dataframe out of outliers/merging to original sql query dataframe
def GESD_out(outlier_dataframe, sql_dataframe, col1, col2, col3):
    outlier_dataframe = pd.DataFrame(outlier_dataframe)
    outlier_dataframe.reset_index(inplace = True)
    outlier_dataframe.columns = [col1,col2]
    outlier_dataframe[col1] = outlier_dataframe[col1].astype(str)
    outlier_dataframe = outlier_dataframe.merge(sql_dataframe, how='inner', on=col1)
    outlier_dataframe.drop(columns = col3, inplace = True)
    return outlier_dataframe


#function for calculating outlier using box plot method
def Zscore(df):
    #Calculate outlier using zscore 1SD   
    df_base = pd.DataFrame(df)
    df_base = df_base.fillna(0)
    df_base = df_base[(np.abs(stats.zscore(df_base)) < 1.5).all(axis=1)]
    df_base = df_base.reset_index()
    df_base.columns = [column_yearmonth, column_metric] 
    df_test = df.reset_index()
    df_test = df_test.fillna(0)
    df_test.columns = [column_yearmonth, column_metric] 
    df_base = df_test[~df_test.column_metric.isin(df_base.column_metric.values)]
    df_result = df_base.sort_values(by=[column_yearmonth]) 
    return df_result


df_query = pd.read_csv('dataset.csv', sep=',')

df_metric = df_query.iloc[:,2:]

column_name = list(df_metric.columns.values)


#Create a list of dealer codes from the output of SQL query
column_dealers = df_query[column_dealer].unique().tolist()

df_final = pd.DataFrame([])

for column in column_name:
    df_SQL = df_query[[column_date, column_dealer, column]]
    df_SQL.columns = [column_yearmonth, column_dealer, column_metric]
    #running the for loop for each dealer and creating a subset dataframe
    for dlr in column_dealers:
        df_loop = df_SQL.loc[df_SQL.column_dealer == dlr]
        df_loop = df_loop.sort_values(by=[column_yearmonth])
        
            #running for only data where dealer recorded data for current yearmonth
        if (len(df_loop) < 12) or (df_loop[column_metric].mean() == 0):                   
            continue
        else:
        
            #transforming dataframe
            df_chunk = df_loop.reset_index()
            df_chunk = df_chunk[[column_yearmonth, column_metric]]
            df_chunk[column_yearmonth] = df_chunk[column_yearmonth].astype(str)
            
            
            #calculate moving average across entire data
            span = 2
            sma = df_chunk.column_metric.rolling(window=span, min_periods=span).mean()[:span]
            rest = df_chunk.column_metric[span:]
            moving_average = pd.concat([sma, rest]).ewm(span=3, adjust=False).mean()
            
            #saving moving average values in a dataframe
            df_ma = moving_average.reset_index()
            df_ma = df_ma.iloc[1:,1:]
            
            
            #calculating 2 standard deviations
            SD_x, SD_y = std_dev(df_ma, metric = column_metric)
            Predicted_Range = '['+str(SD_x)+', '+str(SD_y)+']'
            
            
            #convert yearmonth to list
            year_month = df_chunk.column_yearmonth.tolist()
            
            #convert data to series to subtract predicted values with original
            df_series = pd.Series(df_chunk[column_metric].values, index=df_chunk.column_yearmonth)
            df_outlier =  pd.DataFrame(df_series.values - moving_average.values)
            
            df_outlier[column_yearmonth] = year_month
            df_outlier.columns = [column_metric,column_yearmonth]
            df_outlier = pd.Series(df_outlier[column_metric].values, index=df_outlier.column_yearmonth)
            df_outlier = df_outlier.fillna(0)
            
            #baseline model
            df_zscore = Zscore(df_outlier)
            #for using grubb model 
            num_outlier = len(df_zscore)
            if num_outlier == 0:
                continue
            elif num_outlier == 1:
                num_outlier = 2
            else:
                num_outlier = len(df_zscore) + 1
            
            out_Grubb = pyasl.generalizedESD(df_outlier, num_outlier, alpha=0.05)    
            out_Grubb[1].sort()
            while 0 in out_Grubb[1]: out_Grubb[1].remove(0)
            while 1 in out_Grubb[1]: out_Grubb[1].remove(1)
            
            #taking data only where there is an outlier
            if len(out_Grubb[1]) == 0:
                continue
            else:
            
                #collecting data till first outlier detected
                Grubb_list = out_Grubb[1]
                first_outlier = df_outlier[0:Grubb_list[0]]
                first_outlier = GESD_out(first_outlier, df_loop, col1= column_yearmonth, col2= column_metric, col3= [column_yearmonth, column_dealer, 'column_metric_x'])
                
                #calculating SD till first outlier detected
                fo_SD_x, fo_SD_y = std_dev(first_outlier, metric = 'column_metric_y')
                
                #keeping outliers that do not fall back to baseline
                for i in out_Grubb[1]: 
                    outliers = df_outlier[[i]]
                    outliers = GESD_out(outliers, df_loop, col1= column_yearmonth, col2= column_metric, col3= 'column_metric_x')
                    outliers['METRIC'] = column
                    outliers['ESTIMATE'] = Predicted_Range
                    
                    if fo_SD_x <= outliers.iloc[0,2] <= fo_SD_y:
                        outliers.drop([0], axis=0)
                        #print('baseline range '+ dlr)
                    else:
                        #appending outliers to empty dataframe
                        df_final = df_final.append(outliers)


    
#sorting data for clean representation in excel     
df_final = df_final.sort_values(by=[column_dealer, 'METRIC'])

  
#writing dataframe to excel file
write_excel(df_final, file_name = 'fraud_detected_in_file')
    

print("My program took", time.time() - start_time, "to run") 
