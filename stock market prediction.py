#!/usr/bin/env python
# coding: utf-8

# In[22]:



import numpy as np
import pandas as pd
import pandas_profiling as pdp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


pwd


# In[8]:


data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML project\TataMotors.csv')
data.head()


# In[9]:


data.tail()


# In[11]:


##Checking numerical and categorical data


# In[12]:


data.dtypes


# In[13]:


##Data type of 'Date' is object. Rest of the data is in numerical form which is appropriate


# In[14]:


#shape of the dataset
data.shape


# In[15]:


#some statistical information about data
data.describe()


# In[23]:


# a broader description of the data
pdp.ProfileReport(data)


# In[24]:


#Checking NULL values
data.isna().sum()


# In[26]:


data[data.isnull().any(axis = 1)]


# In[27]:


#Dropping rows with null values
data.dropna(axis = 0,inplace = True)


# In[28]:


data.isnull().sum()


# In[29]:


##Now there are no NaN values in the dataset


# In[30]:


##Formatting data(Making datatypes compatible)
##In our dataset only 'Date' are in object form which need to be changed to 'DateTime' format


# In[31]:


data.dtypes


# In[32]:


data['Date'] = pd.to_datetime(data['Date'])


# In[33]:


#Again check the datatypes
data.dtypes


# In[34]:


#sort the data according to 'Date'
data.sort_values('Date',inplace = True)


# In[35]:


data.head()


# In[36]:


##Univariate analysis
##Data analysis using Data Visualization


# In[37]:


#boxplot to know the statistic of each column


# In[40]:


sb.boxplot(data.Open)
plt.show()


# In[43]:


sb.boxplot(data['High'])
plt.show()


# In[44]:


sb.boxplot(data.Low)
plt.show()


# In[45]:


sb.boxplot(data.Close)
plt.show()


# In[46]:


sb.boxplot(data['Adj Close'])
plt.show()


# In[47]:


sb.boxplot(data['Volume'])
plt.show()


# In[48]:


##Check the distribution of data


# In[49]:


sb.distplot(data.Open)
plt.show()


# In[50]:


sb.distplot(data.High)
plt.show()


# In[51]:


sb.distplot(data.Low)
plt.show()


# In[52]:


sb.distplot(data.Close)
plt.show()


# In[53]:


sb.distplot(data['Adj Close'])
plt.show()


# In[54]:


sb.distplot(data.Volume)
plt.show()


# In[55]:


##Bivariate Analysis


# In[56]:


#store the columns of the data in a list
columns = data.columns
#remove column of date from the list
columns = columns[1:]
columns


# In[57]:


#plotting scatter plot between each column to know the bivariate distribution of the data
for i in range(len(columns)):
    for j in range(i+1,len(columns)):
        if i == j:
            continue
        else:
            sb.scatterplot(data[columns[i]],data[columns[j]],legend = 'brief')
            plt.show()


# In[58]:


#Checking for outliers


# In[59]:


#Import required packages


# In[60]:


from scipy import stats


# In[61]:


# Here we use zscore to identify outliers.
# If zscore is greater than 3 or less than -3 then it will be considered as outlier


# In[62]:


### Here we use zscore to identify outliers.
# If zscore is greater than 3 or less than -3 then it will be considered as outlier


# In[63]:


z_open = stats.zscore(data.Open)


# In[64]:


#Check for minimum and maximum zscore
print(z_open.min())
print(z_open.max())


# In[65]:


##Outliers in Column 'High'


# In[66]:


z_high = stats.zscore(data.High)
print(z_high.min())
print(z_high.max())


# In[67]:


##Outliers in Column 'Low'


# In[68]:


z_low = stats.zscore(data.Low)
print(z_low.min())
print(z_low.max())


# In[69]:


##Outliers in Column 'Close'


# In[70]:


z_close = stats.zscore(data.Close)
print(z_close.min())
print(z_close.max())


# In[71]:


##Outliers in Column 'Adj Close'


# In[72]:


z_aclose = stats.zscore(data['Adj Close'])
print(z_aclose.min())
print(z_aclose.max())


# In[73]:


##Outliers in Column 'Volume'


# In[74]:


z_vol = stats.zscore(data.Volume)
print(z_vol.min())
print(z_vol.max())


# In[75]:


##Removing Outliers


# In[76]:


##In our dataset,only 'Volume' column consists of outliers. So we will remove outliers from 'Volume' Column


# In[77]:


z_vol = abs(z_vol)
outlier = np.where(z_vol > 3)
outlier = list(outlier[0])
print(outlier)


# In[78]:


data.shape


# In[79]:


data.drop(outlier,inplace = True)


# In[80]:


data.shape


# In[81]:


##Correlation Matrix to Identify Relevent Columns


# In[82]:


data.corr()


# In[83]:


##Correlation Matrix


# In[84]:


corr_mat = data.corr()


# In[85]:


f,ax = plt.subplots(figsize = (8,8))
sb.heatmap(corr_mat,ax = ax,cmap = 'cividis_r')


# In[86]:


##These are highly correlated with each other

##Open with High,Low,Close,Adj Close
##High with Open,Low,Close,Adj Close
##Low with Open,High,Close,Adj Close
##Close with Open,High,Low,Adj Close
##Adj Close with Open,High,Low,Close
##Below are very less correlated

##Volume and all other columns


# In[87]:


##Final Model


# In[88]:


test_size = 0.1 #Data used for testing model
cv_size = 0.1 #Data used to select the best number of samples for predictions
max_sample = 30 #Maximum Number of days used to predict the stock price


# In[89]:


from sklearn.linear_model import LinearRegression
def predictPrice(dataset,num_of_samples,offset):
    '''
    dataset : dateset with values you want to predict
    num_of_samples : number of samples used to fit the linear model
    offset : We will be doing predictions for data after the offset i.e. dataset[offset:]. in other words it is our test data
    '''
    #Create Linear Regression object
    reg = LinearRegression()
    
    #List of predicted values for different number of samples
    pred_list = []
    
    #fit the linear model and predict prices for different days
    for i in range(offset,len(dataset['Adj Close'])):
        
        #Divide the dataset into train and test part
        x_train = np.array(range(len(dataset['Adj Close'][i-num_of_samples:i]))) #converting the dates into simple indices
        y_train = np.array(dataset['Adj Close'][i-num_of_samples:i]) #Taking values of Adj_close
        
        x_train = x_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        #print(x_train)
        #print(y_train)
        #print('shpae x ',x_train.shape)
        #print('shpae y ',y_train.shape)
        #fit the model into the LinearRegression Object and train the model
        reg.fit(x_train,y_train)
        
        #predict the price on num_of_sample day
        pred = reg.predict(np.array(num_of_samples).reshape(-1,1))
        pred_list.append(pred[0][0])
    
    return pred_list


# In[90]:


num_test = int(len(data)*test_size) #test data size
num_cv = int(len(data)*cv_size) #cross-validation data size
num_train = len(data) - num_test - num_cv #train data size


#splitting the data into train,test and cross-validation
train_data = data[:num_train].copy()
cv_data = data[num_train:num_train+num_cv].copy()
train_cv = data[:num_train + num_cv].copy()
test_data = data[num_train+num_cv:].copy()

print("Train data shape ",train_data.shape)
print("Train and CV data shape ",train_cv.shape)
print("Test data shape ",test_data.shape)
print("CV data shape ",cv_data.shape)


# In[91]:


#A view of train,cv and test data
mpl.rcParams['figure.figsize'] = 15,6
ax = train_data.plot(x = 'Date',y = 'Adj Close',style = 'b-')
ax = cv_data.plot(x = 'Date',y = 'Adj Close',style = 'g-',ax = ax)
ax = test_data.plot(x = 'Date',y = 'Adj Close',style = 'r-',ax = ax)
plt.legend(['train','cv','test'])
plt.grid()
plt.show()


# In[92]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
rmse = [] #list to store the root mean squared error for different samples
r2_score_err = [] #List to store the r2_score_error for different samples
#predict the price for different number of samples
for num_of_samples in range(1,max_sample+1):
    '''
    train_cv : Dataset consisting training and cross validation dataset
    num_of_samples : Number of samples used to train the model
    num_train : Size of train dataset
    
    return :Estimated price for cv dataset
    '''
    pred = predictPrice(train_cv,num_of_samples,num_train)
    #Add the current predicted value to the cv dataset
    cv_data.loc[:,'pred_for_N_'+str(num_of_samples)] = pred
    
    #Calculate the RMSE value and append it to the RMSE list
    rmse.append(mse(pred,cv_data['Adj Close'],squared = False))
    
    #Calculate the r2_score and append it to r2_score_err list
    r2_score_err.append(r2_score(pred,cv_data['Adj Close']))


# In[93]:


cv_data.head()


# In[94]:


print(rmse)


# In[95]:


##Graph of RMSE vs num_of_samples


# In[96]:


x = np.arange(1,max_sample+1)
plt.plot(x,rmse,'d-')
plt.xlabel("Number Of Samples")
plt.ylabel("RMSE")
plt.title("RMSE vs Number of Samples",size = 20)
plt.grid()
plt.show()


# In[97]:


##Graph of r2_score vs num_of_samples


# In[98]:


plt.plot(x,r2_score_err,'d-')
plt.xlabel("Number Of Samples")
plt.ylabel("r2_score")
plt.title("r2_score vs Number of Samples",size = 20)
plt.grid()
plt.show()


# In[99]:


##Graph between actual value and predicted value for num_of_samples = 5


# In[100]:


ax = cv_data.plot(x='Date',y='Adj Close')
ax = cv_data.plot(x='Date',y='pred_for_N_5',ax = ax)
plt.grid()
plt.legend(['Actual Value','Predicted Value for N = 5'])
plt.show()


# In[101]:


##Predictions on Test data


# In[102]:


num_of_samples = 5
pred = predictPrice(data,num_of_samples,num_train+num_cv)
test_data.loc[:,"Predicted Adj Close"] = pred


# In[103]:


test_data.head()


# In[104]:


#root_mean_squared_error
rmse = mse(test_data['Adj Close'],pred)


# In[105]:


#Visualization on test data and predicted value
ax = test_data.plot(x='Date',y='Adj Close')
ax = test_data.plot(x='Date',y='Predicted Adj Close',ax = ax)
plt.grid()
plt.legend()
plt.xlabel("Date")
plt.ylabel("Adj Close Price")
plt.title("Graph for test data",size = 20)
plt.show()


# In[106]:


mpl.rcParams['figure.figsize'] = 20,8
ax = train_data.plot(x = 'Date',y = 'Adj Close',style = 'bd',markersize = 3)
ax = cv_data.plot(x = 'Date',y = 'Adj Close',style = 'gd',ax = ax,markersize = 3)
ax = cv_data.plot(x = 'Date',y = 'pred_for_N_5',style = 'cd',ax = ax,markersize = 3)
ax = test_data.plot(x = 'Date',y = 'Adj Close',style = 'rd',ax = ax,markersize = 3)
ax = test_data.plot(x = 'Date',y = 'Predicted Adj Close',style = 'md',ax = ax,markersize = 3)
plt.legend(['Train Data','CV Data','Prediction on CV Data','Test Data','Prediction on Test Data'])
plt.grid()
plt.show()


# In[107]:


##Predictions for a specific day


# In[ ]:


from tkinter import *

class Placeholder_State(object):
     __slots__ = 'normal_color', 'normal_font', 'placeholder_text', 'placeholder_color', 'placeholder_font', 'with_placeholder'

def add_placeholder_to(entry, placeholder, color="grey", font=None):
    normal_color = entry.cget("fg")
    normal_font = entry.cget("font")
    
    if font is None:
        font = normal_font

    state = Placeholder_State()
    state.normal_color=normal_color
    state.normal_font=normal_font
    state.placeholder_color=color
    state.placeholder_font=font
    state.placeholder_text = placeholder
    state.with_placeholder=True

    def on_focusin(event, entry=entry, state=state):
        if state.with_placeholder:
            entry.delete(0, "end")
            entry.config(fg = state.normal_color, font=state.normal_font)
        
            state.with_placeholder = False

    def on_focusout(event, entry=entry, state=state):
        if entry.get() == '':
            entry.insert(0, state.placeholder_text)
            entry.config(fg = state.placeholder_color, font=state.placeholder_font)
            
            state.with_placeholder = True

    entry.insert(0, placeholder)
    entry.config(fg = color, font=font)

    entry.bind('<FocusIn>', on_focusin, add="+")
    entry.bind('<FocusOut>', on_focusout, add="+")
    
    entry.placeholder_state = state

    return state

main_window = Tk()
main_window.geometry('500x280')
main_window.title('Stock Market Price Predictor')

def openResultWindow():
    
    date = e1.get()
    
    max_samples = int(e2.get())
        
    try:
        text.set('')
        date = pd.to_datetime(date)

        if(max_samples < 0 or max_samples >30):
            raise Exception()
            
        result_window = Toplevel(main_window)
        height = max_samples
        width = 400

        if height <= 5 :
            size = str(width)+"x"+str(130)
        else :
            size = str(width)+"x"+str(30+int(height*20))

        result_window.geometry(size)
        result_window.title('Predicted Result')
        lbl5 = Label(result_window, text='PREDICTED RESULT')
        lbl5.grid(row=3,column=0,columnspan=3,pady=20)

        #Take the dataset before the date
        df = data[data.Date < date]
        offset = np.where(data.Date < date)
        offset = offset[-1][-1]
        #print(offset)
        #print(data.iloc[offset])

        #if max_samples < 5 
        if(max_samples < 5):
            max_samples = 5

        #predic the prices for given number of samples
        pred_list = []
        for num_of_samples in range(5,max_samples+1):
            pred = predictPrice(df,num_of_samples,offset)
            pred_list.append(pred)

        result_list = []

        for day in range(len(pred_list)):
            result_list.append(pred_list[day])
            result_text = "Predictions using "+str(5+day)+" days is "+str(pred_list[day])
            Label(result_window, text=result_text).grid(row=4+day,column=0)

        bestResult_value = max(result_list)

        for i in range(len(result_list)):
            if(bestResult_value == result_list[i]) :
                bestResult_days = i

        bestResult_text = "Highest Predicted Value is using "+str(5+bestResult_days)+" days and Value is "+str(bestResult_value)
        Label(result_window, text=bestResult_text).grid(row=5+max_samples,column=0)

        result_window.mainloop()
        
    except:
        if(max_samples < 0 or max_samples >30):
            text.set('Enter the number within specified range!!!')
        else:
            text.set('Enter the date in the specified format!!!')
        resetAllEntry()
    
    
def resetAllEntry():
    e1.delete(0,END)
    add_placeholder_to(e1,'yyyy-mm-dd')
    e2.delete(0,END)
    add_placeholder_to(e2,'number<=30')
    
    

    
    
    
lbl1 = Label(main_window, text='STOCK MARKET PRICE PREDICTOR')
lbl1.grid(row=3,column=0,columnspan=3,pady=20)
lbl2 = Label(main_window, text='Enter Date : * ')
lbl2.grid(row=5,column=0,columnspan=2,padx=10,pady=10) 
lbl3 = Label(main_window, text='Enter Number of Days to be used for predictions : * ')
lbl3.grid(row=7,column=0,columnspan=2,padx=10,pady=10) 
e1 = Entry(main_window)
e1.grid(row=5,column=2,padx=10)
add_placeholder_to(e1,'yyyy-mm-dd')
e2 = Entry(main_window)
e2.grid(row=7,column=2,padx=10)
add_placeholder_to(e2,'number<=30')
b1 = Button(main_window, text='Predict', width=25,command=openResultWindow)
b1.grid(row=9,column=1,pady=10)
b2 = Button(main_window, text='Reset', width=25,command=resetAllEntry)
b2.grid(row=9,column=2,pady=10)
lbl4 = Label(main_window, text='* Denotes Mandatory Fields')
lbl4.grid(row=11,column=0,columnspan=3,padx=10,pady=10) 
text = StringVar()
lbl6 = Label(main_window, textvariable = text)
text.set('')
lbl6.grid(row=13,column=0,columnspan=3,padx=10,pady=10) 
main_window.mainloop()


# In[ ]:




