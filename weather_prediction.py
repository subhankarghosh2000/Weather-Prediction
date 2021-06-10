import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv("austin_weather.csv")
df=df[['TempAvgF','DewPointAvgF','HumidityAvgPercent','SeaLevelPressureAvgInches','VisibilityAvgMiles','WindAvgMPH']]
df['pt']=df['TempAvgF'].shift(1)
df['pdp']=df['DewPointAvgF'].shift(1)
df['phap']=df['HumidityAvgPercent'].shift(1)
df['pslpvi']=df['SeaLevelPressureAvgInches'].shift(1)
df['pvam']=df['VisibilityAvgMiles'].shift(1)
df['pwav']=df['WindAvgMPH'].shift(1)
df=df.dropna()


x=np.array(df[['pt','pdp','phap','pslpvi','pvam','pwav']])

y_temp=np.array(df['TempAvgF'])
y_dew=np.array(df['DewPointAvgF'])
y_hum=np.array(df['HumidityAvgPercent'])
y_pre=np.array(df['SeaLevelPressureAvgInches'])
y_vis=np.array(df['VisibilityAvgMiles'])
y_win=np.array(df['WindAvgMPH'])


X_train, X_test, y_temp_train, y_temp_test = train_test_split(x, y_temp, test_size=.3, random_state=42)
X_train, X_test, y_dew_train, y_dew_test = train_test_split(x, y_dew, test_size=.3, random_state=42)
X_train, X_test, y_hum_train, y_hum_test = train_test_split(x, y_hum, test_size=.3, random_state=42)
X_train, X_test, y_pre_train, y_pre_test = train_test_split(x, y_pre, test_size=.3, random_state=42)
X_train, X_test, y_vis_train, y_vis_test = train_test_split(x, y_vis, test_size=.3, random_state=42)
X_train, X_test, y_win_train, y_win_test = train_test_split(x, y_win, test_size=.3, random_state=42)


clf1=LinearRegression()
clf2=LinearRegression()
clf3=LinearRegression()
clf4=LinearRegression()
clf5=LinearRegression()
clf6=LinearRegression()

clf1.fit(X_train,y_temp_train)
clf2.fit(X_train,y_dew_train)
clf3.fit(X_train,y_hum_train)
clf4.fit(X_train,y_pre_train)
clf5.fit(X_train,y_vis_train)
clf6.fit(X_train,y_win_train)

y_temp_predicted=clf1.predict(X_test)
y_dew_predicted=clf2.predict(X_test)
y_hum_predicted=clf3.predict(X_test)
y_pre_predicted=clf4.predict(X_test)
y_vis_predicted=clf5.predict(X_test)
y_win_predicted=clf6.predict(X_test)


output=[list(y_temp_test),list(y_temp_predicted),list(y_dew_test),list(y_dew_predicted),list(y_hum_test),list(y_hum_predicted),list(y_pre_test),list(y_pre_predicted),list(y_vis_test),list(y_vis_predicted),list(y_win_test),list(y_win_predicted)]
output1=np.transpose(output)

out_file=open("predicted_dataset.csv","w")
out_row="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}".format("Actual Temperature","Predicted Temperature","Actual Dew Point","Predicted Dew Point","Actual Humidity","Predicted Humidity","Actual Sea Level Pressure","Predicted Sea Level Pressure","Actual Visibility","Predicted Visibility","Actual Wind Speed","Predicted Wind Speed")+"\n"
out_file.write(out_row)
for row in output1:
	out_row=",".join(map(str,row))+"\n"
	out_file.write(out_row)
out_file.close()



X=[]
for i in range(len(X_test)):
    X.append(i)
    

plt.figure(figsize=(50,15))
plt.plot(X,y_temp_predicted,color="red",label="Predicted Temperature")
plt.plot(X,y_temp_test,color="blue",label="Actual Temperature")
plt.title("TEMPERATURE PREDICTION",fontsize=30)
plt.xlabel("Cases",fontsize=25)
plt.ylabel("Temperature",fontsize=25)
leg = plt.legend(fontsize=25,loc=1);
plt.show()


plt.figure(figsize=(50,15))
plt.plot(X,y_dew_predicted,color="red",label="Predicted Dew Point")
plt.plot(X,y_dew_test,color="blue",label="Actual Dew Point")
plt.title("DEW POINT PREDICTION",fontsize=30)
plt.xlabel("Cases",fontsize=25)
plt.ylabel("Dew Point",fontsize=25)
leg = plt.legend(fontsize=30,loc=4)
plt.show()


plt.figure(figsize=(50,15))
plt.plot(X,y_hum_predicted,color="red",label="Predicted Humidity")
plt.plot(X,y_hum_test,color="blue",label="Actual Humidity")
plt.title("HUMIDITY PREDICTION",fontsize=30)
plt.xlabel("Cases",fontsize=25)
plt.ylabel("Humidity",fontsize=25)
leg = plt.legend(fontsize=30,loc=4)
plt.show()


plt.figure(figsize=(50,10))
plt.plot(X,y_pre_predicted,color="red",label="Predicted Sea Level Pressure")
plt.plot(X,y_pre_test,color="blue",label="Actual Sea Level Pressure")
plt.title("SEA LEVEL PRESSURE PREDICTION",fontsize=30)
plt.xlabel("Cases",fontsize=25)
plt.ylabel("Sea Level Pressure",fontsize=25)
leg = plt.legend(fontsize=30,loc=4)
plt.show()


plt.figure(figsize=(50,15))
plt.plot(X,y_vis_predicted,color="red",label="Predicted Visibility")
plt.plot(X,y_vis_test,color="blue",label="Actual Visibility")
plt.title("VISIBILITY PREDICTION",fontsize=30)
plt.xlabel("Cases",fontsize=25)
plt.ylabel("Visibility",fontsize=25)
leg = plt.legend(fontsize=30,loc=4)
plt.show()


plt.figure(figsize=(50,15))
plt.plot(X,y_win_predicted,color="red",label="Predicted Wind Speed")
plt.plot(X,y_win_test,color="blue",label="Actual Wind Speed")
plt.title("WIND SPEED PREDICTION",fontsize=30)
plt.xlabel("Cases",fontsize=25)
plt.ylabel("Wind Speed",fontsize=25)
leg = plt.legend(fontsize=30,loc=1)
plt.show()


