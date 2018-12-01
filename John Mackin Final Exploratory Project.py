
# coding: utf-8

# In[ ]:


wget https://www.dropbox.com/s/xtms9g31aicstvv/train.csv?dl=1


# In[ ]:


plot(arange(5))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().magic('matplotlib inline')
df = pd.read_csv("/home/kunal/Downloads/Loan_Prediction/train.csv") #Reading the dataset in a dataframe using 
Pandas


# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# In[ ]:


df['Property_Area'].value_counts()


# In[ ]:


df['ApplicantIncome'].hist(bins=50)


# In[ ]:


df.boxplot(column='ApplicantIncome')


# In[ ]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[ ]:


df['LoanAmount'].hist(bins=50)


# In[ ]:


df.boxplot(column='LoanAmount')


# In[ ]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: 
x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)
print ('\nProbility of getting loan for each Credit History class:')
print (temp2)


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')
ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# In[ ]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[ ]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[ ]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


# In[ ]:


df['Self_Employed'].fillna('No',inplace=True)


# In[ ]:


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[ ]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[ ]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

