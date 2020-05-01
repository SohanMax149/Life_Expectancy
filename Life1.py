#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv(r'C:\ML\Life\Life.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",
                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",
                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",
                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",
                   "Total expenditure":"Tot_Exp"},inplace=True)


# In[7]:


df.head()


# In[8]:


df['Country'] = df['Country'].replace(['Afghanistan' , 'Albania' , 'Algeria' , 'Angola' , 'Antigua and Barbuda' , 'Argentina' , 'Armenia' , 'Australia' , 'Austria' , 'Azerbaijan' , 'Bahamas' , 'Bahrain' , 'Bangladesh' , 'Barbados' , 'Belarus' , 'Belgium' , 'Belize' , 'Benin' , 'Bhutan' , 'Bolivia (Plurinational State of)' , 'Bosnia and Herzegovina' , 'Botswana' , 'Brazil' , 'Brunei Darussalam' , 'Bulgaria' , 'Burkina Faso' , 'Burundi' , "Côte d'Ivoire" , 'Cabo Verde' , 'Cambodia' , 'Cameroon' , 'Canada' , 'Central African Republic' , 'Chad' , 'Chile' , 'China' , 'Colombia' , 'Comoros' , 'Congo' , 'Costa Rica' , 'Croatia' , 'Cuba' , 'Cyprus' , 'Czechia' , "Democratic People's Republic of Korea" , 'Democratic Republic of the Congo' , 'Denmark' , 'Djibouti' , 'Dominican Republic' , 'Ecuador' , 'Egypt' , 'El Salvador' , 'Equatorial Guinea' , 'Eritrea' , 'Estonia' , 'Ethiopia' , 'Fiji' , 'Finland' , 'France' , 'Gabon' , 'Gambia' , 'Georgia' , 'Germany' , 'Ghana' , 'Greece' , 'Grenada' , 'Guatemala' , 'Guinea' , 'Guinea-Bissau' , 'Guyana' , 'Haiti' , 'Honduras' , 'Hungary' , 'Iceland' , 'India' , 'Indonesia' , 'Iran (Islamic Republic of)' , 'Iraq' , 'Ireland' , 'Israel' , 'Italy' , 'Jamaica' , 'Japan' , 'Jordan' , 'Kazakhstan' , 'Kenya' , 'Kiribati' , 'Kuwait' , 'Kyrgyzstan' , "Lao People's Democratic Republic" , 'Latvia' , 'Lebanon' , 'Lesotho' , 'Liberia' , 'Libya' , 'Lithuania' , 'Luxembourg' , 'Madagascar' , 'Malawi' , 'Malaysia' , 'Maldives' , 'Mali' , 'Malta' , 'Mauritania' , 'Mauritius' , 'Mexico' , 'Micronesia (Federated States of)' , 'Mongolia' , 'Montenegro' , 'Morocco' , 'Mozambique' , 'Myanmar' , 'Namibia' , 'Nepal' , 'Netherlands' , 'New Zealand' , 'Nicaragua' , 'Niger' , 'Nigeria' , 'Norway' , 'Oman' , 'Pakistan' , 'Panama' , 'Papua New Guinea' , 'Paraguay' , 'Peru' , 'Philippines' , 'Poland' , 'Portugal' , 'Qatar' , 'Republic of Korea' , 'Republic of Moldova' , 'Romania' , 'Russian Federation' , 'Rwanda' , 'Saint Lucia' , 'Saint Vincent and the Grenadines' , 'Samoa' , 'Sao Tome and Principe' , 'Saudi Arabia' , 'Senegal' , 'Serbia' , 'Seychelles' , 'Sierra Leone' , 'Singapore' , 'Slovakia' , 'Slovenia' , 'Solomon Islands' , 'Somalia' , 'South Africa' , 'South Sudan' , 'Spain' , 'Sri Lanka' , 'Sudan' , 'Suriname' , 'Swaziland' , 'Sweden' , 'Switzerland' , 'Syrian Arab Republic' , 'Tajikistan' , 'Thailand' , 'The former Yugoslav republic of Macedonia' , 'Timor-Leste' , 'Togo' , 'Tonga' , 'Trinidad and Tobago' , 'Tunisia' , 'Turkey' , 'Turkmenistan' , 'Uganda' , 'Ukraine' , 'United Arab Emirates' , 'United Kingdom of Great Britain and Northern Ireland' , 'United Republic of Tanzania' , 'United States of America' , 'Uruguay' , 'Uzbekistan' , 'Vanuatu' , 'Venezuela (Bolivarian Republic of)' , 'Viet Nam' , 'Yemen' , 'Zambia' , 'Zimbabwe' , 'Cook Islands' , 'Dominica' , 'Marshall Islands' , 'Monaco' , 'Nauru' , 'Niue' , 'Palau' , 'Saint Kitts and Nevis' , 'San Marino' , 'Tuvalu'], [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,103 ,104 ,105 ,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,133 ,134 ,135 ,136 ,137 ,138 ,139 ,140 ,141 ,142 ,143 ,144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,152 ,153 ,154 ,155 ,156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,166 ,167 ,168 ,169 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184 ,185 ,186 ,187 ,188 ,189 ,190 ,191 ,192 ,193])


# In[9]:


df['Status'] = df['Status'].replace(['Developing', 'Developed'],[1, 2])


# In[10]:


df.head()


# In[11]:


df = df.fillna(df.mean())


# In[12]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('Year')['Year'].count().index,df.groupby('Year')['Life_Expectancy'].mean(),alpha=0.65)
plt.xlabel("Year",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t Year")
plt.show()


# In[13]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('Infant_Deaths')['Infant_Deaths'].count().index,df.groupby('Infant_Deaths')['Adult_Mortality'].mean(),alpha=0.65)
plt.xlabel("Infant_Deaths",fontsize=12)
plt.ylabel("Avg Adult Mortality",fontsize=12)
plt.title("Adult_Mortality w.r.t Infant_Deaths")
plt.show()


# In[14]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('Infant_Deaths')['Infant_Deaths'].count().index,df.groupby('Infant_Deaths')['Life_Expectancy'].mean(),alpha=0.65)
plt.xlabel("Infant_Deaths",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t Infant_Deaths")
plt.show()


# In[15]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('Adult_Mortality')['Adult_Mortality'].count().index,df.groupby('Adult_Mortality')['Life_Expectancy'].mean(),alpha=0.65)
plt.xlabel("Adult_Mortality",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t Adult_Mortality")
plt.show()


# In[16]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('HIV/AIDS')['HIV/AIDS'].count().index,df.groupby('HIV/AIDS')['Life_Expectancy'].mean(),alpha=0.65)
plt.xlabel("HIV/AIDS",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t HIV/AIDS")
plt.show()


# In[17]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('Alcohol')['Alcohol'].count().index,df.groupby('Alcohol')['Life_Expectancy'].mean(),alpha=0.65)
plt.xlabel("Alcohol",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t Alcohol")
plt.show()


# In[18]:


plt.figure(figsize=(7,5))
plt.bar(df.groupby('Alcohol')['Alcohol'].count().index,df.groupby('Alcohol')['Adult_Mortality'].mean(),alpha=0.65)
plt.xlabel("Alcohol",fontsize=12)
plt.ylabel("Avg Adult_Mortality",fontsize=12)
plt.title("Adult_Mortality w.r.t Alcohol")
plt.show()


# In[19]:


sns.factorplot('Life_Expectancy',col='Status',data=df,kind='count')


# In[20]:


plt.subplot(1,3,1)
plt.scatter(df["Schooling"], df["Adult_Mortality"])
plt.title("Schooling vs AdultMortality")


# In[21]:


plt.subplot(1,3,2)
plt.scatter(df["Life_Expectancy"], df["Adult_Mortality"])
plt.title("LifeExpectancy vs AdultMortality")


# In[22]:


corr_matrix=df.corr()


# In[23]:


plt.figure(figsize=(15,15))
sns.heatmap(data=corr_matrix, annot=True, linewidths=0.2)


# In[24]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)


# In[25]:


train_x = train.loc[:, train.columns != "Life_Expectancy"]
test_x = test.loc[:, test.columns != "Life_Expectancy"]
train_y = train["Life_Expectancy"]
test_y = test["Life_Expectancy"]


# In[26]:


coef=LinearRegression()
coef.fit(train_x, train_y)


# In[27]:


Y_pred = coef.predict(test_x)


# In[28]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(coef,train_x,train_y)
print(scores)


# In[29]:


print('Variance score: %.2f' % r2_score(test_y, Y_pred))


# In[30]:


print(mean_squared_error(test_y, Y_pred))


# In[ ]:




