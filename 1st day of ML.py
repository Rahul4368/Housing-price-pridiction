#!/usr/bin/env python
# coding: utf-8

# # ML 1st day starting

# In[1]:


13*9


# In[2]:


'''
Hello My self rahul Srivastav
'''


# In[3]:


10/4


# # Check the datatype:-

# In[4]:


type(1)


# In[5]:


type("Rahul")


# In[6]:


type(True)


# # Variable_namevalue

# In[7]:


a=10


# In[8]:


type(a)


# # Mathematical operations

# In[9]:


a=10
b=20


# In[10]:


print(a*b)
print(a/b)
print(a+b)
print(a-b)


# # various ways of printing

# In[11]:


print("Hello world")


# In[12]:


first_name='Rahul'
last_name='Srivastav'


# In[13]:


print("My first name is {} and last name is {}".format(first_name,last_name))


# In[14]:


print("My first name is {first} and last name is {last}".format(first=first_name,last=last_name))


# # In built function

# In[15]:


bool()


# In[16]:


type(False)


# In[17]:


my_str="Rahul Srivastav" #number not present in function


# In[18]:


my_str.isalnum()


# In[19]:


my_str="Rahul2303Srivastav" #number in the function


# In[20]:


my_str.isalnum()


# In[21]:


my_str='Rahul2303Srivastav' 


# In[22]:


print(my_str.isdigit()) #test if string contains digits 
print(my_str.istitle()) #test if string contains title words
print(my_str.isupper()) #test if string contains upper case
print(my_str.islower()) #test if string contains lower case
print(my_str.isspace()) #test if string contains spaces 
print(my_str.endswith('v')) #test if string endswith a d
print(my_str.startswith('R')) #test if string startswith H


# # Boolean and Logical operations

# In[23]:


True and True


# In[24]:


True and False


# In[25]:


True or False


# In[26]:


True or True


# In[27]:


str_example='Srivastav'
my_str='Rahul'


# In[28]:


my_str.isalpha() or str_example.isstr()


# # List:- IT's a data structure in python that is mutable or changable ordered sequence of elements

# In[29]:


lst_example=[]


# In[30]:


type(lst_example)


# In[31]:


lst=list()


# In[32]:


type(lst)


# In[33]:


lst=['Math','Chemistry',100,500,800,700]


# In[34]:


len(lst)


# In[35]:


type(lst)


# # Append:-Its used to add element in the list

# In[36]:


lst.append("Rahul")


# In[37]:


lst


# # Insert:- Its in specific order

# In[38]:


lst.insert(1,"Srivastav")


# In[39]:


lst


# In[40]:


## Indexing in list
lst[7]


# In[41]:


lst[2:8]


# # Insert

# In[42]:


## Insert in a specific order
lst.insert(1,"Manish")


# In[43]:


lst


# # Various opes that we can perform in list

# In[44]:


lst=[1,2,3,4,5]


# In[45]:


sum(lst)


# In[46]:


lst*2


# # Pop() Method

# In[47]:


lst.pop()


# In[48]:


lst


# In[49]:


lst.pop(1)


# In[50]:


lst


# # Count():- Cal total occurrence of given element of List

# In[51]:


lst=[1,2,1,5,8,7]
lst.count(1)


# In[52]:


## Length:Cal total length of List
len(lst)


# In[53]:


## Index:-Returns the index of 1st occurrence.Start & End index are not necessary parameters
lst.index(1,1,4)


# In[54]:


## Min and Max
min(lst)


# In[55]:


max(lst)


# # SETS:-Its unordered collections data type that is iterable,mutable and has no duplicate elements. Python's set class represents the mathematical notion of a set.its based on a data structure known as hash table

# In[56]:


## Defining on empy set

set_var=set()
print(set_var)
print(type(set_var))


# In[57]:


set_var={1,2,3,2}


# In[58]:


set_var


# In[59]:


set_var={"Rahul","Manish","Srivastav"}
print(set_var)
type(set_var)


# In[60]:


## Inbuilt functions in sets
set_var.add("Raj")


# In[ ]:





# In[61]:


print(set_var)


# In[62]:


set1={"Rahul","Srivastav","Manish","Rishu"}
set2={"Rahul","Srivastav","Manish","Rishu","Kumar"}


# In[63]:


set2.intersection_update(set1)


# In[64]:


set2


# In[65]:


## Difference
set2.difference(set1)


# In[66]:


## Difference update
set2.difference_update(set1)


# In[67]:


print(set2)


# # Dictionaries:-Its collection which is unordered,changeable and indexed

# In[68]:


dic={}


# In[69]:


type(dic)


# In[70]:


dic={1,3,5,7,9}


# In[71]:


type(dic)


# In[72]:


## Let create a dictionary

my_dict={"Car1":"Audi","Car2":"benz","Car3":"G-Wagen","Car4":"BMW"}


# In[73]:


type(my_dict)


# In[74]:


## Access the item values based on keys
my_dict['Car1']


# In[75]:


## We can Even loop through the dictionaries keys

for x in my_dict:
    print(x)


# In[76]:


## We can even loop through the dictionaries values

for x in my_dict.values():
    print(x)


# In[77]:


## We can also check both keys and values
for x in my_dict.items():
    print(x)


# In[78]:


## Adding items in Dictionaries

my_dict['car5']='Tesla'


# In[79]:


my_dict


# # Nested Dictionary

# In[80]:


car1_model={'Mercedes':1970}
car2_model={'Audi':1975}
car3_model={'Ambassador':1980}

car_type={'car1':car1_model,'car2':car2_model,'car3':car3_model}


# In[81]:


print(car_type)


# In[82]:


## Accessing the items in the dictionary:-

print(car_type['car1'])


# In[83]:


print(car_type['car1']['Mercedes'])


# # Tuples

# In[84]:


## Create an empty Tuples

my_tuple=tuple()


# In[85]:


type(my_tuple)


# In[86]:


my_tuple=()


# In[87]:


type(my_tuple)


# In[88]:


my_tuple=("Rahul","Srivastav","Manish")


# In[89]:


my_tuple[0]


# In[90]:


print(type(my_tuple))
print(my_tuple)


# In[91]:


type(my_tuple)


# In[92]:


## Inbuilt function

my_tuple.count('Rahul')


# In[93]:


my_tuple.index('Srivastav')


# #  Numpy:- It's a general purpose array processing package.it provides a high performance multidimensional array object,It's the fundamental package for scientific computing in python

# # Array:-It's a DSA that store values of same data type.
# ## List contain values corresponding to diff data type.
# ## Array contain values corresponding to same data type.

# In[94]:


## Initially lets import numpy

import numpy as np


# In[95]:


my_lst=[1,3,5,7,9]

arr=np.array(my_lst)


# In[96]:


type(arr)


# In[97]:


arr


# In[98]:


print(arr)


# In[99]:


arr.shape


# In[100]:


type(arr)


# In[101]:


## Multinested array

my_lst1=[1,2,3,4,5]
my_lst2=[2,4,5,6,9]
my_lst3=[0,6,3,1,7]

arr=np.array([my_lst1,my_lst2,my_lst3])


# In[102]:


arr


# In[103]:


arr.shape


# In[104]:


arr.reshape(1,15)


# In[105]:


arr.reshape(5,3)


# In[106]:


## Check the shape of array

arr.shape


# # Indexing

# In[107]:


## Accessing the array element

arr=np.array([1,3,2,5,6,7])


# In[108]:


arr[3]


# In[109]:


arr[:4]


# In[110]:


arr=np.arange(0,10) ## 0 starting index  /  10 Ending Index


# In[111]:


arr


# In[112]:


np.linspace(1,20,30) ## its using in Deep Learning


# In[113]:


## Copy() function and broadcasting

arr[3:]=100


# In[114]:


arr


# In[115]:


arr1=arr


# In[116]:


arr1[3:]=500
print(arr1)


# In[117]:


arr1=arr.copy()


# In[118]:


print(arr)
arr1[3:]=1000
print(arr1)


# In[119]:


arr


# In[120]:


## Some conditions very useful in Exploratory Data Analysis

val=2

arr[arr<600]


# In[121]:


## Create arrays and reshapes

np.arange(0,10).reshape(5,2)


# In[122]:


arr1=np.arange(0,10).reshape(2,5)


# In[123]:


arr2=np.arange(0,10).reshape(2,5)


# In[124]:


arr1*arr2


# In[125]:


np.ones(4,dtype=float)


# In[126]:


np.ones((2,5),dtype=int)


# In[127]:


## random distribution

np.random.rand(3,3)


# In[128]:


arr_ex=np.random.randn(4,4)


# In[129]:


arr_ex


# In[130]:


import seaborn as sns
import pandas as pd


# In[131]:


sns.displot(pd.DataFrame(arr_ex.reshape(16,1)))


# In[132]:


np.random.randint(0,100,8)


# In[133]:


np.random.randint(0,100,8).reshape(2,4)


# np.random.randint(0,100,8).reshape(4,2)

# In[134]:


np.random.random_sample((1,5))


# # Pandas:- It's a open source BSD-licensed library providing high performance,easy to use DSA & Data Analysis

# In[135]:


## First Step is to import pandas

import pandas as pd
import numpy as np


# In[136]:


## playing with Dataframe

df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','Row5'],columns=["Column1","Column2","Column3","Column4"])


# In[137]:


type(df['Column3'])


# In[138]:


df[['Column3','Column4']]


# In[139]:


df.head()


# In[140]:


df.to_csv('Test1.csv')  ##to create a CSV file in  notebook


# In[141]:


## Accessing the elements
## 1.) .loc-
## 2.) .iloc

df.loc['Row1']


# In[142]:


## Check the type **

type(df.loc['Row1'])


# In[143]:


df.iloc[0:3,0:2]


# In[144]:


type(df.iloc[0:3,0:2])


# In[145]:


df.iloc[0:2,0:1]


# In[146]:


## Take the element from the column2
df.iloc[:,1:]


# In[147]:


## Convert Dataframes into array
df.iloc[:,1:].values


# In[148]:


df.iloc[:,1:].values.shape


# In[149]:


df.isnull().sum()


# In[150]:


df['Column1'].value_counts()


# In[151]:


df['Column1'].unique()


# In[152]:


test_df=pd.read_csv('test1.csv',sep=';')  ## for replace(:)into(;) in csv file


# In[153]:


test_df.head()


# In[154]:


import pandas as pd
df=pd.read_csv('mercedesbenz.csv')


# In[155]:


df.head()


# In[156]:


df.info()


# In[157]:


df.describe() 


# In[158]:


## Get the unique category counts
df['X0'].value_counts()


# In[159]:


df[df['y']>100]


# In[160]:


df.corr() ##Select only valid columns or specify the value of numeric_only to silence this warning.


# In[161]:


df['X11'].value_counts()


# In[162]:


import numpy as np


# In[163]:


lst_data=[[1,2,3],[3,4,np.nan],[np.nan,np.nan,np.nan]]


# In[164]:


df=pd.DataFrame(lst_data)


# In[165]:


df.head()


# In[166]:


## Handling Missing Values

## Drop non values

df.dropna(axis=0)


# In[167]:


df.dropna(axis=1)


# In[168]:


df = pd.DataFrame(np.random.randn(5,3), index=['a','c','r','s','h'],
                     columns=['one','two','three'])


# In[169]:


df.head()


# In[170]:


df2=df.reindex(['a','b','c','r','d','s','m'])


# In[171]:


df2


# In[172]:


df2.dropna(axis=0)


# In[173]:


pd.isna(df2['one'])


# In[174]:


df2.fillna('Missing')


# In[175]:


df2['one'].values


# In[176]:


## Reading diff Data Source with the help of pandas 


# # CSV

# In[177]:


from io import StringIO,BytesIO


# In[178]:


data = ('col1,col2,col3\n'
               'x,y,1\n'
                'c,d,3')


# In[179]:


type(data)


# In[180]:


StringIO()


# In[181]:


pd.read_csv(StringIO(data))


# In[182]:


## Read from specific columns
df=pd.read_csv(StringIO(data),usecols=lambda x:x.upper() in ['col1','col3'])


# In[183]:


df


# In[184]:


df.to_csv('Test2.csv')


# In[185]:


## Specifying columns data type

data = ('a,b,c,d\n'
           '1,2,3,4\n'
            '5,6,7,8\n'
           '9,10,11,12')


# In[186]:


print(data)


# In[187]:


from io import StringIO,BytesIO
data = ('a,b,c,d\n'
           '1,2,3,4\n'
            '5,6,7,8\n'
           '9,10,11,12')
df=pd.read_csv(StringIO(data),dtype=int)


# In[188]:


df


# In[189]:


df['a']


# In[190]:


df=pd.read_csv(StringIO(data),dtype={'b':int,'c':int,'a':'Int64'})


# In[191]:


df


# In[192]:


df['a'][1]


# In[193]:


## Check the datatype
df.dtypes


# In[194]:


## Index Column and Traning delimiters


# In[ ]:





# In[195]:


data=('index,a,b,c\n'
     '4,apple,bat,5.8\n'
     '8,orange,cow,10')


# In[196]:


pd.read_csv(StringIO(data),index_col=0)


# In[197]:


data=('a,b,c\n'
     '4,apple,bat,\n'
     '8,orange,cow,')


# In[198]:


pd.read_csv(StringIO(data))


# In[199]:


pd.read_csv(StringIO(data),index_col=False)


# In[200]:


pd.read_csv(StringIO(data),usecols=['b','c'],index_col=False)


# In[201]:


## Quoting and Escape Characters very useful in NLP

data='a,b\n"Hello,\\"Bob\\",nice to see you",5'


# In[202]:


pd.read_csv(StringIO(data),escapechar='\\')


# In[203]:


## URL to CSV

df=pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.item',
                  sep='\t')


# In[207]:


df.head()


# # Read Json to CSV

# In[208]:


Data = '{"employee_name":"Rahul Srivastav","email":"rahulkumarrahul2002a@gmail.com","Job_profile":[{"title1":"Team Lead","title2":"Data Engg"}]}'
df1=pd.read_json(Data)


# In[209]:


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)


# In[210]:


df.head()


# In[211]:


## Convert json to csv


# In[212]:


df.to_csv('wine.csv')


# In[213]:


# Convert json to different json formate

df.to_json(orient="index")


# In[214]:


# Convert json to different json formate

df1.to_json()


# In[ ]:





# # Reading HTML content ** Read once again

# In[215]:


import html5lib
import pandas as pd


# In[216]:


url = 'https://www.fdic.gov/bank/individual/failed/banklist.html'
dfs = pd.read_html(url)


# In[217]:


type(dfs)


# In[218]:


url_mcc='https://en.wikipedia.org/wiki/mobile_country_code'
dfs=pd.read_html(url_mcc,match='Country',header=0)


# In[219]:


dfs[0]


# # Reading Excel Files ** Read once again

# In[220]:


df_excel=pd.read_excel('Excel_Sample.xlsx')


# In[221]:


df_excel.head()


# # Pickling:-All pandas obj are equipped with to pickle method which use pyhton's cPickle module to save data structures to disk using the pickle formate ** Read once again

# In[222]:


df_excel.to_pickle('df_excel')


# In[223]:


df=pd.read_pickle('df_excel')


# In[224]:


df.head()


# #  MatplotLib Tutorial:-It's numerical mathematics extension Numpy,it provides an object_orieanted API for embedding plots into apps using general -purpose GUI toolkits like Tkinter,wxPython,Qt,or GTK+
# 
# ## Iteasy to started for simple plots
# ## support for custom labels and texts
# ## Great control of every element in figure
# ## High quality output in many formats
# ## very customizable in general
# 

# In[225]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[226]:


import numpy as np


# In[227]:


## Simple Examples

x=np.arange(0,10)
y=np.arange(11,21)


# In[228]:


a=np.arange(40,50)
b=np.arange(50,60)


# In[229]:


##plotting using matplotlib 

##plt scatter

plt.scatter(x,y,c='b')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Graph in 2D')
plt.savefig('Test.png')


# In[230]:


y=x*x


# In[231]:


## plt plot

plt.plot(x,y,'r*',linestyle='dashed',linewidth=2, markersize=12)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('2d Diagram')


# In[232]:


## Creating Subplots

plt.subplot(2,2,1)
plt.plot(x,y,'r--')
plt.subplot(2,2,2)
plt.plot(x,y,'g*--')
plt.subplot(2,2,3)
plt.plot(x,y,'bo')
plt.subplot(2,2,4)
plt.plot(x,y,'go')


# In[233]:


x = np.arange(1,11) 
y = 7 * x + 60 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()


# In[234]:


np.pi


# In[235]:


# Compute the x and y coordinates for points on a sine curve 
x = np.arange(0, 4 * np.pi, 0.2) 
y = np.sin(x) 
plt.title("sine wave form") 

# Plot the points using matplotlib 
plt.plot(x, y) 
plt.show() 


# In[236]:


#Subplot()
# Compute the x and y coordinates for points on sine and cosine curves 
x = np.arange(0, 5 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
plt.subplot(2, 1, 1)
   
# Make the first plot 
plt.plot(x, y_sin,'b') 
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 1, 2) 
plt.plot(x, y_cos,'b') 
plt.title('Cosine')  
   
# Show the figure. 
plt.show()


# In[237]:


## Bar plot

x = [2,8,10] 
y = [11,16,9]  

x2 = [3,9,11] 
y2 = [6,15,7] 
plt.bar(x, y) 
plt.bar(x2, y2, color = 'g') 
plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis')  

plt.show()


# # Histograms

# In[238]:


a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a,bins=20)
plt.title("histogram")
plt.show


# In[239]:


data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# rectangular box plot
plt.boxplot(data,vert=True,patch_artist=True);  


# In[240]:


data


# # Pie chart

# In[241]:


# Data to plot

labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']


# explode 1st slice
explode = (0.2, 0, 0, 0)  



# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False)
plt.axis('equal')
plt.show()


# # Seaborn Tutorial
# 
# 
# 
# # Distribution plots
# 
# # .distplot
# # .jionplot
# # .pairplot
# 
# # Practise problem on IRIS Dataset

# In[242]:


import seaborn as sns


# In[243]:


df=sns.load_dataset("tips")


# In[244]:


df.head()


# In[245]:


df.dtypes


# # Correlation with Heatmap
# 
# # A correlation heatmap uses colored cells, typically in a monochromatic scale, to show a 2D correlation matrix (table) between two discrete dimensions or event types. It is very important in Feature Selection

# In[246]:


df.corr()


# In[247]:


sns.heatmap(df.corr())


# In[248]:


sns.jointplot(x='tip',y='total_bill',data=df,kind='hex')


# # Pair plot
# 
# A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value, like this: Pairs plots are just elaborations on this, showing all variables paired with all the other variables

# In[249]:


sns.pairplot(df)


# In[250]:


sns.pairplot(df,hue='sex')


# # Dist plot
# 
# Dist plot helps us to check the distribution of the columns feature

# In[251]:


sns.distplot(df['tip'])


# In[252]:


sns.distplot(df['tip'],kde=False,bins=10)


# # Categorical Plots
# 
# 
# Seaborn also helps us in doing the analysis on Categorical Data points. In this section we will discuss about
# .boxplot
# .violinplot
# .countplot
# .bar plot

# In[253]:


## Count plot

sns.countplot('day', data=df)


# In[255]:


## Count plot

sns.countplot(x='sex',data=df)


# In[256]:


##Bar plot

sns.barplot(x='total_bill',y='sex',data=df)


# # sns.barplot(x='sex',y='total_bill',data=df)

# In[257]:


sns.barplot(y='total_bill',x='smoker',data=df)


# In[258]:


df.head()


# # Box plot
# 
# A box and whisker plot (sometimes called a boxplot) is a graph that presents information from a five-number summary.

# In[259]:


sns.boxplot(x='sex',y='total_bill', data=df)


# In[260]:


sns.boxplot(x="day", y="total_bill", data=df)


# In[261]:


sns.boxplot(data=df,orient='v')


# In[262]:


# categorize my data based on some other categories


sns.boxplot(x="total_bill", y="day", hue="smoker",data=df)


# # Violin Plot
# 
# Violin plot helps us to see both the distribution of data in terms of Kernel density estimation and the box plot

# In[263]:


sns.violinplot(x="total_bill", y="day", data=df,palette='rainbow')


# In[264]:


## Practise Homework

iris = sns.load_dataset('iris')


# # Logistic Regression with Python
# For this lecture we will be working with the Titanic Data Set from Kaggle. This is a very famous data set and very often is a student's first step in machine learning!
# 
# We'll be trying to predict a classification- survival or deceased. Let's begin our understanding of implementing Logistic Regression in Python for classification.
# 
# We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
# 
# # Import Libraries
# Let's import some libraries to get started!

# In[265]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # The Data
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[266]:


train = pd.read_csv('titanic_train.csv')


# In[267]:


train.head()


# # Exploratory Data Analysis
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# # Missing Data
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[268]:


train.isnull()


# In[269]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[270]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[271]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[272]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[273]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[274]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[275]:


sns.countplot(x='SibSp',data=train)


# In[276]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# # Cufflinks for plots
# Let's take a quick moment to show an example of cufflinks!

# In[277]:


import cufflinks as cf
cf.go_offline()


# #train['Fare'].iplot(kind='hist',bins=30,color='green')

# # Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class. For example:

# In[278]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[279]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# Now apply that function!

# In[280]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Now let's check that heat map again!

# In[281]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

# In[282]:


train.drop('Cabin',axis=1,inplace=True)


# In[283]:


train.head()


# In[284]:


train.dropna(inplace=True)


# # Converting Categorical Features
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[285]:


train.info()


# In[286]:


pd.get_dummies(train['Embarked'],drop_first=True).head() ##dummy variable


# In[287]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[288]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[289]:


train.head()


# In[290]:


train = pd.concat([train,sex,embark],axis=1)


# In[291]:


train.head()


# Great! Our data is ready for our model!

# # Building a Logistic Regression model
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 
# # Train Test Split

# In[292]:


train.drop('Survived',axis=1).head()


# In[293]:


train['Survived'].head()


# In[294]:


from sklearn.model_selection import train_test_split


# In[295]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# # Training and Predicting

# In[296]:


from sklearn.linear_model import LogisticRegression


# In[297]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[298]:


predictions = logmodel.predict(X_test)


# In[299]:


from sklearn.metrics import confusion_matrix


# In[300]:


accuracy=confusion_matrix(y_test,predictions)


# In[301]:


accuracy


# In[302]:


from sklearn.metrics import accuracy_score


# In[303]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[304]:


predictions


# Let's move on to evaluate our model!

# # Evaluation
# We can check precision,recall,f1-score using classification report!

# In[305]:


from sklearn.metrics import classification_report


# In[306]:


print(classification_report(y_test,predictions))


# Not so bad! You might want to explore other feature engineering and the other titanic_text.csv file, some suggestions for feature engineering:
# 
# Try grabbing the Title (Dr.,Mr.,Mrs,etc..) from the name as a feature
# 
# Maybe the Cabin letter could be a feature
# 
# Is there any info you can get from the ticket?
# 
# 

# # Functions in Python
# 
#  Why Functions
#  
#  Function Definition
#  
#  Positional and keyword arguments in functions
#  

# In[307]:


num=24

if num%2==0:
    print("The no is even")
else:
     print("The no is odd")
    


# In[308]:


def even_odd(num):
    if num%2==0:
        print("The no is even")
    else:
         print("The no is odd") 


# In[309]:


even_odd(26)


# In[310]:


#print vs return

def hello_world():
    print("Hello Welcome")


# In[311]:


hello_world()


# In[312]:


val=hello_world()


# In[313]:


print(val)


# In[314]:


def hello_world():
    return "Hello Welcome"


# In[315]:


val=hello_world()


# In[316]:


print(val)


# In[317]:


def add_number(num1,num2):
    return num1+num2


# In[318]:


val=add_number(2,4)


# In[319]:


val


# In[320]:


#position argument
#keyword argument

def hello(name,age=21):
    print("My name is {} and age is {}".format(name,age))


# In[321]:


hello('Rahul')


# In[322]:


def hello(*args,**kwargs):
    print(args)
    print(kwargs)


# In[323]:


hello("Rahul","Srivastav",age=21,DOB=2002)


# In[324]:


lst=['Rahul', 'Srivastav']
dict_args={'age': 21, 'DOB': 2002}


# In[325]:


hello(*lst,**dict_args)


# In[326]:


lst=[1,2,3,4,5,6,7]


# In[327]:


def evenoddsum(lst):
    even_sum=0
    odd_sum=0
    for i in lst:
        if i%2==0:
            even_sum=even_sum+i
        else:
            odd_sum=odd_sum+i
    return even_sum,odd_sum               


# In[328]:


evenoddsum(lst)


# # Lamda function
# 
# anonymous function.
# 
# A function with no name

# In[329]:


def addition(a,b):
    return a+b


# In[330]:


addition(7,67)


# In[331]:


addition=lambda a,b:a+b


# In[332]:


addition(12,32)


# In[333]:


def even(num):
    if num%2==0:
        return True


# In[334]:


even(46)


# In[335]:


even1=lambda a:a%2==0


# In[336]:


even(12)


# In[337]:


def addition(x,y,z):
    return x+y+z


# In[338]:


addition(3,5,1)


# In[339]:


addition=lambda x,y,z:x+y+z


# In[340]:


addition(5,7,2)


# # Map Function in Python 

# In[341]:


def even_or_odd(num):
    if num%2==0:
        return "The number {} is Even".format(num)
    else:
        return "The number {} is odd".format(num)


# In[342]:


even_or_odd(12)


# In[343]:


lst=[1,2,3,4,5,6,7,8,9,24,56,78]


# In[344]:


list(map(even_or_odd,lst))
#Map function returns an iterator.If you want to get the result as a list.


# # Filter Function in Python 

# In[345]:


def even(num):
    if num%2==0:
        return True


# In[346]:


lst=[1,2,3,4,5,6,7,8,9,0]


# In[347]:


list(filter(even,lst))


# In[348]:


list(filter(lambda num:num%2==0,lst))


# In[349]:


list(map(lambda num:num%2==0,lst))


# # List Comprehension
# 
# It provide a concise way to Create list.it consist of brackets containing an
# expression followed by a for clause,then zero or more for or if clauses.it means you can put in all kinds of object in list

# In[350]:


lst1=[]
def lst_square(lst):
    for i in lst:
        lst1.append(i*i)
    return lst1


# In[351]:


lst_square([1,2,3,4,5,6,7])


# In[352]:


lst=[1,2,3,4,5,6,7]


# In[353]:


[i*i for i in lst]


# # List Comprehension Example

# In[354]:


lst1=[i*i for i in lst if i%2==0]


# In[355]:


print(lst1)


# In[356]:


lst1=[i*i for i in lst if i%2!=0]


# In[357]:


print(lst1)


# # String Formatting in Python

# In[358]:


print("Hello Everyone")


# In[359]:


def greeting(name):
            return "Hello {}.Welcome the community".format(name)


# In[360]:


greeting("Rahul")


# In[361]:


def welcome_email(name, age):
    return "Welcome {name}. Your age is {age}".format(name=name, age=age)


# In[362]:


welcome_email('Rahul',21)


# # Python List Ilerables vs Iterators

# In[363]:


# List is Iterable

lst=[1,2,3,4,5,6,7]

for i in lst:
    print(i)


# In[364]:


lst=iter(lst)


# In[365]:


lst


# In[366]:


next(lst)


# In[367]:


for i in lst:
    print(i)


# # OPPs Tutorial in Python

# In[368]:


class Car():
    def __init__(self,window,door,enginetype):
        self.windows = window
        self.doors = door
        self.enginetype = enginetype
    def self_driving(self):
        return "This is a {} car".format(self.enginetype)


# In[369]:


car1 = Car(4, 5, "Petrol")


# In[370]:


car1.self_driving()


# In[371]:


car2 = Car(3, 4, "Diesel")


# In[372]:


print(car1.windows)


# In[373]:


print(car2.doors)


# In[374]:


print(car2.enginetype)


# In[ ]:





# In[375]:


dir(car1)


# In[ ]:





# In[376]:


car1.windows=5
car1.doors=4


# In[377]:


print(car1.windows)


# In[378]:


car2=car()


# In[379]:


car2.windows=3
car2.doors=2


# In[380]:


print(car2.windows)


# In[381]:


car2.engintype="Petrol"


# In[382]:


print(car2.engintype)


# # Python Exception Handling

# In[383]:


try:
    ##code block where exception can occur
    a=1
    b="s"
    c=a+b
except NameError as ex1:
    print("The user have not defined the variable")
except Exception as ex:
    print(ex)


# In[384]:


a=b


# In[385]:


a=1
b="s"
c=a+b


# In[ ]:


try:
    ##code block where exception can occur
    a=int(input("Enter the 1st no" ))
    b=int(input("Enter the 2nd no" ))
    c=a/b
    d=a*b
    e=a+b
except NameError:
    print("The user have not defined the variable")
except ZeroDivisionError:
    print("please provide number greater than 0")
except TypeError:
    print("Try to make the data type similar")
except Exception as ex:
    print(ex)
else:
    print(c)
finally:
    print("The execution is done")


# # Custom Exception

# In[ ]:


class Error(Exception):
    pass

class dobException(Error):
    pass

class customgeneric(Error):
    pass


# In[ ]:


year=int(input("Enter the year of Birth"))
age=2024-year
try:
    if age<=30 & age>20:
         print("The year age is valid. you can apply")
    else:
        raise dobException
except dobException:
    print("The year age is not valid. you can't apply")


# # Pyforest:-

# In[ ]:


pip install pyforest


# In[ ]:





# In[ ]:


df=pd.read_csv("medals.csv")


# In[ ]:


df.head()


# In[ ]:


import pandas as pd


# In[ ]:


lst1=[1,2,3,4,5]
lst2=[3,4,5,6,7]


# In[ ]:


plt.plot(lst1,lst2)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()


# In[ ]:


np.array([1,2,3,4,5])


# In[ ]:


df1=pd.read_csv('mercedesbenz.csv')


# In[ ]:


df1.head()


# In[ ]:


sns.distplot(df1['y'])


# # Ptyhon OOps-Inheritances

# In[ ]:


## All the class variable are public
## Car Blueprint
class Car():
    def __init__(self,window,door,enginetype):
        self.windows = window
        self.doors = door
        self.enginetype = enginetype
    def drive(self):
        print("The Person Drive The Car")


# In[ ]:


car=Car(4,5,"Diesel")


# In[ ]:


car.windows


# In[ ]:


car.drive()


# In[ ]:


class Car:
    def __init__(self, window, door, engine_type):
        self.window = window
        self.door = door
        self.engine_type = engine_type

class Audi(Car):
    def __init__(self, window, door, engine_type, enable_ai):
        super().__init__(window, door, engine_type)
        self.enable_ai = enable_ai

    def self_driving(self):
        print("Audi supports self-driving")


# In[ ]:


audi_q7 = Audi(4, 5, "Diesel", True)


# In[ ]:


dir(audi_q7)


# In[ ]:


audi_q7.self_driving()


# # Univariate,Bivariate and MultiVariate Analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# # Univariate Analysis

# In[ ]:


df_setosa=df.loc[df['species']=='setosa']


# In[ ]:


df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']


# In[ ]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('Petal length')
plt.show()


# # Bivariate Analysis

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.FacetGrid(df, hue="species", height=5).map(plt.scatter, "petal_length", "sepal_width").add_legend()
plt.show()


# # Multivariate Analysis 

# In[ ]:


sns.pairplot(df,hue="species",size=3)


# # Ridge and LAsso Regression implementation //////Read once again**

# In[ ]:


from sklearn.datasets import load_boston
housing = fetch_boston_house_prices()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df=fetch_boston_house_prices.csv()


# In[ ]:


df


# In[ ]:


dataset=pd.DataFrame(df.data)
print(dataset.head())


# In[ ]:


datset.columns=df.feature_names


# In[ ]:


detaset.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Linear Regression //read once again**

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[ ]:





# In[ ]:





# # Lasso Regression

# In[ ]:


# Example code to define X and y
import numpy as np
from sklearn.linear_model import Lasso

# Generate some example data
np.random.seed(0)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.rand(100)  # 100 target values

# Create a Lasso regressor object
lasso_regressor = Lasso(alpha=0.1)

# Fit the model using the defined X and y
lasso_regressor.fit(X, y)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[ ]:


lasso = Lasso()


# In[ ]:


parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}


# In[ ]:


lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)


# In[ ]:


lasso_regressor.fit(X, y)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

# Instantiate Lasso regression model
lasso = Lasso()

# Define parameters grid for Lasso
lasso_parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

# Instantiate GridSearchCV for Lasso
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring='neg_mean_squared_error', cv=5)

# Fit Lasso model
lasso_regressor.fit(X, y)

# Print best parameters and score for Lasso
print("Lasso Best Parameters:", lasso_regressor.best_params_)
print("Lasso Best Score:", lasso_regressor.best_score_)

# Instantiate Ridge regression model
ridge = Ridge()

# Define parameters grid for Ridge
ridge_parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}

# Instantiate GridSearchCV for Ridge
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring='neg_mean_squared_error', cv=5)

# Fit Ridge model
ridge_regressor.fit(X, y)

# Print best parameters and score for Ridge
print("Ridge Best Parameters:", ridge_regressor.best_params_)
print("Ridge Best Score:", ridge_regressor.best_score_)

# Make predictions using Lasso and Ridge models
prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)


# In[ ]:


prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)


# In[ ]:


import seaborn as sns

sns.distplot(y_test-prediction_lasso)


# In[ ]:


import seaborn as sns

sns.distplot(y_test-prediction_ridge)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Multicollinearity in Linear Regression

# In[ ]:


import pandas as pd


# In[ ]:


pip install statsmodels


# In[ ]:


import statsmodels.api as sm
df_adv = pd.read_csv('Advertising (1).csv', index_col=0)
X = df_adv[['TV', 'radio', 'newspaper']]
y = df_adv['sales']
df_adv.head()


# In[ ]:


X


# In[ ]:


## fit aOLS model with intercept on TV and Ratio

X = sm.add_constant(X)
model=sm.OLS(y,X).fit()


# In[ ]:


model.summary()


# In[ ]:


import matplotlib.pyplot as plt
X.iloc[:,1:].corr()


# In[ ]:


df_salary = pd.read_csv('Salary_Data.csv')
df_salary.head()


# In[ ]:


X = df_salary[['YearsExperience', 'Age']]
y = df_salary['Salary']


# In[ ]:


## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
model= sm.OLS(y, X).fit()


# In[ ]:


model.summary()


# In[ ]:


X.iloc[:,1:].corr()


# # T Test
# At-test is a type of inferentail statistic which is used to determind if there is a significant diff between  the means of two group which may be related in certain features.
# 
# it has two type:-
# 
# one-sampled t-test
# two-sampled t-test

# In[ ]:


ages=[10,20,35,50,28,40,55,18,16,55,30,25,43,18,30,28,14,24,16,17,32,35,26,27,65,18,43,23,21,20,19,70]


# In[ ]:


len(ages)


# In[ ]:


import numpy as np
ages_mean=np.mean(ages)
print(ages_mean)


# In[ ]:


## lets take sample:-

sample_size = 10
ages_sample = np.random.choice(ages, sample_size)


# In[ ]:


ages_sample


# In[ ]:


from scipy.stats import ttest_1samp


# In[ ]:


ttest,p_value=ttest_1samp(ages_sample,30)


# In[ ]:


print(p_value)


# In[ ]:


if p_value < 0.05:  #alpha value is 0.05 or 5%
    print("we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")  


# # Some More Example
# Consider the age of student in a college and in Class A

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import math
np.random.seed(6)
school_ages=stats.poisson.rvs(loc=18,mu=35,size=1500)
classA_ages=stats.poisson.rvs(loc=18,mu=30,size=60)


# In[ ]:


classA_ages.mean()


# In[ ]:


_,p_values=stats.ttest_1samp(a=classA_ages,popmean=school_ages.mean())


# In[ ]:


p_values


# In[ ]:


school_ages.mean()


# In[ ]:


if p_values < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# # Two-sample T-test With Python
# The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different. The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test

# In[ ]:


np.random.seed(12)
ClassB_ages=stats.poisson.rvs(loc=18,mu=33,size=60)
ClassB_ages.mean()


# In[ ]:


_,p_value=stats.ttest_ind(a=classA_ages,b=ClassB_ages,equal_var=False)


# In[ ]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# # Paired T-test With Python
# When you want to check how different samples from the same group are, you can go for a paired T-test

# In[ ]:


weight1=[25,30,28,35,28,34,26,29,30,26,28,32,31,30,45]
weight2=weight1+stats.norm.rvs(scale=5,loc=-1.25,size=15)


# In[ ]:


print(weight1)
print(weight2)


# In[ ]:


weight_df=pd.DataFrame({"weight_10":np.array(weight1),
                         "weight_20":np.array(weight2),
                       "weight_change":np.array(weight2)-np.array(weight1)})


# In[ ]:


weight_df


# In[ ]:


_,p_value=stats.ttest_rel(a=weight1,b=weight2)


# In[ ]:


print(p_value)


# In[ ]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# # Correlation

# In[ ]:


import seaborn as sns
df=sns.load_dataset('iris')


# In[ ]:


df.shape


# In[ ]:


df.corr()


# In[ ]:


sns.pairplot(df)


# # Anova Test(F-Test)
# The t-test works well when dealing with two groups, but sometimes we want to compare more than two groups at the same time.
# 
# For example, if we wanted to test whether petal_width age differs based on some categorical variable like species, we have to compare the means of each level or group the variable

# # One Way F-test(Anova) :-
# It tell whether two or more groups are similar or not based on their mean similarity and f-score.
# 
# Example : there are 3 different category of iris flowers and their petal width and need to check whether all 3 group are similar or not

# In[ ]:


import seaborn as sns
df1=sns.load_dataset('iris')


# In[ ]:


df1.head()


# In[ ]:


df_anova = df1[['petal_width','species']]


# In[ ]:


# Assuming df_anova is a DataFrame with a 'species' column
grps = df_anova['species'].unique()


# In[ ]:


d_data = {grp: df_anova['petal_width'][df_anova.species == grp] for grp in grps}


# In[ ]:


d_data


# In[ ]:


F, p = stats.f_oneway(d_data['setosa'], d_data['versicolor'], d_data['virginica'])


# In[ ]:


print(p)


# In[ ]:


if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


# # Chi-Square Test-
# The test is applied when you have two categorical variables from a single population. It is used to determine whether there is a significant association between the two variables.

# In[ ]:


import scipy.stats as stats


# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
dataset=sns.load_dataset('tips')


# In[ ]:


dataset.head()


# In[ ]:


dataset_table=pd.crosstab(dataset['sex'],dataset['smoker'])
print(dataset_table)


# In[ ]:


dataset_table.values


# In[ ]:


#Observed Values

Observed_Values = dataset_table.values 
print("Observed Values:\n", Observed_Values)


# In[ ]:


val=stats.chi2_contingency(dataset_table)


# In[ ]:


val


# In[ ]:


Expected_Values=val[3]


# In[ ]:


no_of_rows=len(dataset_table.iloc[0:2,0])
no_of_columns=len(dataset_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05


# In[ ]:


from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]


# In[ ]:


print("chi-square statistic:-",chi_square_statistic)


# In[ ]:


critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)


# In[ ]:


#p-value

p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('p-value:',p_value)


# In[ ]:


if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# # Building Machine Learning Pipelines: Data Analysis Phase
#  we will focus on creating Machine Learning Pipelines considering all the life cycle of a Data Science Projects. This will be important for professionals who have not worked with huge dataset.

# # Project Name: House Prices: Advanced Regression Techniques
# The main aim of this project is to predict the house price based on various features which we will discuss as we go ahead
# 
# Dataset to downloaded from the below link                              
# 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# # All the Lifecycle In A Data Science Projects
# 
# Data Analysis                                                             
# Feature Engineering                                                           
# Feature Selection                                                            
# Model Building                                                              
# Model Deployment                                                          

# In[ ]:


## Data Analysis Phase
## Main aim is to understand more about the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
## Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)


# In[ ]:


dataset=pd.read_csv('train.csv')

## print shape of dataset with rows and columns
print(dataset.shape)


# In[ ]:


## print the top5 records
dataset.head()


# # In Data Analysis We will Analyze To Find out the below stuff
# 
# Missing Values                                                               
# All The Numerical Variables                                                   
# Distribution of the Numerical Variables                                      
# Categorical Variables                          
# Cardinality of Categorical Variables                               
# Outliers                                                              
# Relationship between independent and dependent feature(SalePrice)

# # Missing Values

# In[ ]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')


# # Since they are many missing values, we need to find the relationship between missing values and Sales Price
# 
# Let's plot some diagram for this relationship

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

for feature in features_with_na:
    data = dataset.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # let's calculate the median SalePrice where the information is missing or present
    median_prices = data.groupby(feature)['SalePrice'].median()
    
    # Plotting
    median_prices.plot(kind='bar')
    plt.title(f"Median SalePrice by {feature}")
    plt.xlabel("Missing Data" if data[feature].dtype == 'int' else "Categories")
    plt.ylabel("Median SalePrice")
    plt.xticks(rotation=0)
    plt.show()


# Here With the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section
# 
# From the above dataset some of the features like Id is not required

# In[ ]:


print("Id of Houses {}".format(len(dataset.Id)))


# # Numerical Variables

# In[ ]:


# list of numerical variables
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
dataset[numerical_features].head()


# # Temporal Variables(Eg: Datetime Variables)
# 
# From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. We will be performing this analysis in the Feature Engineering which is the next video

# In[ ]:


# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature


# In[ ]:


# let's explore the content of these year variables
for feature in year_feature:
    print(feature, dataset[feature].unique())


# In[ ]:


## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[ ]:


year_feature


# In[ ]:


## Here we will compare the difference between All years feature with SalePrice

for feature in year_feature:
    if feature!='YrSold':
        data=dataset.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[ ]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[ ]:


discrete_feature


# In[ ]:


dataset[discrete_feature].head()


# In[ ]:


## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


## There is a relationship between variable number and SalePrice


# # Continuous Variable

# In[ ]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[ ]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# # In Data Analysis We will Analyze To Find out the below stuff
# Missing Values                                                               
# All The Numerical Variables                                                   
# Distribution of the Numerical Variables                                       
# Categorical Variables                                                        
# Cardinality of Categorical Variables                                         
# Outliers                                                                
# Relationship between independent and dependent feature(SalePrice)

# # Missing Values

# In[ ]:





# In[ ]:


import numpy as np

# Here we will check the percentage of nan values present in each feature
# Step 1: make the list of features which have missing values
features_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1]

# Step 2: print the feature name and the percentage of missing values
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean() * 100, 2), '% missing values')


# In[ ]:


import matplotlib.pyplot as plt

for feature in year_feature:
    if feature != 'YrSold':
        data = dataset.copy()
        # Calculate the difference between the year variable and the year the house was sold for
        data[feature] = data['YrSold'] - data[feature]

        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(f'Difference between {feature} and YrSold')
        plt.ylabel('SalePrice')
        plt.title(f'Scatter plot for {feature} vs SalePrice')
        plt.show()


# In[ ]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[ ]:


discrete_feature


# In[ ]:


dataset[discrete_feature].head()


# In[ ]:


## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


## There is a relationship between variable number and SalePrice


# # Continuous Variable

# In[ ]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[ ]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[ ]:





# # Exploratory Data Analysis Part 2

# In[ ]:


## We will be using logarithmic transformation


for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# # Outliers

# In[ ]:


for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# # Categorical Variables

# In[ ]:


categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
categorical_features


# In[ ]:


dataset[categorical_features].head()


# In[ ]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))


# In[ ]:


## Find out the relationship between categorical variable and dependent feature SalesPrice


# In[ ]:


for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# # Advanced Housing Prices- Feature Engineering
# The main aim of this project is to predict the house price based on various features which we will discuss as we go ahead
# 
# Dataset to downloaded from the below link                         
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data     
# 
# We will be performing all the below steps in Feature Engineering
# 
# Missing values                                                               
# Temporal variables                                                          
# Categorical variables: remove rare labels                                     
# Standarise the values of the variables to the same range                     

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


dataset=pd.read_csv('train.csv')
dataset.head()


# In[ ]:


## Always remember there way always be a chance of data leakage so we need to split the data first and then apply feature
## Engineering
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)


# In[ ]:


X_train.shape, X_test.shape


# # Missing Values

# In[ ]:


## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[ ]:


## Replace missing value with a new label
def replace_cat_feature(dataset,features_nan):
    data=dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(dataset,features_nan)

dataset[features_nan].isnull().sum()


# In[ ]:


dataset.head()


# In[ ]:





# In[ ]:


## Now lets check for numerical variables the contains missing values
numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))


# In[ ]:





# In[ ]:


## Replacing the numerical Missing Values

for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    median_value=dataset[feature].median()
    
    ## create a new feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
    
dataset[numerical_with_nan].isnull().sum()


# In[ ]:


dataset.head(51)


# In[ ]:


## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[ ]:


dataset.head()


# In[ ]:


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# # Numerical Variables
# Since the numerical variables are skewed we will perform log normal distribution

# In[ ]:


dataset.head()


# In[ ]:


import numpy as np
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[ ]:


dataset.head()


# # Handling Rare Categorical Feature
# We will remove categorical variables that are present less than 1% of the observations

# In[ ]:


categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']


# In[ ]:


categorical_features


# In[ ]:


for feature in categorical_features:
    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')
    


# In[ ]:


dataset.head(100)


# In[ ]:


for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# In[ ]:


dataset.head(10)


# In[ ]:


scaling_feature=[feature for feature in dataset.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature)


# In[ ]:


scaling_feature


# In[ ]:


dataset.head()


# # Feature Scaling

# In[ ]:


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[ ]:


scaler.transform(dataset[feature_scale])


# In[ ]:


# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                    axis=1)


# In[ ]:


data.head()


# In[ ]:


data.to_csv('X_train.csv',index=False)


# # Feature Selection Advanced House Price Prediction
# 
# The main aim of this project is to predict the house price based on various features which we will discuss as we go ahead
# 
# Dataset to downloaded from the below link                       
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## for feature slection

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


dataset=pd.read_csv('X_train.csv')


# In[ ]:


dataset.head()


# In[ ]:


## Capture the dependent feature
y_train=dataset[['SalePrice']]


# In[ ]:


## drop dependent feature from dataset
X_train=dataset.drop(['Id','SalePrice'],axis=1)


# In[ ]:


### Apply Feature Selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)


# In[ ]:


feature_sel_model.get_support()


# In[ ]:


# Convert NumPy array to a list of selected feature indices
selected_indices = np.where(feature_sel_model.get_support())[0]

# Get the names of the selected features
selected_feat = [X_train.columns[i] for i in selected_indices]

# Print some stats
print('Total features: {}'.format(X_train.shape[1]))
print('Selected features: {}'.format(len(selected_feat)))
print('Features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))


# In[ ]:


selected_feat


# In[ ]:


X_train=X_train[selected_feat]


# In[ ]:


X_train.head()


# # Perform Metrics on MutiClass Classification Problems

# In[ ]:


from sklearn import metrics


# In[ ]:


C="Cat"
D="Dog"
F="Fox"


# the precision for the cat is the no of correctly predicated cat out of all predicted Cat
# 
# the recall for Cat is the no of correctly predicated cat photo out of the no of actual Cat

# In[ ]:


# Import necessary library
from sklearn import metrics

# True values
y_true = ['C', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
# Predicted values
y_pred = ['C', 'C', 'C', 'C', 'D', 'F', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'F', 'F', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D']

# Print the confusion matrix
print(metrics.confusion_matrix(y_true, y_pred))

# Print the precision, recall, F1-score, and support for each class
print(metrics.classification_report(y_true, y_pred, digits=3))


# # K Nearest Neighbors with Python
# You've been given a classified data set from a company! They've hidden the feature column names but have given you the data and the target classes.
# 
# We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.
# 
# Let's grab it and use it!
# 
# # Import Libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Get the Data
# 
# Set index_col=0 to use the first column as the index.

# In[ ]:


df = pd.read_csv("Classified Data",index_col=0)


# In[ ]:


df.head()


# # Standardize the Variables
# Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.
# 
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[ ]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[ ]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# # Using KNN
# 
# Remember that we are trying to come up with a model to predict whether someone will TARGET CLASS or not. We'll start with k=1.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# # Predictions and Evaluations
# 
# Let's evaluate our KNN model!

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# # Choosing a K Value
# 
# Let's go ahead and use the elbow method to pick a good K Value:

# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Here we can see that that after arouns K>23 the error rate just tends to hover around 0.06-0.05 Let's retrain the model with that and check the classification report!

# In[ ]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# # K Nearest Neighbors with Python
# You've been given a classified data set from a company! They've hidden the feature column names but have given you the data and the target classes.
# 
# We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.
# 
# Let's grab it and use it!
# 
# # Import Libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Get the Data
# Set index_col=0 to use the first column as the index.

# In[ ]:


df = pd.read_csv("Classified Data",index_col=0)


# In[ ]:


df.head()


# # Standardize the Variables
# Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[ ]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[ ]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # Pair Plot

# In[ ]:


import seaborn as sns

sns.pairplot(df,hue='TARGET CLASS')


# # Train Test Split
# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# # Using KNN
# Remember that we are trying to come up with a model to predict whether someone will TARGET CLASS or not. We'll start with k=1.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value:

# In[ ]:


accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    accuracy_rate.append(score.mean())


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    error_rate.append(1-score.mean())


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
#plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
  #       markerfacecolor='red', markersize=10)
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Here we can see that that after arouns K>23 the error rate just tends to hover around 0.06-0.05 Let's retrain the model with that and check the classification report!
# 
# 

# In[ ]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# # Credit Card Kaggle- Fixing Imbalanced Dataset
# 
# # Context
# It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
# 
# # Content
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# # Inspiration
# Identify fraudulent credit card transactions.
# 
# Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
# 
# # Acknowledgements
# The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project

# In[471]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[472]:


data = pd.read_csv('creditcard.csv',sep=',')
data.head()


# In[473]:


df.info()


# In[474]:


#Create independent and Dependent Features
columns = data.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data[columns]
Y = data[target]
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# # Exploratory Data Analysis

# In[475]:


data.isnull().values.any()


# In[476]:


count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[477]:


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[478]:


print(fraud.shape,normal.shape)


# In[481]:


pip install imbalanced-learn


# In[ ]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss


# In[ ]:




