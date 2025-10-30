# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output

#code:
```
import numpy as np
import pandas as pd
data=pd.read_csv("/content/Data_set.csv")
data
```

#output:

<img width="1668" height="651" alt="image" src="https://github.com/user-attachments/assets/41979e4a-c2e8-4feb-9507-4853b7b7b372" />


#code:
```
data.head(4)
```

#output:


<img width="1797" height="270" alt="image" src="https://github.com/user-attachments/assets/f58e2e81-6909-4a9e-be59-58e251fecd85" />


#code:
```
data.tail(7)
```


#output:


<img width="1745" height="383" alt="image" src="https://github.com/user-attachments/assets/b70e7a89-747e-4690-a413-aedb6880babc" />


#code:
```
data.isnull()
```

#output:


<img width="1417" height="581" alt="image" src="https://github.com/user-attachments/assets/4b7ec091-2f6f-4675-8ad6-09d5308b535c" />



#code:
```
data.notnull()
```

#output:


<img width="1391" height="580" alt="image" src="https://github.com/user-attachments/assets/eaa6cd08-8b55-4034-840a-c7dfea7a2416" />


#code:
```
data.isnull().sum()
```

#output:

<img width="492" height="495" alt="image" src="https://github.com/user-attachments/assets/7dfe7dd5-4d18-4eb1-b35f-4d9d360aa3ef" />



#code:
```
data.isnull().any()
```


#output:


<img width="497" height="503" alt="image" src="https://github.com/user-attachments/assets/1c440b0b-fe00-4b4a-b64c-d9d4d7ce11c1" />


#code:
```
data.dropna(axis=1)
```


#output:


<img width="742" height="565" alt="image" src="https://github.com/user-attachments/assets/49bc5294-42cb-4ca5-a031-6039a5bf8d26" />


#code:
```
data.dropna(axis=0)
```

#output:


<img width="1628" height="575" alt="image" src="https://github.com/user-attachments/assets/1361cd8b-0246-4250-914e-8c91876252d1" />



#code:
```
data.fillna(0)
```

#output:


<img width="1617" height="568" alt="image" src="https://github.com/user-attachments/assets/d7aec580-582f-48d8-b3c6-96990419c50c" />



#code:
```
data.fillna(method="ffill")
```

#output:


<img width="1695" height="617" alt="image" src="https://github.com/user-attachments/assets/4347fcd9-b3ea-419a-9fc8-756628bde85a" />


#code:
```
data.bfill()
```
#output:


<img width="1583" height="567" alt="image" src="https://github.com/user-attachments/assets/3afe162c-e511-4290-a52b-de51b3cd700f" />



#code:
```
data.fillna({'REGNO':0, 'NAME':'PRAVEEN'})
```


#output:



<img width="1622" height="581" alt="image" src="https://github.com/user-attachments/assets/d34c59e1-3ece-4607-8c15-e4bc4617fa11" />



#code:
```
ir= pd.read_csv("iris.csv")
ir
```

#output:



<img width="800" height="612" alt="image" src="https://github.com/user-attachments/assets/199dee45-7339-4c32-ba9d-605cf32d6734" />


#code:
```
ir.describe()
```


#output:



<img width="678" height="426" alt="image" src="https://github.com/user-attachments/assets/fa703221-9188-4936-9563-2a7e4ec6c872" />



#code:
```
import seaborn as sns
sns.boxplot(x="sepal_width",data=ir)
```

#output:



<img width="852" height="661" alt="image" src="https://github.com/user-attachments/assets/8a1bf480-b409-402d-88c7-b315d3e4b8dc" />



#code:
```
q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```


#output:



<img width="467" height="175" alt="image" src="https://github.com/user-attachments/assets/ee2eaa60-cdbd-446a-af18-3665cea3911c" />




#code:
```
rid=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid['sepal_width']
```



#output:




<img width="852" height="338" alt="image" src="https://github.com/user-attachments/assets/c508de5b-e6eb-4021-8798-6c2d0f458981" />




#code:
```
rid=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid
```



#output:



<img width="902" height="602" alt="image" src="https://github.com/user-attachments/assets/e9ded005-d69f-4f3e-a8eb-27d17050b704" />




#code:
```
rid=ir[((ir.sepal_width>(q1-1.5*iqr))&(ir.sepal_width<(q3+1.5*iqr)))]
rid['sepal_width']
```



#output:



<img width="867" height="652" alt="image" src="https://github.com/user-attachments/assets/99a25d4c-742f-4ef2-a441-dc012fba46df" />




#code:
```
import numpy as np
import scipy.stats as stats
z=np.abs(stats.zscore(ir.sepal_width))
z
```



#output:




<img width="923" height="808" alt="image" src="https://github.com/user-attachments/assets/db914b4a-fbc1-44cd-a31b-8b5491cf8ae6" />







# Result:
Thus the programs are executed successfully.

