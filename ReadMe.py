#!/usr/bin/env python
# coding: utf-8

# In[40]:


#description: c'est un système qui détecte le cancer de sein basée sur le dataset....
#import librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


#read the data
df=pd.read_csv('C:/Users/lenovo/Desktop/MachineLearning/data.csv')


# In[42]:


df


# In[43]:


#afficher les 7 lignes du dataset
df.head(7)


# In[44]:


#calculer le nombre des lignes et des colonnes du data set
df.shape


# In[45]:


#calculer le nombre des valeurs nulles (NaN,NAN,na)
df.isna().sum()


# In[46]:


#supprimer la colonne qui contient les valeurs nulles
df=df.dropna(axis=1)


# In[47]:


#réafficher le nouveau nombre des lignes et des colonnes
df.shape


# In[48]:


#afficher le nombre des cellules Malognant/malin(M) ou Benign/benin(B)
df['diagnosis'].value_counts()


# In[49]:


#visualiser le diagnosis
sns.countplot(df['diagnosis'],label='count')


# In[50]:


#afficher les types de données
df.dtypes


# In[51]:


#encode the categorical data values

from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1]


# In[52]:


#créer une paire de plot
#visualiser le diagnostic dans les autres colonnes
sns.pairplot(df.iloc[:,1:5], hue='diagnosis')


# In[53]:


#afficher les 5 lignes de notre nouvelle data
df.head(5)


# In[54]:


#la corrélation des colonnes
df.iloc[:,1:12]


# In[55]:


#visualiser la correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(),annot=True,fmt='.0%')


# In[56]:


#diviser le dataset en independent(X) and dependent(Y) data sets
#les changer en arrays
#les features qui peuvent detecter si le patient a le cancer ou pas
X=df.iloc[:,2:31].values
#target(si le patient a breast cancer ou nn)
Y=df.iloc[:,1].values


# In[57]:


#diviser le dataset en 75% training et 25% testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=0)


# In[58]:


#uniformisation d'echelle+normalisation
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_train


# In[79]:


#creation des modèles
def models(X_train,Y_train):
    #on commence par Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    #Desicion Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)
    #afficher lmodels occuracy
    print('[0]Logistic Regression Training Accuracy',log.score(X_train,Y_train))
    print('[1]Desicion Tree Classifier Training Accuracy',tree.score(X_train, Y_train))
    return log,tree


# In[67]:


#avoir tt les modeles
model=models(X_train,Y_train)


# In[71]:


#tester le model accuracy de testing data sur la matrice de confusion
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    print('Model',i)
    cm=confusion_matrix(Y_test,model[i].predict(X_test))
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[1][0]
    FP=cm[0][1]
    print(cm)
    print('Testing Accuracy du modele=',(TP+TN)/(TP+TN+FN+FP))
    print


# In[75]:


#afficher la precision le recall le f1-score et le support + accuracy des 2 modeles
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
    print('Modele', i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print(accuracy_score(Y_test,model[i].predict(X_test)))


# In[77]:


#afficher le prédiction de Desicion Tree Classifier
pred=model[1].predict(X_test)
print('les valeurs prédites sont', pred)
print()
print('les valeurs réelles sont', Y_test)


# In[78]:


#afficher le prédiction de Logistic Regression
pred=model[0].predict(X_test)
print('les valeurs prédites sont', pred)
print()
print('les valeurs réelles sont', Y_test)


# In[ ]:




