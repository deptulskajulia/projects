import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#pliki csv uzyskane z poprzedniego projektu
df_z = pd.read_csv('C:/Users/Julia\Desktop/studia/5sem/neuronowe/projekt_1/rytm_serca/dane/Dane_zdrowi.csv')
df_nad = pd.read_csv('C:/Users/Julia\Desktop/studia/5sem/neuronowe/projekt_1/rytm_serca/dane/Dane_nadcisnienie.csv')
#%% dodanie targetu 
df_z['Target'] = pd.Series([1 for x in range(17)]).values
df_nad['Target'] = pd.Series([2 for x in range(36)]).values
#%% polaczenie obu kolumn i przygotowanie
df = df_z.append(df_nad)
df.rename(columns = {'Unnamed: 0':'Pacjent'}, inplace = True)
df.reset_index(drop=True, inplace=True)
#%% #korelacja
df_korelacja = df.corr(method ='pearson')

#%% skalowanie i wycentrowanie
from sklearn.preprocessing import StandardScaler

temp_n = df.loc[:, 'Srednie cisnienie skurczowe':'Ilosc spadków cisnienia skurczowego SBP przy wydechu [w %]'].values #bez targetu i nazw osob
#usuwamy nazyw pacjentow i target
temp = StandardScaler().fit_transform(temp_n) 


#sprawdzamy czy rozklad jest normalny
print(np.mean(temp),np.std(temp))
#%% PCA - 2 skladowe
from sklearn.decomposition import PCA

pca_df2_ = PCA(n_components = 2)
p = pca_df2_.fit_transform(temp)

pca_df2 = pd.DataFrame(data = p, columns = ['Skladowa glowna 1', 'Skladowa glowna 2'])

print('Wyjasniona zmiennosc przez glowne skladowe: {}'.format(pca_df2_.explained_variance_ratio_))

#%% wykres 2 skladowych
plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Skladowa glowna- 1')
plt.ylabel('Skladowa glowna - 2')
targets = [1,2]
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = df['Target'] == target
    plt.scatter(pca_df2.loc[indicesToKeep, 'Skladowa glowna 1']
               , pca_df2.loc[indicesToKeep, 'Skladowa glowna 2'], c = color, s = 50)

plt.legend(['Zdrowi','Nadcisnienie'],prop={'size': 15})

#%% pca dla wszystkich kolumn
pca = PCA()
p = pca.fit_transform(temp)

kolumny_1 =[]
for i in range(0,17): 
    kolumny_1.insert(i+2,"{}".format(i))
    
#%% funkcja biplot - pokazanie powiazan miedzy zmiennymi objasniajacymi i ich wplyw
def biplot(score,coef,labels=None):
 
    xs = score[:,0]
    ys = score[:,1]
    n = coef.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,
                s=25, 
                color='orange')
 
    for i in range(n):
        plt.arrow(0, 0, coef[i,0], 
                  coef[i,1],color = 'purple',
                  alpha = 0.5)
        plt.text(coef[i,0]* 1.15, 
                 coef[i,1] * 1.15, 
                 labels[i], 
                 color = 'darkblue', 
                 ha = 'center', 
                 va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))    
 
 
    plt.figure()
#%% biplot dla wszytskich kolumn
biplot(p, 
       np.transpose(pca.components_), 
       list(kolumny_1))
#%% chcemy osiagnac 95% wariancji
pca_df_fit = PCA(svd_solver='full', n_components=0.95)

p = pca_df_fit.fit_transform(temp)
pca_dffit = pd.DataFrame(data = p)

#%% WYKRES OSYPISKOWY
pca_df_fit2 = PCA().fit(temp)

plt.rcParams["figure.figsize"] = (8,5)

fig, ax = plt.subplots()
xi = np.arange(1, 18, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='-', color='black')

plt.xlabel('Liczba komponentów')
plt.xticks(np.arange(1, 18, step=1)) 
plt.ylabel('Wyjasniona zmiennosc przez glowne skladowe')
plt.title('Wizualizacja skumulowanej wartości wariancji w zależności od liczby komponentów.')

plt.axhline(y=0.95, color='grey', linestyle='--')
plt.text(1.1, 1, '95% całkowitej wariancji', color = 'black', fontsize=11)

ax.grid(axis='x')
plt.tight_layout()

plt.show()

plt.gcf().set_size_inches(7, 5)

#zalamanie na 9 skladowej - wiec wyjasniamy 95% wariancji wlasnie na niej

#%% PODZIAL NA TRENUJACY I TESTUJACY SET
from sklearn.model_selection import train_test_split

x = p
y = df['Target']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.3,random_state=42)

trainY_1 = (trainY == 1)
testY_1 = (testY == 1)

#%%% STARTOWA PRÓBKA  klasyfikacji

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=50, tol=-np.infty,
                       loss="log_loss", penalty="l2")
sgd_clf_1 = SGDClassifier(max_iter=50, tol=-np.infty,
                       loss="hinge", penalty="l2")
sgd_clf_2 = SGDClassifier(max_iter=50, tol=-np.infty,
                       loss="modified_huber", penalty="l2")

sgd_clf_3 = SGDClassifier(max_iter=50, tol=-np.infty,
                       loss="log_loss", penalty="l1")
sgd_clf_4 = SGDClassifier(max_iter=50, tol=-np.infty,
                       loss="hinge", penalty="l1")
sgd_clf_5 = SGDClassifier(max_iter=50, tol=-np.infty,
                       loss="modified_huber", penalty="l1")


sgd_clf.fit(trainX, trainY_1)
sgd_clf_1.fit(trainX, trainY_1)
sgd_clf_2.fit(trainX, trainY_1)
sgd_clf_3.fit(trainX, trainY_1)
sgd_clf_4.fit(trainX, trainY_1)
sgd_clf_5.fit(trainX, trainY_1)

print('Współczynniki regresji liniowej')
print('loss: log_loss; penalty: l2: ',sgd_clf.intercept_, '+' , sgd_clf.coef_, '\n') 
print('loss: hinge; penalty: l2: ',sgd_clf_1.intercept_, '+' , sgd_clf_1.coef_,'\n') 
print('loss: modified_huber; penalty: l2: ',sgd_clf_2.intercept_, '+' , sgd_clf_2.coef_,'\n') 
print('loss: log_loss; penalty: l1: ',sgd_clf_3.intercept_, '+' , sgd_clf_3.coef_,'\n') 
print('loss: hinge; penalty: l1: ',sgd_clf_4.intercept_, '+' , sgd_clf_4.coef_,'\n') 
print('loss: modified_huber; penalty: l1: ',sgd_clf_5.intercept_, '+' , sgd_clf_5.coef_,'\n') 

#%% podstawowe informacje o klasyfikacji

from sklearn.model_selection import cross_val_score
moje_cv = 3
lista_klasyfikacji = [sgd_clf, sgd_clf_1, sgd_clf_2, sgd_clf_3, sgd_clf_4, sgd_clf_5]
nazwy_klasyfikacji = ['loss: log_loss; penalty: l2','loss: hinge; penalty: l2','loss: modified_huber; penalty: l2'
                      'loss: log_loss; penalty: l1','loss: hinge; penalty: l1','loss: modified_huber; penalty: l1']

for i,j in zip(lista_klasyfikacji,nazwy_klasyfikacji):
    
    print('---------- METODA: ', j,'----------','\n')
    napis= 'Dokladnosć: ilosc dobrych predykcji'
    dokladnosc_klasyfikacji = cross_val_score(sgd_clf, trainX, trainY_1, 
                                                  cv=moje_cv, scoring="accuracy")
    print(napis, dokladnosc_klasyfikacji,'\n')
    
    napis= 'Prezycja : TP/(TP+NP) ilosc dobrych predykcji w klasie pozytwnej'
    precyzja_klasyfikacji = cross_val_score(sgd_clf, trainX, trainY_1, 
                                                  cv=moje_cv, scoring="precision")
    print(napis, precyzja_klasyfikacji,'\n')
    
    
    napis= 'Recall : TP(TP+FN) ile instancji klasy pozytywnej zostalo dobrze rozpoznane'
    czulosc_klasyfikacji = cross_val_score(sgd_clf, trainX, trainY_1, 
                                                  cv=moje_cv, scoring="recall")
    
    print(napis, czulosc_klasyfikacji,'\n')
    
    
    print("srednia i std dokładnosci %.3f"% dokladnosc_klasyfikacji.mean(), 
          " +/- %.3f"% dokladnosc_klasyfikacji.std())
    
    print("srednia i std precyzji  %.3f"% precyzja_klasyfikacji.mean() ,
          " +/- %.3f"%precyzja_klasyfikacji.std())
    
    
    print("srednia i std czułosci %.3f"%czulosc_klasyfikacji.mean(),
          " +'- %.3f"%czulosc_klasyfikacji.std(),'\n')

#%% walidacja krzyzowa dla roznych modeli
from sklearn.model_selection import cross_val_predict

y_train_predict = []
for i in lista_klasyfikacji:
    x = cross_val_predict(i, trainX, trainY_1, cv= moje_cv)
    y_train_predict.append(x)

#%% tworzymy macierz pomylek
from sklearn.metrics import confusion_matrix
for i,j in zip(y_train_predict,nazwy_klasyfikacji):
    confusion =  confusion_matrix(trainY_1, i)
    
    print('---------- METODA: ', j,'----------','\n')
    print('confusion matrix: \n', confusion, '\n')
    print( 'TN=', confusion[0,0], '\tFP=', confusion[0,1])
    print( 'FN=', confusion[1,0], '\tTP=', confusion[1,1])
    print('\nprecision=  %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
    print('recall= %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
    
    from sklearn.metrics import precision_score, recall_score
    print('\n Osoby zdrowe wsrod sklasyfikowanych jako osoby zdrowe to %.2f'% precision_score(trainY_1,i))
    print('\n Osoby zdrowe rozpoznano poprawnie w %.2f przypadkach  '% recall_score(trainY_1,i),'\n')


#%%  KOMPROMIS
#użycie metody decision_function() glownego obiektu
y_scores = []
for i in y_train_predict:
    x = cross_val_predict(sgd_clf, trainX, trainY_1, cv=3,
                             method="decision_function")
    y_scores.append(x)

                        
#%% 
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1.5])
#%% wykresy dla roznych modeli

for i in y_scores:
    precisions, recalls, thresholds = precision_recall_curve(trainY_1, i)
    
    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xlim([-200, 200])
    plt.savefig("precision_recall_vs_threshold_plot dla " ,dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.plot(recalls, precisions )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall vs Precision')
    plt.show()
    
#%%
for i in y_scores:
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    y_train_pred_90 = (i >= threshold_90_precision)
    print(precision_score(trainY_1, y_train_pred_90))
    print(recall_score(trainY_1, y_train_pred_90))

#%% WYSZUKIWANIE NAKLEPSZYCH WARTOŚCI HIPERPARAMETROW popularna metoda

from sklearn.model_selection import GridSearchCV

params = {
    'loss' : ['hinge', 'log_loss', 'squared_hinge'] ,#'modified_huber', 'perceptron'],
    'alpha' : [ 0.01, 0.1],
    'penalty' :['l2','l1']#,'elasticnet','none']
        }

sgd_clf= SGDClassifier(max_iter =100)
grid = GridSearchCV ( sgd_clf, param_grid =params, cv=moje_cv, scoring ='f1',
                     return_train_score=True)

grid.fit( trainX, trainY_1)

cv_res = grid.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(mean_score, params)
    
#%% Ostateczne przetwarzanie
from sklearn.metrics import precision_score, recall_score, f1_score

final_model = grid.best_estimator_


final_model_param = final_model.get_params()
final_training = final_model.fit(trainX, trainY_1)

final_predictions = final_training.predict(testX)

final_f1 = f1_score(testY_1,final_predictions)
final_recall = recall_score(testY_1,final_predictions)
final_precision = precision_score(testY_1,final_predictions)

print('f1 najlepszego rozwiazania SGDClassifier ',"%.2f"%final_f1)
print('recall najlepszego rozwiazania SGDClassifier ',"%.2f"%final_recall)
print('presision najlepszego rozwiazania SGDClassifier ',"%.2f"%final_precision)
#%% przetwarzanie dla zbioru testowego
y_scores_test = cross_val_predict(sgd_clf, testX, testY_1, cv=3,
                             method="decision_function")
#%%
precisions_1, recalls_1, thresholds_1 = precision_recall_curve(testY_1, y_scores_test)

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions_1, recalls_1, thresholds_1)
plt.xlim([-200, 150])
plt.savefig("precision_recall_vs_threshold_plot",dpi=300)
plt.show()


#%%
plt.figure(figsize=(8, 4))
plt.plot(recalls_1, precisions_1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall vs Precision')
plt.show

threshold_90_precision_1 = thresholds_1[np.argmax(precisions_1 >= 0.90)]
y_train_pred_90_test = (y_scores_test >= threshold_90_precision_1)
print(precision_score(testY_1, y_train_pred_90_test))
print(recall_score(testY_1, y_train_pred_90_test))













