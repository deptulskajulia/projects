# -*- coding: utf-8 -*-
"""
Hypertension: ocena arytmii oddechowej (RSA) u ludzi zdrowych i z nadciśnieniem tętniczym
__________________________________________________________________________________________
Głównymi zadaniami projektu była praca z danymi dotyczącymi oddechu oraz rytmu serca pacjentów z dwóch grup: 
zdrowych oraz z nadciśnieniem tętniczym, na pewnym przedziale czasowym. 
Należało sprawdzić jakość oraz podstawowe statystyczne informacje o uzyskanych danych, 
wyznaczyć czasowe własności sygnału dla pewnych zmiennych, a także poszukać wzorców, 
czyli własności symbolicznych sygnału. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Ustawiamy katalogi pracy
import os
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"rytm_serca")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"dane")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
KATALOG_HISTOGRAMOW = os.path.join(KATALOG_WYKRESOW, "histogramy")
KATALOG_ODDECH = os.path.join(KATALOG_WYKRESOW, "krzywe oddechowe")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)
os.makedirs(KATALOG_DANYCH, exist_ok=True)
os.makedirs(KATALOG_HISTOGRAMOW, exist_ok=True)
os.makedirs(KATALOG_ODDECH, exist_ok=True)


SKAD_POBIERAC = ['C:/Users/Julia/Desktop/studia/5sem/neuronowe/projekt_1/healthy_decades', 'C:/Users/Julia/Desktop/studia/5sem/neuronowe/projekt_1/HTX_LVH/', 'C:/Users/Julia/Desktop/studia/5sem/neuronowe/projekt_1/hypertension_RR_SBP/',
                 'C:/Users/Julia/Desktop/studia/5sem/neuronowe/projekt_1/hypertension/']
                 
czytamy = SKAD_POBIERAC[3]
print ('\nPrzetwarzamy katalog', czytamy)
pliki = os.listdir (czytamy)

    

def load_serie(skad , co, ile_pomin = 0, kolumny =['RR_interval', 'Num_contraction']):
    csv_path = os.path.join(skad, co )
    print ( skad, co)
    seria = [pd.read_csv(csv_path, sep='\t', header =None,
                        skiprows= ile_pomin, names= kolumny)]
    if skad == SKAD_POBIERAC[2]:
        seria = pd.read_csv(csv_path, sep='\t',  decimal=',' )
    return seria


pomin=5
kolumny = ['time','R_peak','Resp','SBP','cus','Resp_1500','Wdech_wydech']

#Tworzymy osobne listy dla osób zdrowych - te osoby mają w nazwie pliku z danymi cyfrę 1
#natomiast osoby z cyfrą 2 w nazwie pliku, chorują na nadcisnienie
pliki_z = []
pliki_nad = []

for i in pliki:
    if i[4] == "2":
        seria = load_serie(skad = czytamy, co = i, ile_pomin = pomin, kolumny = kolumny)
        pliki_nad.append(seria)
    else:
        seria = load_serie(skad = czytamy, co = i, ile_pomin = pomin, kolumny = kolumny)
        pliki_z.append(seria)
        
#%%
#Tworzymy kopie gotowych już list
pliki_kopia = pliki.copy()
pliki_z_kopia = pliki_z.copy()
pliki_nad_kopia = pliki_nad.copy()
   
#%%
#Stworzone wyżej kopie używamy by stworzyć nowe listy bez osób, których wyników nie jestesmy użyć w dalszej 
#analizie np. w przypadku osoby z indeksem 19 występowały problemy z przesyłem danych, co skutkowało 
# pojawieniem się przerw oraz danymi nieliczbowymi m.in. w kolumnie R_peak 
indeksy = [19,25,29,30,33,35,51,55,61]
for index in sorted(indeksy, reverse=True):
    del pliki_kopia[index]
    
indeksy_1 = [7,11,13,14,15,17,21,23]
for indexj in sorted(indeksy_1, reverse=True):
    del pliki_z_kopia[indexj]

del pliki_nad_kopia[36]


#Od teraz korzystamy z nowych, 'poprawionych' tabel

#%%
#Podstawowe informacje na temat danych każdej osoby m.in ilosc niezerowych wierszy, typ danych 
for i in pliki_z_kopia:
    print(i[0].info())
for i in pliki_nad_kopia:
    print(i[0].info())        


#%%
#Ogólna informacja kompletnosci danych - suma zerowych danych, a także podstawowe
#narzędzia statystyki jak srednia czy maksymalna wartosc w poszczególnych kolumnach danych pacjentów

for i in pliki_z_kopia:
    print('\n Pacjent:\n')
    print(i[0].isnull().sum())
    print('\n')
    print(i[0].describe())

for j in pliki_nad_kopia:   
    print(j[0].isnull().sum())
    print('\n')
    print(j[0].describe())

#%% 
#Kwantowanie - zliczanie wartosci 0 i 1 z kolumny R_peak dla poszczególnych osób
for i in pliki_z_kopia:
    print(i[0]['R_peak'].value_counts().sort_index())

for j in pliki_nad_kopia:
    print(j[0]['R_peak'].value_counts().sort_index())
    
#%%
#Importujemy do list nazwy plików zawierające dane poszczególnych osób
pliki_nad_dane = []
pliki_z_dane = []

for i in pliki_kopia:
    if i[4] == "2":
        pliki_nad_dane.append(i)
    else:
        pliki_z_dane.append(i)
#%%
#Usuwamy z nazw przyrostek '.txt'
pliki_z_dane = [k[:-4] for k in pliki_z_dane]
pliki_nad_dane = [k[:-4] for k in pliki_nad_dane]
#%% wizualizacja danych : DataFrame.plot
#Tworzymy histogramy dla wybranej kolumny

for i,j in zip(pliki_z_kopia,pliki_z_dane):
    i[0]['Resp'].hist(bins=50, figsize=(9,6))
    plt.tight_layout()
    plt.title("histogram wartosci dla " + j)
    plt.savefig(os.path.join(KATALOG_HISTOGRAMOW,'histogramy ' + j +'.jpg'), dpi=300 ) 
    plt.show() 

for i,j in zip(pliki_nad_kopia,pliki_nad_dane):
    i[0]['Resp'].hist(bins=50, figsize=(9,6))
    plt.tight_layout()
    plt.title("histogram wartosci dla " + j)
    plt.savefig(os.path.join(KATALOG_HISTOGRAMOW,'histogramy ' + j +'.jpg'), dpi=300 ) 
    plt.show() 

#%%___________________________DRUGI PODPUNKT________________________________________________
#WYZNACZENIE CZASOWYCH WLASNOSCI SYNGALU INTERWALOW RR: SDNN,RMSSD,PNN50,PNN20

#interwały RR - róznice czasowe pomiędzy wystąpieniem skurczy, czyli 1 w komunie 'R_peak'
rr_z = []
rr_nad=[]
for i in range(0,17):
    x = np.diff(pliki_z_kopia[i][0].query("R_peak == 1")["time"])
    rr_z.append(x)

for j in range(0,36):
    y = np.diff(pliki_nad_kopia[j][0].query("R_peak == 1")["time"])
    rr_nad.append(y)
#%% zliczenie interwałów RR
rr_z_zliczenie = []
rr_nad_zliczenie = []

for i in rr_z:
    x = np.prod(i.shape) + 1
    rr_z_zliczenie.append(x)

for j in rr_nad:
    y = np.prod(j.shape) + 1
    rr_nad_zliczenie.append(y)
#%% srednia wartosc interwałów
mean_rr_z = []
mean_rr_nad = []

for i in rr_z:
    x = np.mean(i)
    mean_rr_z.append(x)

for j in rr_nad:
    y = np.mean(j)
    mean_rr_nad.append(y)

#%% SSDN - odchylenie standardowe wszystkich odstępów RR
sdnn_z = []
sdnn_nad = []

for i in rr_z:
    x = np.std(i)
    sdnn_z.append(x)

for j in rr_nad:
    y = np.std(j)
    sdnn_nad.append(y)

#%% RMSSD - pierwiastek kwadratowy ze sredniej sumy kwadratów różnic między kolejnymi odstępami RR
rmssd_z = []
rmssd_nad = []

for i in rr_z:
    x = (np.mean((np.diff(i))**2))**(1/2)
    rmssd_z.append(x)

for j in rr_nad:
    y = (np.mean((np.diff(j))**2))**(1/2)
    rmssd_nad.append(y)
#%%
print(1/len(rmssd_z)*sum(rmssd_z))
print(1/len(rmssd_nad)*sum(rmssd_nad))
#Obie grupy mają podobną srednia wartosc RMSSD 
#%% pNN50 - odsetek różnic między kolejnymi odstępami RR przekraczającymi 50 ms
pnn50_z = []
pnn50_nad = []

for i in rr_z:
    x = (np.sum((np.abs(np.diff(i)) > 0.05))) * 1/len(rr_z)
    pnn50_z.append(x)

for j in rr_nad:
    y = (np.sum((np.abs(np.diff(j)) > 0.05))) * 1/len(rr_nad)
    pnn50_nad.append(y)

#%%
print(1/len(pnn50_z)*sum(pnn50_z))
print(1/len(pnn50_nad)*sum(pnn50_nad))
#Wiekszy sredni odsetek roznic pomiedzy odstepami RR > 0.05s mają osoby zdrowe
#Obie wartosci wynosza powyzej 3%, co rokuje pozytywnie na obie grupy
#%% pNN20 - odsetek różnic między kolejnymi odstępami RR przekraczającymi 20 ms
pnn20_z = []
pnn20_nad = []

for i in rr_z:
    x = (np.sum((np.abs(np.diff(i)) > 0.02))) * 1/len(rr_z)
    pnn20_z.append(x)

for j in rr_nad:
    y = (np.sum((np.abs(np.diff(j)) > 0.02))) * 1/len(rr_nad)
    pnn20_nad.append(y)

#%%
print(1/len(pnn20_z)*sum(pnn20_z))
print(1/len(pnn20_nad)*sum(pnn20_nad))
#Wiekszy sredni odsetek roznic pomiedzy odstepami RR > 0.02s mają osoby zdrowe
#Dla osob chorujacych na nadcisniene mamy wartosc zwrotna mniejsza niz 20%, co może być podstawą do niepokoju
#%% SBP - skurczowe ciśnienie krwi
#By obliczyć jego srednia wartosc, usuwamy wszystkie wartosci równe 0 oraz nieliczbowe
mean_sbp_z = []
mean_sbp_nad = []
listabez0 = []
listabez0_1 =[]

for j in range(0,17):
    x = [i for i in pliki_z_kopia[j][0]['SBP'] if i > 0]
    listabez0.append(x)
    
for k in range(0,36):
    y = [i for i in pliki_nad_kopia[k][0]['SBP'] if i > 0]
    listabez0_1.append(y)
 
#srednie SBP

for i in listabez0:
    x = np.mean(i)
    mean_sbp_z.append(x)

for j in listabez0_1:
    y = np.mean(j)
    mean_sbp_nad.append(y)
#%%
print(1/len(mean_sbp_z)*sum(mean_sbp_z))
print(1/len(mean_sbp_nad)*sum(mean_sbp_nad))
#Zgodnie z przewidywaniami - srednie cisnienie skurczowe ma wysze grupa osob z nadcisnieniem    

#%% 
#By podzielić fale oddechowa na wdechy i wydechy, wygładzamy zmienna Resp za 
#pomocą sredniej kroczacej
rolling_resp = []
rolling_resp_2 = []

for i in pliki_z_kopia: 
    x= i[0].Resp.rolling(1500, min_periods=1).mean()
    rolling_resp.append(x)
    
for j in pliki_nad_kopia: 
    y= j[0].Resp.rolling(1500, min_periods=1).mean()
    rolling_resp_2.append(y)

#%%
#Dodajemy "wygładzony" Resp do tabeli
for i in range(0,17):
    pliki_z_kopia[i][0]['Resp_1500'] = rolling_resp[i]
    
for j in range(0,36):
    pliki_nad_kopia[j][0]['Resp_1500'] = rolling_resp_2[j]

#%%
#Obliczamy roznice pomiedzy kolejnymi "wygladzonymi" oddechami, by dowiedziec
#sie gdzie wystepuje wdech a gdzie wydech
ww_z = []
ww_nad = []
for i in range(0,17):
    x = np.diff(pliki_z_kopia[i][0]['Resp_1500'])
    ww_z.append(x)

for j in range(0,36):
    y = np.diff(pliki_nad_kopia[j][0]['Resp_1500'])
    ww_nad.append(y)

#%%
#konieczna edycja by moc dodac dane do tabeli
for i in range(0,17):
    ww_z[i] = np.insert(ww_z[i],0,0)

for j in range(0,36):
    ww_nad[j] = np.insert(ww_nad[j],0,0)

#%% 
#dodajemy do tabeli
for i in range(0,17):
    pliki_z_kopia[i][0]['Wdech_wydech'] = ww_z[i]
    
for j in range(0,36):
    pliki_nad_kopia[j][0]['Wdech_wydech'] = ww_nad[j]
#%% _______________ TRZECI PODPUNKT _____________________________________
# > ilosc wystapien pikow R przy wdechu i wydechu
#dodajemy do listy wszystkie wartosci Wdech_wydech przy, których R_peak = 1
calosc_ww_z = []
calosc_ww_nad = []

for i in range(0,17):
    x = pliki_z_kopia[i][0].query("R_peak == 1")["Wdech_wydech"]
    calosc_ww_z.append(x)


for j in range(0,36):
    y = pliki_nad_kopia[j][0].query("R_peak == 1")["Wdech_wydech"]
    calosc_ww_nad.append(y)
    
#%%
#usuwamy indeksy z list powyzej
non_ind_z = []
non_ind_nad = []

for i in range(0,17):
    x = calosc_ww_z[i].values
    non_ind_z.append(x)
    
for j in range(0,36):
    y = calosc_ww_nad[j].values
    non_ind_nad.append(y)

#%% 
#wartosci uzyskane w powyzszych listach dzielimy na wdechy, jesli sa wieksze
# od zera i wydechy, jesli sa mniejsze niz zero
wwr_wdech_z = []
wwr_wydech_z = []
wwr_wdech_nad = []
wwr_wydech_nad = []

for i in range(0,17):
    y = [k for k in non_ind_z[i] if k > 0]
    z = [k for k in non_ind_z[i] if k < 0]
    wwr_wydech_z.append(z)
    wwr_wdech_z.append(y)
    
#%%
for j in range(0,36):
    x = [m for m in non_ind_nad[j] if m > 0]
    d = [m for m in non_ind_nad[j] if m < 0]
    wwr_wydech_nad.append(d)
    wwr_wdech_nad.append(x)
    
#%%  zliczamy ilosc wdechow i wydechow, gdzie wystepuje pik R dla obu grup 
wwr_wdech_z_zliczenie = []
wwr_wydech_z_zliczenie = []
wwr_wdech_nad_zliczenie = []
wwr_wydech_nad_zliczenie = []
 
for i in range(0,17):
    x = len(wwr_wdech_z[i])
    y = len(wwr_wydech_z[i])
    wwr_wdech_z_zliczenie.append(x)
    wwr_wydech_z_zliczenie.append(y)

for j in range(0,36):
    k = len(wwr_wdech_nad[j])
    d = len(wwr_wydech_nad[j])
    wwr_wdech_nad_zliczenie.append(k)
    wwr_wydech_nad_zliczenie.append(d)
#%% procentowa ilosc wdechów i wydechów
wwr_wdech_z_procent = []
wwr_wydech_z_procent = []
wwr_wdech_nad_procent = []
wwr_wydech_nad_procent = []

for i in range(0,17):
    x = wwr_wdech_z_zliczenie[i]/len(non_ind_z[i])
    y = wwr_wydech_z_zliczenie[i]/len(non_ind_z[i])
    wwr_wdech_z_procent.append(x)
    wwr_wydech_z_procent.append(y)

for j in range(0,36):
    k = wwr_wdech_nad_zliczenie[j]/len(non_ind_nad[j])
    d = wwr_wydech_nad_zliczenie[j]/len(non_ind_nad[j])
    wwr_wdech_nad_procent.append(k)
    wwr_wydech_nad_procent.append(d)



#%% > ilosc przyspieszen i zwolnien rytmu serca przy wdechu i wydechu
#obliczamy roznice pomiedzy wartosciami Wdech_wydech dla których R_peak = 1
#roznice zaliczamy do wdechu lub wydechu na podstawie wartosci odjemnej
diff_wdech_z = []
diff_wydech_z = []

for i in range(0,17):
    temp_lista_z = []
    temp_lista1_z = []
    for j in range(0,len(non_ind_z[i])-1):
        x = non_ind_z[i][j+1] - non_ind_z[i][j]
        if non_ind_z[i][j+1] > 0:
            temp_lista_z.append(x)
        elif non_ind_z[i][j+1] <0:
            temp_lista1_z.append(x)
    diff_wdech_z.append(temp_lista_z)
    diff_wydech_z.append(temp_lista1_z)

diff_wdech_nad = []
diff_wydech_nad = []

for k in range(0,36):
    temp_lista2_nad = []
    temp_lista3_nad = []
    for d in range(0,len(non_ind_nad[k])-1):
        y = non_ind_nad[k][d+1] - non_ind_nad[k][d]
        if non_ind_nad[k][d+1] > 0:
            temp_lista2_nad.append(y)
        elif non_ind_nad[k][d+1] <0:
            temp_lista3_nad.append(y)
    diff_wdech_nad.append(temp_lista2_nad)
    diff_wydech_nad.append(temp_lista3_nad)  
      
#%% 
#dzielimy uzyskane roznice na przyspieszenia i zwolnienia akcji serca na podstawie
#ich wartosci
przyspieszenie_wdechy_z = []
zwolnienie_wdechy_z = []
przyspieszenie_wydechy_z = []
zwolnienie_wydechy_z = []


for i in range(0,17):
    x = [k for k in diff_wdech_z[i] if k > 0]
    y = [k for k in diff_wdech_z[i] if k < 0]
    zwolnienie_wdechy_z.append(y)
    przyspieszenie_wdechy_z.append(x)

for j in range(0,17):
    z = [m for m in diff_wydech_z[j] if m > 0]
    d = [m for m in diff_wydech_z[j] if m < 0]
    zwolnienie_wydechy_z.append(d)
    przyspieszenie_wydechy_z.append(z)
#%%
przyspieszenie_wdechy_nad = []
zwolnienie_wdechy_nad = []
przyspieszenie_wydechy_nad = []
zwolnienie_wydechy_nad = []

for l in range(0,36):
    x = [k for k in diff_wdech_nad[l] if k > 0]
    y = [k for k in diff_wdech_nad[l] if k < 0]
    przyspieszenie_wdechy_nad.append(x)
    zwolnienie_wdechy_nad.append(y)   

for n in range(0,36):
    z = [m for m in diff_wydech_nad[n] if m > 0]
    d = [m for m in diff_wydech_nad[n] if m < 0]
    przyspieszenie_wydechy_nad.append(z)
    zwolnienie_wydechy_nad.append(d)
#%%
#zliczamy ilosc przyspieszen i zwolnien akcji serca
przyspieszenie_wdechy_z_zliczenie = []
zwolnienie_wdechy_z_zliczenie = []
przyspieszenie_wydechy_z_zliczenie = []
zwolnienie_wydechy_z_zliczenie = []  

for i in range(0,17):
    x = len(przyspieszenie_wdechy_z[i])
    y = len(zwolnienie_wdechy_z[i])
    z = len(przyspieszenie_wydechy_z[i])
    d = len(zwolnienie_wydechy_z[i])
    przyspieszenie_wdechy_z_zliczenie.append(x)
    zwolnienie_wdechy_z_zliczenie.append(y)
    przyspieszenie_wydechy_z_zliczenie.append(z)
    zwolnienie_wydechy_z_zliczenie.append(d) 
#%%
przyspieszenie_wdechy_z_procent = []
zwolnienie_wdechy_z_procent = []
przyspieszenie_wydechy_z_procent = []
zwolnienie_wydechy_z_procent = []  

for i in range(0,17):
    x = przyspieszenie_wdechy_z_zliczenie[i]/wwr_wdech_z_zliczenie[i]
    y = zwolnienie_wdechy_z_zliczenie[i]/wwr_wdech_z_zliczenie[i]
    z = przyspieszenie_wydechy_z_zliczenie[i]/wwr_wydech_z_zliczenie[i]
    d = zwolnienie_wydechy_z_zliczenie[i]/wwr_wydech_z_zliczenie[i]
    przyspieszenie_wdechy_z_procent.append(x)
    zwolnienie_wdechy_z_procent.append(y)
    przyspieszenie_wydechy_z_procent.append(z)
    zwolnienie_wydechy_z_procent.append(d)
    
#%%
przyspieszenie_wdechy_nad_zliczenie = []
zwolnienie_wdechy_nad_zliczenie = []
przyspieszenie_wydechy_nad_zliczenie = []
zwolnienie_wydechy_nad_zliczenie = []

for i in range(0,36):
    x = len(przyspieszenie_wdechy_nad[i])
    y = len(zwolnienie_wdechy_nad[i])
    z = len(przyspieszenie_wydechy_nad[i])
    d = len(zwolnienie_wydechy_nad[i])
    przyspieszenie_wdechy_nad_zliczenie.append(x)
    zwolnienie_wdechy_nad_zliczenie.append(y)
    przyspieszenie_wydechy_nad_zliczenie.append(z)
    zwolnienie_wydechy_nad_zliczenie.append(d)

#%%
przyspieszenie_wdechy_nad_procent = []
zwolnienie_wdechy_nad_procent = []
przyspieszenie_wydechy_nad_procent = []
zwolnienie_wydechy_nad_procent = []  

for i in range(0,36):
    x = przyspieszenie_wdechy_nad_zliczenie[i]/wwr_wdech_nad_zliczenie[i]
    y = zwolnienie_wdechy_nad_zliczenie[i]/wwr_wdech_nad_zliczenie[i]
    z = przyspieszenie_wydechy_nad_zliczenie[i]/wwr_wydech_nad_zliczenie[i]
    d = zwolnienie_wydechy_nad_zliczenie[i]/wwr_wydech_nad_zliczenie[i]
    przyspieszenie_wdechy_nad_procent.append(x)
    zwolnienie_wdechy_nad_procent.append(y)
    przyspieszenie_wydechy_nad_procent.append(z)
    zwolnienie_wydechy_nad_procent.append(d)
    
    
#%% > ilosc wzrostow i spadkow cisnienia skurczowego SBP przy wdechu i wydechu
#biore wartosci Wdech_wydech, gdzie sbp > 0
czesc_ww_z = []
czesc_ww_nad =[]
for i in range(0,17):
    x = pliki_z_kopia[i][0].query("SBP > 0")["Wdech_wydech"]
    czesc_ww_z.append(x)


for j in range(0,36):
    y = pliki_nad_kopia[j][0].query("SBP > 0")["Wdech_wydech"]
    czesc_ww_nad.append(y)
#%%
#bierzemy wartosci z list powyzej
sbp_non_ind =[]
sbp_non_ind2 = []
for i in range(0,17):
    x = czesc_ww_z[i].values
    sbp_non_ind.append(x)
    
for j in range(0,36):
    y = czesc_ww_nad[j].values
    sbp_non_ind2.append(y)

#%%
#obliczamy roznice pomiedzy wartosciami SBP 
#roznice zaliczamy do wdechu lub wydechu na podstawie wartosci odjemnej
#korzystamy z list zawierajacych wartosci SBP > 0 uzytych juz wczesniej
diff_sbpwdech_z = []
diff_sbpwydech_z = []

for i in range(0,17):
    temp_slista_z = []
    temp_slista1_z = []
    for j in range(0,len(sbp_non_ind[i])-1):
        x = listabez0[i][j+1] - listabez0[i][j]
        if sbp_non_ind[i][j+1] > 0:
            temp_slista_z.append(x)
        elif sbp_non_ind[i][j+1] <0:
            temp_slista1_z.append(x)
    diff_sbpwdech_z.append(temp_slista_z)
    diff_sbpwydech_z.append(temp_slista1_z)

diff_sbpwdech_nad= []
diff_sbpwydech_nad = []

for k in range(0,36):
    temp_slista2_nad = []
    temp_slista3_nad = []
    for d in range(0,len(sbp_non_ind2[k])-1):
        y = listabez0_1[k][d+1] - listabez0_1[k][d]
        if sbp_non_ind2[k][d+1] > 0:
            temp_slista2_nad.append(y)
        elif sbp_non_ind2[k][d+1] <0:
            temp_slista3_nad.append(y)
    diff_sbpwdech_nad.append(temp_slista2_nad)
    diff_sbpwydech_nad.append(temp_slista3_nad) 

#%% 
#dzielimy uzyskane roznice na wzrosty i spadki SBP dla wdechow i wydechow na podstawie
#ich wartosci
wzrost_SBP_wdechy_z = []
spadek_SBP_wdechy_z = []
wzrost_SBP_wydechy_z = []
spadek_SBP_wydechy_z = []

for i in range(0,17):
    x = [k for k in diff_sbpwdech_z[i] if k > 0]
    y = [k for k in diff_sbpwdech_z[i] if k < 0]
    spadek_SBP_wdechy_z.append(y)
    wzrost_SBP_wdechy_z.append(x)

for j in range(0,17):
    z = [m for m in diff_sbpwydech_z[j] if m > 0]
    d = [m for m in diff_sbpwydech_z[j] if m < 0]
    spadek_SBP_wydechy_z.append(d)
    wzrost_SBP_wydechy_z.append(z)
#%%
wzrost_SBP_wdechy_nad = []
spadek_SBP_wdechy_nad = []
wzrost_SBP_wydechy_nad = []
spadek_SBP_wydechy_nad = []

for l in range(0,36):
    x = [k for k in diff_sbpwdech_nad[l] if k > 0]
    y = [k for k in diff_sbpwdech_nad[l] if k < 0]
    wzrost_SBP_wdechy_nad.append(x)
    spadek_SBP_wdechy_nad.append(y)   

for n in range(0,36):
    z = [m for m in diff_sbpwydech_nad[n] if m > 0]
    d = [m for m in diff_sbpwydech_nad[n] if m < 0]
    wzrost_SBP_wydechy_nad.append(z)
    spadek_SBP_wydechy_nad.append(d)
#%%
#zliczamy wartosci
wzrost_SBP_wdechy_z_zliczenie = []
spadek_SBP_wdechy_z_zliczenie = []
wzrost_SBP_wydechy_z_zliczenie = []
spadek_SBP_wydechy_z_zliczenie = []  

for i in range(0,17):
    x = len(wzrost_SBP_wdechy_z[i])
    y = len(spadek_SBP_wdechy_z[i])
    z = len(wzrost_SBP_wydechy_z[i])
    d = len(spadek_SBP_wydechy_z[i])
    wzrost_SBP_wdechy_z_zliczenie.append(x)
    spadek_SBP_wdechy_z_zliczenie.append(y)
    wzrost_SBP_wydechy_z_zliczenie.append(z)
    spadek_SBP_wydechy_z_zliczenie.append(d) 

#%%
wzrost_SBP_wdechy_z_procent = []
spadek_SBP_wdechy_z_procent = []
wzrost_SBP_wydechy_z_procent = []
spadek_SBP_wydechy_z_procent = []  

for i in range(0,17):
    x = wzrost_SBP_wdechy_z_zliczenie[i]/len(diff_sbpwdech_z[i])
    y = spadek_SBP_wdechy_z_zliczenie[i]/len(diff_sbpwdech_z[i])
    z = wzrost_SBP_wydechy_z_zliczenie[i]/len(diff_sbpwydech_z[i])
    d = spadek_SBP_wydechy_z_zliczenie[i]/len(diff_sbpwydech_z[i])
    wzrost_SBP_wdechy_z_procent.append(x)
    spadek_SBP_wdechy_z_procent.append(y)
    wzrost_SBP_wydechy_z_procent.append(z)
    spadek_SBP_wydechy_z_procent.append(d) 
    
#%%
wzrost_SBP_wdechy_nad_zliczenie = []
spadek_SBP_wdechy_nad_zliczenie = []
wzrost_SBP_wydechy_nad_zliczenie = []
spadek_SBP_wydechy_nad_zliczenie = []

for i in range(0,36):
    x = len(wzrost_SBP_wdechy_nad[i])
    y = len(spadek_SBP_wdechy_nad[i])
    z = len(wzrost_SBP_wydechy_nad[i])
    d = len(spadek_SBP_wydechy_nad[i])
    wzrost_SBP_wdechy_nad_zliczenie.append(x)
    spadek_SBP_wdechy_nad_zliczenie.append(y)
    wzrost_SBP_wydechy_nad_zliczenie.append(z)
    spadek_SBP_wydechy_nad_zliczenie.append(d)

#%%

wzrost_SBP_wdechy_nad_procent = []
spadek_SBP_wdechy_nad_procent = []
wzrost_SBP_wydechy_nad_procent = []
spadek_SBP_wydechy_nad_procent = []

for i in range(0,36):
    x = wzrost_SBP_wdechy_nad_zliczenie[i]/len(diff_sbpwdech_nad[i])
    y = spadek_SBP_wdechy_nad_zliczenie[i]/len(diff_sbpwdech_nad[i])
    z = wzrost_SBP_wydechy_nad_zliczenie[i]/len(diff_sbpwydech_nad[i])
    d = spadek_SBP_wydechy_nad_zliczenie[i]/len(diff_sbpwydech_nad[i])
    wzrost_SBP_wdechy_nad_procent.append(x)
    spadek_SBP_wdechy_nad_procent.append(y)
    wzrost_SBP_wydechy_nad_procent.append(z)
    spadek_SBP_wydechy_nad_procent.append(d)

#%%
#Tworzymy wykres zależnosci wygładzonej zmiennej 'Resp' od czasu 

for i,j in zip(pliki_z_kopia,pliki_z_dane):
    plt.plot(i[0]['time'], i[0]['Resp_1500'])
    plt.title("Wykres zaleznosci oddechu od czasu dla " + j)
    plt.show()

for i,j in zip(pliki_nad_kopia,pliki_nad_dane):
    plt.plot(i[0]['time'], i[0]['Resp_1500'])
    plt.title("Wykres zaleznosci oddechu od czasu dla " + j)
    plt.show()

#%% fala oddechowa
start = 10000
koniec= 30000
for i,j in zip(pliki_z_kopia,pliki_z_dane):
    i[0]['R_peak'] = i[0]['R_peak']* i[0]['Resp_1500']
    i[0]['R_peak'].where(i[0]['R_peak'] >0,np.nan, inplace = True)
    
    plt.plot(i[0]['time'][start : koniec], i[0]['Resp_1500'][start : koniec],'g.',
            markersize=4 )
    plt.plot(i[0]['time'][start : koniec], i[0]['R_peak'][start : koniec],'rx',
             markersize=10)
    plt.title("oddech " + j)
    plt.savefig(os.path.join(KATALOG_ODDECH, 'Oddech ' + j + '.jpg'), dpi=300 ) 
    plt.show()

for i,j in zip(pliki_nad_kopia,pliki_nad_dane):
    i[0]['R_peak'] = i[0]['R_peak']* i[0]['Resp_1500']
    i[0]['R_peak'].where(i[0]['R_peak'] >0,np.nan, inplace = True)
    
    plt.plot(i[0]['time'][start : koniec], i[0]['Resp_1500'][start : koniec],'g.',
            markersize=4 )
    plt.plot(i[0]['time'][start : koniec], i[0]['R_peak'][start : koniec],'rx',
             markersize=10)
    plt.title("oddech " + j)
    plt.savefig(os.path.join(KATALOG_ODDECH, 'Oddech ' + j + '.jpg'), dpi=300 ) 
    plt.show()

#%% Charakterystyka grupowa w postaci tabel

data = {'Srednie cisnienie skurczowe': mean_sbp_z,
        'Zliczone skurcze R': rr_z_zliczenie,
        'Srednia wartosc interwalow RR': mean_rr_z,
        'Wartosci SSDN':sdnn_z,
        'Wartosci RMSSD': rmssd_z,
        'Wartosci pNN50': pnn50_z,
        'Wartosci pNN20':pnn20_z,
        'Ilosc skurczow R przy wdechu [w %]': wwr_wdech_z_procent,
        'Ilosc skurczow R przy wydechu [w %]': wwr_wydech_z_procent,
        'Ilosc przyspieszen rytmu serca przy wdechu [w %]': przyspieszenie_wdechy_z_procent,
        'Ilosc zwolnien rytmu serca przy wdechu [w %]': zwolnienie_wdechy_z_procent,
        'Ilosc przyspieszen rytmu serca przy wydechu [w %]': przyspieszenie_wydechy_z_procent,
        'Ilosc zwolnien rytmu serca przy wydechu [w %]': zwolnienie_wydechy_z_procent,
        'Ilosc wzrostów cisnienia skurczowego SBP przy wdechu [w %]':wzrost_SBP_wdechy_z_procent,
        'Ilosc spadków cisnienia skurczowego SBP przy wdechu [w %]':spadek_SBP_wdechy_z_procent,
        'Ilosc wzrostów cisnienia skurczowego SBP przy wydechu [w %]':wzrost_SBP_wydechy_z_procent,
        'Ilosc spadków cisnienia skurczowego SBP przy wydechu [w %]':spadek_SBP_wydechy_z_procent
        }

data2 = {'Srednie cisnienie skurczowe': mean_sbp_nad,
        'Zliczone skurcze R': rr_nad_zliczenie,
        'Srednia wartosc interwalow RR': mean_rr_nad,
        'Wartosci SSDN':sdnn_nad,
        'Wartosci RMSSD': rmssd_nad,
        'Wartosci pNN50': pnn50_nad,
        'Wartosci pNN20': pnn20_nad,
        'Ilosc skurczow R przy wdechu [w %]': wwr_wdech_nad_procent,
        'Ilosc skurczow R przy wydechu [w %]': wwr_wydech_nad_procent,
        'Ilosc przyspieszen rytmu serca przy wdechu [w %]': przyspieszenie_wdechy_nad_procent,
        'Ilosc zwolnien rytmu serca przy wdechu [w %]': zwolnienie_wdechy_nad_procent,
        'Ilosc przyspieszen rytmu serca przy wydechu [w %]': przyspieszenie_wydechy_nad_procent,
        'Ilosc zwolnien rytmu serca przy wydechu [w %]': zwolnienie_wydechy_nad_procent,
        'Ilosc wzrostów cisnienia skurczowego SBP przy wdechu [w %]':wzrost_SBP_wdechy_nad_procent,
        'Ilosc spadków cisnienia skurczowego SBP przy wdechu [w %]':spadek_SBP_wdechy_nad_procent,
        'Ilosc wzrostów cisnienia skurczowego SBP przy wydechu [w %]':wzrost_SBP_wydechy_nad_procent,
        'Ilosc spadków cisnienia skurczowego SBP przy wydechu [w %]':spadek_SBP_wydechy_nad_procent
        
        }

df_z = pd.DataFrame(data, index=pliki_z_dane)
df_nad = pd.DataFrame(data2, index=pliki_nad_dane)

df_z.to_csv(os.path.join(KATALOG_DANYCH, "Dane_zdrowi.csv"))
df_nad.to_csv(os.path.join(KATALOG_DANYCH, "Dane_nadcisnienie.csv"))