import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import accuracy_score



#Réucpération des données
#On a le choix de prendre les vecteurs des features linguistiques
#ou bien issus de l'indice TFIDF

def get_data(path_file, sep=','):
    return pd.read_csv(path_file, sep=sep).fillna(0)



def data_to_features():
    global X, y, X_predict
    data = get_data("features_X.csv")
    predict = get_data("features_X_predict.csv")

    X = data.drop(columns= ["Pourcentage_Familier", "Pourcentage_Courant","Pourcentage_Soutenu","Pourcentage_Poubelle"]).to_numpy()
    y = data[["Pourcentage_Familier", "Pourcentage_Courant","Pourcentage_Soutenu","Pourcentage_Poubelle"]].to_numpy()
    X_predict = predict.to_numpy()



def data_to_tfidf():
    global X, y, X_predict
    data = get_data("tfidf_X.csv")
    predict = get_data("tfidf_X_predict.csv")

    X = data.drop(columns= ["Pourcentage_Familier", "Pourcentage_Courant","Pourcentage_Soutenu","Pourcentage_Poubelle"]).to_numpy()
    y = data[["Pourcentage_Familier", "Pourcentage_Courant","Pourcentage_Soutenu","Pourcentage_Poubelle"]].to_numpy()
    X_predict = predict.drop(columns= ["Pourcentage_Familier", "Pourcentage_Courant","Pourcentage_Soutenu","Pourcentage_Poubelle"]).to_numpy()

  




data_to_features()


#transformation des labels en 1/0

def transform_labels(y):
    y[y>0]=1
    return y
y = transform_labels(y)
y


# Sueil des prédictions:


def transform_prediction(predictions):
    predictions[predictions>=0.50] = 1
    predictions[predictions<0.50] = 0
    return predictions


#Mise en place du modèle d'apprentissage automatique et des paramètres

def keras_RN(X_train, 
             y_train, 
             nb_classes, 
             hidden_layer_sizes, 
             dropout, 
             epochs, 
             verbose,
             input_dim
             ):
    """
    input: X_train, Y_train
    ouput: predict values [[proba1, proba2, proba3, proba4]]
    """
    model = Sequential()
    model.add(Dense(hidden_layer_sizes, input_dim=input_dim,
                    activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_sizes, input_dim=input_dim,
                    activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_sizes, input_dim=input_dim,
                    activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_sizes, input_dim=input_dim,
                    activation='relu'))

    model.add(Dense(nb_classes, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=500, shuffle=True )
    return model





def transform_pred(y_pred, seuil_reliability=0.9):
    y_pred[y_pred>=seuil_reliability] = 1
    y_pred[y_pred<seuil_reliability] = 0
    return y_pred




def get_reliable_data_inter_classifier(y_pred, y_train, X_train):
    """
    """
    y_to_add = list()
    x_to_add = list()
    for i in range(len(y_pred)):
        if sum(y_pred[i]) != 0:
            y_to_add.append(y_pred[i])
            x_to_add.append(X_predict[i])
    if len(y_to_add) != 0 :
        y_train = np.append(y_train, y_to_add, axis=0)
        X_train = np.append(X_train, x_to_add, axis=0)
    return X_train, y_train, len(y_to_add)



nb_classes = 4
hidden_layer_sizes = 10 
dropout = .2 
epochs = 50 #100
verbose = 1 
input_dim = len(X[0])


#mise en place d'une fonction pour 
def seuil_iter(seuil, nb_iter):
    global nbr_iter, seuil_reliability
    seuil_reliability = seuil
    nbr_iter = nb_iter
    

seuil_iter(0.9, 10)

# # Validation croisée et exécution du modèle d'AA
# 
#  * Lors de la validation croisée le modèle est testé sur *f* folds:
#       * A chaque fold, le modèle itère *n* fois et : 
#           1. s'entraîne à partir de la graine 
#           2. prédit le registre pour les données à annoter automatiquement
#           3. le modèle sélectionne selon un seuil fixé par l'utilisateur les textes jugés fiables 
#           4. Ces textes sont alors rajoutés à la graine
#           5. une nouvelle itération s'amorce
#           
#           
#  Par exemple, pour une validation croisée à 5 folds et un modèle à 10 itérations vous executerez au total 50 fois le modèle.



set_features = list()
set_fold = list()
set_mse = list()
set_mae = list()
set_accuracy = list()
set_cp_text_added = list()


#mise en place de la validation croisée, choix de n folds
def folds(num):
    global fld
    fld = KFold(n_splits=num, shuffle=True, random_state=42)

folds(10)

#exécution du modèle


fold = 0

for train, test in fld.split(X=X, y=y):
    
    all_id = list(train) + list(test)
    train_index = round((len(all_id)/5) * 3)
    test_index = int(round((len(all_id)/5) / 2, 0))

    train = all_id[:train_index]
    test = all_id[-(test_index*2):-test_index]
    dev = all_id[-test_index:]

    X_train = X[train]
    X_test = X[test]
    X_dev = X[dev]

    # familier
    y_train = y[train]
    y_test = y[test]
    y_dev = y[dev]
    
    print("-----"*10)
    print("\t\tfold : {}/5".format(fold+1))
    print("-----"*10)

    for iter in range(nbr_iter):

        print("\n\titer : {}/{}".format(iter+1, nbr_iter))
        set_fold.append(fold)

        if iter < nbr_iter-1:

            print("-"*70)
            print("\tfold : {}/5 \titer : {}/{}".format(fold+1, iter+1, nbr_iter))
            print("-"*70)
            
            predictor = keras_RN(X_train, y_train, nb_classes, hidden_layer_sizes, dropout, epochs, verbose, input_dim)


          
            
            y_pred = predictor.predict(X_predict)
            y_pred = transform_pred(np.array(y_pred), seuil_reliability)
            y_pred_test = predictor.predict(X_test)
            y_pred_test = transform_pred(np.array(y_pred_test), seuil_reliability)

            set_mse.append(mean_squared_error(y_pred_test, y_test))
            set_mae.append(mean_absolute_error(y_pred_test, y_test))

            X_train, y_train, cp_text_added = get_reliable_data_inter_classifier(y_pred, y_train, X_train)

            set_cp_text_added.append(cp_text_added)

        if iter == nbr_iter-1:
            
            print("-"*70)
            print("\tfold : {}/5 \titer : {}/{}".format(fold+1, iter+1, nbr_iter))
            print("-"*70)
            
            predictor = keras_RN(X_train, 
                         y_train, 
                         nb_classes, 
                         hidden_layer_sizes, 
                         dropout, 
                         epochs, 
                         verbose, 
                         input_dim)
            
            y_pred_test = predictor.predict(X_dev)
            y_pred_test = transform_pred(np.array(y_pred_test))
            
            set_mse.append(mean_squared_error(y_pred_test, y_dev))
            set_mae.append(mean_absolute_error(y_pred_test, y_dev))
            
    fold+=1
             
            


#Sauvegarde du classifier en local
predictor.save("M2_semi_supervised_multilabels-expertFeatures_RN.h5")


#On charge le modèle
load_predictor = keras.models.load_model("M2_semi_supervised_multilabels-expertFeatures_RN.h5")

#on obtient les prédictions sur les données de test
predictions = load_predictor.predict(X_test)
annotation_automatique = transform_prediction(predictions)

#on calcule la précision
score = load_predictor.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (load_predictor.metrics_names[1], score[1]*100))




#écriture des résultats
dict_out = dict()
dict_out["fold"] = set_fold
dict_out["text added"] = set_cp_text_added
dict_out["mse"] = set_mse
dict_out["mae"] = set_mae

df = pd.DataFrame.from_dict(dict_out, orient="index").fillna(0)
name_to_save = "semi_supervised_multilabels".format(epochs, str(seuil_reliability).replace(".","-"), nbr_iter)
df.to_csv("{}.csv".format(name_to_save), sep="\t", encoding="utf-8")
df = pd.read_csv("{}.csv".format(name_to_save), sep="\t", encoding="utf-8")


#création d'un dataframe avec les données automatiquement annotées
df1 = pd.DataFrame(X_test) 
df2 = pd.DataFrame(annotation_automatique)
result = pd.concat([df1, df2], axis=1).reindex(df1.index)
result.to_csv("automatic_annotation_results.csv")



