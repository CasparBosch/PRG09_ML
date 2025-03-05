"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9
Deze code is geschreven in Python3
Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- scikit-learn
"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import tree

def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn """
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "1004288"

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# teken de punten
for i in range(len(x)):
    plt.plot(x[i], y[i], 'k.') # k = zwart

plt.axis([min(x), max(x), min(y), max(y)])
plt.show()

# TODO: print deze punten uit en omcirkel de mogelijke clusters
# Zie ../Figure_1.png

# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes
km = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

color_theme = np.array(['gray', 'yellow', 'blue'])

plt.subplot(1,2,1)
plt.title("K-means")
centroids = np.array(km.cluster_centers_)
plt.scatter(x=x, y=y, c=color_theme[km.labels_])
plt.scatter(centroids[:,0], centroids[:,1], marker="x", color="red")
plt.show()

# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X_data = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y_data = extract_from_json_as_np_array("y", classification_training)

# teken de punten
for i in range(len(X_data)):
    if Y_data[i] == 0:
        plt.plot(X_data[...,0][i], X_data[...,1][i], 'r.')
    else:
        plt.plot(X_data[..., 0][i], X_data[..., 1][i], 'b.')

plt.title("Classification")
plt.show()

# TODO: leer de classificaties
lr = LogisticRegression()
lr.fit(X_data, Y_data)

dt = tree.DecisionTreeClassifier(max_depth=3)
dt = dt.fit(X_data, Y_data)

# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict

Y_pred = lr.predict(X_data)
tree_pred = dt.predict(X_data)

# TODO:vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
acc_class = accuracy_score(Y_data, Y_pred)
acc_tree = accuracy_score(Y_data, tree_pred)

print("Accuracy dec tree: {:.2f}".format(acc_tree))
print("Accuracy log reg: {:.2f}".format(acc_class))

tree.plot_tree(dt)

# haal data op om te testen
classification_test = data.classification_test()

# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

# Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

Z = dt.predict(X_test)
ZZ = lr.predict(X_test)

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie dec tree (test): " + str(classification_test))

classification_test = data.classification_test(ZZ.tolist())
print("Classificatie accuratie log reg (test): " + str(classification_test))