import numpy as np

x_entry = np.array(([3, 1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1.5]),dtype=float) # Longueur et Largeur en entrée de l'Ia sous forme de tableau
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float) # Données de sortie = 1:rouge | 0:blue

x_entry_formated = x_entry/np.max(x_entry, axis=0) # Retourne des valeur proportinelles entre 0 et 1 (comme le map en BluePrint)

X = np.split(x_entry_formated, [8])[0] # Recupere seulement les 8 premieres valeur du tableau formaté et les stock dans X
xPrediction = np.split(x_entry_formated, [8])[1] # Donne la derniere valeur pour la prediction

class Neural_Network(object):
    def __init__(self) -> None:
        self.inputSize = 2 # Nombre de neuronnes d'entré.
        self.outputSize = 1 # Nombre de neuronnes de sortie.
        self.hiddenSize = 3 # Nombre de neuronnes caché.

        # Génératoin aleatoire des poids des synapses
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # Matrice 2x3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # Matrice 3x1

    ## Fonction de propagation avant
    ## Permet de multiplier les valeurs des neuronnes par celle des synapse pour additionner les nombre par la suite et utiliser la fonction Sigmoid
    def forward(self,x):

        ## z = valeur entrée
        ## z2 = valeur caché
        ## z3 = valeur sortie

        # Calcules des matrices avec le poid des synapse et les entrées (Calcule matricielle) | Calcule entre la couche d'entrée et la couche cachée
        self.z = np.dot(x, self.W1)

        # Application de la fonction Sigmoid
        self.z2 = self.sigmoid(self.z)

        # Deuxieme calcules de matrice entre la couche cachée et la couche de sortie
        self.z3 = np.dot(self.z2, self.W2)
        
        # Deuxieme application de la fonction Sigmoid pour donner la valeur de sortie
        o = self.sigmoid(self.z3)

        return o

    ## Fonction de retro-propagation grace a la dérivé de la Sigmoid
    def backward(self,X,y,o):

        ## X = valeur entrée 
        ## y = valeur cible
        ## o = valeur sortie

        # Calcule de la marge d'erreur de sortie
        self.o_error = y - o

        # Calcule de l'erreur delta
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        # Calcules des erreurs des neuronnes cachées
        self.z2_error = np.dot(self.o_delta, self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        # Mise a jour des poids
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    # Fonction Sigmoid
    def sigmoid(self,s):
        return 1 / (1 + np.exp(-s))
    
    # Dérivé de la fonction Sigmoid
    def sigmoidPrime(self,s):
        return s * (1-s)

    ## Fonction d'entrainement de l'IA en repettant plusieurs fois la fonction de retro-propagation
    def train(self,X,y):
        # Recupération d'une valeur de sortie
        o = self.forward(X)

        # Utilisation de la retro-propagation a partir de la valeur de sortie recupérée
        self.backward(X,y,o)

    # Transciption des valeurs chiffré en valeur comprehensible par l'utilisateur
    def predict(self):
        print("Donné predite apres l'entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        if(self.forward(xPrediction) < 0.5):
            print("La fleure est BLEU")
        else:
            print("La fleure est ROUGE !")
    
NN = Neural_Network()

output = NN.forward(X)

for i in range(100000):
    print("# " + str(i) + "\n")
    print("Valeur d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))

    # Arrondi les valeurs au deux décimales apres la virgules 
    print("Sortie predite: \n" + str(np.matrix.round(NN.forward(X),2)))
    NN.train(X,y)

NN.predict()

print("----------------------------------------------")

def predict_after_train():

    # Demande a l'utilisateur de saisir les infos du pétal
    l = float(input("Veuillez saisir la longueur de votre pétale : "))
    L = float(input("Veuillez saisir la largeur de votre pétale : "))

    # Combination des valeurs dans un tableau
    user_input = np.array([[l, L]])

    # Formatation des valeurs saisies
    user_input_formatted = user_input / np.max(user_input, axis=0)

    # Prediction a partir des valeurs de l'utilisateur
    user_predict_result = NN.forward(user_input_formatted)

    print("La valeur de sortie est de : " + str(user_predict_result))

    if(user_predict_result) < 0.5:
        print("La fleure est BLEU")
    else:
        print("La fleure est ROUGE !")

while True:
    predict_after_train()