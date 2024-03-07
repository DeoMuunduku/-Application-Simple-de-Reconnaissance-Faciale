# -Application-Simple-de-Reconnaissance-Faciale
##deo##
Ce code est une application de reconnaissance faciale simple qui utilise la bibliothèque dlib et OpenCV pour détecter et reconnaître les visages en temps réel à partir d'une webcam ou d'une vidéo. Voici ce que fait chaque partie du code :

Import des bibliothèques et des modules : Importe les bibliothèques nécessaires telles que cv2, dlib, numpy, argparse, etc.

Analyse des arguments en ligne de commande : Utilise le module argparse pour permettre à l'utilisateur de spécifier le répertoire contenant les visages connus.

Importation des modèles pré-entraînés : Importe les modèles pré-entraînés pour la détection de visage, la prédiction de la pose du visage (5 points et 68 points) et l'encodage des visages.

Définition de fonctions utilitaires :

transform(image, face_locations): Transforme les coordonnées des boîtes englobantes de visage en coordonnées de coin supérieur gauche et inférieur droit.
encode_face(image): Utilise les modèles pré-entraînés pour détecter les visages, extraire leurs caractéristiques et encoder les visages.
easy_face_reco(frame, known_face_encodings, known_face_names): Effectue la reconnaissance faciale sur le cadre d'image donné en comparant les visages détectés aux visages connus.
Point d'entrée principal :

Analyse les arguments en ligne de commande pour importer les visages connus à partir du répertoire spécifié.
Charge les visages connus, encode les caractéristiques de ces visages et les stocke.
Démarre la webcam et détecte les visages en temps réel.
Applique la reconnaissance faciale en superposant les noms des visages connus sur les visages détectés dans le flux vidéo en direct.
Le processus continue jusqu'à ce que l'utilisateur appuie sur la touche 'q' pour quitter l'application.
Ce code est utile pour créer rapidement une application de reconnaissance faciale en utilisant des modèles pré-entraînés et en les intégrant avec OpenCV pour une interface utilisateur en temps réel.
