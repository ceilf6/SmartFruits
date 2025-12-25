# Reconnaissance du type de déchets intelligents : fichier d’entraînement du modèle et analyse technique

---

## 1. Définition de la tâche et organisation des données

### 1.1 Définition de la tâche

- **Entrée** : une image de déchet (papier, plastique, métal, verre, biodéchets, etc.).
- **Sortie** : la catégorie du déchet \(y \in \{0,1,\dots,C-1\}\).
- **Objectif d’apprentissage** : minimiser la perte de classification (entropie croisée), améliorer l’exactitude sur l’ensemble de validation et conserver une bonne capacité de généralisation en test / en conditions réelles.

### 1.2 Structure recommandée du répertoire de données (ImageFolder)

Le script d’entraînement utilise `torchvision.datasets.ImageFolder`, la structure du dataset est donc recommandée comme suit :

```text
your_dataset/
  train/
    plastic/
      xxx.jpg
    paper/
      yyy.jpg
  val/
    plastic/
      a.jpg
    paper/
      b.jpg
```

Note : **le nom de chaque sous-dossier = le nom de la classe**. Le script génère automatiquement `classes.json` pour enregistrer la correspondance index ↔ classe.

---

## 2. Conception de l’algorithme d’entraînement du modèle

### 2.1 Pourquoi utiliser le transfert d’apprentissage (Transfer Learning)

Problèmes fréquents en classification de déchets :

- Volume de données limité, scènes complexes (variations d’éclairage, arrière-plan, occultations, angles de prise de vue).
- Un entraînement “from scratch” sur-ajuste facilement et converge plus lentement.

Principe du transfert d’apprentissage :

- Utiliser un backbone pré-entraîné sur ImageNet (ResNet / MobileNet / EfficientNet) comme extracteur de caractéristiques.
- Remplacer la dernière couche de classification par une couche adaptée au nombre de classes \(C\), puis affiner (fine-tuning).

Avantages :

- **Convergence plus rapide**, **meilleur point de départ**, plus robuste quand les données sont peu nombreuses.

### 2.2 Choix de l’architecture réseau

Le script propose trois backbones possibles :

- `resnet18` : entraînement stable, vitesse correcte, bon modèle de base (baseline) pour comparer.
- `mobilenet_v3_small` : très léger, adapté à un déploiement edge / embarqué.
- `efficientnet_b0` : potentiel de précision plus élevé, coût d’entraînement légèrement supérieur.

Dans le rapport, on peut réaliser une comparaison : nombre de paramètres, vitesse, exactitude.

---

## 3. Prétraitement des données et augmentation (data augmentation)

Objectifs des augmentations à l’entraînement (`RandomResizedCrop` + `HorizontalFlip` + `ColorJitter`) :

- Simuler différentes distances/angles : recadrage + redimensionnement aléatoires.
- Simuler des variations de point de vue : flip horizontal.
- Simuler des variations de lumière/couleur : perturbations luminosité/contraste/saturation.

En validation, on applique seulement `Resize + CenterCrop` pour éviter l’aléatoire et rendre les métriques comparables.

La normalisation `Normalize` utilise la moyenne/écart-type ImageNet pour rester cohérent avec les poids pré-entraînés.

---

## 4. Fonction de perte, optimiseur et stratégie de taux d’apprentissage

### 4.1 Fonction de perte : entropie croisée (Cross Entropy)

Pour une classification multi-classes à étiquette unique :
\[
\mathcal{L} = - \sum_{i=1}^{C} y_i \log p_i
\]

Le script supporte `label_smoothing` :

- Rôle : réduire l’excès de confiance du modèle, atténuer le surapprentissage sur petit dataset et améliorer la généralisation.
- Plage conseillée : 0 à 0.2 (souvent 0.05 / 0.1).

### 4.2 Optimiseur : AdamW

Raisons du choix :

- Adam (taux d’apprentissage adaptatif) converge plus vite.
- W (weight decay découplé) se combine mieux à la régularisation.

### 4.3 Planification du taux d’apprentissage : CosineAnnealingLR

La décroissance cosinus est robuste pour les tâches de classification de taille moyenne/petite :

- Au début : LR plus grand, baisse rapide de la perte.
- En fin d’entraînement : LR plus petit, convergence fine et amélioration de la performance en validation.

---

## 5. Pipeline d’entraînement et détails d’ingénierie

Le script `train_garbage_classifier.py` inclut les choix d’ingénierie suivants :

- **Graine aléatoire fixée** : reproductibilité.
- **Sauvegarde automatique de la configuration** : `config.json` (utile pour le rapport et la reproduction).
- **Sauvegarde des poids** :
  - `checkpoints/last.pt` : dernière époque
  - `checkpoints/best.pt` : meilleur score en validation
- **Early Stopping** : arrêt si pas d’amélioration sur `patience` époques (évite surapprentissage et gaspillage).
- **AMP (optionnel)** : accélération sur CUDA et réduction de l’usage mémoire.
- **Logs d’entraînement** : `metrics.csv` (loss/acc train & val, LR, temps par époque).
- **Matrice de confusion** : export `confusion.npy` en fin d’entraînement pour analyser les confusions de classes.

---

## 6. Métriques d’évaluation et méthodes d’analyse

### 6.1 Indicateurs

- **Top-1 Accuracy** : taux de classification correcte le plus direct.
- (Option) Precision/Recall/F1 : plus pertinent si certaines classes sont rares ou si le coût des erreurs est élevé.

### 6.2 Analyse via matrice de confusion

La matrice de confusion permet de répondre à :

- Quelles classes se ressemblent visuellement (ex. plastique transparent vs verre) et sont souvent confondues ?
- Existe-t-il des problèmes d’annotation ou des frontières de classes ambiguës ?

Recommandation dans le rapport :

- Montrer des exemples typiques de confusion (captures d’écran).
- Proposer des pistes d’amélioration : ajouter des données, affiner les classes, s’aider de segmentation/détection, etc.

---

## 7. Comment lancer l’entraînement

À exécuter depuis la racine du projet :

```bash
python 机器学习/ext-智能垃圾类型识别/train_garbage_classifier.py \
  --data-dir /path/to/your_dataset \
  --arch resnet18 \
  --epochs 30 \
  --batch-size 32 \
  --img-size 224 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --patience 7 \
  --amp
```

Les sorties d’entraînement se trouvent dans :

- `./runs_garbage/日期时间/`

Fichiers clés :

- `checkpoints/best.pt`
- `classes.json`
- `metrics.csv`

---

## 8. Pistes d’amélioration (bonus)

- **Données** : enrichir les scènes (lumière/arrière-plan/occlusion), équilibrer le nombre d’échantillons par classe (ou ré-échantillonnage / pondération).
- **Modèle** : essayer EfficientNet-B0, ajouter Dropout, renforcer les augmentations (RandAugment/MixUp/CutMix).
- **Entraînement** : warmup du LR, geler les premières couches puis dégeler progressivement, validation croisée K-fold.
- **Montée en complexité** : si une image contient plusieurs déchets ou si la localisation est nécessaire, utiliser la détection (YOLO) ou la segmentation.
