# Reconnaissance du type de déchets intelligents : fichier d’entraînement du modèle et analyse technique

```python
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        total_acc += accuracy_top1(logits, targets)
        n_batches += 1

        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(targets.detach().cpu())

    if n_batches == 0:
        return {"loss": float("nan"), "acc": float("nan"), "confusion": None}

    preds = torch.cat(all_preds)
    targs = torch.cat(all_targets)
    num_classes = int(max(preds.max(), targs.max()).item()) + 1 if preds.numel() else 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targs.tolist(), preds.tolist()):
        confusion[t, p] += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches, "confusion": confusion}


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "resnet18":
        # 兼容 torchvision 旧版本（0.8.x）: 使用 pretrained=True
        try:
            weights = getattr(models, "ResNet18_Weights").DEFAULT  # torchvision>=0.13
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        if not hasattr(models, "mobilenet_v3_small"):
            raise ValueError("当前 torchvision 版本不支持 mobilenet_v3_small，请升级 torchvision 或使用 --arch resnet18")
        try:
            weights = getattr(models, "MobileNet_V3_Small_Weights").DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        if not hasattr(models, "efficientnet_b0"):
            raise ValueError("当前 torchvision 版本不支持 efficientnet_b0，请升级 torchvision 或使用 --arch resnet18")
        try:
            weights = getattr(models, "EfficientNet_B0_Weights").DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"不支持的 arch: {arch}（可选: resnet18 / mobilenet_v3_small / efficientnet_b0）")


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf

```

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
