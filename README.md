
# Facial Expression Recognition Challenge 

## პროექტის აღწერა
Kaggle-ის FER Challenge-ის გადაწყვეტა PyTorch-ით.
7 ემოციის კლასიფიკაცია: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## WandB Project
[ექსპერიმენტების ნახვა](https://wandb.ai/tgura23-free-university-of-tbilisi-/fer-challenge)

## მონაცემები
- Train: ~28,000 სურათი (48x48 grayscale)
- Validation: ~7,000 სურათი
- კლასები: 7 (imbalanced — Disgust ყველაზე ცოტა)

---

## არქიტექტურები და გადაწყვეტილებები

### მოდელი 1: SmallCNN (Baseline / Underfitting)
```
Conv(1→32) → ReLU → MaxPool
Conv(32→64) → ReLU → MaxPool
Linear(9216→256) → Dropout(0.5) → Linear(256→7)
```
**გადაწყვეტილება:** საწყისი წერტილი. მარტივი არქიტექტურა რომ დავინახოთ სად ვართ. მოსალოდნელია underfitting რადგან ძალიან ცოტა პარამეტრი აქვს.

**შედეგი:** Val Acc ~50% — underfit ✅

---

### მოდელი 2: MediumCNN (BatchNorm დამატება)
```
Conv(1→32) → BN → ReLU → MaxPool
Conv(32→64) → BN → ReLU → MaxPool
Conv(64→128) → BN → ReLU → MaxPool
Linear(4608→512) → Dropout(0.4) → Linear(512→256) → Linear(256→7)
```
**გადაწყვეტილება:** BatchNorm დაემატა რადგან:
1. training სტაბილური გახდება
2. უფრო სწრაფად ისწავლის
3. regularization ეფექტი აქვს

**შედეგი:** Val Acc ~60% — გაუმჯობესება ✅

---

### მოდელი 3: DeepCNN + Residual Blocks
```
Conv(1→64) → BN → ReLU
ResBlock(64) → MaxPool
Conv(64→128) → BN → ReLU
ResBlock(128) → MaxPool
Conv(128→256) → BN → ReLU
ResBlock(256) → MaxPool
AdaptiveAvgPool → Linear(256→128) → Dropout(0.5) → Linear(128→7)
```
**გადაწყვეტილება:** Residual connections დაემატა რადგან:
1. Gradient vanishing პრობლემა მოიხსნება
2. Deep network უკეთ ისწავლის
3. Skip connections identity-ს ინარჩუნებს

**შედეგი:** Val Acc ~65% ✅

---

### მოდელი 4: OverfitCNN (განზრახ Overfit)
```
Conv(1→64) → BN → ReLU
ResBlock(64) → MaxPool
Conv(64→128) → BN → ReLU
ResBlock(128) → MaxPool
Conv(128→256) → BN → ReLU
ResBlock(256) → MaxPool
AdaptiveAvgPool → Linear(256→128) → Linear(128→7)
```
**გადაწყვეტილება:** იგივე DeepCNN, მაგრამ Dropout გარეშე და მეტი epoch. მიზანია overfitting-ის დემონსტრაცია — Train Acc >> Val Acc.

**შედეგი:** Train Acc ~90%, Val Acc ~55% — overfit ✅

---

## ჰიპერპარამეტრების ექსპერიმენტები

| Run | LR | Weight Decay | Dropout | Val Acc |
|-----|----|-------------|---------|---------|
| SmallCNN | 0.001 | 1e-4 | 0.5 | ~50% |
| MediumCNN | 0.001 | 1e-4 | 0.4 | ~60% |
| DeepCNN | 0.0005 | 1e-3 | 0.5 | ~65% |
| DeepCNN_overfit | 0.001 | 0 | 0 | overfit |

---

## Forward & Backward შემოწმება

ყველა მოდელისთვის პირველ epoch-ზე:
- **Forward Check:** output shape = (batch_size, 7) ✅
- **Backward Check:** gradient norm გამოითვლება ✅
- Gradient clipping: max_norm=1.0

---

## დასკვნა

| მოდელი | Train Acc | Val Acc | დიაგნოზი |
|--------|-----------|---------|----------|
| SmallCNN | ~46% | ~50% | Underfit |
| MediumCNN | ~62% | ~60% | კარგი |
| DeepCNN | ~70% | ~65% | კარგი |
| DeepCNN_overfit | ~90% | ~55% | Overfit |

BatchNorm და Residual connections ყველაზე დიდი გაუმჯობესება მოიტანა.
