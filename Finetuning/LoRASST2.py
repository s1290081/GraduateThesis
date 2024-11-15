from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# モデルとトークナイザーの読み込み
model_name = "./llama-68m"  # 使用するモデル名
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2つのクラスに設定 (SST-2の場合)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# パディングトークンを設定
tokenizer.pad_token = tokenizer.eos_token  # EOSトークンをパディングトークンとして設定
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA設定
lora_config = LoraConfig(
    r=8,  # ランク
    lora_alpha=32,  # スケーリング係数
    lora_dropout=0.1,  # ドロップアウト
    task_type="SEQ_CLS",  # タスクタイプ（分類の場合）
)

# LoRA適用
model = get_peft_model(model, lora_config)

# DataCollatorを使用してデータコラトレーションを行う
data_collator = DataCollatorWithPadding(tokenizer)

# SST-2データセットの準備
sst2_dataset = load_dataset("glue", "sst2")

# データセットの前処理
def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

# データセットの前処理
sst2_train_dataset = sst2_dataset['train'].map(preprocess, batched=True)
sst2_test_dataset = sst2_dataset['validation'].map(preprocess, batched=True)

# 評価メトリクスの定義
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainerの設定
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)

# Trainerインスタンス作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sst2_train_dataset,
    eval_dataset=sst2_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# トレーニングの実行
trainer.train()

# モデルの保存
trainer.save_model("./final_model")
