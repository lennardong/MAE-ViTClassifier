import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel, TrainingArguments, Trainer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_metric, Features, ClassLabel, Array3D
from transformers import ViTConfig, ViTModel


# 1. Load custom folder of images
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = ImageFolder(root='./data/WBC_100/train/data', transform=transform)

# 2. Infer categories
class_names = dataset.classes
num_labels = len(class_names)

# 3. Train/test split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 4. Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Feature Extractor
feature_extractor = ViTFeatureExtractor(size=(224, 224))

# 6. Model
# Create the configuration
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_classes=num_labels,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)

# Custom Model Class
class ViTForImageClassification2(nn.Module):
    def __init__(self, config):
        super(ViTForImageClassification2, self).__init__()
        self.vit = ViTModel(config)  # Initialize with config
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.num_labels = config.num_classes

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Initialize the custom model
model = ViTForImageClassification2(config)


# 7. TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

def compute_metrics(p):
    return {"accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 8. Train and get logs
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# 9. Plot train/val loss
plt.plot(train_results.metrics['train_loss'], label='train_loss')
plt.plot(train_results.metrics['eval_loss'], label='val_loss')
plt.legend()
plt.show()
