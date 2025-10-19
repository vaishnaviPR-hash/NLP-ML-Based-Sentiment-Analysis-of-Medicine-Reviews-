"""
#Local code
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# -------------------------
# Multi-Task BERT Model
# -------------------------
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        super(MultiTaskBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.effectiveness_head = nn.Linear(hidden_size, num_labels)
        self.price_head = nn.Linear(hidden_size, num_labels)
        self.side_effects_head = nn.Linear(hidden_size, num_labels)
        self.overall_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return {
            "effectiveness": self.effectiveness_head(cls_output),
            "price": self.price_head(cls_output),
            "side_effects": self.side_effects_head(cls_output),
            "overall": self.overall_head(cls_output)
        }

# -------------------------
# Load Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")

model = MultiTaskBERT()
model.load_state_dict(torch.load("multi_task_bert_laptop.pt", map_location="cpu"))
model.eval()

# -------------------------
# Prediction Function
# -------------------------
def predict_review(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=48,
        padding="max_length",
        truncation=True
    )
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    predictions = {k: int(torch.argmax(v, dim=1)) for k,v in outputs.items()}
    return predictions
"""


"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"

import os
import torch
from transformers import AutoTokenizer
import torch.nn as nn

# -------------------------
# MultiTaskBERT definition
# -------------------------
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        super(MultiTaskBERT, self).__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))
        hidden_size = self.bert.config.hidden_size
        self.effectiveness_head = nn.Linear(hidden_size, num_labels)
        self.price_head = nn.Linear(hidden_size, num_labels)
        self.side_effects_head = nn.Linear(hidden_size, num_labels)
        self.overall_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return {
            "effectiveness": self.effectiveness_head(cls_output),
            "price": self.price_head(cls_output),
            "side_effects": self.side_effects_head(cls_output),
            "overall": self.overall_head(cls_output)
        }

# -------------------------
# Load tokenizer & model from Hugging Face
# -------------------------
HF_REPO = "VaishnaviPR123/multi-task-bert-laptop"  # private HF repo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(HF_REPO, use_auth_token=os.getenv("HF_TOKEN"))
model = MultiTaskBERT(model_name=HF_REPO)
model.to(DEVICE)
model.eval()

# -------------------------
# Prediction function
# -------------------------
def predict_review(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=48,
        padding="max_length",
        truncation=True
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    predictions = {k: int(torch.argmax(v, dim=1)) for k, v in outputs.items()}
    return predictions

# -------------------------
# Local test
# -------------------------
'''
if __name__ == "__main__":
    sample_text = "The battery life of this medicine is excellent."
    print("Predictions:", predict_review(sample_text))
'''