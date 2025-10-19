import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# -------------------------
# MultiTaskBERT definition
# -------------------------
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        super(MultiTaskBERT, self).__init__()
        from transformers import AutoModel
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
# Tokenizer (lightweight)
# -------------------------
HF_REPO = "VaishnaviPR123/multi-task-bert-laptop"  # Replace with your private Hugging Face repo
DEVICE = torch.device("cpu")           # CPU only for Render Free plan
tokenizer = AutoTokenizer.from_pretrained(HF_REPO, use_auth_token=os.getenv("HF_TOKEN"))

# -------------------------
# Lazy-load quantized model
# -------------------------
model = None

def get_model():
    global model
    if model is None:
        # Download quantized model from Hugging Face
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="multi_task_bert_quantized.pt",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        model = MultiTaskBERT()
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
    return model

# -------------------------
# Prediction function
# -------------------------
def predict_review(text):
    mdl = get_model()  # model loads only when first request comes
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=48,
        padding="max_length",
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = mdl(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    predictions = {k: int(torch.argmax(v, dim=1)) for k, v in outputs.items()}
    return predictions


