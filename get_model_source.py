import inspect
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
)

print(inspect.getsource(model.forward))
