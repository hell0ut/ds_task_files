import timm
import torchvision.transforms as transforms
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch
import os
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import io
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse



_ = os.path.join

BASE_PATH = os.curdir

MODEL_PATH = _(BASE_PATH,'model_last_state_dict.pth')
EMBEDDINGS_PATH = _(BASE_PATH,'list_256_emb_dump_last.pkl')
LABELS_PATH = _(BASE_PATH,'labels_last.pkl')

transformations = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
])

# load model


class Network(nn.Module):
    def __init__(self, emb_dim=256,base_model=None):
        super(Network, self).__init__()
        self.base_model = base_model.forward_features
        
        self.fc = nn.Sequential(
            nn.Linear(1280*8*8, 2048),
            nn.PReLU(),
            nn.Linear(2048, emb_dim)
        )
        
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(-1, 1280*8*8)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


def load_model():
    base_model = timm.create_model('tf_efficientnetv2_s',pretrained=True)
    base_model.eval()
    base_model.to('cpu')
    model = Network(256,base_model=base_model)
    model = model.to('cpu')
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
    return model


def load_embeddings_database():
    with open(EMBEDDINGS_PATH,'rb') as f:
        embeddings = pickle.load(f)
    with open(LABELS_PATH,'rb') as f:
        labels = pickle.load(f)
    labels_df = pd.DataFrame(labels,columns=['class'])
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    return embeddings,labels_df,labels

def train_classifier(embeddings):
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(embeddings)
    return nbrs
    

def transform_image(img_binary):
    img_raw = Image.open(io.BytesIO(img_binary))
    test_im = transformations(img_raw)
    return test_im.unsqueeze(0)


def predict(nbrs,model,image_tensor_4d,labels):
    with torch.no_grad():
        res_emb = model(image_tensor_4d)
    D,I = nbrs.kneighbors(res_emb.reshape([1,-1]))
    print(D,I)
    dft = pd.DataFrame(np.concatenate([D,I]).T,columns=['D','I'])
    dft['label'] = labels[dft['I'].astype(int)]
    label_predicted = dft.groupby('label').mean().sort_values('D').index[0]
    return label_predicted



MODEL = load_model()
EMBEDDINGS, LABELS_DF,LABELS = load_embeddings_database()
NBRS = train_classifier(EMBEDDINGS)

app = FastAPI()

@app.get("/")
async def root(request: Request):
    return HTMLResponse("""
        <html>
            <body>
                <form action="/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <input type="submit" value="Upload">
                </form>
            </body>
        </html>
    """)


@app.post("/")
async def upload_file(request: Request):
    form = await request.form()
    file = form["file"].file.read()
    label = predict(NBRS,MODEL,transform_image(file),LABELS)
    return HTMLResponse(f"""
        <html>
            <body>
                <form action="/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <input type="submit" value="Upload">
                </form>
                <p>label is : {label}</p> 
            </body>
        </html>
    """)