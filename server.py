#
# adapted from github.com/0D0AResearch
#

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.text import *
import uvicorn
import aiohttp
import os


# needed to load learner 
@np_func
def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1), average='weighted')

class WeightedLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight, eps:float=0.1, reduction='mean'):
        super().__init__()
        self.weight,self.eps,self.reduction = weight,eps,reduction
        
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, weight=self.weight, reduction=self.reduction)
        
!mkdir -p /root/.fastai/data/arwiki/corpus2_100/tmp/

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def predict_image_from_bytes(input_bytes):
    img = open_image(BytesIO(input_bytes))
    pred_class, pred_idx, losses = learn.predict(img)
    print(pred_class)
    print({"prediction": str(pred_class), "scores": sorted(zip(learn.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)})
    return JSONResponse({"prediction": str(pred_class), "scores": sorted(zip(learn.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)})


app = Starlette(debug=True)
classes = ['foot', 'hand']
defaults.device = torch.device('cpu')
learn = load_learner('models')


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


@app.route('/')
def form(request):
    return HTMLResponse("""
    
    
<style>
    * {
        box-sizing: border-box;
       }

    #blueBox {
        width: 700px;
        padding: 40px;  
        border: 12px solid blue;
        text-align: left;
        position: absolute;
        left: 25%;
        }

    #redBox {
        width: 500px;
        padding: 10px;  
        border: 2px solid red;
        }
</style>


    <div id="blueBox">       
    <div style="text-align:center">
    <h1> Suffering from Phalanges Agnosia? </h2>
    
    <p>
    <h3> Suffer no more, this web-app can help you tell the difference between a hand and a foot </h2>
    </div>
    <p>
    <p>
    <div id="redBox">
    <form action ="/upload" method="post" enctype="multipart/form-data">
        Select an image to upload:
        <input type="file" name="file">
        <input type="submit" value="Upload Image">
    </form>
    </div>
    <p>
    <p>
    
    <div id="redBox">
    Or submit the URL of a picture:
    
    <form action ="/classify-url" method="get">
        <input type ="url" name ="url">
        <input type="submit" value="Fetch and analyse an image">
    </form>
    </div>
    
    </div>
    """)


if __name__ == '_main__':
    port1 = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port1)
