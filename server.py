#
# adapted from github.com/0D0AResearch
#

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.text import *
import uvicorn
import aiohttp
import os
import shutil


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
        
#!mkdir -p /root/.fastai/data/arwiki/corpus2_100/tmp/
data_path = Config.data_path()
name = f'/arwiki/corpus2_100/tmp/'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)
shutil.copy('models/spm.model', path)

export_file_url = 'https://www.googleapis.com/drive/v3/files/1--scwn8SjaGBtIukFF1_K32QucNbAhIe?alt=media&key=AIzaSyArnAhtI95SoFCexh97Xyi0JHI03ghd-_0'
export_file_name = 'ar_classifier_hard_sp15_multifit.pkl'

def predict_sentiment(txt):
    txt =  "كان المكان نظيفا والطعام جيدا. أوصي به للأصدقاء." #  (category 1)
    pred_class, pred_idx, losses = learn.predict(txt)
    print(pred_class)
    print({"prediction": str(pred_class), "scores": sorted(zip(learn.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)})
    return JSONResponse({"prediction": str(pred_class), "scores": sorted(zip(learn.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)})

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


app = Starlette(debug=True)
classes = ['-1', '1']
defaults.device = torch.device('cpu')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#learn = load_learner('models')


@app.route("/classify", methods=["GET"])
async def classify(request):
    the_text = await get_text(request.query_params["sentenc"])
    return predict_sentiment(the_text)


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
    <h1> Sentiment Classifier </h2>
    </div>

    
    <div id="redBox">
    Enter your text:  
    <form action ="/classify-url" method="get">
        <input type ="text" name ="sentence">
        <input type="submit" value="Get Sentiment">
    </form>
    </div>
    
    </div>
    """)


if __name__ == '_main__':
    port1 = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port1)
