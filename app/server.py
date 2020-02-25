#
# adapted from github.com/0D0AResearch and https://github.com/piegu/
#

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from fastai.text import *
import uvicorn
import aiohttp
import asyncio
import os
import shutil
import requests
import re
        
#!mkdir -p /root/.fastai/data/arwiki/corpus2_100/tmp/
data_path = Config.data_path()
name = f'arwiki/corpus2_100/tmp/'
path_t = data_path/name
path_t.mkdir(exist_ok=True, parents=True)
shutil.copy('./app/models/spm.model', path_t)

path = Path(__file__).parent

export_file_url = 'https://www.googleapis.com/drive/v3/files/1D48EeJVzEUAf2YiomqZHZJaYlPYTOabk?alt=media&key=AIzaSyArnAhtI95SoFCexh97Xyi0JHI03ghd-_0'
export_file_name = 'ar_classifier_hard_sp15_multifit.pkl'


app = Starlette(debug=True)
classes = ['Negative', 'Positive']
defaults.device = torch.device('cpu')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app . mount ( '/ static' , StaticFiles ( directory = 'app / static' ))

#
async  def  download_file ( url , dest ):
    if  dest . exists (): return
    async  with  aiohttp . ClientSession () as  session :
        async  with  session . get ( url ) as  response :
            data  =  await  response . read ()
            with  open ( dest , 'wb' ) as  f :
                f . write ( data )
                       

accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel (kashida) to remove  
arabic_punc = re.compile(r'[\u0621-\u063A\u0641-\u064A\u061b\u061f\u060c\u003A\u003D\u002E\u002F\u007C]+') # to keep 
def clean_text(x):
    return ' '.join(arabic_punc.findall(accents.sub('',x)))


def predict_sentiment(txt):
    if not txt or len(txt.strip()) < 5:
        return JSONResponse({"prediction": "Invalid Entry", "scores": "None", "key": "1 = positive, -1 = negative"})
    txt_clean = clean_text(txt)
    if len(txt_clean.split()) < 2:
        return JSONResponse({"prediction": "Invalid Entry", "scores": "None", "key": "1 = positive, -1 = negative"})
    pred_class, pred_idx, losses = learn.predict(txt_clean)
    print(pred_class)
    print({"prediction": str(pred_class), "scores": sorted(zip(learn.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)})
    return JSONResponse({"prediction": str(pred_class), "scores": sorted(zip(learn.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True), "key": "1 = positive, -1 = negative"})



def setup_learner():
    download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        raise


# needed to load learner 
@np_func
def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1), average='weighted')

learn = setup_learner()


@app.route("/classify", methods=["GET"])
def classify(request):
    the_text = request.query_params["sentence"]
    return predict_sentiment(the_text)


@app.route('/')
def form(request):
    return HTMLResponse("""
    
    
<style>
    * {
        box-sizing: border-box;
       }

    #blackBox {
        width: 800px;
        padding: 20px;  
        border: 4px solid black;
        text-align: left;
        position: absolute;
        left: 25%;
        }

    #greenBox {
        width: 600px;
        padding: 10px;  
        border: 2px solid green;
        }
        
     #cleared {
        margin-top:20px;
        border: 1px solid navy;
        clear:both;
        width: 600px;
        padding: 4px;
     }
</style>


    <div id="blackBox">       
    <div style="text-align:center">
    <h1> Sentiment Classifier </h2>
    </div>

    
    <div id="greenBox">
    Enter your text:  
    <form action ="/classify" method="get">
        <input type ="text" name ="sentence" minlength="5" maxlength="100" size="50" value= "كان المكان نظيفا والطعام جيدا. أوصي به للأصدقاء.">
        <input type="submit" value="Get Sentiment">
    </form>
    </div>
    
    <div id="cleared">
    By Abed Khooli (Twitter: @akhooli)
    </div>
    
    </div>

    """)

if __name__ == '__main__':
    if 'serve' in sys.argv:
        port = int(os.getenv('PORT', 5042))
        uvicorn.run(app=app, host='0.0.0.0', port=port)
   
