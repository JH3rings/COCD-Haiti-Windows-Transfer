from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent/'results'
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/api':
   rows=[]
   for p in ROOT.glob('*/progress.json'):
    try:rows.append(dict(name=p.parent.name,**json.loads(p.read_text())))
    except (OSError,ValueError):pass
   body=json.dumps(rows).encode();mime='application/json'
  else:
   body='''<meta charset="utf-8"><title>DLR shared SAR retrieval</title><style>body{font:18px system-ui;margin:40px}progress{width:400px}article{padding:16px;border-bottom:1px solid #ddd}</style><h1>DLR共享SAR检索训练</h1><p>100 epochs · val every 5 · no test access</p><main></main><script>async function update(){let r=await fetch('/api'),a=await r.json(),m=document.querySelector('main');m.replaceChildren();for(let x of a){let e=document.createElement('article');e.textContent=x.name+' | epoch '+x.epoch+'/100 | batch '+x.batch+'/'+x.batches+' | best val IoU '+(x.best_validation_IoU<0?'—':(100*x.best_validation_IoU).toFixed(2)+'%')+' | KD '+x.KD_enabled;let p=document.createElement('progress');p.max=100;p.value=x.epoch-1+x.batch/x.batches;e.append(document.createElement('br'),p);m.append(e)}}update();setInterval(update,3000)</script>'''.encode();mime='text/html; charset=utf-8'
  self.send_response(200);self.send_header('Content-Type',mime);self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
ThreadingHTTPServer(('127.0.0.1',8782),Handler).serve_forever()
