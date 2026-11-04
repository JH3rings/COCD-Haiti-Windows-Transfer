from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
import json
R=Path(__file__).resolve().parent/'results'
def read(p,default):
 try:return json.loads(p.read_text(encoding='utf-8'))
 except (OSError,ValueError):return default
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path.startswith('/status'):
   rows={}
   for a in ['T_multimodal','T_optical','base','A','B','unet']:
    h=read(R/a/'history.json',[]);s=read(R/a/'selection.json',{});t=read(R/a/'test_metrics.json',{})
    rows[a]={'epochs':h[-1]['epoch'] if h else 0,'best_val':s.get('validation_IoU',max([v['validation']['IoU'] for v in h if v.get('validation')],default=None)),'test_IoU':t.get('IoU')}
   data=json.dumps({'rows':rows,'progress':read(R/'progress.json',{}),'error':read(R/'error.json',{})}).encode();ctype='application/json'
  else:
   data='''<!doctype html><meta charset="utf-8"><title>SAR A/B</title><style>body{font:18px system-ui;max-width:950px;margin:40px auto;color:#234}td,th{padding:14px}progress{width:100%;height:25px}</style><h1>DLR BCE＋Dice 1:1 · 六组重新训练</h1><p>基础损失BCE:Dice=1:1，六组均重新训练100轮，val每5轮；没有复用旧训练结果。</p><progress id="bar" max="600"></progress><table><thead><tr><th>模型</th><th>完成轮次</th><th>最佳验证IoU</th><th>测试IoU</th></tr></thead><tbody id="rows"></tbody></table><pre id="status"></pre><script>async function u(){let d=await(await fetch('/status')).json(),sum=0;let f=v=>typeof v==='number'?(100*v).toFixed(2)+'%':'—';document.getElementById('rows').innerHTML=Object.entries(d.rows).map(([a,r])=>{sum+=r.epochs;return '<tr><td>'+a+'</td><td>'+r.epochs+'/100</td><td>'+f(r.best_val)+'</td><td>'+f(r.test_IoU)+'</td></tr>'}).join('');document.getElementById('bar').value=sum;document.getElementById('status').textContent=JSON.stringify(d.progress,null,2)+(Object.keys(d.error).length?JSON.stringify(d.error):'');}u();setInterval(u,3000);</script>'''.encode();ctype='text/html; charset=utf-8'
  self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*a):pass
if __name__=='__main__':HTTPServer(('127.0.0.1',8768),H).serve_forever()
