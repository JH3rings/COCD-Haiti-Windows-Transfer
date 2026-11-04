from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import json
R=Path(__file__).resolve().parent/'results';ARMS=['T_residual','base','MFSAD','E1','E3']
def read(p,default):
 try:return json.loads(p.read_text(encoding='utf-8'))
 except (OSError,ValueError):return default
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path.startswith('/status'):
   rows={}
   for a in ARMS:
    h=read(R/a/'history.json',[]);s=read(R/a/'selection.json',{});t=read(R/a/'test_metrics.json',{});p=read(R/'progress_by_arm'/f'{a}.json',{})
    rows[a]={'epochs':len(h),'best_val':s.get('validation_IoU',max([x['validation']['IoU'] for x in h if x.get('validation')],default=None)),'test_IoU':t.get('IoU'),'selected_epoch':s.get('selected_epoch'),'progress':p}
   data=json.dumps({'rows':rows,'controller':read(R/'controller_status.json',{}),'complete':read(R/'complete.json',{}),'errors':{p.stem:read(p,{}) for p in (R/'errors').glob('*.json')}}).encode();ctype='application/json'
  else:
   data='''<!doctype html><meta charset="utf-8"><title>DLR 3+1</title><style>body{font:17px system-ui;max-width:1100px;margin:32px auto;color:#234}td,th{padding:10px}progress{width:100%;height:24px}pre{white-space:pre-wrap;font-size:13px}</style><h1>DLR 3+1 光学蒸馏</h1><p>官方 3+1：Teacher、SAR Baseline、M-FSAD、注意力蒸馏、注意力加 Haar。各 100 轮，每 5 轮验证。</p><progress id="bar" max="500"></progress><table><thead><tr><th>实验</th><th>完成轮次</th><th>最佳验证IoU</th><th>最佳Epoch</th><th>测试IoU</th></tr></thead><tbody id="rows"></tbody></table><pre id="details"></pre><p>所有新模型冻结后才测试；44张是120张的子集；官方划分的已知重叠保留。</p><script>async function u(){let d=await(await fetch('/status')).json(),sum=0;let f=v=>typeof v==='number'?(100*v).toFixed(2)+'%':'—';document.getElementById('rows').innerHTML=Object.entries(d.rows).map(([a,r])=>{sum+=r.epochs;return '<tr><td>'+a+'</td><td>'+r.epochs+'/100</td><td>'+f(r.best_val)+'</td><td>'+(r.selected_epoch??'—')+'</td><td>'+f(r.test_IoU)+'</td></tr>'}).join('');document.getElementById('bar').value=sum;let details={controller:d.controller,progress:Object.fromEntries(Object.entries(d.rows).map(([k,v])=>[k,v.progress])),errors:d.errors};document.getElementById('details').textContent=JSON.stringify(details,null,2);}u();setInterval(u,3000);</script>'''.encode();ctype='text/html; charset=utf-8'
  self.send_response(200);self.send_header('Cache-Control','no-store');self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*a):pass
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8769),H).serve_forever()


