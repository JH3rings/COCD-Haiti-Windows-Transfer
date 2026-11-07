"""Read-only local progress page for the retrieval experiment."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"results"
PAGE="""<!doctype html><meta charset="utf-8"><title>DLR retrieval study</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#f4f6f8;color:#202b34}
table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}
progress{width:220px}#state{padding:16px;background:#dce8ef;margin-bottom:16px}</style>
<h1>DLR optical → SAR retrieval</h1><div id="state"></div><table><thead><tr>
<th>Model</th><th>Progress</th><th>Epoch / batch</th><th>Best validation IoU</th><th>Loss</th>
</tr></thead><tbody id="rows"></tbody></table><p>100 epochs · validation every 5 · BCE + Dice · fixed official split</p>
<script>
async function update(){let r=await fetch('/api',{cache:'no-store'}),d=await r.json();
document.getElementById('state').textContent=JSON.stringify(d.state);
let b=document.getElementById('rows');b.replaceChildren();
for(let x of d.models){let tr=document.createElement('tr');let name=document.createElement('td');name.textContent=x.name;tr.append(name);
let td=document.createElement('td'),p=document.createElement('progress');p.max=100;p.value=x.epoch;td.append(p);tr.append(td);
for(let value of [x.epoch+' / 100; '+x.batch+' / '+x.batches,
x.best<0?'—':(x.best*100).toFixed(2)+'%',x.loss===null?'—':x.loss.toFixed(4)]){let c=document.createElement('td');c.textContent=value;tr.append(c)}b.append(tr)}
}update();setInterval(update,5000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/api":
            state=json.loads((ROOT/"status.json").read_text()) if (ROOT/"status.json").exists() else {}
            models=[]
            for method in ("M0","M1","M2","M3","M4"):
                for role in ("sar","teacher","student"):
                    if method=="M2" and role!="student":
                        continue
                    name=f"{method}_{role}_seed42"
                    folder=ROOT/name
                    p=json.loads((folder/"progress.json").read_text()) if (folder/"progress.json").exists() else {}
                    s=json.loads((folder/"selection.json").read_text()) if (folder/"selection.json").exists() else {}
                    loss=p.get("losses",{}).get("seg")
                    models.append(dict(name=name,epoch=p.get("epoch",0),batch=p.get("batch",0),
                                       batches=p.get("batches",23),best=s.get("validation_IoU",p.get("best_validation_IoU",-1)),loss=loss))
            data=json.dumps(dict(state=state,models=models)).encode()
            mime="application/json"
        else:
            data=PAGE.encode();mime="text/html; charset=utf-8"
        self.send_response(200);self.send_header("Content-Type",mime)
        self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(data)

    def log_message(self,*args):
        pass


if __name__=="__main__":
    ThreadingHTTPServer(("127.0.0.1",8781),Handler).serve_forever()
