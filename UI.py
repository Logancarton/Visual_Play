"""Node-centric Visual_Play laboratory: fields are depth boundaries; dots are neurons."""
from __future__ import annotations
import math, textwrap, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol
import cv2, numpy as np
from experiment_spec import MECHANISMS,SENSORY_LABELS,SENSORY_SOURCES,VISUALIZATION_MODES,ExperimentSpec
from graph_layout import field_depths
from vision_features import VisionFeatureExtractor, VisionFeatures

EXPERIMENT_FILE=Path("experiment.local.json")

@dataclass
class NeuralFrameResult:
    layer_views: Dict[str,Dict[str,np.ndarray]]
    diagnostics: Dict[str,float]
    visual_projection: Optional[np.ndarray]=None
    projection_provenance: str=""
    node_positions: Dict[str,np.ndarray]=field(default_factory=dict)
    synapses: Dict[str,np.ndarray]=field(default_factory=dict)

class NeuralEngine(Protocol):
    def configure(self,spec:ExperimentSpec)->None: ...
    def process(self,features:VisionFeatures,*,learn:bool=True)->NeuralFrameResult: ...
    def reset_activity(self)->None: ...

class VisualExperimentUI:
    WINDOW="Visual Play - Node Laboratory"; WIDTH=1500; HEADER_H=58; BUILDER_H=500; OBS_H=300; FOOTER_H=62
    LEFT_W=220; RIGHT_W=360; CENTER_W=WIDTH-LEFT_W-RIGHT_W
    def __init__(self,extractor=None,engine=None,spec=None):
        self.extractor=extractor or VisionFeatureExtractor(cols=64,rows=36,mirror=True,motion_decay=.70,contrast_gain=6.5,motion_gain=7.5,edge_gain=4.0)
        self.engine=engine; self.spec=spec or ExperimentSpec.default(); self.cap=None; self.camera_on=False; self.paused=False; self.learning_on=True
        self.last_frame=None; self.last_features=None; self.last_neural=None; self.selected_kind="layer"; self.selected_id=self.spec.layers[0].id if self.spec.layers else None
        self.connect_mode=False; self.pending_source=None; self.controls={}; self.sensor_rects={}; self.field_rects={}; self.path_segments={}; self.note_target=None; self.note_buffer=""
        self.fps=0.; self._last=time.time(); self.status="Node view ready: field boundaries do not move."
        if engine: engine.configure(self.spec)
    def _blank(self,h,w,v=9): return np.full((h,w,3),v,np.uint8)
    def _text(self,c,t,x,y,s=.38,col=(225,225,230),th=1): cv2.putText(c,str(t),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,s,col,th,cv2.LINE_AA)
    def _button(self,c,key,r,label,active=False,enabled=True,s=.32):
        self.controls[key]=r; x1,y1,x2,y2=r; fill=(48,78,54) if active else (38,34,35); border=(105,210,125) if active else (82,76,78); text=(235,235,240)
        if not enabled: fill,border,text=(25,25,29),(50,50,56),(95,95,102)
        cv2.rectangle(c,(x1,y1),(x2,y2),fill,-1); cv2.rectangle(c,(x1,y1),(x2,y2),border,1); self._text(c,label,x1+7,y1+(y2-y1)//2+4,s,text)
    def _panel(self,c,title,sub,x,y,w):
        cv2.rectangle(c,(x,y),(x+w-1,y+38),(19,17,18),-1); self._text(c,title,x+10,y+22,.46); self._text(c,sub,x+10,y+35,.25,(140,145,154))
    def _wrap(self,c,t,x,y,width=44,lines=4,s=.27):
        for i,line in enumerate(textwrap.wrap(" ".join(str(t).split()),width=width)[:lines]): self._text(c,line,x,y+i*17,s,(160,165,175))
    def _open_camera(self):
        modes=[(0,cv2.CAP_AVFOUNDATION)] if hasattr(cv2,"CAP_AVFOUNDATION") else []; modes.append((0,cv2.CAP_ANY))
        for i,b in modes:
            cap=cv2.VideoCapture(i,b)
            if cap.isOpened(): cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720); return cap
            cap.release()
    def toggle_camera(self):
        if self.camera_on:
            self.cap.release() if self.cap else None; self.cap=None; self.camera_on=False; self.status="Camera disconnected."
        else:
            self.cap=self._open_camera(); self.camera_on=bool(self.cap and self.cap.isOpened()); self.status="Camera connected." if self.camera_on else "Camera unavailable."
    def _changed(self,msg):
        self.spec.validate(); self.last_neural=None; self.status=msg
        if self.engine:self.engine.configure(self.spec)
    def _header(self):
        c=self._blank(self.HEADER_H,self.WIDTH,12); self._button(c,"camera",(12,10,170,47),"CAMERA [k]",self.camera_on); self._button(c,"pause",(180,10,330,47),"RESUME [p]" if self.paused else "PAUSE [p]",self.paused)
        self._button(c,"reset",(340,10,480,47),"RESET INPUT"); self._button(c,"learning",(490,10,640,47),"PLASTICITY [l]",self.learning_on and self.engine is not None,self.engine is not None)
        self._button(c,"save",(650,10,755,47),"SAVE [s]"); self._button(c,"load",(765,10,870,47),"LOAD [o]")
        self._text(c,f"FPS {self.fps:4.1f} | fields {len(self.spec.layers)} | paths {len(self.spec.connections)} | engine {'CONNECTED' if self.engine else 'NOT BUILT'}",910,34,.39,(160,220,220)); return c
    def _tools(self,c):
        self._panel(c,"EXPERIMENT","build neuron fields and pathways",0,0,self.LEFT_W); y=48
        for key,label in [("add","+ ADD FIELD [a]"),("branch","+ NEXT / BRANCH [b]"),("connect","CONNECT PATH [c]"),("delete","DELETE SELECTED"),("notes","EXPERIMENT NOTES [e]")]:
            self._button(c,key,(12,y,self.LEFT_W-12,y+32),label,(key=="connect" and self.connect_mode) or (key=="notes" and self.selected_kind=="experiment")); y+=38
        self._text(c,"ENGINE VIEW",12,y+8,.30,(165,170,180)); y+=17
        for i,m in enumerate(VISUALIZATION_MODES):
            row,col=divmod(i,2); x=12+col*98; yy=y+row*29; self._button(c,f"view:{m}",(x,yy,x+92,yy+23),m.upper(),self.spec.visualization_mode==m,s=.23)
        self._wrap(c,"No whole-field pull. Future plastic movement belongs to individual neurons/synapses.",12,self.BUILDER_H-58,width=29,lines=3,s=.25)
    def _depth_rects(self):
        d=field_depths(self.spec) if self.spec.layers else {}; groups={}
        for l in self.spec.layers: groups.setdefault(d.get(l.id,0),[]).append(l)
        x0=self.LEFT_W+8; y0=84; w=self.CENTER_W-16; h=self.BUILDER_H-94; bands=max(d.values(),default=0)+1; bh=max(92,h//max(1,bands)); out={}
        for depth,items in sorted(groups.items()):
            items=sorted(items,key=lambda z:(z.name,z.id)); gap=10; fw=max(120,(w-20-gap*(len(items)-1))//len(items)); y1=y0+depth*bh+12; y2=min(y0+(depth+1)*bh-8,self.BUILDER_H-8)
            for i,l in enumerate(items): x1=x0+10+i*(fw+gap); out[l.id]=(x1,y1,min(x1+fw,x0+w-8),y2)
        return out,d
    def _sensor_nodes(self,c):
        start=self.LEFT_W+14; gap=7; w=(self.CENTER_W-28-gap*4)//5; nodes={}; self.sensor_rects={}
        for i,sid in enumerate(SENSORY_SOURCES):
            x1=start+i*(w+gap); r=(x1,44,x1+w,76); self.sensor_rects[sid]=r; sel=self.selected_kind=="sensor" and self.selected_id==sid
            cv2.rectangle(c,(r[0],r[1]),(r[2],r[3]),(38,42,44),-1); cv2.rectangle(c,(r[0],r[1]),(r[2],r[3]),(110,205,235) if sel else (75,85,92),2 if sel else 1); self._text(c,SENSORY_LABELS[sid].upper(),r[0]+5,64,.26)
            xs=np.linspace((r[0]+r[2])//2-28,(r[0]+r[2])//2+28,16,dtype=int); nodes[sid]=np.column_stack([xs,np.full_like(xs,82)])
            c[nodes[sid][:,1],nodes[sid][:,0]]=(145,175,185)
        return nodes
    def _nodes(self,layer,r):
        x1,y1,x2,y2=r; px=12; pt=26; pb=8; w=max(1,x2-x1-2*px); h=max(1,y2-y1-pt-pb)
        if self.last_neural is not None:
            p=np.asarray(self.last_neural.node_positions.get(layer.id,[]),np.float32)
            if p.ndim==2 and p.shape==(layer.unit_count,2): return np.column_stack([x1+px+np.clip(p[:,0],0,1)*w,y1+pt+np.clip(p[:,1],0,1)*h]).astype(int)
        xs=np.linspace(x1+px,x2-px,max(1,layer.cols)); ys=np.linspace(y1+pt,y2-pb,max(1,layer.rows)); xx,yy=np.meshgrid(xs,ys); return np.column_stack([xx.ravel(),yy.ravel()]).astype(int)
    def _draw_field(self,c,l,r,depth):
        sel=self.selected_kind=="layer" and self.selected_id==l.id; cv2.rectangle(c,(r[0],r[1]),(r[2],r[3]),(10,12,12),-1); cv2.rectangle(c,(r[0],r[1]),(r[2],r[3]),(110,220,135) if sel else (45,55,54),2 if sel else 1)
        self._text(c,f"DEPTH {depth+1} · {l.name} · {l.rows}×{l.cols} = {l.unit_count:,} neurons",r[0]+6,r[1]+16,.27,(180,190,188)); p=self._nodes(l,r)
        if len(p): c[p[:,1],p[:,0]]=(145,175,150)
        if l.unit_count<=2500:
            for x,y in p: cv2.circle(c,(int(x),int(y)),1,(160,205,170),-1)
        return p
    def _draw_path(self,c,path,src,dst):
        if not len(src) or not len(dst): return
        n=min(42,len(src),len(dst)); si=np.linspace(0,len(src)-1,n,dtype=int); ti=np.linspace(0,len(dst)-1,n,dtype=int) if path.pattern=="one_to_one" else (si*37+11)%len(dst); col=(175,120,220) if path.signal=="inhibitory" else (90,145,100)
        if self.selected_kind=="connection" and self.selected_id==path.id: col=(105,220,245)
        for a,b in zip(si,ti): cv2.line(c,tuple(src[a]),tuple(dst[b]),col,1,cv2.LINE_AA)
        A=tuple(np.mean(src,axis=0).astype(int)); B=tuple(np.mean(dst,axis=0).astype(int)); self.path_segments[path.id]=(A,B)
    def _graph(self,c):
        self._panel(c,"NEURON / SYNAPSE GRAPH","every dot is a neuron; boundaries only mark structural depth",self.LEFT_W,0,self.CENTER_W); self.path_segments={}
        srcs=self._sensor_nodes(c); rects,d=self._depth_rects(); self.field_rects=rects; pts={}
        for l in self.spec.layers:
            if l.id in rects: pts[l.id]=self._draw_field(c,l,rects[l.id],d.get(l.id,0))
        for p in self.spec.connections:self._draw_path(c,p,srcs.get(p.source_id,pts.get(p.source_id,np.empty((0,2),int))),pts.get(p.target_id,np.empty((0,2),int)))
        self._text(c,"NODE POSITIONS ARE STRUCTURAL; ACTIVITY / PLASTIC MOVEMENT NOT SIMULATED",self.LEFT_W+12,self.BUILDER_H-9,.25,(130,135,145))
    def _inspector(self,c):
        x=self.LEFT_W+self.CENTER_W; self._panel(c,"INSPECTOR","selected field/path",x,0,self.RIGHT_W)
        if self.selected_kind=="layer" and self.selected_id:
            l=self.spec.layer_by_id(self.selected_id)
            if l:
                self._text(c,l.name.upper(),x+12,68,.46); self._text(c,f"{l.unit_count:,} individual neuron positions",x+12,91,.29,(155,165,170)); self._text(c,f"ROWS {l.rows}   COLS {l.cols}",x+12,120,.32)
                self._text(c,"MECHANISMS — CONFIG ONLY",x+12,156,.29,(180,185,195)); yy=172
                for i,m in enumerate(MECHANISMS):
                    row,col=divmod(i,2); xx=x+12+col*168; y=yy+row*31; self._button(c,f"mech:{m}",(xx,y,xx+160,y+25),m.replace('_',' ').upper(),l.mechanisms.get(m,False),s=.23)
                self._text(c,"NOTES",x+12,384,.28,(165,170,180)); self._wrap(c,l.notes,x+12,405,width=48,lines=2,s=.24); self._button(c,"edit_note",(x+12,456,x+self.RIGHT_W-12,487),"EDIT FIELD NOTES [n]")
        elif self.selected_kind=="connection" and self.selected_id:
            p=self.spec.connection_by_id(self.selected_id)
            if p:
                self._text(c,"PATHWAY",x+12,68,.46); self._text(c,f"{self._name(p.source_id)} → {self._name(p.target_id)}",x+12,93,.29); self._text(c,f"pattern {p.pattern} | signal {p.signal}",x+12,122,.29)
                self._wrap(c,"Canvas lines are sampled configured routes. Exact learned synapses will come only from a real engine.",x+12,160,width=49,lines=5); self._button(c,"edit_note",(x+12,456,x+self.RIGHT_W-12,487),"EDIT PATH NOTES [n]")
        elif self.selected_kind=="experiment":
            self._text(c,"EXPERIMENT LOGIC",x+12,68,.46); self._wrap(c,self.spec.notes,x+12,105,width=50,lines=12); self._button(c,"edit_note",(x+12,456,x+self.RIGHT_W-12,487),"EDIT NOTES [n]")
    def _name(self,i):
        if i in SENSORY_LABELS:return SENSORY_LABELS[i]
        l=self.spec.layer_by_id(i)
        if l:return l.name
        p=self.spec.connection_by_id(i)
        return f"{self._name(p.source_id)} → {self._name(p.target_id)}" if p else str(i)
    def _map(self,a): a=np.clip(np.nan_to_num(np.asarray(a,np.float32)),0,1); return cv2.cvtColor((a*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    def _tile(self,img,title,sub,w,h,smooth=False):
        t=self._blank(h,w,7); hh=31; t[hh:]=cv2.resize(img,(w,h-hh),interpolation=cv2.INTER_AREA if smooth else cv2.INTER_NEAREST); cv2.rectangle(t,(0,0),(w-1,hh-1),(20,20,24),-1); self._text(t,title,6,14,.29); self._text(t,sub,6,27,.22,(145,150,160)); return t
    def _observation(self):
        c=self._blank(self.OBS_H,self.WIDTH,7); self._panel(c,"LIVE OBSERVATION","measured input and future network-derived output",0,0,self.WIDTH); gap=7; tw=(self.WIDTH-gap*7)//6; sh=116
        cam=cv2.flip(self.last_frame,1) if self.last_frame is not None and self.camera_on else self._blank(80,100,4); items=[("WEBCAM",cam,"physical input",True)]
        if self.last_features is not None:
            f=self.last_features; items += [("BRIGHTNESS",self._map(f.brightness),"direct light",False),("CONTRAST",self._map(f.contrast),"diagnostic",False),("MOTION",self._map(f.motion),"diagnostic",False),("HORIZONTAL",self._map(f.horizontal),"diagnostic",False),("VERTICAL",self._map(f.vertical),"diagnostic",False)]
        while len(items)<6: items.append(("WAITING",self._blank(80,100,4),"no frame",False))
        for i,(a,b,d,e) in enumerate(items[:6]): x=gap+i*(tw+gap); c[42:42+sh,x:x+tw]=self._tile(b,a,d,tw,sh,e)
        ly=165; hw=(self.WIDTH-gap*3)//2; lh=self.OBS_H-ly-7; L=self._blank(lh,hw,8); R=self._blank(lh,hw,8); self._text(L,"SELECTED SIGNAL",10,19,.36); self._text(R,"VISUAL LOGIC OUTPUT",10,19,.36)
        self._text(L,"NO NEURAL SIGNAL YET",185,84,.38,(125,135,150)); self._text(R,"NO FABRICATED RECONSTRUCTION",145,84,.38,(120,170,255)); c[ly:ly+lh,gap:gap+hw]=L; c[ly:ly+lh,gap*2+hw:gap*2+hw*2]=R; return c
    def _footer(self):
        c=self._blank(self.FOOTER_H,self.WIDTH,11)
        if self.note_target:
            self._text(c,"EDIT NOTES — ENTER saves, ESC cancels",14,21,.36,(120,205,255)); self._text(c,self.note_buffer[-170:],14,48,.32); return c
        self._text(c,f"{self.selected_kind}: {self._name(self.selected_id) if self.selected_id else ''} | {self.status[:150]}",14,28,.32); return c
    def _compose(self):
        self.controls={}; b=self._blank(self.BUILDER_H,self.WIDTH,8); cv2.line(b,(self.LEFT_W,0),(self.LEFT_W,self.BUILDER_H),(52,52,60),1); cv2.line(b,(self.LEFT_W+self.CENTER_W,0),(self.LEFT_W+self.CENTER_W,self.BUILDER_H),(52,52,60),1); self._tools(b); self._graph(b); self._inspector(b); return np.vstack([self._header(),b,self._observation(),self._footer()])
    def _select(self,i): self.selected_kind="sensor" if i in SENSORY_SOURCES else "layer"; self.selected_id=i
    def _add_branch(self):
        if self.selected_kind not in {"sensor","layer"} or not self.selected_id:self.status="Select a source first."; return
        src=self.spec.layer_by_id(self.selected_id); n=self.spec.next_layer_index; l=self.spec.add_layer(name=f"Field {n}",rows=src.rows if src else 100,cols=src.cols if src else 100); p=self.spec.add_connection(self.selected_id,l.id)
        if src:
            p.pattern="one_to_one"; p.density=1.
        self.selected_kind,self.selected_id="layer",l.id; self._changed(f"Added {l.name}.")
    def _begin_note(self):
        if self.selected_kind=="experiment": self.note_target=("experiment",None); self.note_buffer=self.spec.notes
        elif self.selected_kind=="layer" and self.selected_id:self.note_target=("layer",self.selected_id); self.note_buffer=self.spec.layer_by_id(self.selected_id).notes
        elif self.selected_kind=="connection" and self.selected_id:self.note_target=("connection",self.selected_id); self.note_buffer=self.spec.connection_by_id(self.selected_id).notes
    def _commit_note(self):
        k,i=self.note_target
        if k=="experiment":self.spec.notes=self.note_buffer
        elif k=="layer":self.spec.layer_by_id(i).notes=self.note_buffer
        else:self.spec.connection_by_id(i).notes=self.note_buffer
        self.note_target=None; self.note_buffer=""; self._changed("Notes updated.")
    def _control(self,key):
        if key=="camera":self.toggle_camera()
        elif key=="pause":self.paused=not self.paused
        elif key=="reset":self.extractor.reset(); self.last_features=self.last_neural=None
        elif key=="learning": self.learning_on=not self.learning_on if self.engine else self.learning_on
        elif key=="save":self.spec.save(EXPERIMENT_FILE); self.status="Saved experiment."
        elif key=="load":self.spec=ExperimentSpec.load(EXPERIMENT_FILE); self.selected_kind="layer"; self.selected_id=self.spec.layers[0].id if self.spec.layers else None; self._changed("Loaded experiment.")
        elif key=="add":n=self.spec.next_layer_index; l=self.spec.add_layer(name=f"Field {n}"); self.selected_kind,self.selected_id="layer",l.id; self._changed(f"Added {l.name}.")
        elif key=="branch":self._add_branch()
        elif key=="connect":self.connect_mode=not self.connect_mode; self.pending_source=None
        elif key=="delete":
            if self.selected_kind=="layer" and self.selected_id:self.spec.remove_layer(self.selected_id)
            elif self.selected_kind=="connection" and self.selected_id:self.spec.remove_connection(self.selected_id)
            self.selected_kind=self.selected_id=None; self._changed("Deleted selection.")
        elif key=="notes":self.selected_kind,self.selected_id="experiment",None
        elif key=="edit_note":self._begin_note()
        elif key.startswith("view:"):self.spec.set_visualization(key.split(':',1)[1])
        elif key.startswith("mech:") and self.selected_kind=="layer":self.spec.toggle_mechanism(self.selected_id,key.split(':',1)[1]); self._changed("Mechanism changed [NOT SIMULATED].")
    def _dist(self,p,s):
        px,py=p;(x1,y1),(x2,y2)=s;dx=x2-x1;dy=y2-y1
        if not dx and not dy:return math.hypot(px-x1,py-y1)
        t=max(0,min(1,((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy))); return math.hypot(px-(x1+t*dx),py-(y1+t*dy))
    def _mouse(self,event,x,y,flags,param):
        if self.note_target or event!=cv2.EVENT_LBUTTONDOWN:return
        by=y-self.HEADER_H; head={"camera","pause","reset","learning","save","load"}
        for k,r in self.controls.items():
            px,py=(x,y) if k in head else (x,by)
            if r[0]<=px<=r[2] and r[1]<=py<=r[3]:self._control(k);return
        for sid,r in self.sensor_rects.items():
            if r[0]<=x<=r[2] and r[1]<=by<=r[3]:return self._node_click(sid)
        for lid,r in self.field_rects.items():
            if r[0]<=x<=r[2] and r[1]<=by<=r[3]:return self._node_click(lid)
        best=(None,10.)
        for cid,s in self.path_segments.items():
            d=self._dist((x,by),s)
            if d<best[1]:best=(cid,d)
        if best[0]:self.selected_kind,self.selected_id="connection",best[0]
    def _node_click(self,i):
        if not self.connect_mode:self._select(i);return
        if self.pending_source is None:self.pending_source=i;self._select(i);return
        if i in SENSORY_SOURCES:self.status="Target must be a field.";return
        try:p=self.spec.add_connection(self.pending_source,i);self.selected_kind,self.selected_id="connection",p.id;self.connect_mode=False;self.pending_source=None;self._changed("Path configured.")
        except ValueError as e:self.status=str(e)
    def _key(self,key):
        key&=255
        if self.note_target:
            if key in (13,10):self._commit_note()
            elif key==27:self.note_target=None;self.note_buffer=""
            elif key in (8,127):self.note_buffer=self.note_buffer[:-1]
            elif 32<=key<=126:self.note_buffer+=chr(key)
            return True
        if key in (ord('q'),27):return False
        mp={ord('k'):"camera",ord('p'):"pause",ord('l'):"learning",ord('s'):"save",ord('o'):"load",ord('a'):"add",ord('b'):"branch",ord('c'):"connect",ord('e'):"notes",ord('n'):"edit_note"}
        if key in mp:self._control(mp[key])
        return True
    def run(self):
        total=self.HEADER_H+self.BUILDER_H+self.OBS_H+self.FOOTER_H; cv2.namedWindow(self.WINDOW,cv2.WINDOW_NORMAL); cv2.resizeWindow(self.WINDOW,self.WIDTH,total); cv2.setMouseCallback(self.WINDOW,self._mouse); self.toggle_camera()
        try:
            run=True
            while run:
                now=time.time();dt=max(now-self._last,1e-6);self._last=now;self.fps=.9*self.fps+.1/dt
                if self.camera_on and not self.paused and self.cap is not None:
                    ok,f=self.cap.read()
                    if ok:self.last_frame=f.copy();self.last_features=self.extractor.extract(f);self.last_neural=self.engine.process(self.last_features,learn=self.learning_on) if self.engine else None
                cv2.imshow(self.WINDOW,self._compose());run=self._key(cv2.waitKey(1))
        finally:
            if self.cap:self.cap.release()
            cv2.destroyAllWindows()

if __name__=="__main__":VisualExperimentUI().run()
