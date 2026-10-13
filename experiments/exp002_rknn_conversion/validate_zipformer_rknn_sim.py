from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from rknn.api import RKNN
root=Path('experiments/exp002_rknn_conversion')
for name in ('encoder','decoder','joiner'):
 src=root/'inputs/zipformer/32'/f'{name}.onnx'
 model=onnx.load(str(src),load_external_data=False)
 metadata=';'.join(f'{x.key}={x.value}' for x in model.metadata_props)
 names=[]; sizes=[]
 for v in model.graph.input:
  names.append(v.name)
  sizes.append([d.dim_value if d.dim_value else 1 for d in v.type.tensor_type.shape.dim])
 sess=ort.InferenceSession(str(src),providers=['CPUExecutionProvider'])
 rng=np.random.default_rng(20261011)
 inputs=[]
 for v,sh in zip(sess.get_inputs(),sizes):
  inputs.append(np.zeros(sh,dtype=np.int64) if v.type=='tensor(int64)' else (rng.standard_normal(sh)*0.02).astype(np.float32))
 expected=sess.run(None,{v.name:a for v,a in zip(sess.get_inputs(),inputs)})
 r=RKNN(verbose=False)
 try:
  assert r.config(target_platform='rk3568',optimization_level=0,custom_string=metadata)==0
  assert r.load_onnx(model=str(src),inputs=names,input_size_list=sizes)==0
  assert r.build(do_quantization=False)==0
  assert r.init_runtime()==0
  actual=r.inference(inputs=inputs,data_format=["nchw" if a.ndim==4 else "nhwc" for a in inputs])
  print(name,'n_outputs',len(actual),flush=True)
  for i,(a,b) in enumerate(zip(expected,actual)):
   a=np.asarray(a); b=np.asarray(b)
   if a.shape==b.shape and a.size:
    av=a.astype(np.float32);bv=b.astype(np.float32)
    max_abs=float(np.max(np.abs(av-bv)))
    cos=float(np.sum(av*bv)/(np.linalg.norm(av)*np.linalg.norm(bv)+1e-12))
   else:max_abs='shape mismatch';cos='shape mismatch'
   print(i,a.shape,b.shape,a.dtype,b.dtype,'max_abs',max_abs,'cosine',cos,flush=True)
 finally:r.release()
