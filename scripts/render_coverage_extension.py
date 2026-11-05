#!/usr/bin/env python3
"""Render already-verified Phase A summaries and fixed-event arrays locally."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from report_coverage_extension import render_aggregate_figures,render_event_hydrographs
from hydra_directflow.reference import require


def read(run,desc):
    require(set(desc)=={'path','bytes','sha256'} and not Path(desc['path']).is_absolute()
            and '..' not in Path(desc['path']).parts,'Bounded private figure descriptor required')
    path=run/desc['path'];info=path.lstat()
    require(path==path.resolve() and path.is_relative_to(run) and stat.S_ISREG(info.st_mode)
            and not getattr(info,'st_flags',0)&0x40000000 and info.st_size==desc['bytes']
            and 0<info.st_size<=256*1024**2,'Resident bounded figure source required')
    raw=path.read_bytes();require(len(raw)==desc['bytes'] and hashlib.sha256(raw).hexdigest()==desc['sha256'],
                                 'Figure source checksum changed')
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();run=args.run.absolute()
    require(run==run.resolve() and args.manifest==args.manifest.resolve() and args.manifest.is_relative_to(run), 'Private phase plot paths required')
    before=args.manifest.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before,'st_flags',0)&0x40000000
            and 0<before.st_size<=1024**2,'Resident bounded plot manifest required before reading')
    raw=args.manifest.read_bytes();after=args.manifest.lstat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(after) and len(raw)==before.st_size
            and hashlib.sha256(raw).hexdigest()==args.manifest_sha256,'Plot manifest changed')
    manifest=json.loads(raw);summary=json.loads(read(run,manifest['summary']))
    require(summary['phase_plan_sha256']==manifest['phase_plan_sha256']
            and summary['prediction_barrier_sha256']==manifest['prediction_barrier_sha256'],'Plot source phase/barrier differs')
    data={}
    for gauge,desc in manifest['gauges'].items():
        body=read(run,desc)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            require(len(z.infolist())<=16 and sum(x.file_size for x in z.infolist())<=256*1024**2,'Plot archive bound exceeded')
        with np.load(io.BytesIO(body),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        arrays.update(phase_plan_sha256=manifest['phase_plan_sha256'],prediction_barrier_sha256=manifest['prediction_barrier_sha256'])
        data[gauge]=arrays
    output=args.output.absolute();require(output==output.resolve() and output.is_relative_to(run) and not output.exists(),'Fresh private figure directory required')
    output.mkdir(parents=True)
    figures=render_aggregate_figures(summary,output,png_previews=True)+render_event_hydrographs(summary,data,output,png_previews=True)
    result=dict(schema='hydra-phase-a-figure-manifest/v1',status='RENDERED_REQUIRES_VISUAL_REVIEW',phase_plan_sha256=manifest['phase_plan_sha256'],
                prediction_barrier_sha256=manifest['prediction_barrier_sha256'],figures=figures,source_manifest_sha256=args.manifest_sha256,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_calls=0,protected_assessment_access=False)
    (output/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],figures=len(figures))))


if __name__=='__main__':main()
