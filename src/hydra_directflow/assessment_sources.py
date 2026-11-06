"""Bounded opaque transport of the explicitly approved assessment sources.

This module does not decode weather/discharge arrays, load models, or authorize
scoring. Development lineage and transport gates remain unchanged. The source
plan, source inventory and separate persistent budget must agree before a
client is constructed. Remote transfers retain the existing precharge/retry
implementation; only assessment-specific lineage and resource guards are new.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import stat
import urllib.parse

from .assessment_contract import origins
from .monthly import cycle_ids_for_origins
from .reference import digest, require
from .local_data import verified_local_bytes
from .staging import (DevelopmentSourceProvider, SourceBudget, GCSClient,
                      object_identity, verified_hit, durable_directory,
                      atomic_json, MAX_ARCHIVE, MAX_COMMIT, RAW_HOTSET)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'hydra-assessment-staging-plan/v1'
INVENTORY_SCHEMA = 'hydra-assessment-source-metadata-inventory/v1'
INDEX_SHA256 = '75469df20f79d94ffb058f439f1ac873f0cf4164c1417eec7298a6641a35baa5'
INDEX_BYTES = 112202255
GIB = 1024**3
LIMITS = dict(download_body_bytes=16*GIB, get_attempts=4000,
              raw_hotset_bytes=2*GIB, archive_response_body_bytes=MAX_ARCHIVE,
              commit_response_body_bytes=MAX_COMMIT, minimum_free_disk_bytes=8*GIB)
MAX_INVENTORY_BYTES = 8*1024**2


def _hex(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def assessment_cycle_ids():
    """All 1,463 possible issued cycles for the full origin year and H18 tail."""
    result=tuple(cycle_ids_for_origins(list(origins())))
    require(len(result)==1463 and result[0]=='20240730_06Z'
            and result[-1]=='20250730_18Z','Assessment source calendar changed')
    return result


def _identity(value,maximum,*,bytes_required,allowed_buckets):
    result=object_identity(value,maximum,bytes_required=bytes_required,
                           allowed_buckets=allowed_buckets)
    if value.get('uri_generation') is not None:
        require(value['uri_generation']==result['durable_uri']+'#'+result['generation'],
                'Conflicting URI/generation source aliases')
    return result


def _bucket(value):
    uri=value.get('durable_uri') or value.get('uri_generation','').rsplit('#',1)[0]
    return urllib.parse.urlsplit(uri).netloc


def _sibling_bundle_uri(commit_uri,cycle):
    suffix='/commits/'+cycle+'.json'
    require(commit_uri.endswith(suffix),'Commit object does not name its source cycle')
    return commit_uri[:-len(suffix)]+'/cycles/'+cycle+'.tar'


def authenticate_assessment_bundle_identity(spec,index_row,*,expected_sealed_row_sha256=None):
    """Authenticate commit metadata only; return (bundle, worker, receipt).

    This is deliberately separate from the development-only lineage resolver.
    It grants no numeric-read or scoring permission. A caller must authenticate
    its release and sealed-index row before using an archive numerically.
    """
    cycle=spec.get('cycle_id')
    require(cycle in assessment_cycle_ids(),'Cycle lies outside the exact assessment source calendar')
    require(index_row.get('cycle')==cycle and index_row.get('location_kind') in ('commit','bundle')
            and index_row.get('original_quality_flags_rewritten') is False
            and type(index_row.get('data_complete')) is bool,'Assessment sealed lineage differs')
    if expected_sealed_row_sha256 is not None:
        require(_hex(expected_sealed_row_sha256) and digest(index_row)==expected_sealed_row_sha256,
                'Assessment sealed row digest differs')
    locations=index_row.get('source_locations')
    require(isinstance(locations,dict) and locations,'Sealed source locations required')
    buckets={_bucket(v) for v in locations.values() if isinstance(v,dict)}
    require(buckets and '' not in buckets,'Sealed source bucket missing')
    indexed=locations.get('bundle');bundle=worker=commit_receipt=None
    if indexed is not None:
        _identity(indexed,MAX_ARCHIVE,bytes_required=True,allowed_buckets=buckets)
        require(indexed.get('verified') is True,'Sealed bundle must retain its verified identity')
        bundle=indexed
    use_commit=index_row['location_kind']=='commit' or spec.get('commit_path') is not None
    if use_commit:
        require(spec.get('commit_path') is not None and locations.get('commit'),
                'Commit-kind lineage requires the original local commit')
        commit_identity=_identity(locations['commit'],MAX_COMMIT,bytes_required=False,allowed_buckets=buckets)
        raw=verified_local_bytes(spec['commit_path'],commit_identity['sha256'],
                                 expected_bytes=commit_identity['bytes'],max_bytes=MAX_COMMIT)
        commit=json.loads(raw)
        require(commit.get('schema')=='hrrr-verified-cloud-cycle/v1' and commit.get('cycle')==cycle,
                'Assessment commit cycle/schema differs')
        bundle=commit.get('bundle')
        normalized=_identity(bundle,MAX_ARCHIVE,bytes_required=True,allowed_buckets=buckets)
        require(bundle.get('verified') is True and normalized['durable_uri']==
                _sibling_bundle_uri(commit_identity['durable_uri'],cycle),
                'Committed bundle must be the same source cycle and namespace')
        worker=commit.get('worker_manifest')
        require(isinstance(worker,dict) and worker.get('cycle')==cycle
                and worker.get('bundle_bytes')==bundle['bytes']
                and worker.get('bundle_sha256')==bundle['sha256']
                and _hex(worker.get('manifest_sha256'))
                and type(worker.get('record_count')) is int and 0<=worker['record_count']<=145,
                'Assessment worker manifest binding differs')
        if indexed is not None:
            require(normalized==_identity(indexed,MAX_ARCHIVE,bytes_required=True,allowed_buckets=buckets),
                    'Index and commit bundle identities conflict')
        commit_receipt=dict(sha256=commit_identity['sha256'],bytes=len(raw))
    require(bundle is not None,'Bundle-kind row lacks its sealed bundle identity')
    require(_identity(bundle,MAX_ARCHIVE,bytes_required=True,allowed_buckets=buckets)['durable_uri']
            .endswith('/cycles/'+cycle+'.tar'),'Bundle object does not name its source cycle')
    receipt=dict(schema='hydra-assessment-lineage-adapter/v1',cycle_id=cycle,
                 index_location_kind=index_row['location_kind'],
                 binding_route='SEALED_INDEX_COMMIT_BUNDLE' if use_commit else 'SEALED_INDEX_BUNDLE',
                 commit=commit_receipt,archive_sha256=bundle['sha256'],archive_bytes=bundle['bytes'],
                 source_summaries_rewritten=False,numeric_values_decoded=False)
    return bundle,worker,receipt


class AssessmentSourceProvider(DevelopmentSourceProvider):
    """Assessment-scoped adapter over the existing opaque transfer machinery."""
    def __init__(self,*,plan,directory,budget,expected_plan_sha256,client=None,max_attempts_per_object=3):
        require(isinstance(plan,dict) and plan.get('schema')==SCHEMA
                and plan.get('status')=='APPROVED_BOUNDED_STAGING','Approved assessment staging plan required')
        require(_hex(expected_plan_sha256) and plan.get('plan_sha256')==expected_plan_sha256
                and digest({k:v for k,v in plan.items() if k!='plan_sha256'})==expected_plan_sha256,
                'Exact approved assessment plan digest required')
        require(plan.get('limits')==LIMITS and all(type(v) is int for v in plan['limits'].values())
                and type(plan.get('approved_staging_ceiling_usd')) in (int,float)
                and plan['approved_staging_ceiling_usd']==5
                and type(plan.get('combined_ceiling_usd')) is int and plan['combined_ceiling_usd']==150,
                'Assessment staging caps must remain 16GiB/4000GETs/5USD within150USD')
        sources=plan.get('sources',{});approval=sources.get('budget_approval',{}).get('sha256')
        inventory_ref=sources.get('source_inventory',{});index=sources.get('qualified_index',{})
        require(_hex(approval) and _hex(inventory_ref.get('sha256'))
                and index.get('sha256')==INDEX_SHA256 and index.get('bytes')==INDEX_BYTES,
                'Pinned approval, source inventory and qualified index required')
        require(plan.get('approval_sha256',approval)==approval
                and plan.get('source_inventory_sha256',inventory_ref['sha256'])==inventory_ref['sha256'],
                'Conflicting assessment source authority aliases')
        self.directory=Path(directory).absolute()
        expected_directory=ROOT/'.local/assessment-v1/source-staging'
        require(self.directory==self.directory.resolve()==expected_directory.resolve()
                and expected_directory==expected_directory.resolve(),
                'Use only the fixed new assessment source-staging directory')
        require(isinstance(budget,SourceBudget) and budget.path==budget.path.resolve()==self.directory/'budget',
                'Assessment needs its own persistent source budget namespace')
        state=budget.snapshot()
        require(state['body_cap']==LIMITS['download_body_bytes'] and state['get_cap']==LIMITS['get_attempts']
                and state['cost_estimate_cap_usd']==5
                and budget.connection.execute('SELECT approval FROM budget WHERE id=1').fetchone()[0]==approval,
                'Persistent assessment budget identity or caps differ')
        require(type(max_attempts_per_object) is int and 1<=max_attempts_per_object<=3,
                'At most three charged source attempts are allowed')
        inventory_path=Path(inventory_ref.get('path','')).absolute()
        require(inventory_path==ROOT/'.local/assessment-preparation-v1/SOURCE_INVENTORY.json'
                and inventory_path==inventory_path.resolve()
                and type(inventory_ref.get('bytes')) is int and 0<inventory_ref['bytes']<=MAX_INVENTORY_BYTES,
                'Exact private source-inventory location and byte bound required')
        raw=verified_local_bytes(inventory_path,inventory_ref['sha256'],expected_bytes=inventory_ref['bytes'],
                                 max_bytes=MAX_INVENTORY_BYTES)
        inventory=json.loads(raw)
        require(inventory.get('schema')==INVENTORY_SCHEMA
                and inventory.get('inventory_content_sha256')==digest({k:v for k,v in inventory.items() if k!='inventory_content_sha256'}),
                'Source-inventory content binding differs')
        source_rows=inventory.get('hrrr',{}).get('cycles',[])
        source_by_cycle={r['cycle_id']:r for r in source_rows}
        planned=plan.get('cycles',[]);self.cycles={r['cycle_id']:r for r in planned}
        expected=assessment_cycle_ids()
        require(len(source_rows)==len(source_by_cycle)==len(planned)==len(self.cycles)==1463
                and tuple(r['cycle_id'] for r in planned)==expected and set(source_by_cycle)==set(expected),
                'Exactly the ordered1463 assessment candidate cycles are required')
        require(inventory['hrrr'].get('qualified_index',{}).get('sha256')==INDEX_SHA256
                and inventory['hrrr']['qualified_index'].get('bytes')==INDEX_BYTES
                and inventory['hrrr']['qualified_index'].get('rows')==61824,
                'Source inventory uses a different qualified index')
        self.allowed_buckets={_bucket(r['commit_identity']) for r in source_rows if r.get('commit_identity')}
        self.allowed_buckets|={_bucket(r['known_bundle_identity']) for r in source_rows if r.get('known_bundle_identity')}
        require(self.allowed_buckets and '' not in self.allowed_buckets,'Inventory source buckets missing')
        if 'allowed_buckets' in plan:require(set(plan['allowed_buckets'])==self.allowed_buckets,'Source bucket whitelist differs')
        archives={}
        for a in inventory['hrrr'].get('archive_candidates',[]):archives.setdefault(a['cycle_id'],set()).add(a['path'])
        self._approved_objects=set()
        for row in planned:
            cycle=row['cycle_id'];source=source_by_cycle[cycle]
            require(_hex(row.get('sealed_row_sha256')) and row['sealed_row_sha256']==source['sealed_row_digest'],
                    'Planned sealed-row identity differs from the inventory')
            for key,source_key,maximum,required in [('commit_identity','commit_identity',MAX_COMMIT,False),
                                                  ('bundle_identity','known_bundle_identity',MAX_ARCHIVE,True)]:
                value,original=row.get(key),source.get(source_key)
                require((value is None)==(original is None),'Planned source identity presence differs')
                if value is not None:
                    normalized=_identity(value,maximum,bytes_required=required,allowed_buckets=self.allowed_buckets)
                    require(normalized==_identity(original,maximum,bytes_required=required,allowed_buckets=self.allowed_buckets),
                            'Planned source object differs from the sealed inventory')
                    self._approved_objects.add(digest(normalized))
            if row.get('commit_identity'):
                uri=_sibling_bundle_uri(_identity(row['commit_identity'],MAX_COMMIT,bytes_required=False,
                    allowed_buckets=self.allowed_buckets)['durable_uri'],cycle)
                if row.get('bundle_identity'):
                    require(_identity(row['bundle_identity'],MAX_ARCHIVE,bytes_required=True,
                            allowed_buckets=self.allowed_buckets)['durable_uri']==uri,
                            'Bundle and commit namespaces differ')
            known_commit={a['path'] for a in source.get('commit_candidates',[])}
            known_commit.update(source.get('cloud_only_commit_metadata_paths',[]))
            if (source.get('bundle_proof') or {}).get('route')=='RESIDENT_COMMIT_AUTHENTICATED_AGAINST_SEALED_INDEX':
                known_commit.add(source['bundle_proof']['path'])
            for key,allowed in [('local_commit_candidates',known_commit),('local_archive_candidates',archives.get(cycle,set()))]:
                candidates=row.get(key,[]);require(isinstance(candidates,list),'Explicit local candidate list required')
                for item in candidates:
                    path=item.get('path') if isinstance(item,dict) else item
                    require(isinstance(path,str) and path in allowed and Path(path).is_absolute(),
                            'Uninventoried local source candidate rejected')
        self.plan=json.loads(json.dumps(plan,sort_keys=True,allow_nan=False))
        self.cycles={r['cycle_id']:r for r in self.plan['cycles']}
        self.budget=budget;self.max_attempts_per_object=max_attempts_per_object
        self.owner=digest(dict(directory=str(self.directory),approval=approval,index=INDEX_SHA256,
                               assessment_plan=expected_plan_sha256,source_inventory=inventory_ref['sha256']))
        self.lock=None;self.hits=self.downloads=self.evictions=0
        self.check_resources()
        durable_directory(self.directory)
        self.check_resources()
        self.client=client if client is not None else GCSClient()

    def check_resources(self,required_write_bytes=0):
        require(type(required_write_bytes) is int and 0<=required_write_bytes<=MAX_ARCHIVE,
                'Bounded pending source write required')
        raw_bytes=0
        for folder in (self.directory/'raw',self.directory/'usgs'):
            if not folder.exists():continue
            require(folder==folder.resolve() and folder.is_dir(),'Raw staging directory alias rejected')
            for p in folder.iterdir():
                st=p.lstat();require(stat.S_ISREG(st.st_mode) and not getattr(st,'st_flags',0)&0x40000000,
                                     'Owned raw hotset must contain resident regular files')
                raw_bytes+=st.st_size
        location=self.directory
        while not location.exists():location=location.parent
        require(raw_bytes+required_write_bytes<=RAW_HOTSET,'Assessment raw hotset exceeds2GiB')
        require(shutil.disk_usage(location).free>=LIMITS['minimum_free_disk_bytes']+2*required_write_bytes,
                'Assessment8GiB free-disk floor reached')
        return dict(raw_hotset_bytes=raw_bytes)

    def __enter__(self):
        self.check_resources();super().__enter__()
        try:
            binding=dict(schema='hydra-assessment-source-provider-binding/v1',owner=self.owner,
                         plan_sha256=self.plan['plan_sha256'],inventory_sha256=self.plan['sources']['source_inventory']['sha256'])
            path=self.directory/'PROVIDER_BINDING.json'
            if path.exists() or path.is_symlink():
                st=path.lstat();require(stat.S_ISREG(st.st_mode) and not getattr(st,'st_flags',0)&0x40000000
                    and st.st_size<8192,'Staging binding must remain resident regular metadata')
                require(json.loads(path.read_text())==binding,'Existing assessment staging binding differs')
            else:atomic_json(path,binding)
            self.check_resources();return self
        except BaseException:
            super().__exit__();raise

    def _transfer(self,identity,destination,maximum,kind):
        require(self.lock is not None,'Assessment source writer lock required')
        require(kind in ('commit','archive') and maximum==(MAX_COMMIT if kind=='commit' else MAX_ARCHIVE)
                and digest(identity) in self._approved_objects,'Transfer outside approved source identities')
        suffix=Path(urllib.parse.urlsplit(identity['durable_uri']).path).name
        expected=self.directory/('commits' if kind=='commit' else 'raw')/suffix
        require(Path(destination).absolute()==expected and expected==expected.resolve(),
                'Transfer destination leaves the owned assessment namespace')
        limit=identity['bytes'] if identity['bytes'] is not None else maximum
        self.check_resources(limit)
        try:return super()._transfer(identity,destination,maximum,kind)
        finally:self.check_resources()

    def __call__(self,cycle_id,sealed_row):
        require(self.lock is not None and cycle_id in self.cycles,'Planned assessment cycle and writer lock required')
        self.check_resources();row=self.cycles[cycle_id]
        require(row['sealed_row_sha256']==digest(sealed_row),'Planned assessment cycle differs from the sealed index')
        spec={'cycle_id':cycle_id}
        try:
            if sealed_row['location_kind']=='commit':
                identity=_identity(sealed_row['source_locations']['commit'],MAX_COMMIT,bytes_required=False,
                                   allowed_buckets=self.allowed_buckets)
                target=self.directory/'commits'/(cycle_id+'.json')
                candidates=[str(target)]+[v['path'] if isinstance(v,dict) else v for v in row.get('local_commit_candidates',[])]
                chosen=next((p for p in candidates if verified_hit(p,identity,MAX_COMMIT)),None)
                if chosen is None:self._transfer(identity,target,MAX_COMMIT,'commit');chosen=str(target)
                else:self.hits+=1
                spec['commit_path']=chosen
            bundle,_,_=authenticate_assessment_bundle_identity(spec,sealed_row,
                                      expected_sealed_row_sha256=row['sealed_row_sha256'])
            identity=_identity(bundle,MAX_ARCHIVE,bytes_required=True,allowed_buckets=self.allowed_buckets)
            if row.get('bundle_identity'):
                require(identity==_identity(row['bundle_identity'],MAX_ARCHIVE,bytes_required=True,
                    allowed_buckets=self.allowed_buckets),'Resolved bundle differs from the approved plan')
            self._approved_objects.add(digest(identity))
            target=self.directory/'raw'/(cycle_id+'.tar')
            candidates=[str(target)]+[v['path'] if isinstance(v,dict) else v for v in row.get('local_archive_candidates',[])]
            chosen=next((p for p in candidates if verified_hit(p,identity,MAX_ARCHIVE)),None)
            if chosen is None:self._transfer(identity,target,MAX_ARCHIVE,'archive');chosen=str(target)
            else:self.hits+=1
            spec['archive_path']=chosen
            return {'status':'READY','spec':spec,'ownership':{'owned_archive':Path(chosen).absolute()==target,
                    'archive_path':chosen,'owner_run_id':self.owner,'archive_sha256':identity['sha256']}}
        finally:self.check_resources()

    def on_cache_committed(self,receipt,provider_result):
        self.check_resources()
        try:return super().on_cache_committed(receipt,provider_result)
        finally:self.check_resources()
