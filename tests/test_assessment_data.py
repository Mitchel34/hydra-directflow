"""Synthetic assessment reader qualification. No real source or model reads."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import hydra_directflow.assessment_data as ad
from hydra_directflow.local_data import GAUGES, GRID_SHA256, CFS_TO_CMS, LocalCycle, iso, sha256
from hydra_directflow.reference import (ASSESSMENT_START as START, ASSESSMENT_END as END,
                                      ASSESSMENT_LAST_TARGET as LAST, HOUR, Observation,
                                      AvailabilityPolicy, history_window, digest)
from hydra_directflow.quality import PRIMARY_APPROVED, quality_admitter


def authority(cycles=(), *, enabled=True, verify=False, rows=None):
    bindings=tuple((cid,digest(rows[cid]) if rows else 'a'*64) for cid in cycles)
    return ad.AssessmentReadAuthority('1'*64,'2'*64,'3'*64,'4'*64,bindings,
            inputs_enabled=enabled,verification_enabled=verify,
            prediction_barrier_sha256='5'*64 if verify else None,allowed_history_hours=(168,24))


def cycle(reference, *, weather=True, soil=True, value=1.0):
    cid=ad.datetime.fromtimestamp(reference,ad.timezone.utc).strftime('%Y%m%d_%HZ')
    records=[];groups=[]
    for group in list(ad.UNITS)+(['soil_moisture'] if soil else []):
        if group!='soil_moisture' and not weather:continue
        leads=[0] if group=='soil_moisture' else range(1,37)
        for lead in leads:
            records.append(dict(record_index=len(records),group=group,lead=lead,
                source={'key':cid+'/'+group},availability_status='AVAILABLE_VERIFIED',
                source_reference_UTC=iso(reference),source_valid_UTC=iso(reference+lead*HOUR)))
        groups.append(dict(group=group,leads=list(leads),record_count=len(list(leads)),status='AVAILABLE_VERIFIED'))
    n=len(records);values=np.full((n,957),value);present=np.ones((n,957),bool)
    means=np.full((n,3),value);mask=np.ones((n,3),bool);coverage=np.ones((n,3))
    if soil:
        for i,r in enumerate(records):
            if r['group']=='soil_moisture':means[i]=.25
    return cid,LocalCycle(reference,values,present,tuple(records),0,{},dict(cycle=cid,groups=groups)),(means,mask,coverage)


def observations(origin=START, length=168, *, extra=0):
    return [Observation(g,t,None,float(10+i),('A',),'wire-'+g)
            for i,g in enumerate(GAUGES)
            for t in range(origin-(length+1)*HOUR,origin-2*HOUR+1+extra*HOUR,HOUR)]


def assembly(origin=START, *, cycles=None, obs=None, origins=None):
    if cycles is None:cycles=[cycle(origin-6*HOUR)]
    cs={cid:c for cid,c,_ in cycles};rs={cid:r for cid,_,r in cycles}
    return ad.assemble_assessment_inputs(cs,rs,observations(origin) if obs is None else obs,
             [origin] if origins is None else origins,authority(cs),source_identity='e'*64)


class AssessmentAssemblyTests(unittest.TestCase):
    def test_complete_inputs_have_expected_geometry_and_no_targets(self):
        r=assembly();a=r['arrays']
        self.assertEqual(a['history'].shape,(3,168,2));self.assertEqual(a['weather'].shape,(3,18,4))
        self.assertTrue(a['input_admitted'].all());self.assertNotIn('target',a);self.assertNotIn('target_mask',a)
        self.assertEqual(a['history_time'][0,0],START-169*HOUR);self.assertEqual(a['history_time'][0,-1],START-2*HOUR)
        self.assertEqual(a['forecast_time'][0,-1],START+18*HOUR);self.assertFalse(a['history'].flags.writeable)
        self.assertEqual(r['manifest']['optimizer_updates'],0);self.assertFalse(r['manifest']['target_values_read'])

    def test_history_matches_scalar_reference_and_retains_qualifiers(self):
        obs=observations();obs[0]=replace(obs[0],quality_flags=('R','A','R'))
        r=assembly(obs=obs)['arrays'];expected=history_window(obs,site=GAUGES[0],origin_anchor=START,cutoff=START,
            lookback_hours=168,history_lag_hours=2,policy=AvailabilityPolicy('x','retrospective_simulated',7200),
            admit_quality=quality_admitter(PRIMARY_APPROVED))
        np.testing.assert_array_equal(r['history'][0,:,0],expected['values'])
        np.testing.assert_array_equal(r['history'][0,:,1],expected['age_hours'])
        self.assertEqual(json.loads(r['history_quality_flags'][0,0]),['R','A','R'])

    def test_gap_is_not_backsearched_filled_or_substituted(self):
        obs=observations();bad=obs.pop(10);obs.append(replace(bad,valid_at=START-170*HOUR,value=999.))
        a=assembly(obs=obs)['arrays'];self.assertFalse(a['input_admitted'][0]);self.assertTrue(a['input_admitted'][1:].all())
        self.assertTrue(np.isnan(a['history'][0,10,0]));self.assertFalse(a['history_mask'][0,10].any())
        self.assertEqual(a['history_exclusion_reason'][0,10],'MISSING_EXACT_HOUR')

    def test_provisional_estimated_and_unknown_history_flags_reject(self):
        for flags in [('P',),('A','e'),('A','unknown'),(),('R',)]:
            with self.subTest(flags=flags):
                obs=observations();obs[0]=replace(obs[0],quality_flags=flags)
                a=assembly(obs=obs)['arrays'];self.assertFalse(a['input_admitted'][0])
                self.assertEqual(json.loads(a['history_quality_flags'][0,0]),list(flags))

    def test_future_observation_values_are_never_read_or_used(self):
        class PoisonFuture:
            def __init__(self):self.site=GAUGES[0];self.valid_at=START+HOUR
            def __getattribute__(self,name):
                if name in ('value','quality_flags','available_at'):raise AssertionError('Future observation inspected')
                return super().__getattribute__(name)
        base=assembly();obs=observations()+[PoisonFuture()]
        actual=assembly(obs=obs)
        self.assertEqual(base['manifest']['arrays_sha256'],actual['manifest']['arrays_sha256'])

    def test_later_origin_history_does_not_change_earlier_admission(self):
        obs=observations(extra=6);cs=[cycle(START-6*HOUR),cycle(START)]
        before=assembly(cycles=cs,obs=obs,origins=[START,START+6*HOUR])['arrays']
        altered=[replace(o,value=999.,quality_flags=('P',)) if o.site==GAUGES[0] and o.valid_at==START+HOUR else o for o in obs]
        after=assembly(cycles=cs,obs=altered,origins=[START,START+6*HOUR])['arrays']
        np.testing.assert_array_equal(before['history'][:3],after['history'][:3])
        np.testing.assert_array_equal(before['input_admitted'][:3],after['input_admitted'][:3])
        self.assertTrue(before['input_admitted'][3]);self.assertFalse(after['input_admitted'][3])

    def test_latest_complete_weather_falls_back_as_one_cycle(self):
        older=cycle(START-12*HOUR,value=2.);newer=cycle(START-6*HOUR,value=3.)
        row=next(i for i,r in enumerate(newer[1].records) if r['group']=='precipitation' and r['lead']==7)
        newer[2][1][row,0]=False;newer[2][2][row,0]=.5;newer[2][0][row,0]=np.nan
        a=assembly(cycles=[older,newer])['arrays'];self.assertEqual(a['source_cycle'][0],older[0]);self.assertEqual(a['source_cycle'][1],newer[0])
        self.assertTrue((a['weather'][0]==2).all());self.assertTrue((a['weather_source_leads'][0]==np.arange(13,31)).all())
        self.assertEqual(a['soil_source_cycle'][0],newer[0])

    def test_weather_cycles_are_not_stitched_and_precipitation_not_zero(self):
        old=cycle(START-12*HOUR);new=cycle(START-6*HOUR)
        for item,group in [(old,'temperature'),(new,'precipitation')]:
            for i,r in enumerate(item[1].records):
                if r['group']==group:item[2][1][i]=False;item[2][2][i]=0;item[2][0][i]=np.nan
        a=assembly(cycles=[old,new])['arrays'];self.assertFalse(a['input_admitted'].any())
        self.assertFalse(a['weather_mask'].any());self.assertTrue(np.isnan(a['weather']).all())

    def test_weather_availability_and_separate_soil_age(self):
        old=cycle(START-6*HOUR);future=cycle(START,value=5.)
        a=assembly(cycles=[old,future])['arrays'];self.assertTrue((a['source_cycle']==old[0]).all())
        a=assembly(origin=START+2*HOUR,cycles=[old,cycle(START,soil=False)],obs=observations(START+2*HOUR))['arrays']
        self.assertTrue(a['state_mask'].all());self.assertTrue((a['state'][:,1]==8).all())
        b=assembly(origin=START+3*HOUR,cycles=[old,cycle(START,soil=False)],obs=observations(START+3*HOUR))['arrays']
        self.assertFalse(b['state_mask'].any());self.assertFalse(b['input_admitted'].any())

    def test_weather_eighteen_hour_age_boundary_and_independent_soil(self):
        weather=cycle(START-6*HOUR);soil=cycle(START+6*HOUR,weather=False)
        a=assembly(START+12*HOUR,cycles=[weather,soil])['arrays']
        self.assertTrue(a['input_admitted'].all());self.assertTrue((a['weather_source_leads'][0]==np.arange(19,37)).all())
        self.assertTrue((a['source_cycle']==weather[0]).all());self.assertTrue((a['soil_source_cycle']==soil[0]).all())
        b=assembly(START+13*HOUR,cycles=[weather,soil])['arrays']
        self.assertFalse(b['weather_mask'].any());self.assertTrue(b['state_mask'].all())
        self.assertTrue((b['state'][:,1]==7).all());self.assertFalse(b['input_admitted'].any())

    def test_last_origin_retains_assessment_h18_tail(self):
        c=END-HOUR;data=cycle((c-2*HOUR)//(6*HOUR)*(6*HOUR))
        a=assembly(c,cycles=[data])['arrays'];self.assertEqual(a['forecast_time'][0,-1],LAST)

    def test_input_gate_and_invalid_origins_precede_values(self):
        class PoisonMap(dict):
            def items(self):raise AssertionError('Read cycle values')
        with self.assertRaises(ValueError):ad.assemble_assessment_inputs(PoisonMap(),{},[],[START],authority(enabled=False),source_identity='a'*64)
        for origin in [START-HOUR,END,START+1]:
            with self.assertRaises(ValueError):ad.assemble_assessment_inputs(PoisonMap(),{},[],[origin],authority(),source_identity='a'*64)

    def test_duplicate_observations_and_unbound_cycles_rejected(self):
        obs=observations();obs.append(obs[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):assembly(obs=obs)
        cid,cyc,reduced=cycle(START-6*HOUR)
        with self.assertRaisesRegex(ValueError,'inventory-bound'):ad.assemble_assessment_inputs({cid:cyc},{cid:reduced},[],[START],authority(),source_identity='e'*64)

    def test_partial_positive_weight_coverage_cannot_claim_complete(self):
        c=cycle(START-6*HOUR);c[2][2][0,0]=.5
        with self.assertRaisesRegex(ValueError,'renormalize'):assembly(cycles=[c])

    def test_target_gate_independent_per_lead_quality_and_no_fill(self):
        obs=[Observation(g,START+h*HOUR,None,20.,('A','R') if h!=3 else ('P',),'wire') for g in GAUGES for h in range(1,19)]
        with self.assertRaises(ValueError):ad.assemble_assessment_targets(obs,[START],authority())
        r=ad.assemble_assessment_targets(obs,[START],authority(verify=True));a=r['arrays']
        self.assertFalse(a['target_mask'][:,2].any());self.assertTrue(a['target_mask'][:,17].all())
        self.assertTrue(np.isnan(a['target'][:,2]).all());self.assertEqual(json.loads(a['target_quality_flags'][0,2]),['P'])

    def test_target_assembly_ignores_values_outside_requested_target_union(self):
        class Outside:
            site=GAUGES[0];valid_at=START+19*HOUR
            @property
            def value(self):raise AssertionError('Outside verification scope')
        r=ad.assemble_assessment_targets([Outside()],[START],authority(verify=True))
        self.assertFalse(r['arrays']['target_mask'].any())
        self.assertTrue(np.isnan(r['arrays']['target']).all())


class AssessmentUsGsTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()
    def row(self,epoch=START,flags=('A','R'),**change):
        r=dict(site=GAUGES[0],sample_epoch=epoch,sample_time_utc=iso(epoch),source_units='ft3/s',discharge_cfs=100.,
            source_qualifiers_json=json.dumps(flags),numeric_valid=True,is_exact_utc_hour=epoch%HOUR==0,is_core=True,
            source_wire_sha256='6'*64,historical_available_at_utc=None);r.update(change);return r
    def files(self,month,rows):
        stream=pa.BufferOutputStream();pq.write_table(pa.Table.from_pylist(rows),stream);raw=stream.getvalue().to_pybytes()
        part=self.root/(month+'.parquet');part.write_bytes(raw)
        manifest=dict(native_partitions=[dict(path=str(part),bytes=len(raw),sha256=sha256(raw))]);body=json.dumps(manifest).encode();mp=self.root/'manifest.json';mp.write_bytes(body)
        native=ad.verified_local_bytes
        def trusted(path,expected,**kw):
            if Path(path)==mp:
                self.assertEqual(expected,ad.USGS_MANIFEST_SHA256);return body
            return native(path,expected,**kw)
        return part,mp,trusted

    def test_source_windows_and_gate_fail_before_io(self):
        with patch.object(ad,'_resident_charge',side_effect=AssertionError('I/O before gate')):
            with self.assertRaises(ValueError):ad.load_assessment_usgs({},'absent',[(START,START)],authority(enabled=False))
            with self.assertRaises(ValueError):ad.load_assessment_usgs({},'absent',[(START,LAST)],authority(),mode='verification')
            with self.assertRaises(ValueError):ad.load_assessment_usgs({},'absent',[(START,LAST)],authority(),mode='input')

    def test_native_input_conversion_and_original_qualifiers_preserved(self):
        part,mp,trusted=self.files('2024-07',[self.row(START-2*HOUR,flags=('R','A','R')),self.row(START+HOUR)])
        charged=[]
        with patch.object(ad,'verified_local_bytes',side_effect=trusted):
            r=ad.load_assessment_usgs({'2024-07':part},mp,[(START-2*HOUR,START-2*HOUR)],authority(),read_budget=charged.append)
        self.assertEqual(len(r['observations']),1);o=r['observations'][0]
        self.assertEqual(o.value,100*CFS_TO_CMS);self.assertEqual(o.quality_flags,('R','A','R'))
        self.assertEqual(r['original_rows'][0]['source_qualifiers_json'],'["R", "A", "R"]')
        self.assertTrue(r['metadata'][0]['parquet_may_decode_full_row_groups']);self.assertEqual(sum(charged),part.stat().st_size+mp.stat().st_size)

    def test_verification_tail_exact_boundary(self):
        part,mp,trusted=self.files('2025-07',[self.row(LAST,is_core=False),self.row(LAST+HOUR,is_core=False)])
        with patch.object(ad,'verified_local_bytes',side_effect=trusted):
            r=ad.load_assessment_usgs({'2025-07':part},mp,[(LAST,LAST)],authority(verify=True),mode='verification')
        self.assertEqual([o.valid_at for o in r['observations']],[LAST])
        self.assertFalse(r['original_rows'][0]['is_core'])
        self.assertEqual(r['metadata'][0]['retained_noncore_rows'],1)
        self.assertEqual(r['metadata'][0]['exact_core_rows'],0)
        with self.assertRaises(ValueError):ad.load_assessment_usgs({'2025-07':part},mp,[(LAST,LAST+HOUR)],authority(verify=True),mode='verification')

    def test_invalid_numeric_and_nonhour_rules_ignore_temporal_core_flag(self):
        rows=[self.row(START-2*HOUR,numeric_valid=False),self.row(START-3*HOUR,is_core=False),self.row(START-2*HOUR-900)]
        part,mp,trusted=self.files('2024-07',rows)
        with patch.object(ad,'verified_local_bytes',side_effect=trusted):r=ad.load_assessment_usgs({'2024-07':part},mp,[(START-3*HOUR,START-2*HOUR)],authority())
        self.assertEqual(len(r['observations']),2);self.assertIsNone(r['observations'][0].value)
        self.assertEqual(r['original_rows'][0]['discharge_cfs'],100.)
        self.assertEqual(r['observations'][1].value,100*CFS_TO_CMS)
        self.assertFalse(r['original_rows'][1]['is_core'])

    def test_neighbor_month_padding_does_not_duplicate_native_owner(self):
        boundary=ad.utc_epoch('2024-08-01T00:00:00Z')
        part,mp,trusted=self.files('2024-07',[self.row(boundary-2*HOUR),self.row(boundary,is_core=False)])
        august=self.root/'2024-08.parquet'
        stream=pa.BufferOutputStream();pq.write_table(pa.Table.from_pylist([self.row(boundary,is_core=False)]),stream)
        raw=stream.getvalue().to_pybytes();august.write_bytes(raw)
        manifest=json.loads(mp.read_bytes());manifest['native_partitions'].append(dict(path=str(august),bytes=len(raw),sha256=sha256(raw)))
        body=json.dumps(manifest).encode();mp.write_bytes(body)
        native=ad.verified_local_bytes
        def trust(path,expected,**kw):return body if Path(path)==mp else native(path,expected,**kw)
        with patch.object(ad,'verified_local_bytes',side_effect=trust):
            r=ad.load_assessment_usgs({'2024-07':part,'2024-08':august},mp,[(boundary-2*HOUR,boundary)],authority())
        self.assertEqual([o.valid_at for o in r['observations']],[boundary-2*HOUR,boundary])
        self.assertEqual(r['original_rows'][1]['source_partition_month'],'2024-08')

    def test_duplicate_source_rows_or_wrong_units_fail(self):
        for rows in [[self.row(START-2*HOUR)]*2,[self.row(START-2*HOUR,source_units='m3/s')]]:
            part,mp,trusted=self.files('2024-07',rows)
            with patch.object(ad,'verified_local_bytes',side_effect=trusted):
                with self.assertRaises(ValueError):ad.load_assessment_usgs({'2024-07':part},mp,[(START-2*HOUR,START-2*HOUR)],authority())

    def test_parquet_row_bound_is_checked_before_numeric_table_decode(self):
        from types import SimpleNamespace
        part,mp,trusted=self.files('2024-07',[self.row(START-2*HOUR)])
        with patch.object(ad,'verified_local_bytes',side_effect=trusted),patch.object(pq,'ParquetFile',return_value=SimpleNamespace(metadata=SimpleNamespace(num_rows=30001))),patch.object(pq,'read_table',side_effect=AssertionError('Decoded too many rows')):
            with self.assertRaisesRegex(ValueError,'bounded reader'):ad.load_assessment_usgs({'2024-07':part},mp,[(START-2*HOUR,START-2*HOUR)],authority())


class AssessmentCycleGateTests(unittest.TestCase):
    def test_weights_gate_charge_and_original_parser_contract(self):
        with patch.object(ad,'_resident_charge',side_effect=AssertionError('I/O before gate')):
            with self.assertRaises(ValueError):ad.read_assessment_weights('absent',authority(enabled=False))
        values=np.full((3,957),1/957)
        with patch.object(ad,'_resident_charge') as charge,patch.object(ad,'load_candidate_weights',return_value=values) as load:
            weights,receipt=ad.read_assessment_weights('/synthetic/weights.csv',authority())
        load.assert_called_once_with('/synthetic/weights.csv',qualification=ad.DEVELOPMENT_QUALIFICATION)
        self.assertEqual(charge.call_args.kwargs['expected_bytes'],62601)
        self.assertFalse(weights.flags.writeable);self.assertEqual(receipt['weights_sha256'],ad.WEIGHTS_SHA256)
        self.assertTrue(receipt['original_candidate_status_unchanged']);self.assertFalse(receipt['partial_coverage_renormalized'])

    def test_bad_release_or_source_row_precedes_any_source_open(self):
        cid='20240730_18Z';row={'cycle':cid}
        with patch.object(ad,'_resident_charge',side_effect=AssertionError('Source opened')):
            with self.assertRaises(ValueError):ad.load_assessment_cycle({'cycle_id':cid},row,authority([cid],enabled=False,rows={cid:row}))
            with self.assertRaises(ValueError):ad.load_assessment_cycle({'cycle_id':cid},row,authority([cid]))
            with self.assertRaises(ValueError):ad.load_assessment_cycle({'cycle_id':'20250731_00Z'},row,authority([cid]))

    def test_disabled_verification_cannot_be_constructed_as_released(self):
        with self.assertRaises(ValueError):ad.AssessmentReadAuthority('1'*64,'2'*64,'3'*64,'4'*64,inputs_enabled=True,verification_enabled=True)

    def test_cycle_calendar_boundaries(self):
        self.assertEqual(ad.assessment_cycle_reference('20240730_06Z'),START-18*HOUR)
        self.assertEqual(ad.assessment_cycle_reference('20250730_18Z'),END-6*HOUR)
        for cid in ['20240730_00Z','20250731_00Z','20240801_01Z','20240230_00Z']:
            with self.assertRaises(ValueError):ad.assessment_cycle_reference(cid)

    def fixture_archive(self,root,*,temperature=False,mutation=None):
        cid='20240801_00Z';reference=ad.assessment_cycle_reference(cid);records=[]
        n=36 if temperature else 0;values=np.full((n,957),280.);bits=np.zeros((n,120),dtype='u1')
        for i in range(n):
            records.append(dict(record_index=i,group='temperature',lead=i+1,units='K',field_source_id='T',
                source_reference_UTC=iso(reference),source_valid_UTC=iso(reference+(i+1)*HOUR),
                source_values_sha256_float64_le=sha256(values[i].astype('<f8').tobytes()),
                missing_bits_sha256=sha256(bits[i].tobytes()),missing_cells=0,APCP_interval_hours=None))
        if mutation:mutation(records,values,bits)
        available=['temperature'] if temperature else []
        groups=[]
        for group in [*ad.UNITS,'soil_moisture']:
            yes=group in available;groups.append(dict(group=group,status='AVAILABLE_VERIFIED' if yes else 'MISSING_SYNTHETIC_SOURCE',
              record_count=36 if yes else 0,leads=list(range(1,37)) if yes else []))
        av=dict(schema='hrrr-group-availability/v1',cycle=cid,data_complete=False,coverage_status='CONFIRMED_GAPS',
                expected_groups=[*ad.UNITS,'soil_moisture'],groups=groups)
        enc=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')).encode()
        members={'AVAILABILITY.json':enc(av),'FIELD_SOURCES.json':enc({'T':{'synthetic':True}} if temperature else {}),
                 'RAW_PROVENANCE.json':enc(records),'POST_FETCH_CACHE_PROVENANCE.json':b'{}'}
        if temperature:
            for name,array in [('raw_values.npy',values),('raw_missing_bits.npy',bits)]:
                stream=io.BytesIO();np.save(stream,array,allow_pickle=False);members[name]=stream.getvalue()
        manifest=dict(schema='HRRR-gap-aware-raw-cycle-acquisition/v1',cycle=cid,record_count=n,groups=available,
            expected_groups=[*ad.UNITS,'soil_moisture'],data_complete=False,coverage_status='CONFIRMED_GAPS',availability=av,
            raw_shape=[n,957] if temperature else None,mask_shape=[n,120] if temperature else None,assets={'grid_sha256':GRID_SHA256})
        manifest['files']={name:dict(bytes=len(raw),sha256=sha256(raw)) for name,raw in members.items()};members['CYCLE_MANIFEST.json']=enc(manifest)
        path=root/'synthetic.tar'
        with tarfile.open(path,'w:') as archive:
            for name,raw in members.items():
                info=tarfile.TarInfo(cid+'/'+name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        raw=path.read_bytes();bundle=dict(durable_uri='gs://synthetic-only/cycles/'+cid+'.tar',generation='123',
            bytes=len(raw),sha256=sha256(raw),verified=True)
        row=dict(cycle=cid,location_kind='bundle',data_complete=False,original_quality_flags_rewritten=False,
                 source_locations={'bundle':bundle})
        return dict(cycle_id=cid,archive_path=str(path)),row

    def test_zero_record_gap_uses_real_tar_authentication_without_invented_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,row=self.fixture_archive(Path(tmp).resolve())
            c,r=ad.load_assessment_cycle(spec,row,authority([spec['cycle_id']],rows={spec['cycle_id']:row}))
            self.assertEqual(c.values.shape,(0,957));self.assertEqual(r['numeric_records_decoded'],0)
            self.assertFalse(r['normalized_summary']['data_complete']);self.assertFalse(r['verification_target_arrays_read'])

    def test_assessment_raw_records_preserve_value_and_clock_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,row=self.fixture_archive(Path(tmp).resolve(),temperature=True)
            with patch.object(ad,'_verify_grid_and_clocks') as grid,patch.object(ad,'_verify_weather_crop') as crop:
                c,r=ad.load_assessment_cycle(spec,row,authority([spec['cycle_id']],rows={spec['cycle_id']:row}))
            self.assertEqual(c.values.shape,(36,957));self.assertTrue((c.values==280.).all())
            self.assertFalse(c.values.flags.writeable);grid.assert_called_once();crop.assert_called_once()
            self.assertEqual(r['normalized_summary']['groups_available'],['temperature'])

    def test_changed_record_clock_or_value_fails_before_source_crop_validation(self):
        mutations=[lambda r,v,b:r[0].update(source_valid_UTC=iso(START)),
                   lambda r,v,b:v.__setitem__((0,0),999.),
                   lambda r,v,b:b.__setitem__((0,119),128)]
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory() as tmp:
                spec,row=self.fixture_archive(Path(tmp).resolve(),temperature=True,mutation=mutate)
                with patch.object(ad,'_verify_grid_and_clocks'),patch.object(ad,'_verify_weather_crop') as crop:
                    with self.assertRaises(ValueError):ad.load_assessment_cycle(spec,row,authority([spec['cycle_id']],rows={spec['cycle_id']:row}))
                    crop.assert_not_called()

    def test_archive_hash_change_rejected_without_any_numeric_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,row=self.fixture_archive(Path(tmp).resolve());p=Path(spec['archive_path']);raw=p.read_bytes();p.write_bytes(b'x'+raw[1:])
            with patch.object(ad,'bounded_npy',side_effect=AssertionError('Decoded unauthenticated bytes')):
                with self.assertRaisesRegex(ValueError,'SHA256'):ad.load_assessment_cycle(spec,row,authority([spec['cycle_id']],rows={spec['cycle_id']:row}))


if __name__=='__main__':unittest.main()
