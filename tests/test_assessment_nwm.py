"""Synthetic NWM preparation tests; no data files, model calls or release."""
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import unittest

from hydra_directflow.assessment_contract import NWM_REACHES, clock
from hydra_directflow.assessment_nwm import (
    NwmSourceQualification, normalize_nwm_row, index_nwm_rows, prepare_paired_matches,
    DECODED_REPRESENTATION, UNITS, NUMERIC_VALID, SOURCE_ABSENT, SOURCE_NULL,
    SOURCE_CSV_NULL, SOURCE_NONFINITE, SOURCE_INVALID_TEXT, SOURCE_DECLARED_FILL)
from hydra_directflow.reference import ASSESSMENT_START, ASSESSMENT_END, HOUR


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace('+00:00', 'Z')


class AssessmentNwmTests(unittest.TestCase):
    def source(self, **changes):
        values = dict(source_id='synthetic-owned-view', snapshot_sha256='a'*64,
                      parity_receipt_sha256='b'*64, units=UNITS,
                      representation=DECODED_REPRESENTATION)
        values.update(changes)
        return NwmSourceQualification(**values)

    def row(self, *, g='03161000', c=ASSESSMENT_START, h=1, flow='12.3400'):
        return dict(feature_id=str(NWM_REACHES[g]), ensemble='0',
                    reference_time=iso(c-2*HOUR), valid_time=iso(c+h*HOUR),
                    lead_hours=str(h+2), source_forecast_offset=str(h+2), streamflow=flow)

    def normalize(self, row=None, **changes):
        values=dict(gauge_id='03161000',origin=ASSESSMENT_START,origin_horizon=1,
                    source=self.source(),row_id='synthetic-row-1')
        values.update(changes)
        return normalize_nwm_row(self.row() if row is None else row,**values)

    def test_decoded_float_is_not_scaled_or_rounded(self):
        r=self.normalize()
        self.assertEqual(r.streamflow_m3_s,12.34)
        self.assertEqual(r.raw_streamflow,'12.3400')
        self.assertTrue(r.numeric_valid)
        self.assertEqual(r.numeric_status,NUMERIC_VALID)

    def test_zero_and_finite_negative_predictions_remain_valid(self):
        for raw,value in [('0.000',0.),('-12.5',-12.5),('-9999',-9999.),('1.25e+2',125.)]:
            r=self.normalize(self.row(flow=raw))
            self.assertEqual(r.streamflow_m3_s,value)
            self.assertTrue(r.numeric_valid)
            self.assertEqual(r.raw_streamflow,raw)

    def test_absent_null_and_empty_csv_are_distinct(self):
        row=self.row();del row['streamflow']
        absent=self.normalize(row);null=self.normalize(self.row(flow=None));blank=self.normalize(self.row(flow=''))
        self.assertEqual([r.numeric_status for r in [absent,null,blank]],
                         [SOURCE_ABSENT,SOURCE_NULL,SOURCE_CSV_NULL])
        self.assertFalse(absent.raw_streamflow_present)
        self.assertTrue(null.raw_streamflow_present)
        self.assertEqual(blank.raw_streamflow,'')
        for r in [absent,null,blank]:
            self.assertFalse(r.numeric_valid);self.assertIsNone(r.streamflow_m3_s)

    def test_nonfinite_tokens_and_float_overflow_are_recorded(self):
        for raw in ['NaN','-NaN','sNaN','Infinity','-inf','+Inf','1e9999']:
            r=self.normalize(self.row(flow=raw))
            self.assertEqual(r.numeric_status,SOURCE_NONFINITE)
            self.assertEqual(r.raw_streamflow,raw)
            self.assertIsNone(r.streamflow_m3_s)

    def test_malformed_text_is_retained_as_source_invalid(self):
        for raw in ['null','None','1_000','0x10',' 2.5','2.5 ','\t','1,000']:
            r=self.normalize(self.row(flow=raw))
            self.assertEqual(r.numeric_status,SOURCE_INVALID_TEXT)
            self.assertEqual(r.raw_streamflow,raw)

    def test_preconverted_values_are_rejected(self):
        for raw in [12.34,0,False,float('nan'),[],{}]:
            with self.assertRaises(ValueError):self.normalize(self.row(flow=raw))

    def test_fills_require_explicit_decoded_qualification(self):
        with self.assertRaises(ValueError):self.source(decoded_fill_values=('-9999',))
        with self.assertRaises(ValueError):self.source(fill_qualification_sha256='c'*64)
        for values in [('-9999','-9999.0'),('NaN',),('1e9999',),(-9999,),' -9999']:
            with self.assertRaises(ValueError):self.source(decoded_fill_values=values,fill_qualification_sha256='c'*64)

    def test_exact_declared_fill_is_not_a_guessed_negative_rule(self):
        s=self.source(decoded_fill_values=('-9999',),fill_qualification_sha256='c'*64)
        for raw in ['-9999','-9999.0','-9.999e3']:
            r=self.normalize(self.row(flow=raw),source=s)
            self.assertEqual(r.numeric_status,SOURCE_DECLARED_FILL)
            self.assertIsNone(r.streamflow_m3_s);self.assertEqual(r.raw_streamflow,raw)
        for raw in ['-999900','-12.5','0']:
            self.assertTrue(self.normalize(self.row(flow=raw),source=s).numeric_valid)

    def test_near_fill_decimal_is_not_masked_by_float_rounding(self):
        s=self.source(decoded_fill_values=('-9999',),fill_qualification_sha256='c'*64)
        r=self.normalize(self.row(flow='-9999.0000000000001'),source=s)
        self.assertTrue(r.numeric_valid)
        self.assertEqual(r.raw_streamflow,'-9999.0000000000001')

    def test_qualification_binds_representation_units_and_evidence(self):
        for changes in [dict(units='cfs'),dict(representation='PACKED_NETCDF_INT'),
                        dict(source_table='another_table'),dict(snapshot_sha256=''),
                        dict(parity_receipt_sha256='B'*64),dict(source_id='')]:
            with self.assertRaises(ValueError):self.source(**changes)
        s=self.source();self.assertEqual(len(s.identity),64)
        self.assertNotEqual(s.identity,self.source(snapshot_sha256='c'*64).identity)
        self.assertNotEqual(s.identity,self.source(parity_receipt_sha256='c'*64).identity)
        with self.assertRaises(FrozenInstanceError):s.units='cfs'

    def test_row_and_source_ids_and_original_offset_are_preserved(self):
        row=self.row();row['source_forecast_offset']='opaque-original-offset'
        r=self.normalize(row,row_id=125)
        self.assertEqual(r.row_id,125);self.assertEqual(r.source.source_id,'synthetic-owned-view')
        self.assertEqual(r.raw_source_forecast_offset,'opaque-original-offset')
        self.assertEqual(dict(r.raw_metadata)['reference_time'],row['reference_time'])
        with self.assertRaises(FrozenInstanceError):r.streamflow_m3_s=1
        for ident in ['',None,False,-1,1.0]:
            with self.assertRaises(ValueError):self.normalize(row,row_id=ident)

    def test_all_gauges_first_last_origins_h1_h16(self):
        for g in NWM_REACHES:
            for c in [ASSESSMENT_START,ASSESSMENT_END-HOUR]:
                for h in [1,16]:
                    r=self.normalize(self.row(g=g,c=c,h=h),gauge_id=g,origin=c,origin_horizon=h)
                    self.assertEqual(r.key,(g,c,h,c+h*HOUR))
        self.assertEqual(clock(ASSESSMENT_END-HOUR)['valid_times'][-1],ASSESSMENT_END+17*HOUR)

    def test_native_integer_metadata_is_also_accepted(self):
        row=self.row();row.update(feature_id=NWM_REACHES['03161000'],ensemble=0,
            reference_time=ASSESSMENT_START-2*HOUR,valid_time=ASSESSMENT_START+HOUR,lead_hours=3)
        self.assertTrue(self.normalize(row).numeric_valid)

    def test_metadata_validation_precedes_any_flow_access(self):
        class Poison(dict):
            def __getitem__(self,key):
                if key=='streamflow':raise AssertionError('Flow accessed before metadata rejection')
                return super().__getitem__(key)
        for field,value in [('feature_id',str(NWM_REACHES['03164000'])),('ensemble','1'),
                            ('reference_time',iso(ASSESSMENT_START-HOUR)),
                            ('valid_time',iso(ASSESSMENT_START+2*HOUR)),('lead_hours','1')]:
            row=Poison(self.row());row[field]=value
            with self.assertRaises(ValueError):self.normalize(row)
        for h in [17,18]:
            with self.assertRaises(ValueError):self.normalize(Poison(self.row(h=h)),origin_horizon=h)

    def test_older_issue_and_noncanonical_metadata_fail(self):
        for field,value in [('reference_time',iso(ASSESSMENT_START-3*HOUR)),
                            ('reference_time','2024-07-30T22:00:00'),
                            ('reference_time','2024-07-30T23:00:00+01:00'),
                            ('valid_time',ASSESSMENT_START+HOUR+1),
                            ('valid_time',float(ASSESSMENT_START+HOUR)),
                            ('lead_hours','3.0'),('lead_hours',True),('ensemble',False),
                            ('feature_id','06892192')]:
            row=self.row();row[field]=value
            with self.assertRaises(ValueError):self.normalize(row)

    def test_unknown_extra_fields_are_not_read(self):
        class Selective(dict):
            def __getitem__(self,key):
                if key=='verification_target':raise AssertionError('Target value accessed')
                return super().__getitem__(key)
            def __iter__(self):raise AssertionError('Unexpected enumeration of source row')
        row=Selective(self.row());row['verification_target']=object()
        self.assertTrue(self.normalize(row).numeric_valid)

    def test_duplicate_keys_never_silently_merge(self):
        r=self.normalize()
        with self.assertRaisesRegex(ValueError,'requires explicit lineage'):index_nwm_rows([r,r])
        variants=[self.normalize(self.row(flow='12.34')),self.normalize(row_id='different-row'),
                  self.normalize(source=self.source(source_id='another-source')),
                  self.normalize(source=self.source(snapshot_sha256='c'*64))]
        row=self.row();row['source_forecast_offset']='changed';variants.append(self.normalize(row))
        for other in variants:
            with self.assertRaisesRegex(ValueError,'conflicts'):index_nwm_rows([r,other])

    def test_index_is_immutable_and_keeps_invalid_rows(self):
        r=self.normalize(self.row(flow='NaN'));indexed=index_nwm_rows([r])
        self.assertEqual(indexed[r.key].numeric_status,SOURCE_NONFINITE)
        with self.assertRaises(TypeError):indexed[r.key]=r

    def test_index_rejects_forged_disposition_or_rescaling(self):
        r=self.normalize()
        for bad in [replace(r,numeric_status=SOURCE_NONFINITE,streamflow_m3_s=None),
                    replace(r,streamflow_m3_s=0.1234),
                    replace(r,streamflow_m3_s=float('nan')),
                    replace(r,raw_streamflow_present=False)]:
            with self.assertRaises(ValueError):index_nwm_rows([bad])

    def test_index_revalidates_crosswalk_and_horizon_after_restore(self):
        r=self.normalize();c=ASSESSMENT_START
        bad_meta=tuple((k,str(NWM_REACHES['03164000']) if k=='feature_id' else v)
                       for k,v in r.raw_metadata)
        for bad in [replace(r,raw_metadata=bad_meta),
                    replace(r,key=('03161000',c,18,c+18*HOUR)),
                    replace(r,key=('03161000',c,1,float(c+HOUR))),
                    replace(r,raw_metadata=r.raw_metadata[:-1])]:
            with self.assertRaises(ValueError):index_nwm_rows([bad])

    def test_nwm_invalid_or_missing_never_changes_hydra_forecasts(self):
        c=ASSESSMENT_START;hydra=tuple(('03161000',c,h,c+h*HOUR) for h in range(1,19))
        rows=[self.normalize(self.row(h=h,flow='NaN' if h==3 else '-1.0'),origin_horizon=h,row_id=h) for h in range(1,17) if h!=2]
        result=prepare_paired_matches(hydra,rows,hydra)
        self.assertEqual(result['independent_hydra_forecast_keys'],frozenset(hydra))
        self.assertEqual(result['independent_hydra_evaluable_keys'],frozenset(hydra))
        self.assertEqual(len(result['paired_evaluable_keys']),14)
        self.assertEqual(result['nwm_source_invalid_keys'],frozenset([hydra[2]]))
        self.assertEqual(dict(result['nwm_source_invalid_counts']),{SOURCE_NONFINITE:1})
        self.assertIn(hydra[-1],result['independent_hydra_evaluable_keys'])
        self.assertNotIn(hydra[-1],result['paired_evaluable_keys'])

    def test_target_missingness_only_changes_evaluable_sets(self):
        c=ASSESSMENT_START;hydra=tuple(('03161000',c,h,c+h*HOUR) for h in range(1,19))
        rows=[self.normalize(self.row(h=h),origin_horizon=h,row_id=h) for h in range(1,17)]
        targets=[key for key in hydra if key[2]!=3]
        result=prepare_paired_matches(hydra,rows,targets)
        self.assertEqual(len(result['paired_evaluable_keys']),15)
        self.assertEqual(len(result['independent_hydra_forecast_keys']),18)
        self.assertEqual(len(result['independent_hydra_evaluable_keys']),17)
        self.assertEqual(len(result['nwm_numeric_valid_keys']),16)


if __name__=='__main__':unittest.main()
