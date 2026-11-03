"""Synthetic geometry/weight tests. No user source files or scientific fits."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/audit_spatial_support.py'
SPEC=importlib.util.spec_from_file_location('spatial_support_audit_tests',PATH)
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


def fixture_rows():
    rows=[]
    for gauge,count in zip(AUDIT.GAUGES,(84,387,43)):
        for index in range(count):
            i,j=index%29+1364,index//29+478
            rows.append(dict(gauge_id=gauge,crop_index=str(index),native_i=str(i),native_j=str(j),native_flat_index=str(j*1799+i),
                intersection_area_m2='1',full_basin_area_m2=str(count),weight=str(1/count),status='CANDIDATE_SCIENTIFIC_HOLD',admitted='false'))
    return rows


def encode(rows):
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


class SpatialAuditTests(unittest.TestCase):
    def test_exact_fraction_formula_and_original_holds_preserved(self):
        result=AUDIT.parse_weights(encode(fixture_rows()))
        self.assertEqual([len(result[g]) for g in AUDIT.GAUGES],[84,387,43])
        self.assertAlmostEqual(sum(v['weight'] for v in result['03161000'].values()),1)

    def test_duplicate_cell_shift_and_changed_admission_rejected(self):
        for field,value in [('crop_index','999'),('native_i','1370'),('admitted','true'),('weight','0.5')]:
            rows=fixture_rows();rows[0][field]=value
            with self.assertRaises(ValueError):AUDIT.parse_weights(encode(rows))
        rows=fixture_rows();rows[1]=dict(rows[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):AUDIT.parse_weights(encode(rows))

    def test_resident_read_verifies_hash_and_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'metadata.json';p.write_bytes(b'{}')
            self.assertEqual(AUDIT.resident_bytes(p,AUDIT.sha(b'{}')),b'{}')
            with self.assertRaisesRegex(ValueError,'SHA'):AUDIT.resident_bytes(p,'0'*64)
            q=Path(directory)/'link.json';q.symlink_to(p)
            with self.assertRaisesRegex(ValueError,'resident regular'):AUDIT.resident_bytes(q,AUDIT.sha(b'{}'))

    def test_dataless_file_refused_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'metadata.json';p.write_bytes(b'{}');info=p.lstat()
            fake=type('Stat',(),dict(st_mode=info.st_mode,st_flags=0x40000000))()
            with patch.object(Path,'lstat',return_value=fake),patch.object(AUDIT.os,'open',side_effect=AssertionError('Dataless opened')):
                with self.assertRaisesRegex(ValueError,'hydration'):AUDIT.resident_bytes(p,AUDIT.sha(b'{}'))

    def test_cell_edges_share_identical_boundary_vertices(self):
        first=AUDIT.cell_ring(0,0,125);second=AUDIT.cell_ring(3000,0,125)
        self.assertEqual({p for p in first if p[0]==1500},{p for p in second if p[0]==1500})
        self.assertEqual(len(first),96)
        with self.assertRaises(ValueError):AUDIT.cell_ring(0,0,128)

    def test_active_missing_cell_policy_does_not_renormalize(self):
        import numpy as np
        from hydra_directflow.local_data import basin_reduce
        weights=np.array([[1e-7,1-1e-7,0.]])
        values=np.array([[999.,10.,999.],[999.,10.,999.]])
        present=np.array([[False,True,True],[True,True,False]],bool)
        means,masks,coverage=basin_reduce(values,present,weights)
        self.assertFalse(masks[0,0]);self.assertTrue(np.isnan(means[0,0]));self.assertAlmostEqual(coverage[0,0],1-1e-7)
        self.assertTrue(masks[1,0]);self.assertAlmostEqual(means[1,0],999e-7+10*(1-1e-7));self.assertAlmostEqual(coverage[1,0],1)

    @unittest.skipUnless(importlib.util.find_spec('shapely'),'GIS runtime needed for synthetic geometry')
    def test_intersection_preserves_full_denominator_and_detects_missing_cell(self):
        from shapely.geometry import box
        basin=box(.2,0,2,1);cells=[box(0,0,1,1),box(1,0,2,1)]
        supplied={0:dict(area=.8,basin=1.8,weight=.8/1.8),1:dict(area=1,basin=1.8,weight=1/1.8)}
        result,_=AUDIT.compare_intersections(basin,cells,supplied)
        self.assertEqual(result['uncovered_area_m2'],0)
        with self.assertRaisesRegex(ValueError,'footprint'):AUDIT.compare_intersections(basin,cells,supplied={0:supplied[0]})

    @unittest.skipUnless(importlib.util.find_spec('shapely'),'GIS runtime needed for synthetic geometry')
    def test_original_empty_upstream_component_is_preserved_without_filling_holes(self):
        from shapely.geometry import Polygon,Point,mapping
        hole=Polygon([(0,0),(3,0),(3,3),(0,3)],holes=[[(1,1),(2,1),(2,2),(1,2)]])
        features=[dict(type='Feature',properties={'scope':scope},geometry=mapping(geom)) for scope,geom in
                  [('pourpoint',Point(.5,.5)),('split_catchment',hole),('adjoint_catchment',Polygon()),('upstream_basin',Polygon())]]
        geo,_,scopes=AUDIT.polygon_geometry(dict(type='FeatureCollection',features=features),'03161000')
        self.assertEqual(geo.area,8);self.assertFalse(geo.covers(Point(1.5,1.5)));self.assertEqual(scopes['upstream_basin'],1)

    @unittest.skipUnless(importlib.util.find_spec('shapely'),'GIS runtime needed for synthetic geometry')
    def test_gages_identity_and_invalid_polygon_not_repaired(self):
        from shapely.geometry import box,mapping
        value=dict(type='Feature',id='03164000',properties={'gage_id':'03164000'},geometry=mapping(box(0,0,1,1)))
        geo,point,_=AUDIT.polygon_geometry(value,'03164000');self.assertEqual(geo.area,1);self.assertIsNone(point)
        with self.assertRaisesRegex(ValueError,'identity'):AUDIT.polygon_geometry({**value,'id':'other'},'03164000')
        value['geometry']=dict(type='Polygon',coordinates=[[(0,0),(1,1),(0,1),(1,0),(0,0)]])
        with self.assertRaisesRegex(ValueError,'Invalid source'):AUDIT.polygon_geometry(value,'03164000')


if __name__=='__main__':unittest.main()
