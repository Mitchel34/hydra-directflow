#!/usr/bin/env python3
"""Select the fastest demonstrated B64 backend policy without running models."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from plan_all_mps_sequence import load_sources,comparison,PINS
from plan_seed_sequence import digest,require,FAMILIES,ARMS
from plan_scientific_workload import _row_rates


def build_plan(parent,full,controls):
    require(parent['plan_sha256']==digest({k:v for k,v in parent.items() if k!='plan_sha256'}),
            'Unchanged seed-order parent required')
    compared=comparison(full)
    devices={row['architecture']:('cpu' if row['CPU_mean_copy_plus_update_seconds']
                                  <=row['MPS_mean_copy_plus_update_seconds'] else 'mps') for row in compared}
    result=deepcopy(parent);result.pop('plan_sha256')
    result.update(schema='hydra-local-seed-sequence/v3',status='USER_REQUESTED_FASTEST_TESTED_LOCAL_POLICY_PREPARED_NOT_EXECUTED',
        parent_plan_sha256=parent['plan_sha256'],evidence_sha256=PINS,
        device_policy='FASTEST_OBSERVED_FULL_L168_B64_MEAN_COPY_PLUS_UPDATE_BY_ARCHITECTURE',
        selected_devices=devices,device_comparison=compared,
        exact_tie_policy='RETAIN_CPU_WITHOUT_CLAIMING_A_MEASURED_SPEED_ADVANTAGE',
        required_worker_environment={'PYTORCH_ENABLE_MPS_FALLBACK':'0'},
        backend_failure_policy='FAIL_AND_PRESERVE_ATTEMPT_NO_AUTOMATIC_BACKEND_SWITCH',
        checkpoint_backend_migration_allowed=False,
        faster_for_every_possible_batch_or_feature_shape_proven=False,
        all_MPS_alternative_is_active=False)
    # The original plan already used this measured policy. Reject a changed
    # winner rather than attaching its old per-fit rates to a different device.
    for job in result['jobs']:
        require(job['device']==devices[job['architecture']], 'Changed backend requires new timing attribution')
        if job['input_arm']=='full' and job['history_hours']==168:
            rows=[x for x in full['cases'] if x['architecture']==job['architecture']
                  and x['device']==job['device'] and x['batch_size']==64]
        else:
            mode='Q_ONLY_L168' if job['input_arm']=='q_only' else 'FULL_L24'
            rows=[x for x in controls['cases'] if x['architecture']==job['architecture']
                  and x['device']==job['device'] and x['batch_size']==64 and x['input_mode']==mode]
        require(len(rows)==1 and rows[0]['environment']['device_fallback_enabled'] is False,
                'One matched actual-backend timing cell without fallback required')
        rate=_row_rates(rows[0]);require(rate['device']==job['device'], 'Backend/timing identity differs')
        job['timing_measurement_case_id']=rate['measurement_case_id']
        job['timing_status']='MATCHED_GEOMETRY_ENGINEERING_RATE_NOT_FULL_SCIENTIFIC_FIT_DURATION'
    result['required_main_backend_preflight_geometries']=[dict(architecture=a,input_arm=arm,history_hours=length,
        batch_size=64,device=devices[a],precision='FP32',actual_B64_current_trainer_recovery_and_best_inference_qualified=False)
        for a in FAMILIES for arm,length in ARMS]
    result['remaining_gates']=[*parent['remaining_gates'],
        'Qualify current primary trainer recovery and best-state inference at the actual planned B64 backend geometries',
        'Set disabled MPS CPU fallback before fresh worker imports; never silently change a fit backend']
    result['implementation_status']='METADATA_ROSTER_ONLY_NOT_A_DURABLE_EXECUTION_CONTROLLER'
    result['plan_sha256']=digest(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=build_plan(*load_sources())
    with args.output.open('x') as out:json.dump(result,out,indent=2,sort_keys=True);out.write('\n')
    print(json.dumps({k:result[k] for k in ('status','selected_devices','plan_sha256')}))


if __name__=='__main__':main()
