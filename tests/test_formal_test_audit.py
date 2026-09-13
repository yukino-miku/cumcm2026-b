"""对正式测试审计作篡改/缺失反例，避免只凭汇总数接受错误轨迹或完成标志。"""
import sys
import zipfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_formal_tests import ARCHIVE, MODELS, SUMMARY, load, read, formal_header, pair_protocol, audit_geometry, collect, verify_archives, summarize


def test_serialized_formal_records_roundtrip():
    # 包括含整数频道键的诊断字典，JSON往返不应造成核验误报。
    runs = collect(verify_archives(False), True)
    assert len(runs) == 6
    assert summarize(runs) == read(SUMMARY)


def raw_first():
    with zipfile.ZipFile(ARCHIVE/'q3form.zip') as z:
        items=[load(x) for x in z.read('http/F3-1/请求响应日志.jsonl').decode('utf-8').splitlines()]
        result=load(z.read('http/F3-1/运行结果.json'))
    return items,result


@pytest.mark.parametrize('mutation',['missing_response','changed_time','changed_bearing'])
def test_protocol_corruption_is_rejected(mutation):
    items,result=raw_first()
    if mutation=='missing_response':
        items.pop(next(i for i,x in enumerate(items) if x['类型']=='响应'))
    elif mutation=='changed_time':
        next(x for x in items if x['类型']=='响应')['response']['virtual_time_s']+=1
    else:
        response=next(x for x in items if x['类型']=='响应' and 'svd_deg' in x['response'])
        response['response']['svd_deg']+=.5
    with pytest.raises(AssertionError):
        pair_protocol(items,result)


def test_formal_magic_does_not_accept_practice_package():
    with zipfile.ZipFile(ARCHIVE/'q3form.zip') as z:
        raw=z.read(next(n for n in z.namelist() if n.endswith('.jlog')))
    assert formal_header(raw)['problem_no']==3
    with pytest.raises(AssertionError):
        formal_header(b'JMBPLOG1'+raw[8:])


def test_missing_actual_negative_observations_invalidates_absence_proof():
    result=read(MODELS/'F3-1.json.gz')['结果']
    c=result['已证明不存在频道'][0]
    result['动作记录']=[e for e in result['动作记录'] if not(e['动作']=='检测' and e['频道']==c)]
    with pytest.raises(AssertionError):
        audit_geometry(result,3)


def test_13_cleared_directional_run_does_not_prove_all_clear():
    result=read(MODELS/'F4-1.json.gz')['结果']
    assert not audit_geometry(result,4)['模型全清证据']
    result['全部完成证据']=True
    with pytest.raises(AssertionError):
        audit_geometry(result,4)
