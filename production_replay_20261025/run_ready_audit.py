"""Run an already-frozen audit once all fixed-end inputs exist.

Only absolute wall-clock waiting loops are removed in memory. No clock is
spoofed, no frozen file is changed, and no strategy/evaluation statement is
altered. Dependency waits remain. This avoids waiting an extra20..40minutes
after settlement records and the immutable end prefix have already arrived.
"""
import ast
import importlib
import inspect
import json
import sys
import time
from .protocol import OUT,save,sha

ALLOWED={'full_forward_review','exact_forward_review','paced_forward_review',
         'delayed_forward_review','data_aware_forward_review','venue_feasibility',
         'live_replay_audit','qualification_review','lot_exact_forward_review'}
END=1792915500


def strip_absolute_waits(source):
    tree=ast.parse(source);removed=[]
    class RemoveWait(ast.NodeTransformer):
        def visit_While(self,node):
            expected=(isinstance(node.test,ast.Compare) and len(node.test.ops)==1
                and isinstance(node.test.ops[0],ast.Lt)
                and ast.unparse(node.test.left)=='time.time()'
                and len(node.test.comparators)==1
                and isinstance(node.test.comparators[0],ast.BinOp)
                and isinstance(node.test.comparators[0].op,ast.Add)
                and ast.unparse(node.test.comparators[0].left)=='END'
                and isinstance(node.test.comparators[0].right,ast.Constant)
                and isinstance(node.test.comparators[0].right.value,int))
            if not expected:return self.generic_visit(node)
            if node.orelse or any(not isinstance(s,ast.Expr) or not isinstance(s.value,ast.Call) or
                ast.unparse(s.value.func) not in {'save','time.sleep'} for s in node.body):
                raise ValueError('Refusing to remove a loop with evaluation work')
            removed.append(ast.unparse(node.test))
            return ast.copy_location(ast.Pass(),node)
    RemoveWait().visit(tree);ast.fix_missing_locations(tree)
    if not 1<=len(removed)<=2:raise ValueError('Unexpected scheduler shape')
    return tree,removed


def accelerate(module):
    source=inspect.getsource(module.run);tree,removed=strip_absolute_waits(source)
    original_hash=sha(module.__file__)
    exec(compile(tree,'<post-end-scheduling-only>','exec'),module.__dict__)
    assert sha(module.__file__)==original_hash
    return dict(module=module.__name__,source_file_sha256=original_hash,removed_absolute_waits=removed)


def main():
    name=sys.argv[1]
    if name not in ALLOWED:raise ValueError('Not a research audit')
    if time.time()<END+125:raise ValueError('Fixed period and capture flush have not ended')
    sealed=json.loads((OUT/'sealed_forward_capture/manifest.json').read_text())
    if sealed['end']!=END:raise ValueError('Wrong sealed period')
    status=json.loads((OUT/'settlement_check/status.json').read_text())
    if status['state']!='complete':raise ValueError('Settlement evidence is incomplete')
    if len(list((OUT/'forward_evaluation').glob('*_funding.json')))!=10:
        raise ValueError('Incomplete market settlement schedules')
    if name=='live_replay_audit':
        while True:
            f=OUT/'exact_forward_review/status.json'
            if f.exists() and json.loads(f.read_text()).get('state')=='complete':break
            if time.time()>END+3300:raise ValueError('Corrected incumbent unavailable')
            time.sleep(10)
    target=importlib.import_module('.'+name,__package__)
    parent=importlib.import_module('.full_forward_review',__package__)
    changes=[accelerate(parent)]
    if target is not parent:changes.append(accelerate(target))
    save(OUT/'scheduling_audit'/f'{name}.json',dict(at=time.time(),fixed_end=END,changes=changes,
        reason='All end-period public settlement records and immutable capture are complete. Remove scheduling delay only; dependency checks, numerical code and frozen source checks remain.',
        wall_clock_spoofed=False,source_files_modified=False,orders_sent=0))
    target.run()


if __name__=='__main__':main()
